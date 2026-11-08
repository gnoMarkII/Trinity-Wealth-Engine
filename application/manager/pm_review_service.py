"""Application use-case service for Morning PM Review & Daily Delta Packet.

Architecture Rule (Layer 2 - Application):
Orchestrates 4-quadrant institutional morning delta briefing.
Enforces institutional rule: Initial packet status is strictly REVIEW_REQUIRED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional
import uuid

from application.manager.ports import PMReviewSignoffRepoPort
from application.thesis.thesis_service import ThesisApplicationService
from application.macro.vintage_service import MacroVintageApplicationService
from core.equity_analytics.factor_exposure import (
    calculate_descriptive_factor_beta,
    calculate_portfolio_exposure,
    MissingPriceError,
    PositionHolding,
    PositionSide,
    ReturnObservation,
)
from core.portfolio_advisory.guardrails import check_late_run_guard, check_missing_price_policy, resolve_holding_currency_and_market
from core.portfolio_advisory.review_sampling import ReviewSampleItem, select_review_sample
from core.portfolio_advisory.state_integrity import (
    StateIntegrityReport,
    verify_state_integrity,
    StateManifest,
)
from core.resilience.failure_atlas import (
    FailureClass,
    FailureIncident,
    triage_failure,
)

log = logging.getLogger(__name__)


class SignoffPersistenceUnavailable(RuntimeError):
    """Raised when required sign-off evidence cannot be durably persisted and read back."""


@dataclass
class MorningReviewPacket:
    """The complete 4-quadrant Morning Delta Briefing for the Portfolio Manager."""
    review_date: str                     # YYYY-MM-DD
    status: str                          # "REVIEW_REQUIRED" or "SIGNED_OFF"
    portfolio_delta: Dict[str, Any]
    macro_delta: Dict[str, Any]
    thesis_exceptions: List[Dict[str, Any]]
    pending_decisions: List[Dict[str, Any]]
    packet_sha256: str
    signoff_info: Optional[Dict[str, Any]] = None
    thesis_coverage: Optional[Dict[str, Any]] = None
    quarantined_holdings: List[Dict[str, Any]] = field(default_factory=list)
    review_sample: List[Dict[str, Any]] = field(default_factory=list)
    run_quality: str = "ON_TIME"         # "ON_TIME" or "LATE_RUN_STALE"
    drift_minutes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_date": self.review_date,
            "status": self.status,
            "portfolio_delta": self.portfolio_delta,
            "macro_delta": self.macro_delta,
            "thesis_exceptions": self.thesis_exceptions,
            "pending_decisions": self.pending_decisions,
            "packet_sha256": self.packet_sha256,
            "signoff_info": self.signoff_info,
            "thesis_coverage": self.thesis_coverage or {},
            "quarantined_holdings": self.quarantined_holdings,
            "review_sample": self.review_sample,
            "run_quality": self.run_quality,
            "drift_minutes": self.drift_minutes,
        }


class PMReviewApplicationService:
    """Assembles morning delta briefings and records human PM sign-off decisions."""

    @staticmethod
    def _positive_decimal(value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        try:
            parsed = Decimal(str(value))
            return parsed if parsed.is_finite() and parsed > Decimal("0") else None
        except Exception:
            return None

    @staticmethod
    def _pending_exception_ids(exception_rows: List[Dict[str, Any]]) -> List[str]:
        pending_ids: List[str] = []
        seen_ids = set()
        for row in exception_rows:
            if not isinstance(row, dict):
                raise ValueError("Current review packet contains a malformed exception.")
            exception_id = row.get("exception_id")
            if not isinstance(exception_id, str) or not exception_id.strip():
                raise ValueError("Current review packet contains an exception without a valid ID.")
            exception_id = exception_id.strip()
            if exception_id in seen_ids:
                raise ValueError("Current review packet contains duplicate exception IDs.")
            seen_ids.add(exception_id)
            exception_status = str(row.get("status") or "PENDING").strip().upper()
            if exception_status not in ("ACKNOWLEDGED", "REVISED"):
                pending_ids.append(exception_id)
        return pending_ids

    def __init__(
        self,
        signoff_repo: PMReviewSignoffRepoPort,
        thesis_service: Optional[ThesisApplicationService] = None,
        portfolio_reader: Optional[Callable[..., Any]] = None,
        macro_reader: Optional[Callable[..., Any]] = None,
        advisory_reader: Optional[Callable[..., Any]] = None,
        performance_reader: Optional[Callable[..., Any]] = None,
        benchmark_return_reader: Optional[Callable[..., Any]] = None,
        state_integrity_checker: Optional[Callable[[], Optional[StateIntegrityReport]]] = None,
        macro_vintage_service: Optional[MacroVintageApplicationService] = None,
        catalyst_service: Optional[Any] = None,
        routine_monitor_service: Optional[Any] = None,
    ) -> None:
        self._signoff_repo = signoff_repo
        self._thesis_service = thesis_service
        self._portfolio_reader = portfolio_reader
        self._macro_reader = macro_reader
        self._advisory_reader = advisory_reader
        self._performance_reader = performance_reader
        self._benchmark_return_reader = benchmark_return_reader
        self._state_integrity_checker = state_integrity_checker
        self._macro_vintage_service = macro_vintage_service
        self._catalyst_service = catalyst_service
        self._routine_monitor_service = routine_monitor_service

    def _compute_packet_checksum(
        self,
        review_date: str,
        portfolio_delta: Dict[str, Any],
        macro_delta: Dict[str, Any],
        thesis_exceptions: List[Dict[str, Any]],
        pending_decisions: List[Dict[str, Any]],
        thesis_coverage: Optional[Dict[str, Any]] = None,
        quarantined_holdings: Optional[List[Dict[str, Any]]] = None,
        review_sample: Optional[List[Dict[str, Any]]] = None,
        run_quality: str = "ON_TIME",
        drift_minutes: int = 0,
    ) -> str:
        """Computes deterministic SHA-256 checksum over the 4 quadrants."""
        payload = {
            "review_date": review_date,
            "portfolio_delta": portfolio_delta,
            "macro_delta": macro_delta,
            "thesis_exceptions": thesis_exceptions,
            "pending_decisions": pending_decisions,
            "thesis_coverage": thesis_coverage or {},
            "quarantined_holdings": quarantined_holdings or [],
            "review_sample": review_sample or [],
            "run_quality": run_quality,
            "drift_minutes": drift_minutes,
        }
        canonical_str = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def build_morning_packet(
        self,
        portfolio_id: str = "default",
        review_date: Optional[str] = None,
    ) -> MorningReviewPacket:
        """Assembles the 2-minute delta packet across 4 quadrants."""
        now_utc = datetime.now(timezone.utc)
        target_date = review_date or now_utc.strftime("%Y-%m-%d")
        try:
            target_end = datetime.strptime(target_date, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc
            )
        except ValueError as exc:
            raise ValueError("review_date must use YYYY-MM-DD") from exc
        packet_cutoff_utc = min(now_utc, target_end).isoformat().replace("+00:00", "Z")

        # 1. Quadrant 1: Portfolio Delta (with Ch. 8 Factor Exposure & Beta)
        is_blocked = False
        portfolio_delta = {
            "total_nav": 0.0,
            "daily_pnl_value": None,
            "daily_pnl_pct": None,
            "daily_pnl_status": "UNAVAILABLE",
            "daily_pnl_reason": "INSUFFICIENT_AS_OF_SNAPSHOTS",
            "daily_pnl_flow_coverage": "MISSING_FLOW_HISTORY",
            "cash_weight_pct": 0.0,
            "bucket_drift_detected": False,
            "top_mover": {"ticker": "N/A", "change_pct": 0.0},
            "gross_market_value": 0.0,
            "net_market_value": 0.0,
            "long_market_value": 0.0,
            "short_market_value": 0.0,
            "long_exposure_pct": 0.0,
            "short_exposure_pct": 0.0,
            "net_exposure_pct": 0.0,
            "portfolio_beta": None,
            "beta_benchmark": "SPY",
            "factor_observations_count": 0,
            "eligible_positions_count": 0,
        }
        raw_holdings: List[Any] = []
        quarantined_items: List[Dict[str, Any]] = []
        if self._portfolio_reader:
            try:
                p_state = self._portfolio_reader(portfolio_id=portfolio_id, refresh_prices=False)
                if p_state:
                    total_nav = 0.0
                    if getattr(p_state, "summary", None):
                        summ = p_state.summary
                        total_nav = float(getattr(summ, "total_value_thb", None) or getattr(summ, "total_value", 0.0))
                        if total_nav > 0:
                            portfolio_delta["total_nav"] = total_nav

                    holdings = getattr(p_state, "holdings", []) or []
                    raw_holdings = holdings
                    if holdings and total_nav > 0:
                        cash_thb = sum(
                            float(getattr(h, "market_value_thb", 0.0) or 0.0)
                            for h in holdings
                            if "CASH" in str(getattr(h, "symbol", "")).upper()
                            or str(getattr(h, "asset_type", "")).lower() == "cash"
                        )
                        portfolio_delta["cash_weight_pct"] = round((cash_thb / total_nav) * 100.0, 2)

                        valid_movers = [
                            h for h in holdings
                            if (getattr(h, "daily_change_pct", None) is not None or getattr(h, "change_pct", None) is not None)
                            and "CASH" not in str(getattr(h, "symbol", "")).upper()
                        ]
                        if valid_movers:
                            valid_movers.sort(key=lambda h: abs(float(getattr(h, "daily_change_pct", None) or getattr(h, "change_pct", 0.0) or 0)), reverse=True)
                            top = valid_movers[0]
                            val = float(getattr(top, "daily_change_pct", None) or getattr(top, "change_pct", 0.0) or 0)
                            portfolio_delta["top_mover"] = {
                                "ticker": top.symbol,
                                "change_pct": round(val, 2),
                            }
                        else:
                            lifetime_movers = [
                                h for h in holdings
                                if getattr(h, "unrealized_pnl_percent", None) is not None
                                and "CASH" not in str(getattr(h, "symbol", "")).upper()
                            ]
                            if lifetime_movers:
                                lifetime_movers.sort(key=lambda h: abs(float(h.unrealized_pnl_percent or 0)), reverse=True)
                                top = lifetime_movers[0]
                                portfolio_delta["top_mover"] = {
                                    "ticker": top.symbol,
                                    "change_pct": round(float(top.unrealized_pnl_percent or 0), 2),
                                    "is_lifetime_fallback": True,
                                }

                        # Chapter 8: Factor Exposure (Gross / Net / Long / Short) & Hard-Fail Missing Price
                        pos_list: List[PositionHolding] = []
                        missing_price_quarantine_items: List[Dict[str, Any]] = []
                        for h in holdings:
                            s_name = str(getattr(h, "symbol", "")).strip().upper()
                            if not s_name or "CASH" in s_name:
                                continue
                            u = getattr(h, "units", None)
                            if u is not None:
                                try:
                                    dec_u = Decimal(str(u))
                                    if dec_u == Decimal("0"):
                                        continue  # Closed position, zero exposure
                                except Exception:
                                    continue
                            else:
                                quarantined_items.append({
                                    "symbol": s_name,
                                    "reason": f"Position '{s_name}' lacks units. Silently imputing 1.0 is prohibited (Chapter 8).",
                                    "guardrail_code": "MISSING_UNITS_EXCEPTION",
                                    "current_value_thb": 0.0,
                                })
                                continue

                            try:
                                dec_u = Decimal(str(u))
                            except Exception:
                                dec_u = Decimal("NaN")
                            if not dec_u.is_finite():
                                quarantined_items.append({
                                    "symbol": s_name,
                                    "reason": f"Position '{s_name}' has non-finite units.",
                                    "guardrail_code": "INVALID_UNITS_EXCEPTION",
                                    "current_value_thb": 0.0,
                                })
                                continue

                            mv_thb = getattr(h, "market_value_thb", None)
                            dec_p: Optional[Decimal] = None
                            try:
                                mv_dec = Decimal(str(mv_thb)) if mv_thb is not None else None
                                if mv_dec is not None and mv_dec.is_finite() and mv_dec != 0:
                                    dec_p = abs(mv_dec) / abs(dec_u)
                            except Exception:
                                dec_p = None

                            if dec_p is None:
                                quoted_thb = self._positive_decimal(getattr(h, "current_price_thb", None))
                                if quoted_thb is not None:
                                    dec_p = quoted_thb
                                else:
                                    _, native_currency = resolve_holding_currency_and_market(h)
                                    if native_currency == "USD":
                                        native_price = self._positive_decimal(getattr(h, "current_price_usd", None))
                                        approved_fx = self._positive_decimal(getattr(h, "fx_rate", None))
                                        if native_price is not None and approved_fx is not None:
                                            dec_p = native_price * approved_fx
                                    else:
                                        dec_p = self._positive_decimal(getattr(h, "current_price", None))

                            side = PositionSide.SHORT if dec_u < 0 else PositionSide.LONG
                            pos_list.append(PositionHolding(symbol=s_name, shares=dec_u, price=dec_p, side=side))

                        portfolio_delta["eligible_positions_count"] = len([p for p in pos_list if p.price is not None and p.price > Decimal("0") and p.shares != Decimal("0")])

                        if pos_list:
                            try:
                                # Chapter 8 Hard-Fail: fail immediately if any holding lacks approved price
                                exposure = calculate_portfolio_exposure(pos_list, fail_on_missing_price=True)
                                portfolio_delta["gross_market_value"] = float(exposure.gross_market_value)
                                portfolio_delta["net_market_value"] = float(exposure.net_market_value)
                                portfolio_delta["long_market_value"] = float(exposure.long_market_value)
                                portfolio_delta["short_market_value"] = float(exposure.short_market_value)
                                portfolio_delta["long_exposure_pct"] = float(exposure.long_exposure_pct)
                                portfolio_delta["short_exposure_pct"] = float(exposure.short_exposure_pct)
                                portfolio_delta["net_exposure_pct"] = float(exposure.net_exposure_pct)
                            except MissingPriceError as mp_err:
                                # Chapter 8 DEL Seeded Missing Test:
                                # Never silently impute zero or guess. Name the missing security in quarantine.
                                log.warning("Hard-fail on missing price for %s: quarantining position", mp_err.symbol)
                                missing_price_quarantine_items.append({
                                    "symbol": mp_err.symbol,
                                    "reason": f"Position '{mp_err.symbol}' lacks an approved as-of price. Silently imputing zero is strictly prohibited (Chapter 8).",
                                    "guardrail_code": "MISSING_PRICE_EXCEPTION",
                                    "current_value_thb": 0.0,
                                })
                                portfolio_delta["missing_price_exception"] = f"Missing approved price for {mp_err.symbol}"
                                # Calculate exposure over verified subset with clear auditability
                                valid_subset = [p for p in pos_list if p.price is not None and p.price > 0]
                                if valid_subset:
                                    sub_exp = calculate_portfolio_exposure(valid_subset, fail_on_missing_price=False)
                                    portfolio_delta["gross_market_value"] = float(sub_exp.gross_market_value)
                                    portfolio_delta["net_market_value"] = float(sub_exp.net_market_value)
                                    portfolio_delta["long_market_value"] = float(sub_exp.long_market_value)
                                    portfolio_delta["short_market_value"] = float(sub_exp.short_market_value)
                                    portfolio_delta["long_exposure_pct"] = float(sub_exp.long_exposure_pct)
                                    portfolio_delta["short_exposure_pct"] = float(sub_exp.short_exposure_pct)
                                    portfolio_delta["net_exposure_pct"] = float(sub_exp.net_exposure_pct)
                            except Exception as err:
                                log.warning("Could not calculate portfolio exposure: %s", err)
            except Exception as err:
                log.warning("Could not read portfolio state for morning delta: %s", err)

        if self._performance_reader:
            try:
                try:
                    perf = self._performance_reader(
                        days=30, portfolio_id=portfolio_id, as_of_utc=packet_cutoff_utc
                    )
                except TypeError:
                    perf = self._performance_reader(days=30, portfolio_id=portfolio_id)
                if isinstance(perf, list) and len(perf) >= 2:
                    dated_perf = [
                        p for p in perf
                        if isinstance(p, dict) and p.get("Date") and p.get("Total_NAV") is not None
                    ]
                    try:
                        target_day = datetime.strptime(target_date[:10], "%Y-%m-%d").date()
                        dated_perf = [
                            row for row in dated_perf
                            if datetime.strptime(str(row.get("Date", ""))[:10], "%Y-%m-%d").date() <= target_day
                        ]
                    except (TypeError, ValueError):
                        dated_perf = []

                    dated_perf.sort(key=lambda x: str(x.get("Date", "")))
                    # Conflicting revisions for one date are ambiguous until the producer exposes versions.
                    by_date: Dict[str, Dict[str, Any]] = {}
                    conflicting_dates = set()
                    for row in dated_perf:
                        date_key = str(row.get("Date", ""))[:10]
                        prior = by_date.get(date_key)
                        if prior is not None and (
                            str(prior.get("Total_NAV")) != str(row.get("Total_NAV"))
                            or str(prior.get("Net_External_Flow_THB", prior.get("Net_Cash_Flow", prior.get("net_external_flow"))))
                            != str(row.get("Net_External_Flow_THB", row.get("Net_Cash_Flow", row.get("net_external_flow"))))
                            or str(prior.get("Flow_Coverage_Status") or "MISSING_FLOW_HISTORY").upper()
                            != str(row.get("Flow_Coverage_Status") or "MISSING_FLOW_HISTORY").upper()
                        ):
                            conflicting_dates.add(date_key)
                        else:
                            by_date[date_key] = row

                    dated_perf = [row for key, row in sorted(by_date.items()) if key not in conflicting_dates]
                    if len(dated_perf) >= 2:
                        last_p = dated_perf[-1]
                        prev_p = dated_perf[-2]
                        last_date_str = str(last_p.get("Date", "")).strip()[:10]
                        prev_date_str = str(prev_p.get("Date", "")).strip()[:10]

                        try:
                            t_d = datetime.strptime(target_date[:10], "%Y-%m-%d").date()
                            l_d = datetime.strptime(last_date_str, "%Y-%m-%d").date()
                            p_d = datetime.strptime(prev_date_str, "%Y-%m-%d").date()
                            days_from_target = (t_d - l_d).days
                            days_between = (l_d - p_d).days
                            fresh = 0 <= days_from_target <= 4 and 1 <= days_between <= 4
                        except (TypeError, ValueError):
                            fresh = False

                        flow_key = next(
                            (key for key in ("Net_External_Flow_THB", "Net_Cash_Flow", "net_external_flow") if key in last_p),
                            None,
                        )
                        flow_coverage = str(last_p.get("Flow_Coverage_Status") or "MISSING_FLOW_HISTORY").upper()
                        portfolio_delta["daily_pnl_flow_coverage"] = flow_coverage
                        if not fresh:
                            portfolio_delta["daily_pnl_reason"] = "STALE_OR_IRREGULAR_SNAPSHOTS"
                        elif last_date_str in conflicting_dates or prev_date_str in conflicting_dates:
                            portfolio_delta["daily_pnl_reason"] = "CONFLICTING_SNAPSHOT_REVISIONS"
                        elif flow_coverage != "COMPLETE" or flow_key is None or last_p.get(flow_key) is None:
                            portfolio_delta["daily_pnl_reason"] = flow_coverage if flow_coverage != "COMPLETE" else "MISSING_EXTERNAL_FLOW_VALUE"
                        else:
                            try:
                                prev_nav = Decimal(str(prev_p.get("Total_NAV_THB") or prev_p.get("Total_NAV")))
                                today_nav = Decimal(str(last_p.get("Total_NAV_THB") or last_p.get("Total_NAV")))
                                net_flow = Decimal(str(last_p[flow_key]))
                                if (
                                    prev_nav.is_finite() and today_nav.is_finite() and net_flow.is_finite()
                                    and prev_nav > Decimal("0")
                                ):
                                    gain = (today_nav - prev_nav) - net_flow
                                    gain_pct = (gain / prev_nav) * Decimal("100")
                                    portfolio_delta["daily_pnl_value"] = float(gain.quantize(Decimal("0.01")))
                                    portfolio_delta["daily_pnl_pct"] = float(gain_pct.quantize(Decimal("0.01")))
                                    portfolio_delta["daily_pnl_status"] = "AVAILABLE"
                                    portfolio_delta["daily_pnl_reason"] = None
                                else:
                                    portfolio_delta["daily_pnl_reason"] = "INVALID_NAV_OR_FLOW"
                            except Exception as d_err:
                                portfolio_delta["daily_pnl_reason"] = "INVALID_NAV_OR_FLOW"
                                log.warning("Decimal NAV calculation unavailable: %s", d_err)
                    # Chapter 8: One-Factor Descriptive Beta
                    if self._benchmark_return_reader:
                        try:
                            bench_obs = self._benchmark_return_reader(perf=perf)
                            if isinstance(bench_obs, list) and len(bench_obs) >= 2:
                                beta_res = calculate_descriptive_factor_beta(bench_obs)
                                portfolio_delta["portfolio_beta"] = round(float(beta_res.beta), 2)
                                portfolio_delta["beta_benchmark"] = beta_res.benchmark_name
                                portfolio_delta["factor_observations_count"] = beta_res.eligible_observations_count
                        except Exception as b_err:
                            log.debug("Benchmark beta calculation skipped: %s", b_err)
            except Exception as err:
                log.warning("Could not calculate daily pnl from performance history: %s", err)

        # 2. Quadrant 2: Macro Delta (with Chapter 6 Point-in-Time Vintage Walk)
        macro_delta = {
            "yield_spread_10y_2y_bps": None,
            "yield_spread_24h_change_bps": None,
            "ofr_stress_index": None,
            "macro_regime_label": "Unavailable",
            "regime_shift_risk": "Unavailable",
            "vintage_walk": [],
        }
        if self._macro_reader:
            try:
                m_state = self._macro_reader()
                if isinstance(m_state, dict):
                    macro_delta.update({
                        k: m_state[k] for k in ["yield_spread_10y_2y_bps", "yield_spread_24h_change_bps", "ofr_stress_index", "macro_regime_label", "regime_shift_risk"] if k in m_state and m_state[k] is not None
                    })
            except Exception as err:
                log.warning("Could not read macro data for morning delta: %s", err)

        try:
            v_service = self._macro_vintage_service or MacroVintageApplicationService()
            walk = v_service.get_vintage_walk(cutoff_date=target_date)
            macro_delta["vintage_walk"] = [r.to_dict() for r in walk]
        except Exception as v_err:
            log.warning("Could not compute macro vintage walk: %s", v_err)

        # 3. Quadrant 3: Thesis Exceptions (from Watchdog scan) & Coverage
        thesis_exceptions: List[Dict[str, Any]] = []
        thesis_coverage: Dict[str, Any] = {
            "total_holdings_count": 0,
            "covered_count": 0,
            "uncovered_count": 0,
            "coverage_pct": 0.0,
            "covered_tickers": [],
            "uncovered_tickers": [],
        }
        if self._thesis_service:
            try:
                observed_by_ticker: Dict[str, Dict[str, Decimal]] = {}
                for h in raw_holdings:
                    sym = getattr(h, "symbol", "")
                    market, native_curr = resolve_holding_currency_and_market(h)
                    if native_curr == "USD":
                        p = getattr(h, "current_price_usd", None) or getattr(h, "current_price", None)
                    else:
                        p = getattr(h, "current_price_thb", None) or getattr(h, "current_price", None)
                    if sym and p is not None:
                        try:
                            dec_p = Decimal(str(p))
                            if dec_p > 0:
                                clean_s = sym.strip().upper()
                                observed_by_ticker[clean_s] = {"current_price": dec_p}
                                observed_by_ticker[clean_s.replace(".BK", "")] = {"current_price": dec_p}
                        except Exception:
                            pass

                raw_exceptions = self._thesis_service.run_watchdog_scan(
                    as_of_date_str=target_date,
                    observed_metrics_by_ticker=observed_by_ticker,
                )
                thesis_exceptions = [e.to_dict() for e in raw_exceptions]
            except Exception as err:
                log.error("Could not run watchdog scan for morning delta: %s", err)
                is_blocked = True
                thesis_exceptions.append({
                    "exception_id": f"EXC-INTEGRITY-{target_date.replace('-', '')}-WATCHDOG",
                    "ticker": "PORTFOLIO",
                    "claim_id": "SYSTEM_INTEGRITY",
                    "falsifier_id": "FAL-WATCHDOG-INTEGRITY",
                    "metric_name": "thesis_integrity",
                    "expected_value": "OPERATIONAL",
                    "observed_value": "ERROR",
                    "source_citation": f"Watchdog scan exception: {err}",
                    "human_action": "REVIEW_REQUIRED",
                    "status": "PENDING",
                    "detected_at_utc": f"{target_date}T00:00:00Z",
                })
                quarantined_items.append({
                    "symbol": "SYSTEM_INTEGRITY",
                    "reason": f"Watchdog scan error: {err}",
                    "guardrail_code": "THESIS_INTEGRITY_BREACH",
                    "current_value_thb": 0.0,
                })

        # Chapter 7: Catalyst Exception Queue Integration
        if self._catalyst_service:
            try:
                cat_exs = self._catalyst_service.scan_exceptions(as_of_utc=f"{target_date}T06:00:00Z")
                for c_ex in cat_exs:
                    if getattr(c_ex, "classification", "") != "NO_CHANGE":
                        thesis_exceptions.append(c_ex.to_dict())
            except Exception as c_err:
                log.warning("Could not scan catalyst exceptions for morning delta: %s", c_err)

        if self._thesis_service:
            try:
                # Extract non-cash portfolio tickers
                portfolio_tickers = [
                    str(getattr(h, "symbol", "")).strip().upper()
                    for h in raw_holdings
                    if getattr(h, "symbol", None) and "CASH" not in str(getattr(h, "symbol", "")).upper()
                ]
                if portfolio_tickers and hasattr(self._thesis_service, "get_thesis_coverage"):
                    thesis_coverage = self._thesis_service.get_thesis_coverage(portfolio_tickers)
            except Exception as err:
                log.error("Could not compute thesis coverage for morning delta: %s", err)
                is_blocked = True
                quarantined_items.append({
                    "symbol": "THESIS_COVERAGE",
                    "reason": f"Thesis coverage calculation failed: {err}",
                    "guardrail_code": "THESIS_COVERAGE_FAILED",
                    "current_value_thb": 0.0,
                })

        # 4. Quadrant 4: Pending Decisions
        pending_decisions: List[Dict[str, Any]] = []
        if self._advisory_reader:
            try:
                try:
                    adv = self._advisory_reader(portfolio_id=portfolio_id)
                except TypeError:
                    adv = self._advisory_reader()

                if isinstance(adv, list):
                    pending_decisions = adv
                elif isinstance(adv, dict) and "alerts" in adv:
                    pending_decisions = adv["alerts"]
                elif hasattr(adv, "holdings_advisory"):
                    # Institutional PortfolioAdvisoryResponseDTO
                    for h in adv.holdings_advisory:
                        action_code = getattr(h, "action_code", "")
                        if action_code and action_code not in ("HOLD_WITHIN_BAND", "HOLD_LOCKED_VEHICLE"):
                            sym = getattr(h, "symbol", "")
                            rationale = getattr(h, "action_rationale", "")
                            is_imm = getattr(h, "is_immediate_execution", False)
                            incident = triage_failure(
                                error=rationale or f"{sym}: {action_code}",
                                context={"ticker": sym, "action_code": action_code},
                            )
                            pending_decisions.append({
                                "decision_id": f"DEC-{sym}-{action_code}",
                                "ticker": sym,
                                "action": rationale or f"{sym}: {action_code}",
                                "action_code": action_code,
                                "urgency": "HIGH" if is_imm or action_code in ("URGENT_REDUCE", "TRIM_OVERWEIGHT") else "NORMAL",
                                "narrative": rationale,
                                "failure_class": incident.failure_class.value,
                                "smallest_repair": incident.smallest_repair,
                            })
            except Exception as err:
                log.warning("Could not read advisory recommendations: %s", err)

        if not pending_decisions and thesis_exceptions:
            # Default decision item based on breached falsifier
            exc_incident = triage_failure("Thesis falsifier condition breached", context={"artifact": "thesis_exceptions"})
            pending_decisions.append({
                "decision_id": f"DEC-{target_date}-01",
                "action": "Acknowledge Falsifier Exceptions",
                "urgency": "HIGH",
                "narrative": f"Review {len(thesis_exceptions)} detected thesis exceptions requiring PM sign-off.",
                "failure_class": exc_incident.failure_class.value,
                "smallest_repair": exc_incident.smallest_repair,
            })

        # 5. Institutional Hard-Fail Missing Price Policy & Quarantined Holdings
        if 'missing_price_quarantine_items' in locals() and missing_price_quarantine_items:
            quarantined_items.extend(missing_price_quarantine_items)

        if 'adv' in locals() and hasattr(adv, "quarantined_holdings") and adv.quarantined_holdings:
            for q in adv.quarantined_holdings:
                quarantined_items.append({
                    "symbol": getattr(q, "symbol", ""),
                    "reason": getattr(q, "reason", ""),
                    "guardrail_code": getattr(q, "guardrail_code", ""),
                    "current_value_thb": float(getattr(q, "current_value_thb", 0.0)),
                })
        elif raw_holdings:
            for h in raw_holdings:
                sym = getattr(h, "symbol", "")
                a_type = getattr(h, "asset_type", "Stock")
                units = getattr(h, "units", 0)
                market, native_curr = resolve_holding_currency_and_market(h)
                if native_curr == "USD":
                    p = getattr(h, "current_price_usd", None) or getattr(h, "current_price", None)
                else:
                    p = getattr(h, "current_price_thb", None) or getattr(h, "current_price", None)
                warn = check_missing_price_policy(p, units, a_type, sym)
                if warn:
                    quarantined_items.append({
                        "symbol": sym,
                        "reason": warn.message,
                        "guardrail_code": warn.guardrail_code,
                        "current_value_thb": float(getattr(h, "market_value_thb", 0.0)),
                    })

        # Deduplicate quarantined items by symbol
        dedup_quarantined: Dict[str, Dict[str, Any]] = {}
        for q in quarantined_items:
            s_key = str(q.get("symbol", "")).strip().upper()
            if s_key and s_key not in dedup_quarantined:
                dedup_quarantined[s_key] = q
        quarantined_items = list(dedup_quarantined.values())

        # 6. State Hash Integrity Fast Check (Ch. 3, Appendix A)
        if self._state_integrity_checker:
            try:
                state_report = self._state_integrity_checker()
                if state_report:
                    portfolio_delta["state_integrity"] = state_report.to_dict()
                    if not state_report.is_valid:
                        is_blocked = True
                        quarantined_items.append({
                            "symbol": "STATE_INTEGRITY",
                            "reason": state_report.message,
                            "guardrail_code": state_report.summary_status,
                            "current_value_thb": 0.0,
                        })
            except Exception as s_err:
                log.error("State integrity check failed to execute: %s", s_err)
                is_blocked = True
                quarantined_items.append({
                    "symbol": "STATE_INTEGRITY",
                    "reason": f"State integrity error: {s_err}",
                    "guardrail_code": "STATE_INTEGRITY_FAILED",
                    "current_value_thb": 0.0,
                })

        # 7. Late-Run Guard (Ch. 10, p. 71)
        if self._routine_monitor_service:
            try:
                r_fresh = self._routine_monitor_service.get_morning_routine_freshness(target_date_str=target_date)
                run_quality = r_fresh.get("status", "ON_TIME")
                drift_minutes = int(r_fresh.get("delay_minutes", 0.0))
            except Exception as r_err:
                log.warning("Could not read routine freshness: %s", r_err)
                run_quality, drift_minutes = check_late_run_guard()
        else:
            run_quality, drift_minutes = check_late_run_guard()

        # 8. Quadrant 4 Governance: Risk-Based Review Sampling (Ch. 9 & App. F, p. 108)
        review_sample_items = select_review_sample(
            holdings=raw_holdings,
            thesis_exceptions=thesis_exceptions,
            quarantined_holdings=quarantined_items,
            pending_decisions=pending_decisions,
            top_mover=portfolio_delta.get("top_mover"),
            review_date=target_date,
            min_sample_size=3,
            max_sample_size=5,
        )
        review_sample = [s.to_dict() for s in review_sample_items]

        if is_blocked:
            run_quality = "BLOCKED"

        checksum = self._compute_packet_checksum(
            review_date=target_date,
            portfolio_delta=portfolio_delta,
            macro_delta=macro_delta,
            thesis_exceptions=thesis_exceptions,
            pending_decisions=pending_decisions,
            thesis_coverage=thesis_coverage,
            quarantined_holdings=quarantined_items,
            review_sample=review_sample,
            run_quality=run_quality,
            drift_minutes=drift_minutes,
        )

        # Stored evidence must be read from repository fields, never reconstructed from notes.
        frozen_payload = {
            "review_date": target_date,
            "portfolio_delta": portfolio_delta,
            "macro_delta": macro_delta,
            "thesis_exceptions": thesis_exceptions,
            "pending_decisions": pending_decisions,
            "thesis_coverage": thesis_coverage or {},
            "quarantined_holdings": quarantined_items,
            "review_sample": review_sample,
            "run_quality": run_quality,
            "drift_minutes": drift_minutes,
            "packet_sha256": checksum,
        }
        save_snapshot = getattr(self._signoff_repo, "save_packet_snapshot", None)
        if callable(save_snapshot):
            save_snapshot(
                packet_id=f"PKT-{portfolio_id.upper()[:8]}-{target_date.replace('-', '')}-{uuid.uuid4().hex[:10].upper()}",
                portfolio_id=portfolio_id,
                review_date=target_date,
                cutoff_utc=packet_cutoff_utc,
                input_fingerprint_sha256=checksum,
                policy_version="pm-review-v1",
                packet_sha256=checksum,
                payload=frozen_payload,
            )

        signoff = self._signoff_repo.get_signoff(target_date, portfolio_id=portfolio_id)
        required_exception_ids = self._pending_exception_ids(thesis_exceptions)
        required_sample_ids = [
            sample.get("sample_id")
            for sample in review_sample
            if isinstance(sample, dict)
            and isinstance(sample.get("sample_id"), str)
            and sample.get("sample_id", "").strip()
        ]
        if len(required_sample_ids) != len(review_sample) or len(set(required_sample_ids)) != len(required_sample_ids):
            required_sample_ids = ["__INVALID_PACKET_SAMPLE__"]

        def stored_ids_cover(field_name: str, required_ids: List[str]) -> bool:
            if not required_ids:
                return True
            stored_ids = signoff.get(field_name) if signoff else None
            if not isinstance(stored_ids, (list, tuple)):
                return False
            normalized_stored = {
                value.strip() for value in stored_ids
                if isinstance(value, str) and value.strip()
            }
            return set(required_ids).issubset(normalized_stored)

        signoff_evidence_complete = (
            stored_ids_cover("verified_sample_ids", required_sample_ids)
            and stored_ids_cover("acknowledged_exception_ids", required_exception_ids)
        )

        # Invariant: status is SIGNED_OFF only when checksum and current evidence match.
        if is_blocked:
            status = "BLOCKED"
        elif signoff and signoff.get("packet_sha256") == checksum and signoff_evidence_complete:
            status = "SIGNED_OFF"
        else:
            status = "REVIEW_REQUIRED"
            if signoff and signoff.get("packet_sha256") != checksum:
                signoff["stale_warning"] = "Underlying packet state changed since prior signoff; re-review required."
            elif signoff and not signoff_evidence_complete:
                signoff["evidence_warning"] = "Stored sign-off lacks current sample or exception acknowledgement evidence."

        return MorningReviewPacket(
            review_date=target_date,
            status=status,
            portfolio_delta=portfolio_delta,
            macro_delta=macro_delta,
            thesis_exceptions=thesis_exceptions,
            pending_decisions=pending_decisions,
            packet_sha256=checksum,
            signoff_info=signoff,
            thesis_coverage=thesis_coverage,
            quarantined_holdings=quarantined_items,
            review_sample=review_sample,
            run_quality=run_quality,
            drift_minutes=drift_minutes,
        )

    def sign_off(
        self,
        review_date: str,
        signed_off_by: str,
        notes: Optional[str] = None,
        packet_sha256: Optional[str] = None,
        verified_sample_ids: Optional[List[str]] = None,
        acknowledged_exception_ids: Optional[List[str]] = None,
        portfolio_id: str = "default",
    ) -> Dict[str, Any]:
        """Records immutable sign-off audit trail for the morning review with checksum & sample validation."""
        now_utc = datetime.now(timezone.utc).isoformat()
        current_packet = self.build_morning_packet(portfolio_id=portfolio_id, review_date=review_date)
        actual_sha = current_packet.packet_sha256
        get_snapshot = getattr(self._signoff_repo, "get_packet_snapshot", None)
        packet_snapshot = get_snapshot(portfolio_id, actual_sha) if callable(get_snapshot) else None
        if not isinstance(packet_snapshot, dict) or packet_snapshot.get("packet_sha256") != actual_sha:
            raise SignoffPersistenceUnavailable(
                "Sign-off is unavailable: the exact frozen packet is not durably stored."
            )
        frozen_payload = packet_snapshot.get("payload")
        if not isinstance(frozen_payload, dict) or frozen_payload.get("packet_sha256") != actual_sha:
            raise SignoffPersistenceUnavailable(
                "Sign-off is unavailable: the stored packet snapshot failed read-back validation."
            )
        frozen_exceptions = frozen_payload.get("thesis_exceptions")
        frozen_samples = frozen_payload.get("review_sample")
        if not isinstance(frozen_exceptions, list) or not isinstance(frozen_samples, list):
            raise SignoffPersistenceUnavailable(
                "Sign-off is unavailable: the frozen packet lacks required review evidence."
            )
        required_exception_ids = self._pending_exception_ids(frozen_exceptions)

        # 0. Reject sign-off if packet is BLOCKED by critical integrity error
        if current_packet.status == "BLOCKED" or any(q.get("guardrail_code") in ("THESIS_INTEGRITY_BREACH", "STATE_INTEGRITY_FAILED", "THESIS_COVERAGE_FAILED") for q in current_packet.quarantined_holdings):
            raise ValueError(
                "Sign-off is BLOCKED: critical thesis integrity or state corruption detected in packet. Repair underlying state before signing."
            )

        # 1. Enforce checksum verification: packet_sha256 is strictly mandatory
        if not packet_sha256 or not str(packet_sha256).strip():
            raise ValueError("Packet SHA-256 checksum is mandatory for PM review sign-off.")
        if packet_sha256 != actual_sha:
            raise ValueError(
                f"Packet SHA-256 mismatch: client submitted checksum {packet_sha256} does not match current packet {actual_sha}"
            )

        # 2. Enforce verified samples validity: if review samples exist, PM must inspect and submit them
        if any(
            not isinstance(row, dict)
            or not isinstance(row.get("sample_id"), str)
            or not row.get("sample_id", "").strip()
            for row in frozen_samples
        ):
            raise ValueError("Frozen review packet contains a malformed sample; rebuild the packet before sign-off.")
        sample_rows = frozen_samples
        sample_id_list = [row["sample_id"].strip() for row in sample_rows]
        actual_sample_ids = set(sample_id_list)
        if len(actual_sample_ids) != len(sample_id_list):
            raise ValueError("Current review packet contains duplicate sample IDs; rebuild the packet before sign-off.")

        supplied_ids = list(verified_sample_ids or [])
        if any(not isinstance(sample_id, str) or not sample_id.strip() for sample_id in supplied_ids):
            raise ValueError("Invalid verified sample ID: IDs must be non-empty strings.")
        supplied_unique_ids = {sample_id.strip() for sample_id in supplied_ids}
        unknown_ids = supplied_unique_ids - actual_sample_ids
        if unknown_ids:
            bad_id = sorted(unknown_ids)[0]
            raise ValueError(f"Invalid verified sample ID '{bad_id}': not in current morning review sample")
        missing_ids = actual_sample_ids - supplied_unique_ids
        if missing_ids:
            raise ValueError(
                "Review sample verification required: PM must inspect every packet sample; "
                f"missing {sorted(missing_ids)}."
            )
        verified_sample_ids = [sample_id for sample_id in sample_id_list if sample_id in supplied_unique_ids]

        supplied_exception_ids = list(acknowledged_exception_ids or [])
        if any(not isinstance(exception_id, str) or not exception_id.strip() for exception_id in supplied_exception_ids):
            raise ValueError("Invalid acknowledged exception ID: IDs must be non-empty strings.")
        supplied_exception_set = {exception_id.strip() for exception_id in supplied_exception_ids}
        unknown_exception_ids = supplied_exception_set - set(required_exception_ids)
        if unknown_exception_ids:
            bad_id = sorted(unknown_exception_ids)[0]
            raise ValueError(f"Invalid acknowledged exception ID '{bad_id}': not a pending exception in the current packet.")
        missing_exception_ids = set(required_exception_ids) - supplied_exception_set
        if missing_exception_ids:
            raise ValueError(
                "Exception acknowledgement required: PM must acknowledge every pending packet exception; "
                f"missing {sorted(missing_exception_ids)}."
            )
        acknowledged_exception_ids = [
            exception_id for exception_id in required_exception_ids
            if exception_id in supplied_exception_set
        ]
        exc_count = len(acknowledged_exception_ids)

        # 3. Idempotent return if already signed off with same checksum
        try:
            existing = self._signoff_repo.get_signoff(review_date, portfolio_id=portfolio_id, packet_sha256=actual_sha)
        except TypeError:
            existing = self._signoff_repo.get_signoff(review_date, portfolio_id=portfolio_id)

        if existing and existing.get("packet_sha256") == actual_sha:
            stored_samples = existing.get("verified_sample_ids")
            stored_exceptions = existing.get("acknowledged_exception_ids")
            samples_complete = not sample_id_list or (
                isinstance(stored_samples, (list, tuple))
                and set(sample_id_list).issubset({x.strip() for x in stored_samples if isinstance(x, str) and x.strip()})
            )
            exceptions_complete = not required_exception_ids or (
                isinstance(stored_exceptions, (list, tuple))
                and set(required_exception_ids).issubset({x.strip() for x in stored_exceptions if isinstance(x, str) and x.strip()})
            )
            if not samples_complete or not exceptions_complete:
                raise ValueError(
                    "Existing sign-off for this packet lacks persisted sample or exception acknowledgement evidence; "
                    "prepare a new packet revision before signing again."
                )
            log.info("Idempotent sign-off returned for %s / %s (packet: %s)", portfolio_id, review_date, actual_sha)
            ret = dict(existing)
            ret["status"] = "SIGNED_OFF"
            ret["verified_sample_ids"] = ret.get("verified_sample_ids") or []
            ret["acknowledged_exception_ids"] = ret.get("acknowledged_exception_ids") or []
            return ret

        signoff_id = f"SIG-{portfolio_id.upper()[:4]}-{review_date.replace('-', '')}-{uuid.uuid4().hex[:6].upper()}"

        final_notes = notes
        request_digest = hashlib.sha256(
            json.dumps(
                {
                    "portfolio_id": portfolio_id,
                    "review_date": review_date,
                    "packet_sha256": actual_sha,
                    "principal_subject_id": signed_off_by.strip(),
                    "verified_sample_ids": verified_sample_ids,
                    "acknowledged_exception_ids": acknowledged_exception_ids,
                    "notes": final_notes,
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

        try:
            persisted = self._signoff_repo.save_signoff(
                signoff_id=signoff_id,
                review_date=review_date,
                signed_off_by=signed_off_by.strip(),
                signed_off_at_utc=now_utc,
                acknowledged_exceptions_count=exc_count,
                notes=final_notes,
                packet_sha256=actual_sha,
                portfolio_id=portfolio_id,
                verified_sample_ids=verified_sample_ids,
                acknowledged_exception_ids=acknowledged_exception_ids,
                packet_id=packet_snapshot.get("packet_id"),
                request_digest=request_digest,
                principal_subject_id=signed_off_by.strip(),
            )
        except TypeError as err:
            if "acknowledged_exception_ids" in str(err):
                message = "Sign-off is unavailable: the configured repository cannot durably store exception acknowledgements."
            else:
                message = "Sign-off is unavailable: the configured repository cannot persist the full frozen packet evidence."
            raise SignoffPersistenceUnavailable(message) from err

        canonical = persisted if isinstance(persisted, dict) and "signoff_id" in persisted else None
        if verified_sample_ids and (
            canonical is None or canonical.get("verified_sample_ids") != verified_sample_ids
        ):
            raise SignoffPersistenceUnavailable(
                "Sign-off is unavailable: verified sample evidence was not returned by durable storage."
            )
        if acknowledged_exception_ids and (
            canonical is None or canonical.get("acknowledged_exception_ids") != acknowledged_exception_ids
        ):
            raise SignoffPersistenceUnavailable(
                "Sign-off is unavailable: exception acknowledgement evidence was not returned by durable storage."
            )
        res_id = canonical["signoff_id"] if canonical else signoff_id
        res_signer = canonical.get("signed_off_by", signed_off_by.strip()) if canonical else signed_off_by.strip()
        res_utc = canonical.get("signed_off_at_utc", now_utc) if canonical else now_utc
        res_notes = canonical.get("notes", final_notes) if canonical else final_notes
        res_samples = canonical.get("verified_sample_ids") if canonical and "verified_sample_ids" in canonical else list(verified_sample_ids or [])
        res_acknowledgements = (
            canonical.get("acknowledged_exception_ids")
            if canonical and "acknowledged_exception_ids" in canonical
            else list(acknowledged_exception_ids or [])
        )

        log.info("PM %s signed off morning review for %s (portfolio: %s, id: %s)", res_signer, review_date, portfolio_id, res_id)
        return {
            "signoff_id": res_id,
            "portfolio_id": portfolio_id,
            "review_date": review_date,
            "status": "SIGNED_OFF",
            "signed_off_by": res_signer,
            "signed_off_at_utc": res_utc,
            "acknowledged_exceptions_count": exc_count,
            "notes": res_notes,
            "packet_sha256": actual_sha,
            "verified_sample_ids": res_samples,
            "acknowledged_exception_ids": res_acknowledgements,
        }

    def list_history(self, limit: int = 30, portfolio_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists historical review sign-offs."""
        return self._signoff_repo.list_signoffs(limit=limit, portfolio_id=portfolio_id)

