"""Portfolio performance snapshots with explicit external-flow coverage and economic NAV."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from core.portfolio.attribution import (
    BrinsonAttributionReport,
    BucketAttributionInput,
    compute_single_period_brinson,
    link_multi_period_brinson_carino,
)
from core.portfolio.dividend_accrual import (
    DividendEntitlementSnapshot,
    DividendScheduleEntry,
    DividendSettlementRecord,
    compute_accrued_dividend_receivable,
    normalize_dividend_schedule_entry,
)
from tools.portfolio.domain.calculations import compute_total_cost, recalc_all
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.dividend_port import DividendHistoryPort
from tools.portfolio.ports.performance_port import PerformanceRepositoryPort
from tools.portfolio.ports.price_port import MarketPricePort
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort


def _require_fx(state) -> float:
    fx = state.fx_rates.get("USDTHB")
    if fx is None or fx <= 0:
        raise ValueError("ไม่พบ fx_rates.USDTHB ที่ valid ใน portfolio")
    return fx


def _decimal(value) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"invalid performance value: {value!r}")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"invalid performance value: {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError("performance value must be finite")
    return parsed


def build_accrued_dividend_context(
    state: Any,
    trade_rows: Optional[Sequence[Any]],
    as_of_date: str,
    fx_dec: Decimal,
    *,
    dividend_schedules: Optional[Sequence[Any]] = None,
    dividend_provider: Optional[DividendHistoryPort] = None,
    refresh_prices: bool = False,
    entitlement_snapshots: Optional[Sequence[Any]] = None,
    settlement_records: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Derives schedules, immutable entitlement snapshots, and settlements for economic NAV."""
    derived_schedules: List[Any] = list(dividend_schedules or ())
    derived_snapshots: List[Any] = list(entitlement_snapshots or ())
    derived_settlements: List[Any] = list(settlement_records or ())

    traded_symbols = {
        str(
            r.get("Symbol", r.get("symbol", r.get("ticker", "")))
            if isinstance(r, dict)
            else getattr(r, "symbol", "")
        )
        .strip()
        .upper()
        .removesuffix(".BK")
        for r in (trade_rows or ())
    }
    traded_symbols.discard("")

    active_equity_symbols: List[str] = []
    for holding in getattr(state, "holdings", ()) or ():
        asset_type = str(getattr(holding, "asset_type", "") or "").strip()
        sym = str(getattr(holding, "symbol", "") or "").strip().upper()
        if not sym or asset_type in ("Cash", "PVD") or sym.startswith("PVD_"):
            continue
        if getattr(holding, "status", "active") == "active" and getattr(holding, "units", 0) > 0:
            active_equity_symbols.append(sym)

        for rnd in getattr(holding, "dividend_rounds", ()) or ():
            rnd_status = str(getattr(rnd, "status", "") or "").strip().lower()
            rnd_ex = getattr(rnd, "ex_date", None)
            rnd_pay = getattr(rnd, "pay_date", None)
            if rnd_status == "received":
                derived_settlements.append(
                    DividendSettlementRecord(
                        ticker=sym,
                        settled_date=str(rnd_pay or rnd_ex or as_of_date),
                        ex_date=str(rnd_ex) if rnd_ex else None,
                        settled_amount_native=_decimal(getattr(rnd, "net_native", 0) or 0),
                        settled_amount_thb=_decimal(getattr(rnd, "net_thb", 0) or 0),
                        currency=str(getattr(rnd, "currency", "THB") or "THB"),
                        status="received",
                    )
                )
            elif rnd_status == "upcoming" and rnd_pay and rnd_ex:
                if dividend_schedules is None:
                    derived_schedules.append(
                        DividendScheduleEntry(
                            ticker=sym,
                            ex_date=str(rnd_ex),
                            pay_date=str(rnd_pay),
                            dps_native=_decimal(getattr(rnd, "dps", 0) or 0),
                            market="US" if str(getattr(rnd, "currency", "")).upper() == "USD" else "TH",
                            currency="USD" if str(getattr(rnd, "currency", "")).upper() == "USD" else "THB",
                            withholding_tax_pct=(
                                _decimal(getattr(rnd, "tax_rate", 0) or 0) * Decimal("100")
                                if _decimal(getattr(rnd, "tax_rate", 0) or 0) <= Decimal("1")
                                else _decimal(getattr(rnd, "tax_rate", 0) or 0)
                            ),
                        )
                    )
                units_held = getattr(rnd, "units_held", 0) or 0
                base_sym = sym.removesuffix(".BK")
                if units_held > 0 and base_sym not in traded_symbols and entitlement_snapshots is None:
                    derived_snapshots.append(
                        DividendEntitlementSnapshot(
                            ticker=sym,
                            ex_date=str(rnd_ex),
                            entitled_units=_decimal(units_held),
                            market="US" if str(getattr(rnd, "currency", "")).upper() == "USD" else "TH",
                            source="dividend_round_snapshot",
                        )
                    )

    if (
        dividend_schedules is None
        and dividend_provider is not None
        and active_equity_symbols
        and (refresh_prices or not getattr(dividend_provider, "requires_network", False))
    ):
        try:
            fetched = dividend_provider.fetch_dividend_history(active_equity_symbols) or {}
            for sym, records in fetched.items():
                for rec in records or ():
                    if isinstance(rec, DividendScheduleEntry):
                        derived_schedules.append(rec)
                        continue
                    if isinstance(rec, dict) and rec.get("pay_date"):
                        derived_schedules.append(
                            normalize_dividend_schedule_entry({"ticker": sym, **rec})
                        )
        except Exception:
            pass

    if not derived_schedules:
        return {
            "accrued_receivable_thb": Decimal("0.00"),
            "accrued_receivable_usd": Decimal("0.00"),
            "entitlement_basis": "none",
            "entitlement_verified": True,
            "warnings": [],
        }

    normalized_schedules = [normalize_dividend_schedule_entry(item) for item in derived_schedules]
    effective_trade_history: Optional[Sequence[Any]] = trade_rows if trade_rows else None
    summary = compute_accrued_dividend_receivable(
        holdings=state.holdings,
        dividend_schedules=normalized_schedules,
        as_of_date=as_of_date,
        usd_thb_fx_rate=fx_dec,
        trade_history=effective_trade_history,
        entitlement_snapshots=derived_snapshots or None,
        settlement_records=derived_settlements or None,
    )
    return {
        "accrued_receivable_thb": summary.accrued_receivable_thb,
        "accrued_receivable_usd": summary.accrued_receivable_usd,
        "entitlement_basis": summary.entitlement_basis,
        "entitlement_verified": summary.entitlement_verified,
        "warnings": summary.warnings,
    }


class PortfolioPerformanceService:
    """Creates versioned snapshots and carries P&L only across covered flow windows."""

    def __init__(
        self,
        repo: PortfolioRepositoryPort,
        perf_repo: PerformanceRepositoryPort,
        price_provider: MarketPricePort,
        dividend_provider: Optional[DividendHistoryPort] = None,
    ) -> None:
        self.repo = repo
        self.perf_repo = perf_repo
        self.price_provider = price_provider
        self.dividend_provider = dividend_provider

    def _history_as_of(self, portfolio_id: str, cutoff_utc: str) -> List[Dict]:
        try:
            return self.perf_repo.read_history(
                portfolio_id=portfolio_id, days=None, as_of_utc=cutoff_utc
            )
        except TypeError:
            return self.perf_repo.read_history(portfolio_id=portfolio_id, days=None)

    def _read_trade_rows_safe(self, portfolio_id: str) -> List[Dict]:
        reader = getattr(self.repo, "read_trade_log", None)
        if not callable(reader):
            return []
        try:
            return list(reader(portfolio_id=portfolio_id) or [])
        except TypeError:
            try:
                return list(reader(portfolio_id) or [])
            except Exception:
                return []
        except Exception:
            return []

    def record_performance_snapshot(
        self,
        refresh_prices: bool = True,
        portfolio_id: str = "default",
        dividend_schedules: Optional[List[Any]] = None,
        *,
        as_of_date: Optional[str] = None,
        trade_history: Optional[Sequence[Any]] = None,
        entitlement_snapshots: Optional[Sequence[Any]] = None,
        settlement_records: Optional[Sequence[Any]] = None,
    ) -> str:
        try:
            pid = validate_portfolio_id(portfolio_id)
            cutoff = datetime.now(timezone.utc)
            cutoff_utc = cutoff.isoformat().replace("+00:00", "Z")
            from tools.portfolio.domain.ledger_change import LedgerChange

            with self.repo.unit_of_work(pid) as uow:
                state = uow.load_state()
                if refresh_prices and self.price_provider:
                    self.price_provider.refresh_portfolio_prices(state)
                recalc_all(state)
                uow.commit(state, LedgerChange(kind="unchanged"))

            fx_raw = _require_fx(state)
            fx_dec = _decimal(fx_raw)
            fx_float = float(fx_dec)
            total_nav = _decimal(state.summary.total_value_thb)

            snapshot_date = (as_of_date[:10] if as_of_date else cutoff.date().isoformat())
            resolved_trades = (
                list(trade_history) if trade_history is not None else self._read_trade_rows_safe(pid)
            )
            accrual_ctx = build_accrued_dividend_context(
                state=state,
                trade_rows=resolved_trades,
                as_of_date=snapshot_date,
                fx_dec=fx_dec,
                dividend_schedules=dividend_schedules,
                dividend_provider=self.dividend_provider,
                refresh_prices=refresh_prices,
                entitlement_snapshots=entitlement_snapshots,
                settlement_records=settlement_records,
            )
            accrued_receivable_thb = accrual_ctx["accrued_receivable_thb"]
            economic_nav = total_nav + accrued_receivable_thb

            total_cost = _decimal(compute_total_cost(state, fx_float))
            unrealized = _decimal(state.summary.total_unrealized_profit)
            cash_balance = sum(
                (_decimal(h.market_value_thb or 0) for h in state.holdings if h.asset_type == "Cash"),
                Decimal("0"),
            )
            asset_class_values: Dict[str, str] = {}
            for holding in state.holdings:
                asset_class = holding.asset_type or "Unknown"
                value = _decimal(holding.market_value_thb or 0).quantize(Decimal("0.01"))
                asset_class_values[asset_class] = str(
                    Decimal(asset_class_values.get(asset_class, "0")) + value
                )

            provenance_reader = getattr(self.repo, "performance_provenance", None)
            provenance = provenance_reader(pid) if callable(provenance_reader) else {}
            current_sequence = provenance.get("source_state_sequence")
            current_hash = provenance.get("source_state_sha256")
            history = self._history_as_of(pid, cutoff_utc)
            versioned = [
                row
                for row in history
                if isinstance(row, dict) and row.get("Cutoff_UTC") and row.get("Snapshot_ID")
            ]
            versioned.sort(key=lambda row: (str(row.get("Cutoff_UTC")), int(row.get("Revision") or 0)))
            previous = versioned[-1] if versioned else None

            flow_coverage = "MISSING_FLOW_HISTORY"
            net_external_flow: Optional[Decimal] = None
            flow_event_ids: list[str] = []
            flow_reader = getattr(self.repo, "list_cash_flow_events", None)
            coverage_reader = getattr(self.repo, "cash_flow_coverage_between", None)
            if (
                previous is not None
                and current_sequence is not None
                and previous.get("Source_State_Sequence") not in (None, "")
                and callable(flow_reader)
                and callable(coverage_reader)
            ):
                previous_sequence = int(previous["Source_State_Sequence"])
                current_sequence = int(current_sequence)
                events = flow_reader(
                    pid, after_sequence=previous_sequence, through_sequence=current_sequence
                )
                all_flows_known = all(
                    event.get("flow_coverage_status") == "COMPLETE"
                    and event.get("amount_thb_decimal") is not None
                    and event.get("flow_kind") == "EXTERNAL_CONTRIBUTION"
                    for event in events
                )
                history_covered = coverage_reader(
                    pid, after_sequence=previous_sequence, through_sequence=current_sequence
                )
                if all_flows_known and history_covered:
                    net_external_flow = sum(
                        (_decimal(event["amount_thb_decimal"]) for event in events),
                        Decimal("0"),
                    ).quantize(Decimal("0.01"))
                    flow_coverage = "COMPLETE"
                    flow_event_ids = [str(event["event_id"]) for event in events]

            row = {
                "Date": snapshot_date,
                "Total_NAV": str(total_nav.quantize(Decimal("0.01"))),
                "Total_Cost": str(total_cost.quantize(Decimal("0.01"))),
                "Unrealized_PnL": str(unrealized.quantize(Decimal("0.01"))),
                "Cash_Balance": str(cash_balance.quantize(Decimal("0.01"))),
                "Realized_PnL_YTD": str(_decimal(state.summary.total_realized_profit_ytd).quantize(Decimal("0.01"))),
                "Passive_Income_YTD": str(_decimal(state.summary.passive_income_ytd).quantize(Decimal("0.01"))),
                "Asset_Class_Values_THB": asset_class_values,
                "Snapshot_ID": f"PERF-{uuid4().hex}",
                "Cutoff_UTC": cutoff_utc,
                "Recorded_At_UTC": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "Source_State_Sequence": current_sequence,
                "Source_State_SHA256": current_hash,
                "Total_NAV_THB": str(total_nav.quantize(Decimal("0.01"))),
                "Accrued_Dividend_Receivable_THB": str(accrued_receivable_thb.quantize(Decimal("0.01"))),
                "Economic_NAV_THB": str(economic_nav.quantize(Decimal("0.01"))),
                "Net_External_Flow_THB": str(net_external_flow) if net_external_flow is not None else None,
                "Flow_Coverage_Status": flow_coverage,
                "Cash_Flow_Event_IDs_JSON": flow_event_ids,
            }
            self.perf_repo.upsert_snapshot(pid, row)
            return (
                f"[PERF SNAPSHOT] บันทึก snapshot วันที่ {row['Date']} "
                f"(NAV: {row['Total_NAV_THB']} THB, Economic NAV: {row['Economic_NAV_THB']} THB, "
                f"flow coverage: {flow_coverage}) สำเร็จ"
            )
        except Exception as exc:
            return f"Error: {exc}"

    def read_performance_history(
        self, days: int = 30, portfolio_id: str = "default", as_of_utc: Optional[str] = None
    ) -> str:
        rows = self.get_structured_performance_history(
            days=days, portfolio_id=portfolio_id, as_of_utc=as_of_utc
        )
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def get_structured_performance_history(
        self,
        days: Optional[int] = None,
        portfolio_id: str = "default",
        as_of_utc: Optional[str] = None,
    ) -> List[Dict]:
        try:
            return self.perf_repo.read_history(
                portfolio_id=portfolio_id, days=days, as_of_utc=as_of_utc
            )
        except TypeError:
            return self.perf_repo.read_history(portfolio_id=portfolio_id, days=days)

    @staticmethod
    def _serialize_brinson_report(report: BrinsonAttributionReport) -> Dict[str, Any]:
        return {
            "period_count": report.period_count,
            "portfolio_gross_return": float(report.portfolio_gross_return),
            "benchmark_gross_return": float(report.benchmark_gross_return),
            "portfolio_net_return": float(report.portfolio_net_return) if report.portfolio_net_return is not None else None,
            "benchmark_net_return": float(report.benchmark_net_return) if report.benchmark_net_return is not None else None,
            "portfolio_cost_rate": float(report.portfolio_cost_rate),
            "benchmark_cost_rate": float(report.benchmark_cost_rate),
            "net_active_return": float(report.net_active_return),
            "allocation_effect_total": float(report.allocation_effect_total),
            "selection_effect_total": float(report.selection_effect_total),
            "interaction_effect_total": float(report.interaction_effect_total),
            "cost_effect": float(report.cost_effect),
            "cash_drag_effect": float(report.cash_drag_effect),
            "reconciliation_error": str(report.reconciliation_error),
            "is_reconciled": report.is_reconciled,
            "rows": [
                {
                    "bucket_id": row.bucket_id,
                    "bucket_name": row.bucket_name,
                    "portfolio_weight": float(row.portfolio_weight),
                    "benchmark_weight": float(row.benchmark_weight),
                    "active_weight": float(row.active_weight),
                    "portfolio_return": float(row.portfolio_return),
                    "benchmark_return": float(row.benchmark_return),
                    "allocation_effect": float(row.allocation_effect),
                    "selection_effect": float(row.selection_effect),
                    "interaction_effect": float(row.interaction_effect),
                    "total_active_contribution": float(row.total_active_contribution),
                }
                for row in report.rows
            ],
        }

    @staticmethod
    def _coerce_bucket_input(item: Any) -> BucketAttributionInput:
        if isinstance(item, BucketAttributionInput):
            return item
        if isinstance(item, dict):
            return BucketAttributionInput(
                bucket_id=str(item["bucket_id"]),
                bucket_name=str(item.get("bucket_name") or item["bucket_id"]),
                portfolio_weight=_decimal(item["portfolio_weight"]),
                benchmark_weight=_decimal(item["benchmark_weight"]),
                portfolio_return=_decimal(item["portfolio_return"]),
                benchmark_return=_decimal(item["benchmark_return"]),
            )
        raise TypeError(f"Unsupported attribution bucket item: {type(item).__name__}")

    def compute_brinson_attribution(
        self,
        buckets: Sequence[Any],
        portfolio_cost_rate: Decimal | float | str = Decimal("0"),
        benchmark_cost_rate: Decimal | float | str = Decimal("0"),
        tolerance: Decimal | float | str = Decimal("0.001"),
    ) -> Dict[str, Any]:
        """Compute single-period Brinson-Fachler attribution report."""
        coerced = [self._coerce_bucket_input(b) for b in buckets]
        report = compute_single_period_brinson(
            buckets=coerced,
            portfolio_cost_rate=_decimal(portfolio_cost_rate),
            benchmark_cost_rate=_decimal(benchmark_cost_rate),
            tolerance=_decimal(tolerance),
        )
        return self._serialize_brinson_report(report)

    def compute_multi_period_attribution(
        self,
        periods: Sequence[Any],
        tolerance: Decimal | float | str = Decimal("0.001"),
    ) -> Dict[str, Any]:
        """Link multiple single-period Brinson reports using Carino logarithmic linking."""
        tol_dec = _decimal(tolerance)
        single_reports: List[BrinsonAttributionReport] = []
        for period in periods:
            if isinstance(period, BrinsonAttributionReport):
                single_reports.append(period)
            elif isinstance(period, dict):
                coerced = [self._coerce_bucket_input(b) for b in period.get("buckets", ())]
                single_reports.append(
                    compute_single_period_brinson(
                        buckets=coerced,
                        portfolio_cost_rate=_decimal(period.get("portfolio_cost_rate", 0)),
                        benchmark_cost_rate=_decimal(period.get("benchmark_cost_rate", 0)),
                        tolerance=tol_dec,
                    )
                )
            else:
                coerced = [self._coerce_bucket_input(b) for b in period]
                single_reports.append(compute_single_period_brinson(buckets=coerced, tolerance=tol_dec))
        linked = link_multi_period_brinson_carino(single_reports)
        return self._serialize_brinson_report(linked)

    def compute_portfolio_bucket_attribution(
        self,
        portfolio_id: str = "default",
        benchmark_weights: Optional[Dict[str, Decimal | float | str]] = None,
        portfolio_returns: Optional[Dict[str, Decimal | float | str]] = None,
        benchmark_returns: Optional[Dict[str, Decimal | float | str]] = None,
        portfolio_cost_rate: Decimal | float | str = Decimal("0"),
        benchmark_cost_rate: Decimal | float | str = Decimal("0"),
    ) -> Dict[str, Any]:
        """Compute single-period Brinson-Fachler attribution from current portfolio buckets."""
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
        recalc_all(state)
        fx_dec = _decimal(_require_fx(state))

        target_names: Dict[str, str] = {}
        target_weights_raw: Dict[str, Decimal] = {}
        for tgt in getattr(state, "allocation_targets", ()) or ():
            bid = str(getattr(tgt, "bucket_id", "") or "").strip()
            if not bid:
                continue
            target_names[bid] = str(getattr(tgt, "name", bid) or bid)
            target_weights_raw[bid] = _decimal(getattr(tgt, "target_percent", 0) or 0) / Decimal("100")

        bucket_values: Dict[str, Decimal] = {}
        bucket_cost: Dict[str, Decimal] = {}
        for holding in getattr(state, "holdings", ()) or ():
            if getattr(holding, "status", "active") != "active" or (getattr(holding, "units", 0) or 0) <= 0:
                continue
            bid = str(getattr(holding, "bucket_id", "") or "").strip()
            if not bid:
                bid = "cash" if str(getattr(holding, "asset_type", "") or "").strip() == "Cash" else "unassigned"
            mv = _decimal(getattr(holding, "market_value_thb", 0) or 0)
            units_dec = _decimal(getattr(holding, "units", 0) or 0)
            avg_thb = getattr(holding, "avg_cost_thb", None)
            avg_usd = getattr(holding, "avg_cost_usd", None)
            if avg_thb is not None and avg_thb > 0:
                cost = units_dec * _decimal(avg_thb)
            elif avg_usd is not None and avg_usd > 0:
                cost = units_dec * _decimal(avg_usd) * fx_dec
            else:
                cost = mv
            bucket_values[bid] = bucket_values.get(bid, Decimal("0")) + mv
            bucket_cost[bid] = bucket_cost.get(bid, Decimal("0")) + cost

        all_bucket_ids = sorted(set(bucket_values.keys()) | set(target_weights_raw.keys()) | set((benchmark_weights or {}).keys()))
        if not all_bucket_ids:
            raise ValueError(f"Portfolio '{pid}' has no buckets or holdings for attribution.")

        total_mv = sum(bucket_values.values(), Decimal("0"))
        if total_mv <= Decimal("0"):
            raise ValueError(f"Portfolio '{pid}' total market value must be positive for attribution.")

        raw_bench: Dict[str, Decimal] = {}
        for bid in all_bucket_ids:
            if benchmark_weights and bid in benchmark_weights:
                raw_bench[bid] = _decimal(benchmark_weights[bid])
            elif bid in target_weights_raw:
                raw_bench[bid] = target_weights_raw[bid]
            else:
                raw_bench[bid] = Decimal("0")
        bench_sum = sum(raw_bench.values(), Decimal("0"))
        if bench_sum <= Decimal("0"):
            even_w = Decimal("1") / Decimal(len(all_bucket_ids))
            raw_bench = {bid: even_w for bid in all_bucket_ids}
            bench_sum = sum(raw_bench.values(), Decimal("0"))

        inputs: List[BucketAttributionInput] = []
        for bid in all_bucket_ids:
            wp = (bucket_values.get(bid, Decimal("0")) / total_mv)
            wb = (raw_bench[bid] / bench_sum)
            if portfolio_returns and bid in portfolio_returns:
                rp = _decimal(portfolio_returns[bid])
            else:
                c_val = bucket_cost.get(bid, Decimal("0"))
                mv_val = bucket_values.get(bid, Decimal("0"))
                rp = ((mv_val - c_val) / c_val) if c_val > Decimal("0") else Decimal("0")
            rb = _decimal(benchmark_returns[bid]) if (benchmark_returns and bid in benchmark_returns) else Decimal("0")
            inputs.append(
                BucketAttributionInput(
                    bucket_id=bid,
                    bucket_name=target_names.get(bid, bid),
                    portfolio_weight=wp,
                    benchmark_weight=wb,
                    portfolio_return=rp,
                    benchmark_return=rb,
                )
            )

        return self.compute_brinson_attribution(
            buckets=inputs,
            portfolio_cost_rate=portfolio_cost_rate,
            benchmark_cost_rate=benchmark_cost_rate,
        )
