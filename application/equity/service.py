"""Application services for Equity research, valuation, and insider views."""
import json
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from application.equity.ports import (
    AnalystCachePort,
    AnalystProviderPort,
    AssetResolverPort,
    InsiderLedgerPort,
    InsiderSyncPort,
    ValuationLedgerPort,
    ValuationSidecarPort,
    FinancialsQueryPort,
)
from application.equity.validation import validate_ticker

log = logging.getLogger(__name__)


class EquityFinancialsApplicationService:
    """Resolve an equity and delegate statement retrieval to the query port.

    Asset classification and market/provider-symbol consistency are business
    rules, so they belong here rather than in the FastAPI router.  The router
    remains an inbound adapter that only maps exceptions to HTTP responses.
    """

    def __init__(self, resolver: AssetResolverPort, financials: FinancialsQueryPort) -> None:
        self._resolver = resolver
        self._financials = financials

    def get_statements(
        self,
        ticker: str,
        market: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Any:
        clean_ticker = validate_ticker(ticker)
        if len(clean_ticker) > 20:
            raise ValueError("Invalid ticker format")

        resolved = self._resolver.resolve(clean_ticker)
        if not resolved:
            raise LookupError(f"Asset not found for ticker: {clean_ticker}")

        provider_symbol = getattr(resolved, "provider_symbol", None) or clean_ticker
        resolved_market = (
            "TH"
            if getattr(resolved, "market", None) == "TH" or provider_symbol.endswith(".BK")
            else "US"
        )
        asset_class = getattr(resolved, "asset_class", "")
        asset_class_value = getattr(asset_class, "value", asset_class)
        if asset_class_value not in {"STOCK_US", "STOCK_TH", "equity"}:
            raise ValueError(
                f"Asset {clean_ticker} is not an equity ({asset_class_value})"
            )

        if market is not None:
            requested_market = market.strip().upper()
            if requested_market not in {"US", "TH"}:
                raise ValueError(f"Invalid market query param: {market}")
            if requested_market != resolved_market:
                raise ValueError(
                    f"Market mismatch: requested market '{requested_market}' "
                    f"does not match authoritative market '{resolved_market}' for {clean_ticker}"
                )

        return self._financials.get_financial_statements(
            ticker=clean_ticker,
            market=resolved_market,
            provider_symbol=provider_symbol,
            force_refresh=force_refresh,
        )


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _positive_int_or_none(value: Any) -> Optional[int]:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


class EquityAnalystApplicationService:
    """Coordinates analyst targets, earnings, and a durable stale cache."""

    _BURST_FAIL_TTL = 10.0
    _MAX_STALE_AGE = 7 * 24 * 3600

    def __init__(
        self,
        resolver: AssetResolverPort,
        cache: AnalystCachePort,
        provider: AnalystProviderPort,
    ) -> None:
        self._resolver = resolver
        self._cache = cache
        self._provider = provider
        self._lock = threading.Lock()
        self._symbol_locks: Dict[str, threading.Lock] = {}
        self._burst_fail: Dict[str, float] = {}

    def _symbol_lock(self, symbol: str) -> threading.Lock:
        with self._lock:
            return self._symbol_locks.setdefault(symbol, threading.Lock())

    @staticmethod
    def _days_to_earnings(next_date: Optional[str], exchange_tz: str) -> Optional[int]:
        if not next_date:
            return None
        try:
            today = datetime.now(ZoneInfo(exchange_tz)).date()
            target = datetime.strptime(next_date, "%Y-%m-%d").date()
            value = (target - today).days
            return value if value >= 0 else None
        except Exception:
            return None

    @staticmethod
    def _cached_payload(cached: Dict[str, Any], exchange_tz: str, *, status: Optional[str] = None) -> Dict[str, Any]:
        payload = dict(cached)
        if status:
            payload["data_status"] = status
        payload["days_to_earnings"] = EquityAnalystApplicationService._days_to_earnings(
            payload.get("next_earnings_date"), exchange_tz
        )
        return payload

    def get_context(self, ticker: str) -> Dict[str, Any]:
        clean_ticker = ticker.upper()
        resolved = self._resolver.resolve(clean_ticker)
        provider_symbol = getattr(resolved, "provider_symbol", None) or clean_ticker
        market = "TH" if (getattr(resolved, "market", None) == "TH" or provider_symbol.endswith(".BK")) else "US"
        currency = "THB" if market == "TH" else "USD"
        exchange_tz = "Asia/Bangkok" if market == "TH" else "America/New_York"

        cached = self._cache.get(clean_ticker)
        now_wall = time.time()
        if cached:
            try:
                cached_at = datetime.fromisoformat(cached["synced_at"]).timestamp()
                if now_wall - cached_at < 24 * 3600:
                    return self._cached_payload(cached, exchange_tz)
            except (KeyError, TypeError, ValueError):
                cached = None

        provider_key = provider_symbol.strip().upper()
        with self._symbol_lock(provider_key):
            cached = self._cache.get(clean_ticker)
            if cached:
                try:
                    cached_at = datetime.fromisoformat(cached["synced_at"]).timestamp()
                    if now_wall - cached_at < 24 * 3600:
                        return self._cached_payload(cached, exchange_tz)
                except (KeyError, TypeError, ValueError):
                    cached = None

            if provider_key in self._burst_fail and now_wall - self._burst_fail[provider_key] < self._BURST_FAIL_TTL:
                if cached:
                    try:
                        if now_wall - datetime.fromisoformat(cached["synced_at"]).timestamp() <= self._MAX_STALE_AGE:
                            return self._cached_payload(cached, exchange_tz, status="stale")
                    except (KeyError, TypeError, ValueError):
                        pass
                return {
                    "ticker": clean_ticker,
                    "provider_symbol": provider_symbol,
                    "market": market,
                    "currency": currency,
                    "exchange_tz": exchange_tz,
                    "target_mean": None,
                    "target_high": None,
                    "target_low": None,
                    "num_analysts": None,
                    "next_earnings_date": None,
                    "days_to_earnings": None,
                    "earnings_history": [],
                    "source_as_of": datetime.now(ZoneInfo(exchange_tz)).isoformat(),
                    "data_status": "unavailable",
                    "provider_tier": "best_effort",
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                }

            errors: List[str] = []
            target_mean = target_high = target_low = None
            num_analysts = None
            next_earnings_date = None
            earnings_history: List[Dict[str, Any]] = []

            try:
                targets = self._provider.price_targets(provider_symbol) or {}
                target_mean = _finite_or_none(targets.get("mean") or targets.get("targetMeanPrice"))
                target_high = _finite_or_none(targets.get("high") or targets.get("targetHighPrice"))
                target_low = _finite_or_none(targets.get("low") or targets.get("targetLowPrice"))
                num_analysts = _positive_int_or_none(targets.get("numberOfAnalystOpinions"))
                if target_mean is None and targets.get("targetMeanPrice") is not None:
                    target_mean = _finite_or_none(targets.get("targetMeanPrice"))
                    target_high = _finite_or_none(targets.get("targetHighPrice"))
                    target_low = _finite_or_none(targets.get("targetLowPrice"))
            except Exception as exc:
                log.warning("Analyst price targets fetch failed for %s: %s", provider_symbol, exc)
                errors.append("analyst_targets")

            try:
                cal = self._provider.calendar(provider_symbol) or {}
                dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
                if isinstance(dates, list):
                    candidates = []
                    today = datetime.now(ZoneInfo(exchange_tz)).date()
                    for value in dates:
                        if hasattr(value, "year"):
                            candidates.append(value.date() if hasattr(value, "hour") else value)
                        elif isinstance(value, str):
                            try:
                                candidates.append(datetime.strptime(value[:10], "%Y-%m-%d").date())
                            except ValueError:
                                pass
                    future = [value for value in candidates if value >= today]
                    if future:
                        next_earnings_date = min(future).strftime("%Y-%m-%d")
            except Exception as exc:
                log.warning("Calendar fetch failed for %s: %s", provider_symbol, exc)
                errors.append("calendar")

            try:
                earnings = self._provider.earnings_history(provider_symbol, exchange_tz)
                earnings_history = list(getattr(earnings, "rows", []) or [])
                if getattr(earnings, "status", "ok") == "failed":
                    errors.append("earnings_history")
            except Exception as exc:
                log.warning("Earnings history fetch failed for %s: %s", provider_symbol, exc)
                errors.append("earnings_history")

            if errors:
                self._burst_fail[provider_key] = time.time()
            if errors and cached:
                try:
                    if now_wall - datetime.fromisoformat(cached["synced_at"]).timestamp() <= self._MAX_STALE_AGE:
                        return self._cached_payload(cached, exchange_tz, status="stale")
                except (KeyError, TypeError, ValueError):
                    pass

            has_target = target_mean is not None
            has_calendar = next_earnings_date is not None
            has_earnings = bool(earnings_history)
            if not has_target and not has_calendar and not has_earnings:
                data_status = "unavailable"
            elif errors or not has_target or not has_calendar:
                data_status = "partial"
            else:
                data_status = "ok"

            source_as_of = datetime.now(ZoneInfo(exchange_tz)).isoformat()
            synced_at = datetime.now(timezone.utc).isoformat()
            payload = {
                "ticker": clean_ticker,
                "provider_symbol": provider_symbol,
                "market": market,
                "currency": currency,
                "exchange_tz": exchange_tz,
                "target_mean": target_mean,
                "target_high": target_high,
                "target_low": target_low,
                "num_analysts": num_analysts,
                "next_earnings_date": next_earnings_date,
                "days_to_earnings": self._days_to_earnings(next_earnings_date, exchange_tz),
                "earnings_history": earnings_history,
                "source_as_of": source_as_of,
                "data_status": data_status,
                "provider_tier": "best_effort",
                "synced_at": synced_at,
            }
            if data_status in ("ok", "partial"):
                self._cache.upsert(clean_ticker, payload)
            return payload


class EquityValuationApplicationService:
    """Loads a DCF ledger record and computes presentation-neutral facts."""

    def __init__(self, ledger: ValuationLedgerPort, sidecar: ValuationSidecarPort) -> None:
        self._ledger = ledger
        self._sidecar = sidecar

    def get_targets(self, ticker: str) -> Dict[str, Any]:
        clean_ticker = ticker.upper()
        row = self._ledger.latest(clean_ticker)
        if row is None:
            model = self._sidecar.latest(clean_ticker)
            dcf = getattr(getattr(model, "quant_signals", None), "dcf_result", None) if model else None
            if model is not None and dcf is not None:
                eval_id = f"eval_{clean_ticker}_{model.analysis_date}_{int(datetime.now(timezone.utc).timestamp())}"
                self._ledger.record(
                    evaluation_id=eval_id,
                    ticker=clean_ticker,
                    market=model.market,
                    evaluated_at=model.quant_signals.evaluated_at or f"{model.analysis_date}T00:00:00Z",
                    scenarios={key: value.model_dump() for key, value in dcf.scenarios.items()},
                    model_version="dcf_v1.0",
                    valuation_price_basis="split_adjusted_only",
                    current_price_at_eval=getattr(model, "current_price", getattr(model.quant_signals, "current_price", None)),
                    wacc_pct=dcf.wacc_pct,
                    valuation_verdict=dcf.valuation_verdict or "unknown",
                    corporate_action_evidence=[],
                    input_snapshot={"analysis_date": model.analysis_date, "observable_refs": dcf.observable_refs},
                )
                row = self._ledger.latest(clean_ticker)

        if row is None:
            return {
                "evaluation_id": f"eval_empty_{clean_ticker}",
                "ticker": clean_ticker,
                "market": "TH" if clean_ticker.endswith(".BK") else "US",
                "currency": "THB" if clean_ticker.endswith(".BK") else "USD",
                "status": "unavailable",
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "as_of_label": "N/A",
                "comparability_status": "unknown",
                "comparability_reasons": ["No DCF evaluation found for this ticker"],
                "scenarios": [],
            }

        market = row["market"]
        currency = "THB" if market == "TH" else "USD"
        evaluated_at = row["evaluated_at"]
        try:
            scenarios_dict = json.loads(row["scenarios_json"])
        except Exception:
            scenarios_dict = {}
        try:
            corp_evidence = json.loads(row["corporate_action_evidence_json"]) if row.get("corporate_action_evidence_json") else []
        except Exception:
            corp_evidence = []
        try:
            input_snapshot = json.loads(row["input_snapshot_json"]) if row.get("input_snapshot_json") else {}
        except Exception:
            input_snapshot = {}
        try:
            eval_dt = datetime.fromisoformat(evaluated_at.replace("Z", "+00:00"))
        except Exception:
            eval_dt = datetime.now(timezone.utc)

        reasons: List[str] = []
        factors: List[Dict[str, Any]] = []
        comparability = "comparable"
        for factor in corp_evidence:
            factor_data = {
                "event_type": factor.get("event_type", "split"),
                "effective_date": factor.get("effective_date", ""),
                "ratio": factor.get("ratio"),
                "amount": factor.get("amount"),
            }
            factors.append(factor_data)
            if factor_data["event_type"] == "split" and factor_data["effective_date"] > evaluated_at[:10]:
                comparability = "not_comparable"
                reasons.append(
                    f"Unadjusted stock split ({factor_data['ratio'] or 'N/A'}) occurred on {factor_data['effective_date']} after DCF evaluation date ({evaluated_at[:10]})"
                )

        scenario_specs = (("base", "DCF Base", "emerald"), ("bull", "DCF Bull", "green"), ("bear", "DCF Bear", "rose"))
        scenarios: List[Dict[str, Any]] = []
        for name, label, color in scenario_specs:
            data = scenarios_dict.get(name)
            if data:
                scenarios.append({
                    "scenario_name": name,
                    "label": label,
                    "target_price": round(float(data.get("target_price", 0.0)), 2),
                    "upside_pct": round(float(data["upside_pct"]), 2) if data.get("upside_pct") is not None else None,
                    "margin_of_safety_pct": round(float(data["margin_of_safety_pct"]), 2) if data.get("margin_of_safety_pct") is not None else None,
                    "color": color,
                })

        order_valid = True
        verdict = row["valuation_verdict"] or "unknown"
        if all(scenarios_dict.get(name) for name in ("bear", "base", "bull")):
            bear = float(scenarios_dict["bear"].get("target_price", 0.0))
            base = float(scenarios_dict["base"].get("target_price", 0.0))
            bull = float(scenarios_dict["bull"].get("target_price", 0.0))
            if not bear <= base <= bull:
                order_valid = False
                verdict = "unknown"
                reasons.append("DCF scenario monotonicity violated: Bear <= Base <= Bull order is inconsistent")

        return {
            "evaluation_id": row["evaluation_id"],
            "ticker": clean_ticker,
            "market": market,
            "currency": currency,
            "chart_price_basis": "provider_proportional_adj_close_ratio",
            "valuation_price_basis": row["valuation_price_basis"],
            "comparability_status": comparability,
            "comparability_reasons": reasons,
            "corporate_action_factors": factors,
            "current_price_at_eval": row["current_price_at_eval"],
            "evaluated_at": evaluated_at,
            "as_of_label": f"as of {eval_dt.strftime('%Y-%m-%d')}",
            "model_version": row["model_version"],
            "valuation_verdict": verdict,
            "wacc_pct": row["wacc_pct"],
            "macro_observable_refs": input_snapshot.get("observable_refs", []),
            "data_quality_flags": [],
            "status": "stale" if (datetime.now(timezone.utc) - eval_dt).days > 30 else "available",
            "scenario_order_valid": order_valid,
            "scenarios": scenarios,
        }


class EquityInsiderApplicationService:
    """Builds insider filing aggregates from the canonical ledger."""

    def __init__(self, ledger: InsiderLedgerPort, sync: InsiderSyncPort) -> None:
        self._ledger = ledger
        self._sync = sync

    def get_filings(self, ticker: str, requested_range: str, interval: str) -> Dict[str, Any]:
        clean_ticker = ticker.upper()
        market = "TH" if clean_ticker.endswith(".BK") else "US"
        range_days = {"5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "max": 3650}.get(requested_range, 365)
        now = datetime.now(timezone.utc)
        since_date = (now - timedelta(days=range_days)).strftime("%Y-%m-%d")
        records = self._ledger.list_records(clean_ticker, since_date)
        if not records and market == "US":
            self._sync.sync(clean_ticker)
            records = self._ledger.list_records(clean_ticker, since_date)

        cutoffs = {days: (now - timedelta(days=days)).strftime("%Y-%m-%d") for days in (30, 90, 180)}
        net = {30: 0.0, 90: 0.0, 180: 0.0}
        buyers: set[str] = set()
        filings: Dict[str, Dict[str, Any]] = {}
        for row in records:
            acc = row["accession_number"]
            shares = float(row["shares"])
            delta = shares if row["acquired_or_disposed"] == "A" else -shares
            for days in (30, 90, 180):
                if row["transaction_date"] >= cutoffs[days]:
                    net[days] += delta
                    if days == 30 and row["acquired_or_disposed"] == "A" and row.get("reporting_owner_name"):
                        buyers.add(row["reporting_owner_name"])
            if acc not in filings:
                try:
                    dt = datetime.strptime(row["transaction_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    timestamp = int(dt.timestamp() * 1000)
                except Exception:
                    timestamp = int(now.timestamp() * 1000)
                filings[acc] = {
                    "accession_number": acc,
                    "issuer_cik": row["issuer_cik"],
                    "ticker": clean_ticker,
                    "filing_url": row["filing_url"],
                    "filed_at": row["filed_at"] or row["transaction_date"],
                    "timestamp": timestamp,
                    "reporting_owner_cik": row["reporting_owner_cik"],
                    "reporting_owner_name": row["reporting_owner_name"],
                    "is_director": bool(row["is_director"]),
                    "is_officer": bool(row["is_officer"]),
                    "is_ten_percent_owner": bool(row["is_ten_percent_owner"]),
                    "officer_title": row["officer_title"],
                    "is_amendment": bool(row["is_amendment"]),
                    "amends_accession_number": row["amends_accession_number"],
                    "is_cluster_buy": False,
                    "transactions": [],
                }
            filings[acc]["transactions"].append({
                "transaction_id": row["transaction_id"],
                "transaction_date": row["transaction_date"],
                "transaction_code": row["transaction_code"],
                "shares": shares,
                "price_per_share": float(row["price_per_share"]),
                "acquired_or_disposed": "A" if row["acquired_or_disposed"] == "A" else "D",
                "shares_owned_following": row["shares_owned_following"],
                "ownership_nature": row["ownership_nature"],
                "is_derivative": bool(row["is_derivative"]),
                "normalized_weight": float(row["normalized_weight"] or 1.0),
            })

        cluster = len(buyers) >= 3
        filing_list = []
        for filing in filings.values():
            if cluster and filing["reporting_owner_name"] in buyers:
                filing["is_cluster_buy"] = True
            filing_list.append(filing)
        filing_list.sort(key=lambda item: item["timestamp"], reverse=True)
        return {
            "ticker": clean_ticker,
            "market": market,
            "requested_range": requested_range,
            "interval": interval,
            "net_shares_30d": round(net[30], 2),
            "net_shares_90d": round(net[90], 2),
            "net_shares_180d": round(net[180], 2),
            "cluster_buy_count": sum(1 for item in filing_list if item["is_cluster_buy"]),
            "total_filings_count": len(filing_list),
            "filings": filing_list,
        }
