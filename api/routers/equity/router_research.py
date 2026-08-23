"""FastAPI Sub-router for Equity Research: Valuation Targets, Insider Filings, Analyst Context, Financials."""
import json
import logging
import re
import time
from typing import Literal, Optional, List
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_session
from api.schemas import (
    ValuationTargetsDTO,
    DCFScenarioLevelDTO,
    CorporateActionFactorDTO,
    InsiderFilingsResponseDTO,
    InsiderFilingDTO,
    InsiderTransactionDTO,
    AnalystContextDTO,
    EarningsHistoryEntryDTO,
    FinancialStatementsDTO,
)
from api.state_db import (
    get_connection,
    get_latest_dcf_evaluation,
    record_dcf_evaluation,
    get_sec_insider_filings_and_transactions,
    get_analyst_context_cache,
    upsert_analyst_context_cache,
)
from tools.market.asset_resolver import resolve_asset
from tools.market.calendar import get_asset_calendar
from tools.market.earnings import fetch_earnings_dates, finite_or_none
from tools.market.financials import get_financial_statements
from tools.market.sec_form4_pipeline import sync_insider_filings_from_yfinance
from api.routers.equity.common import (
    _validate_ticker,
    _get_equity_files,
    _get_latest_sidecar_for_ticker,
    _get_analyst_lock,
    positive_int_or_none,
    _ANALYST_LOCK,
    _ANALYST_BURST_FAIL_CACHE,
    ANALYST_BURST_FAIL_TTL,
    MAX_STALE_AGE_SECONDS,
)

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{ticker}/valuation-targets", response_model=ValuationTargetsDTO)
def get_equity_valuation_targets(ticker: str) -> ValuationTargetsDTO:
    clean_ticker = _validate_ticker(ticker)

    conn = get_connection()
    row = get_latest_dcf_evaluation(conn, clean_ticker)

    if row is None:
        files = _get_equity_files(clean_ticker)
        res = _get_latest_sidecar_for_ticker(files, clean_ticker, strict=True)
        if res is not None:
            model, _ = res
            if model.quant_signals and model.quant_signals.dcf_result:
                dcf = model.quant_signals.dcf_result
                eval_id = f"eval_{clean_ticker}_{model.analysis_date}_{int(datetime.now(timezone.utc).timestamp())}"
                record_dcf_evaluation(
                    conn=conn,
                    evaluation_id=eval_id,
                    ticker=clean_ticker,
                    market=model.market,
                    evaluated_at=model.quant_signals.evaluated_at or f"{model.analysis_date}T00:00:00Z",
                    scenarios={k: v.model_dump() for k, v in dcf.scenarios.items()},
                    model_version="dcf_v1.0",
                    valuation_price_basis="split_adjusted_only",
                    current_price_at_eval=getattr(model, "current_price", getattr(model.quant_signals, "current_price", None)),
                    wacc_pct=dcf.wacc_pct,
                    valuation_verdict=dcf.valuation_verdict or "unknown",
                    corporate_action_evidence=[],
                    input_snapshot={"analysis_date": model.analysis_date, "observable_refs": dcf.observable_refs},
                )
                row = get_latest_dcf_evaluation(conn, clean_ticker)

    if row is None:
        return ValuationTargetsDTO(
            evaluation_id=f"eval_empty_{clean_ticker}",
            ticker=clean_ticker,
            market="TH" if clean_ticker.endswith(".BK") else "US",
            currency="THB" if clean_ticker.endswith(".BK") else "USD",
            status="unavailable",
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            as_of_label="N/A",
            comparability_status="unknown",
            comparability_reasons=["No DCF evaluation found for this ticker"],
            scenarios=[],
        )

    eval_id = row["evaluation_id"]
    market = row["market"]
    currency: Literal["USD", "THB"] = "THB" if market == "TH" else "USD"
    evaluated_at_str = row["evaluated_at"]
    model_version = row["model_version"]
    val_basis = row["valuation_price_basis"]
    current_price_eval = row["current_price_at_eval"]
    wacc_pct = row["wacc_pct"]
    valuation_verdict = row["valuation_verdict"] or "unknown"

    try:
        scenarios_dict = json.loads(row["scenarios_json"])
    except Exception:
        scenarios_dict = {}

    try:
        corp_evidence = json.loads(row["corporate_action_evidence_json"]) if row["corporate_action_evidence_json"] else []
    except Exception:
        corp_evidence = []

    try:
        input_snap = json.loads(row["input_snapshot_json"]) if row["input_snapshot_json"] else {}
    except Exception:
        input_snap = {}

    macro_refs = input_snap.get("observable_refs", [])

    try:
        eval_dt = datetime.fromisoformat(evaluated_at_str.replace("Z", "+00:00"))
    except Exception:
        eval_dt = datetime.now(timezone.utc)

    days_elapsed = (datetime.now(timezone.utc) - eval_dt).days
    status: Literal["available", "unavailable", "stale"] = "stale" if days_elapsed > 30 else "available"
    as_of_label = f"as of {eval_dt.strftime('%Y-%m-%d')}"

    comparability_status: Literal["comparable", "not_comparable", "unknown"] = "comparable"
    comparability_reasons: list[str] = []
    corp_factors: list[CorporateActionFactorDTO] = []

    for factor in corp_evidence:
        f_type = factor.get("event_type", "split")
        f_date = factor.get("effective_date", "")
        f_ratio = factor.get("ratio")
        f_amt = factor.get("amount")
        corp_factors.append(
            CorporateActionFactorDTO(
                event_type=f_type,
                effective_date=f_date,
                ratio=f_ratio,
                amount=f_amt,
            )
        )
        if f_type == "split" and f_date > evaluated_at_str[:10]:
            comparability_status = "not_comparable"
            comparability_reasons.append(f"Unadjusted stock split ({f_ratio or 'N/A'}) occurred on {f_date} after DCF evaluation date ({evaluated_at_str[:10]})")

    scenario_order_valid = True
    scenarios: list[DCFScenarioLevelDTO] = []

    base_data = scenarios_dict.get("base")
    bull_data = scenarios_dict.get("bull")
    bear_data = scenarios_dict.get("bear")

    if base_data:
        scenarios.append(
            DCFScenarioLevelDTO(
                scenario_name="base",
                label="DCF Base",
                target_price=round(float(base_data.get("target_price", 0.0)), 2),
                upside_pct=round(float(base_data.get("upside_pct", 0.0)), 2) if base_data.get("upside_pct") is not None else None,
                margin_of_safety_pct=round(float(base_data.get("margin_of_safety_pct", 0.0)), 2) if base_data.get("margin_of_safety_pct") is not None else None,
                color="emerald",
            )
        )
    if bull_data:
        scenarios.append(
            DCFScenarioLevelDTO(
                scenario_name="bull",
                label="DCF Bull",
                target_price=round(float(bull_data.get("target_price", 0.0)), 2),
                upside_pct=round(float(bull_data.get("upside_pct", 0.0)), 2) if bull_data.get("upside_pct") is not None else None,
                margin_of_safety_pct=round(float(bull_data.get("margin_of_safety_pct", 0.0)), 2) if bull_data.get("margin_of_safety_pct") is not None else None,
                color="green",
            )
        )
    if bear_data:
        scenarios.append(
            DCFScenarioLevelDTO(
                scenario_name="bear",
                label="DCF Bear",
                target_price=round(float(bear_data.get("target_price", 0.0)), 2),
                upside_pct=round(float(bear_data.get("upside_pct", 0.0)), 2) if bear_data.get("upside_pct") is not None else None,
                margin_of_safety_pct=round(float(bear_data.get("margin_of_safety_pct", 0.0)), 2) if bear_data.get("margin_of_safety_pct") is not None else None,
                color="rose",
            )
        )

    if bear_data and base_data and bull_data:
        bear_p = float(bear_data.get("target_price", 0.0))
        base_p = float(base_data.get("target_price", 0.0))
        bull_p = float(bull_data.get("target_price", 0.0))
        if not (bear_p <= base_p <= bull_p):
            scenario_order_valid = False
            valuation_verdict = "unknown"
            comparability_reasons.append("DCF scenario monotonicity violated: Bear <= Base <= Bull order is inconsistent")

    return ValuationTargetsDTO(
        evaluation_id=eval_id,
        ticker=clean_ticker,
        market=market,
        currency=currency,
        chart_price_basis="provider_proportional_adj_close_ratio",
        valuation_price_basis=val_basis,
        comparability_status=comparability_status,
        comparability_reasons=comparability_reasons,
        corporate_action_factors=corp_factors,
        current_price_at_eval=current_price_eval,
        evaluated_at=evaluated_at_str,
        as_of_label=as_of_label,
        model_version=model_version,
        valuation_verdict=valuation_verdict,
        wacc_pct=wacc_pct,
        macro_observable_refs=macro_refs,
        data_quality_flags=[],
        status=status,
        scenario_order_valid=scenario_order_valid,
        scenarios=scenarios,
    )


@router.get("/{ticker}/insider-filings", response_model=InsiderFilingsResponseDTO)
def get_equity_insider_filings(
    ticker: str,
    range: str = "1y",
    interval: str = "1d",
) -> InsiderFilingsResponseDTO:
    clean_ticker = _validate_ticker(ticker)
    market = "TH" if clean_ticker.endswith(".BK") else "US"

    conn = get_connection()
    now = datetime.now(timezone.utc)
    range_days_map = {
        "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "max": 3650
    }
    days = range_days_map.get(range, 365)
    since_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")

    records = get_sec_insider_filings_and_transactions(conn, clean_ticker, since_date=since_date)
    if not records and market == "US":
        sync_insider_filings_from_yfinance(conn, clean_ticker)
        records = get_sec_insider_filings_and_transactions(conn, clean_ticker, since_date=since_date)

    filing_map: dict[str, dict] = {}

    d30_cutoff = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    d90_cutoff = (now - timedelta(days=90)).strftime("%Y-%m-%d")
    d180_cutoff = (now - timedelta(days=180)).strftime("%Y-%m-%d")

    net_shares_30d = 0.0
    net_shares_90d = 0.0
    net_shares_180d = 0.0

    buyers_in_30d: set[str] = set()

    for r in records:
        acc = r["accession_number"]
        tx_date = r["transaction_date"]
        acq_disp = r["acquired_or_disposed"]
        shares = float(r["shares"])
        tx_code = r["transaction_code"]

        share_delta = shares if acq_disp == "A" else -shares
        if tx_date >= d30_cutoff:
            net_shares_30d += share_delta
            if acq_disp == "A" and r["reporting_owner_name"]:
                buyers_in_30d.add(r["reporting_owner_name"])
        if tx_date >= d90_cutoff:
            net_shares_90d += share_delta
        if tx_date >= d180_cutoff:
            net_shares_180d += share_delta

        if acc not in filing_map:
            try:
                dt = datetime.strptime(tx_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                ts = int(dt.timestamp() * 1000)
            except Exception:
                ts = int(now.timestamp() * 1000)

            filing_map[acc] = {
                "accession_number": acc,
                "issuer_cik": r["issuer_cik"],
                "ticker": clean_ticker,
                "filing_url": r["filing_url"],
                "filed_at": r["filed_at"] or tx_date,
                "timestamp": ts,
                "reporting_owner_cik": r["reporting_owner_cik"],
                "reporting_owner_name": r["reporting_owner_name"],
                "is_director": bool(r["is_director"]),
                "is_officer": bool(r["is_officer"]),
                "is_ten_percent_owner": bool(r["is_ten_percent_owner"]),
                "officer_title": r["officer_title"],
                "is_amendment": bool(r["is_amendment"]),
                "amends_accession_number": r["amends_accession_number"],
                "is_cluster_buy": False,
                "transactions": [],
            }

        filing_map[acc]["transactions"].append(
            InsiderTransactionDTO(
                transaction_id=r["transaction_id"],
                transaction_date=tx_date,
                transaction_code=tx_code,
                shares=shares,
                price_per_share=float(r["price_per_share"]),
                acquired_or_disposed="A" if acq_disp == "A" else "D",
                shares_owned_following=r["shares_owned_following"],
                ownership_nature=r["ownership_nature"],
                is_derivative=bool(r["is_derivative"]),
                normalized_weight=float(r["normalized_weight"] or 1.0),
            )
        )

    cluster_buy_signal = len(buyers_in_30d) >= 3
    filings_list: list[InsiderFilingDTO] = []
    cluster_buy_count = 0

    for f_data in filing_map.values():
        if cluster_buy_signal and f_data["reporting_owner_name"] in buyers_in_30d:
            f_data["is_cluster_buy"] = True
            cluster_buy_count += 1
        filings_list.append(InsiderFilingDTO(**f_data))

    filings_list.sort(key=lambda x: x.timestamp, reverse=True)

    return InsiderFilingsResponseDTO(
        ticker=clean_ticker,
        market=market,
        requested_range=range,
        interval=interval,
        net_shares_30d=round(net_shares_30d, 2),
        net_shares_90d=round(net_shares_90d, 2),
        net_shares_180d=round(net_shares_180d, 2),
        cluster_buy_count=cluster_buy_count,
        total_filings_count=len(filings_list),
        filings=filings_list,
    )


@router.get("/{ticker}/analyst-context", response_model=AnalystContextDTO)
def get_equity_analyst_context(ticker: str) -> AnalystContextDTO:
    """ดึงข้อมูล Consensus Target Price, Next Earnings Date Countdown, และ EPS History พร้อม Cache 24h"""
    clean_ticker = _validate_ticker(ticker)
    resolved = resolve_asset(clean_ticker)
    provider_symbol = resolved.provider_symbol or clean_ticker
    market: Literal["TH", "US"] = "TH" if (resolved.market == "TH" or provider_symbol.endswith(".BK")) else "US"
    currency: Literal["USD", "THB"] = "THB" if market == "TH" else "USD"
    exchange_tz = "Asia/Bangkok" if market == "TH" else "America/New_York"

    def _days_to_earnings(next_date_str: str | None) -> int | None:
        if not next_date_str:
            return None
        try:
            tz = ZoneInfo(exchange_tz)
            today = datetime.now(tz).date()
            target = datetime.strptime(next_date_str, "%Y-%m-%d").date()
            diff = (target - today).days
            return diff if diff >= 0 else None
        except Exception:
            return None

    conn = get_connection()
    now_wall = time.time()
    now_mono = time.monotonic()
    CACHE_TTL = 24 * 3600

    cached = get_analyst_context_cache(conn, clean_ticker)
    if cached:
        cached_at = datetime.fromisoformat(cached["synced_at"]).timestamp()
        if (now_wall - cached_at) < CACHE_TTL:
            return AnalystContextDTO(
                **{**cached, "days_to_earnings": _days_to_earnings(cached.get("next_earnings_date"))}
            )

    clean_prov_sym = provider_symbol.strip().upper()
    key_lock = _get_analyst_lock(clean_prov_sym)

    with key_lock:
        cached = get_analyst_context_cache(conn, clean_ticker)
        if cached:
            cached_at = datetime.fromisoformat(cached["synced_at"]).timestamp()
            if (now_wall - cached_at) < CACHE_TTL:
                return AnalystContextDTO(
                    **{**cached, "days_to_earnings": _days_to_earnings(cached.get("next_earnings_date"))}
                )

        with _ANALYST_LOCK:
            if clean_prov_sym in _ANALYST_BURST_FAIL_CACHE:
                if (now_mono - _ANALYST_BURST_FAIL_CACHE[clean_prov_sym]) < ANALYST_BURST_FAIL_TTL:
                    if cached and ((now_wall - datetime.fromisoformat(cached["synced_at"]).timestamp()) <= MAX_STALE_AGE_SECONDS):
                        return AnalystContextDTO(
                            **{**cached, "data_status": "stale", "days_to_earnings": _days_to_earnings(cached.get("next_earnings_date"))}
                        )
                    return AnalystContextDTO(
                        ticker=clean_ticker,
                        provider_symbol=provider_symbol,
                        market=market,
                        currency=currency,
                        exchange_tz=exchange_tz,
                        target_mean=None,
                        target_high=None,
                        target_low=None,
                        num_analysts=None,
                        next_earnings_date=None,
                        days_to_earnings=None,
                        earnings_history=[],
                        source_as_of=datetime.now(ZoneInfo(exchange_tz)).isoformat(),
                        data_status="unavailable",
                        provider_tier="best_effort",
                        synced_at=datetime.now(timezone.utc).isoformat(),
                    )

        target_mean: float | None = None
        target_high: float | None = None
        target_low: float | None = None
        num_analysts: int | None = None
        next_earnings_date: str | None = None
        eps_history: list[dict] = []
        fetch_errors: list[str] = []

        import sys
        routes_equity = sys.modules.get("api.routes_equity")
        yf_mod = getattr(routes_equity, "yf", yf) if routes_equity else yf
        cal_fn = getattr(routes_equity, "get_asset_calendar", get_asset_calendar) if routes_equity else get_asset_calendar
        earn_fn = getattr(routes_equity, "fetch_earnings_dates", fetch_earnings_dates) if routes_equity else fetch_earnings_dates

        try:
            tk = yf_mod.Ticker(provider_symbol)
            apt = tk.get_analyst_price_targets()
            if apt and isinstance(apt, dict):
                target_mean = finite_or_none(apt.get("mean") or apt.get("targetMeanPrice"))
                target_high = finite_or_none(apt.get("high") or apt.get("targetHighPrice"))
                target_low = finite_or_none(apt.get("low") or apt.get("targetLowPrice"))
            info = tk.info or {}
            num_analysts = positive_int_or_none(info.get("numberOfAnalystOpinions"))
            if target_mean is None and info.get("targetMeanPrice") is not None:
                target_mean = finite_or_none(info.get("targetMeanPrice"))
                target_high = finite_or_none(info.get("targetHighPrice"))
                target_low = finite_or_none(info.get("targetLowPrice"))
        except Exception as e:
            log.warning("Analyst price targets fetch failed for %s: %s", provider_symbol, e)
            fetch_errors.append("analyst_targets")

        try:
            cal = cal_fn(provider_symbol)
            earnings_dates_raw = cal.get("Earnings Date") if isinstance(cal, dict) else None
            if isinstance(earnings_dates_raw, list) and earnings_dates_raw:
                from datetime import date as _date
                tz = ZoneInfo(exchange_tz)
                today = datetime.now(tz).date()
                candidates: list[_date] = []
                for d in earnings_dates_raw:
                    if hasattr(d, "year"):
                        candidates.append(d.date() if hasattr(d, "hour") else d)
                    elif isinstance(d, str):
                        try:
                            candidates.append(datetime.strptime(d[:10], "%Y-%m-%d").date())
                        except ValueError:
                            pass
                future = [c for c in candidates if c >= today]
                if future:
                    next_earnings_date = min(future).strftime("%Y-%m-%d")
        except Exception as e:
            log.warning("Calendar fetch failed for %s: %s", provider_symbol, e)
            fetch_errors.append("calendar")

        earn_result = earn_fn(provider_symbol, exchange_tz)
        eps_history = earn_result.rows
        if earn_result.status == "failed":
            fetch_errors.append("earnings_history")

        if fetch_errors:
            with _ANALYST_LOCK:
                _ANALYST_BURST_FAIL_CACHE[clean_prov_sym] = time.monotonic()

        if fetch_errors and cached:
            cached_at = datetime.fromisoformat(cached["synced_at"]).timestamp()
            if (now_wall - cached_at) <= MAX_STALE_AGE_SECONDS:
                return AnalystContextDTO(
                    **{**cached, "data_status": "stale", "days_to_earnings": _days_to_earnings(cached.get("next_earnings_date"))}
                )

        has_fresh_target = target_mean is not None
        has_fresh_calendar = next_earnings_date is not None
        has_fresh_eps = bool(eps_history)

        if not has_fresh_target and not has_fresh_calendar and not has_fresh_eps:
            data_status: Literal["ok", "partial", "stale", "unavailable"] = "unavailable"
        elif fetch_errors or not has_fresh_target or not has_fresh_calendar:
            data_status = "partial"
        else:
            data_status = "ok"

        source_as_of = datetime.now(ZoneInfo(exchange_tz)).isoformat()
        synced_at_str = datetime.now(timezone.utc).isoformat()

        if data_status in ("ok", "partial"):
            upsert_analyst_context_cache(
                conn,
                clean_ticker,
                {
                    "provider_symbol": provider_symbol,
                    "market": market,
                    "currency": currency,
                    "exchange_tz": exchange_tz,
                    "target_mean": target_mean,
                    "target_high": target_high,
                    "target_low": target_low,
                    "num_analysts": num_analysts,
                    "next_earnings_date": next_earnings_date,
                    "earnings_history": eps_history,
                    "source_as_of": source_as_of,
                    "data_status": data_status,
                    "synced_at": now_wall,
                },
            )

        return AnalystContextDTO(
            ticker=clean_ticker,
            provider_symbol=provider_symbol,
            market=market,
            currency=currency,
            exchange_tz=exchange_tz,
            target_mean=target_mean,
            target_high=target_high,
            target_low=target_low,
            num_analysts=num_analysts,
            next_earnings_date=next_earnings_date,
            days_to_earnings=_days_to_earnings(next_earnings_date),
            earnings_history=[EarningsHistoryEntryDTO(**r) for r in eps_history],
            source_as_of=source_as_of,
            data_status=data_status,
            provider_tier="best_effort",
            synced_at=synced_at_str,
        )


@router.get("/{ticker}/financials", response_model=FinancialStatementsDTO)
def get_equity_financial_statements(
    ticker: str,
    market: Optional[str] = None,
    force_refresh: bool = False,
    session: dict = Depends(require_session),
) -> FinancialStatementsDTO:
    """ดึงข้อมูลงบการเงินย้อนหลัง (Income Statement, Balance Sheet, Cash Flow) พร้อมระบบ Dual-Provider (EDGAR/yfinance)"""
    clean_ticker = ticker.strip().upper()
    if not clean_ticker or len(clean_ticker) > 20 or not re.match(r"^[A-Z0-9.\-_]+$", clean_ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker format")

    resolved = resolve_asset(clean_ticker)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Asset not found for ticker: {clean_ticker}")

    provider_symbol = resolved.provider_symbol or clean_ticker
    resolved_market: Literal["US", "TH"] = "TH" if (resolved.market == "TH" or provider_symbol.endswith(".BK")) else "US"

    asset_class_str = resolved.asset_class.value if hasattr(resolved.asset_class, "value") else str(resolved.asset_class)
    if asset_class_str not in ["STOCK_US", "STOCK_TH", "equity"]:
        raise HTTPException(status_code=400, detail=f"Asset {clean_ticker} is not an equity ({asset_class_str})")

    if market:
        req_market = market.strip().upper()
        if req_market not in ["US", "TH"]:
            raise HTTPException(status_code=400, detail=f"Invalid market query param: {market}")
        if req_market != resolved_market:
            raise HTTPException(
                status_code=400,
                detail=f"Market mismatch: requested market '{req_market}' does not match authoritative market '{resolved_market}' for {clean_ticker}",
            )

    provider_symbol = resolved.provider_symbol or clean_ticker
    return get_financial_statements(
        ticker=clean_ticker,
        market=resolved_market,
        provider_symbol=provider_symbol,
        force_refresh=force_refresh,
    )
