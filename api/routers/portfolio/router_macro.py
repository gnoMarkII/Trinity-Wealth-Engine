"""FastAPI Sub-router for Macro & Strategy Endpoints."""
import json
import concurrent.futures
from datetime import datetime, timezone, date, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_session
from api.schemas import (
    MacroIndicatorSeriesDTO,
    MacroDashboardDTO,
    PortfolioDTO,
    NewsFunnelPendingItemDTO,
    NewsFunnelFilteredItemDTO,
    CalendarEventDTO,
    PortfolioCalendarDTO,
    macro_dashboard_dto_from_raw,
    portfolio_dto_from_raw,
)
from tools.archivist.core import VAULT_PATH
from tools.macro.dashboard import load_indicator_series
from api.routers.portfolio.common import _latest_strategy_json, handle_portfolio_exceptions

router = APIRouter(dependencies=[Depends(require_session)])


@router.get("/api/portfolio/latest", response_model=PortfolioDTO)
def get_latest_portfolio() -> PortfolioDTO:
    return portfolio_dto_from_raw(_latest_strategy_json())


@router.get("/api/macro/dashboard", response_model=MacroDashboardDTO)
def get_macro_dashboard() -> MacroDashboardDTO:
    return macro_dashboard_dto_from_raw(_latest_strategy_json())


@router.get("/api/macro/indicators/{indicator_id}/series", response_model=MacroIndicatorSeriesDTO)
def get_macro_indicator_series(indicator_id: str, range: str = "3m") -> MacroIndicatorSeriesDTO:
    raw = _latest_strategy_json()
    indicators = raw.get("dashboard_indicators", [])
    indicator = next(
        (
            item
            for item in indicators
            if isinstance(item, dict) and item.get("indicator_id") == indicator_id
        ),
        None,
    )
    if indicator is None:
        raise HTTPException(status_code=404, detail="Macro indicator not found in the latest report")
    if range not in {"1m", "3m", "1y"}:
        raise HTTPException(status_code=422, detail="range must be one of: 1m, 3m, 1y")

    try:
        import sys
        routes_portfolio = sys.modules.get("api.routes_portfolio")
        vault_path = getattr(routes_portfolio, "VAULT_PATH", VAULT_PATH) if routes_portfolio else VAULT_PATH
        points = load_indicator_series(vault_path, str(indicator.get("series_key", "")), range)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Macro indicator series is unavailable") from exc

    return MacroIndicatorSeriesDTO(
        indicator_id=indicator_id,
        series_key=str(indicator.get("series_key", "")),
        label=str(indicator.get("label", "")),
        unit=str(indicator.get("unit", "")),
        range=range,
        points=points,
    )


@router.get("/api/macro/news_funnel/pending", response_model=list[NewsFunnelPendingItemDTO])
def get_news_funnel_pending() -> list[NewsFunnelPendingItemDTO]:
    from tools.macro import news_funnel_store
    events = news_funnel_store.get_pending_high_impact_events()
    return [NewsFunnelPendingItemDTO.model_validate(e) for e in events]


@router.get("/api/macro/news_funnel/filtered", response_model=list[NewsFunnelFilteredItemDTO])
def get_news_funnel_filtered() -> list[NewsFunnelFilteredItemDTO]:
    from tools.macro import news_funnel_store
    events = news_funnel_store.get_filtered_or_rejected_events()
    return [NewsFunnelFilteredItemDTO.model_validate(e) for e in events]


@router.delete("/api/macro/news_funnel/pending/{event_id}")
def delete_news_funnel_pending(event_id: str) -> dict:
    from tools.macro import news_funnel_store
    from tools.macro.news_funnel import get_synthesis_period
    from api.news_funnel_cards import upsert_news_funnel_card

    news_funnel_store.update_events_status(rejected_ids=[event_id])
    remaining = news_funnel_store.get_pending_high_impact_events()
    upsert_news_funnel_card(get_synthesis_period(), remaining)
    return {"ok": True, "remaining_count": len(remaining)}


@router.get("/api/portfolio/calendar", response_model=PortfolioCalendarDTO)
def get_portfolio_calendar(portfolio_id: str = "default") -> PortfolioCalendarDTO:
    import sys
    routes_mod = sys.modules.get("api.routes_portfolio")
    portfolio_core = getattr(routes_mod, "portfolio_core", None) if routes_mod else None
    if portfolio_core is None:
        from tools.portfolio import core as portfolio_core

    portfolio_watchlist = getattr(routes_mod, "portfolio_watchlist", None) if routes_mod else None
    if portfolio_watchlist is None:
        from tools.portfolio import watchlist as portfolio_watchlist

    from tools.market.asset_resolver import resolve_asset
    from tools.market.calendar import get_asset_calendar

    CASH_SYMBOLS = {"CASH_THB", "CASH_USD", "CASH"}
    holding_symbols = []
    holding_names = {}
    with handle_portfolio_exceptions("Portfolio lock timeout"):
        state = portfolio_core.get_structured_portfolio_state(portfolio_id=portfolio_id)
        holding_symbols = [
            h.symbol for h in state.holdings
            if h.symbol and h.symbol not in CASH_SYMBOLS and h.asset_type != "Cash"
        ]
        holding_names = {
            h.symbol.strip().upper(): h.company_name for h in state.holdings
            if h.symbol and h.symbol not in CASH_SYMBOLS and h.asset_type != "Cash"
        }

    watchlist_symbols = []
    with handle_portfolio_exceptions("Watchlist lock timeout"):
        wl = portfolio_watchlist.get_structured_watchlist(portfolio_id=portfolio_id)
        watchlist_symbols = [
            w.symbol for w in wl.items
            if w.symbol and w.symbol not in CASH_SYMBOLS and w.asset_type != "Cash"
        ]

    ticker_items = []
    seen = set()

    for sym in holding_symbols:
        clean = sym.strip().upper()
        if clean and clean not in seen:
            seen.add(clean)
            resolved = resolve_asset(clean)
            if resolved and resolved.confidence == "low":
                continue
            provider_sym = resolved.provider_symbol if resolved else clean
            company_name = holding_names.get(clean)
            ticker_items.append({
                "symbol": clean,
                "provider_symbol": provider_sym,
                "company_name": company_name,
                "bucket": "holding"
            })

    for sym in watchlist_symbols:
        clean = sym.strip().upper()
        if clean and clean not in seen:
            seen.add(clean)
            resolved = resolve_asset(clean)
            if resolved and resolved.confidence == "low":
                continue
            provider_sym = resolved.provider_symbol if resolved else clean
            company_name = None
            ticker_items.append({
                "symbol": clean,
                "provider_symbol": provider_sym,
                "company_name": company_name,
                "bucket": "watchlist"
            })

    if not ticker_items:
        return PortfolioCalendarDTO(
            generated_at=datetime.now(timezone.utc).isoformat(),
            events=[],
            tickers_fetched=0,
            tickers_failed=[]
        )

    today = date.today()
    events: list[CalendarEventDTO] = []
    tickers_failed: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(get_asset_calendar, item["provider_symbol"]): item
            for item in ticker_items
        }

        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                cal = future.result()
                if not isinstance(cal, dict):
                    continue

                # 1. Parse Earnings Date
                earnings_dates = cal.get("Earnings Date")
                if isinstance(earnings_dates, list) and len(earnings_dates) > 0:
                    e_date = earnings_dates[0]
                    if isinstance(e_date, str):
                        try:
                            e_date = datetime.strptime(e_date, "%Y-%m-%d").date()
                        except ValueError:
                            e_date = None
                    elif isinstance(e_date, datetime):
                        e_date = e_date.date()

                    if isinstance(e_date, date):
                        e_iso = e_date.strftime("%Y-%m-%d")
                        d_until = (e_date - today).days

                        eps_avg = cal.get("Earnings Average")
                        eps_low = cal.get("Earnings Low")
                        eps_high = cal.get("Earnings High")
                        rev_avg = cal.get("Revenue Average")
                        rev_low = cal.get("Revenue Low")
                        rev_high = cal.get("Revenue High")

                        events.append(CalendarEventDTO(
                            ticker=item["symbol"],
                            company_name=item["company_name"],
                            event_type="earnings",
                            event_date=e_iso,
                            days_until=d_until,
                            bucket=item["bucket"],
                            earnings_avg=float(eps_avg) if eps_avg is not None else None,
                            earnings_low=float(eps_low) if eps_low is not None else None,
                            earnings_high=float(eps_high) if eps_high is not None else None,
                            revenue_avg=float(rev_avg) if rev_avg is not None else None,
                            revenue_low=float(rev_low) if rev_low is not None else None,
                            revenue_high=float(rev_high) if rev_high is not None else None,
                        ))

                # 2. Parse Ex-Dividend Date
                ex_div_date = cal.get("Ex-Dividend Date")
                if isinstance(ex_div_date, str):
                    try:
                        ex_div_date = datetime.strptime(ex_div_date, "%Y-%m-%d").date()
                    except ValueError:
                        ex_div_date = None
                elif isinstance(ex_div_date, datetime):
                    ex_div_date = ex_div_date.date()

                if isinstance(ex_div_date, date):
                    ex_iso = ex_div_date.strftime("%Y-%m-%d")
                    d_until = (ex_div_date - today).days
                    events.append(CalendarEventDTO(
                        ticker=item["symbol"],
                        company_name=item["company_name"],
                        event_type="ex_dividend",
                        event_date=ex_iso,
                        days_until=d_until,
                        bucket=item["bucket"]
                    ))

            except Exception:
                tickers_failed.append(item["symbol"])

    events.sort(key=lambda x: x.event_date)

    return PortfolioCalendarDTO(
        generated_at=datetime.now(timezone.utc).isoformat(),
        events=events,
        tickers_fetched=len(ticker_items) - len(tickers_failed),
        tickers_failed=tickers_failed
    )
