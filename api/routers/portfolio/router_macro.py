"""Inbound HTTP adapter for Macro and Portfolio Calendar queries."""
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_session
from api.dependencies import (
    get_macro_service,
    get_portfolio_calendar_service,
)
from api.error_mapping import handle_domain_exceptions as handle_portfolio_exceptions
from api.schemas import (
    CalendarEventDTO,
    MacroDashboardDTO,
    MacroIndicatorSeriesDTO,
    NewsFunnelFilteredItemDTO,
    NewsFunnelPendingItemDTO,
    PortfolioCalendarDTO,
    PortfolioDTO,
    macro_dashboard_dto_from_raw,
    portfolio_dto_from_raw,
)
from application.macro.service import MacroApplicationService, PortfolioCalendarApplicationService

router = APIRouter(dependencies=[Depends(require_session)])


_NO_STRATEGY_DETAIL = (
    "ยังไม่มีรายงาน Macro Strategy ที่มี JSON sidecar — รอรายงานถัดไปหลัง Phase 0 อัปเดต"
)


@router.get("/api/portfolio/latest", response_model=PortfolioDTO)
def get_latest_portfolio(
    service: MacroApplicationService = Depends(get_macro_service),
) -> PortfolioDTO:
    try:
        return portfolio_dto_from_raw(service.latest_portfolio())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_NO_STRATEGY_DETAIL) from exc


@router.get("/api/macro/dashboard", response_model=MacroDashboardDTO)
def get_macro_dashboard(
    service: MacroApplicationService = Depends(get_macro_service),
) -> MacroDashboardDTO:
    try:
        return macro_dashboard_dto_from_raw(service.dashboard())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_NO_STRATEGY_DETAIL) from exc


@router.get("/api/macro/indicators/{indicator_id}/series", response_model=MacroIndicatorSeriesDTO)
def get_macro_indicator_series(
    indicator_id: str,
    range: str = "3m",
    service: MacroApplicationService = Depends(get_macro_service),
) -> MacroIndicatorSeriesDTO:
    try:
        return MacroIndicatorSeriesDTO.model_validate(service.indicator_series(indicator_id, range))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=_NO_STRATEGY_DETAIL) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/api/macro/news_funnel/pending", response_model=list[NewsFunnelPendingItemDTO])
def get_news_funnel_pending(
    service: MacroApplicationService = Depends(get_macro_service),
) -> list[NewsFunnelPendingItemDTO]:
    return [NewsFunnelPendingItemDTO.model_validate(item) for item in service.pending_news()]


@router.get("/api/macro/news_funnel/filtered", response_model=list[NewsFunnelFilteredItemDTO])
def get_news_funnel_filtered(
    service: MacroApplicationService = Depends(get_macro_service),
) -> list[NewsFunnelFilteredItemDTO]:
    return [NewsFunnelFilteredItemDTO.model_validate(item) for item in service.filtered_news()]


@router.delete("/api/macro/news_funnel/pending/{event_id}")
def delete_news_funnel_pending(
    event_id: str,
    service: MacroApplicationService = Depends(get_macro_service),
) -> dict:
    return service.reject_news(event_id)


@router.get("/api/portfolio/calendar", response_model=PortfolioCalendarDTO)
def get_portfolio_calendar(
    portfolio_id: str = "default",
    service: PortfolioCalendarApplicationService = Depends(get_portfolio_calendar_service),
) -> PortfolioCalendarDTO:
    with handle_portfolio_exceptions("Portfolio lock timeout"):
        return PortfolioCalendarDTO.model_validate(service.get_calendar(portfolio_id=portfolio_id))
