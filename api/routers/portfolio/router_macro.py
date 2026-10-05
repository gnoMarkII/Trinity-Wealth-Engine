"""Inbound HTTP adapter for Macro and Portfolio Calendar queries."""
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from datetime import datetime, timezone

from api.auth import require_session
from api.dependencies import (
    get_macro_service,
    get_portfolio_calendar_service,
    get_sector_rotation_service,
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
from application.macro.sector_rotation_service import SectorRotationApplicationService
from api.schemas.sector_rotation import (
    SectorRotationHistoryDTO,
    SectorRotationResponseDTO,
)

router = APIRouter(dependencies=[Depends(require_session)])


@router.get("/api/macro/sector-rotation/latest", response_model=SectorRotationResponseDTO)
def get_sector_rotation_latest(
    response: Response,
    timeframe: str = Query("weekly", pattern="^(daily|weekly)$"),
    tail: int = Query(12, ge=1, le=60),
    service: SectorRotationApplicationService = Depends(get_sector_rotation_service),
) -> SectorRotationResponseDTO:
    try:
        payload = service.latest(timeframe=timeframe, tail=tail)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if payload["snapshot"] is None and payload["refresh_state"] == "running":
        response.status_code = 202
    elif payload["snapshot"] is None and payload["refresh_state"] == "failed":
        response.status_code = 503
    return SectorRotationResponseDTO.model_validate(payload)


@router.post("/api/macro/sector-rotation/refresh", response_model=SectorRotationResponseDTO)
def refresh_sector_rotation(
    response: Response,
    timeframe: str = Query("weekly", pattern="^(daily|weekly)$"),
    tail: int = Query(12, ge=1, le=60),
    service: SectorRotationApplicationService = Depends(get_sector_rotation_service),
) -> SectorRotationResponseDTO:
    service.request_refresh(force=True)
    payload = service.latest(timeframe=timeframe, tail=tail)
    if payload["snapshot"] is None and payload["refresh_state"] == "running":
        response.status_code = 202
    return SectorRotationResponseDTO.model_validate(payload)


@router.get("/api/macro/sector-rotation/history", response_model=SectorRotationHistoryDTO)
def get_sector_rotation_history(
    snapshot_id: str = Query(..., min_length=1, max_length=96),
    timeframe: str = Query("weekly", pattern="^(daily|weekly)$"),
    range_name: str = Query("1y", alias="range", pattern="^(3m|6m|1y|2y)$"),
    service: SectorRotationApplicationService = Depends(get_sector_rotation_service),
) -> SectorRotationHistoryDTO:
    try:
        payload = service.history(snapshot_id, timeframe=timeframe, range_name=range_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={"code": "sector_snapshot_integrity_failure"}) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail={"code": "sector_snapshot_not_found"})
    return SectorRotationHistoryDTO.model_validate(payload)


@router.get("/api/macro/sector-rotation/snapshots/{snapshot_id}", response_model=SectorRotationResponseDTO)
def get_sector_rotation_snapshot(
    snapshot_id: str,
    timeframe: str = Query("weekly", pattern="^(daily|weekly)$"),
    tail: int = Query(12, ge=1, le=60),
    service: SectorRotationApplicationService = Depends(get_sector_rotation_service),
) -> SectorRotationResponseDTO:
    try:
        snapshot = service.get_snapshot(snapshot_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={"code": "sector_snapshot_integrity_failure"}) from exc
    if snapshot is None:
        raise HTTPException(status_code=404, detail={"code": "sector_snapshot_not_found"})
    payload = service._view(snapshot, timeframe, tail)
    return SectorRotationResponseDTO.model_validate({
        "capability_status": "enabled", "refresh_state": "idle", "served_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "timeframe": timeframe, "tail": tail, "summary": service.summary(snapshot, timeframe=timeframe), "snapshot": payload,
    })


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


@router.get("/api/macro/reports/{strategy_report_id}", response_model=MacroDashboardDTO)
def get_macro_report_by_id(
    strategy_report_id: str,
    service: MacroApplicationService = Depends(get_macro_service),
) -> MacroDashboardDTO:
    try:
        return macro_dashboard_dto_from_raw(service.report_by_id(strategy_report_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail={"code": "macro_report_not_found", "message": str(exc)}) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={"code": "macro_report_integrity_failure"}) from exc


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
