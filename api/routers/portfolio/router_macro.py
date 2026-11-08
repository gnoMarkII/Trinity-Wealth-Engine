"""Inbound HTTP adapter for Macro and Portfolio Calendar queries."""
from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from datetime import datetime, timezone

from api.auth import require_session

from api.dependencies import (
    get_digest_service,
    get_macro_notebooklm_export_service,
    get_macro_service,
    get_portfolio_calendar_service,
    get_sector_rotation_service,
    get_macro_vintage_service,
)
from api.error_mapping import handle_domain_exceptions as handle_portfolio_exceptions
from api.schemas import (
    DigestCandidatesResponseDTO,
    DailyDigestResponseDTO,
    DailyDigestSynthesizeRequestDTO,
    WeeklyDigestResponseDTO,
    WeeklyDigestSynthesizeRequestDTO,
    DigestReadStateRequestDTO,
    DigestReadStateResponseDTO,
    CalendarEventDTO,
    MacroAsOfRowDTO,
    MacroVintageWalkResponseDTO,
    MacroDashboardDTO,
    MacroIndicatorSeriesDTO,
    MacroNotebookLMExportRequestDTO,
    MacroNotebookLMExportResponseDTO,
    MacroNotebookLMExportStatusDTO,
    macro_export_record_to_dto,
    NewsFunnelFilteredItemDTO,
    NewsFunnelPendingItemDTO,
    PortfolioCalendarDTO,
    PortfolioDTO,
    macro_dashboard_dto_from_raw,
    portfolio_dto_from_raw,
)
from application.macro.service import MacroApplicationService, PortfolioCalendarApplicationService
from application.macro.sector_rotation_service import SectorRotationApplicationService
from application.macro.vintage_service import MacroVintageApplicationService
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


from application.macro.notebooklm_export_service import MacroNotebookLMExportService


@router.post(
    "/api/macro/notebooklm/exports",
    status_code=202,
    response_model=MacroNotebookLMExportResponseDTO,
)
def export_macro_to_notebooklm(
    payload: Optional[MacroNotebookLMExportRequestDTO] = None,
    service: MacroNotebookLMExportService = Depends(get_macro_notebooklm_export_service),
) -> MacroNotebookLMExportResponseDTO:
    mode = payload.mode if payload else "all_retained"
    try:
        record = service.request_export(mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to prepare macro export: {exc}") from exc
    return MacroNotebookLMExportResponseDTO(
        export_id=record.export_id,
        job_id=record.job_id,
        state=record.state,
        stage=record.stage,
        message="Macro export bundle prepared and dispatched to NotebookLM worker",
    )


@router.get(
    "/api/macro/notebooklm/exports/latest",
    response_model=Optional[MacroNotebookLMExportStatusDTO],
)
def get_latest_macro_notebooklm_export(
    service: MacroNotebookLMExportService = Depends(get_macro_notebooklm_export_service),
) -> Optional[MacroNotebookLMExportStatusDTO]:
    record = service.get_latest_export()
    if record is None:
        return None
    return macro_export_record_to_dto(record)


@router.get(
    "/api/macro/notebooklm/exports/{export_id}",
    response_model=MacroNotebookLMExportStatusDTO,
)
def get_macro_notebooklm_export_by_id(
    export_id: str,
    service: MacroNotebookLMExportService = Depends(get_macro_notebooklm_export_service),
) -> MacroNotebookLMExportStatusDTO:
    record = service.get_export(export_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "macro_export_not_found", "message": f"Export '{export_id}' not found"},
        )
    return macro_export_record_to_dto(record)


@router.post(
    "/api/macro/notebooklm/exports/{export_id}/retry",
    status_code=202,
    response_model=MacroNotebookLMExportResponseDTO,
)
def retry_macro_notebooklm_export(
    export_id: str,
    service: MacroNotebookLMExportService = Depends(get_macro_notebooklm_export_service),
) -> MacroNotebookLMExportResponseDTO:
    try:
        record = service.retry_export(export_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "macro_export_not_found", "message": str(exc)},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retry macro export: {exc}") from exc
    return MacroNotebookLMExportResponseDTO(
        export_id=record.export_id,
        job_id=record.job_id,
        state=record.state,
        stage=record.stage,
        message="Macro export retry dispatched to NotebookLM worker",
    )


# ---------------------------------------------------------------------------
# Daily and Weekly Market Intelligence Digest Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/api/macro/digest/candidates",
    response_model=DigestCandidatesResponseDTO,
)
def get_digest_candidates(
    days: int = Query(7, ge=1, le=14),
    force_refresh: bool = Query(False),
    max_news_age_hours: Optional[int] = Query(None, ge=1, le=336),
    service: Any = Depends(get_digest_service),
) -> DigestCandidatesResponseDTO:
    """Fetch pre-LLM news and YouTube candidates with freshness verification."""
    try:
        data = service.get_candidates(days=days, force_refresh=force_refresh, max_news_age_hours=max_news_age_hours)
        return DigestCandidatesResponseDTO.model_validate(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch candidates: {exc}") from exc


@router.get(
    "/api/macro/digest/daily",
    response_model=DailyDigestResponseDTO,
)
def get_daily_digest(
    date: Optional[str] = Query(None, description="Optional ISO date string YYYY-MM-DD"),
    service: Any = Depends(get_digest_service),
) -> DailyDigestResponseDTO:
    """Get latest or historical daily digest with freshness and read status."""
    try:
        data = service.get_daily_digest(target_date=date)
        return DailyDigestResponseDTO.model_validate(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load daily digest: {exc}") from exc


@router.post(
    "/api/macro/digest/daily/synthesize",
    response_model=DailyDigestResponseDTO,
)
def synthesize_daily_digest(
    payload: DailyDigestSynthesizeRequestDTO,
    service: Any = Depends(get_digest_service),
) -> DailyDigestResponseDTO:
    """Trigger single-call one-shot LLM synthesis for Daily Digest."""
    try:
        data = service.synthesize_daily(
            selected_video_ids=payload.selected_video_ids,
            excluded_news_ids=payload.excluded_news_ids,
            custom_youtube_urls=payload.custom_youtube_urls,
            custom_news_items=payload.custom_news_items,
        )
        return DailyDigestResponseDTO.model_validate(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to synthesize daily digest: {exc}") from exc


@router.get(
    "/api/macro/digest/weekly",
    response_model=WeeklyDigestResponseDTO,
)
def get_weekly_digest(
    date: Optional[str] = Query(None, description="Optional target date YYYY-MM-DD"),
    service: Any = Depends(get_digest_service),
) -> WeeklyDigestResponseDTO:
    """Get latest or historical weekly digest."""
    try:
        data = service.get_weekly_digest(target_date=date)
        return WeeklyDigestResponseDTO.model_validate(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load weekly digest: {exc}") from exc


@router.post(
    "/api/macro/digest/weekly/synthesize",
    response_model=WeeklyDigestResponseDTO,
)
def synthesize_weekly_digest(
    payload: WeeklyDigestSynthesizeRequestDTO,
    service: Any = Depends(get_digest_service),
) -> WeeklyDigestResponseDTO:
    """Synthesize weekly multi-asset intelligence matrix."""
    try:
        data = service.synthesize_weekly(
            days=payload.days,
            selected_video_ids=payload.selected_video_ids,
            excluded_news_ids=payload.excluded_news_ids,
            custom_youtube_urls=payload.custom_youtube_urls,
            custom_news_items=payload.custom_news_items,
        )
        return WeeklyDigestResponseDTO.model_validate(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to synthesize weekly digest: {exc}") from exc


@router.post(
    "/api/macro/digest/read-state",
    response_model=DigestReadStateResponseDTO,
)
def update_digest_read_state(
    payload: DigestReadStateRequestDTO,
    service: Any = Depends(get_digest_service),
) -> DigestReadStateResponseDTO:
    """Mark digest items as read or unread."""
    try:
        data = service.update_read_state(
            item_ids=payload.item_ids,
            mark_as_read=payload.mark_as_read,
        )
        return DigestReadStateResponseDTO.model_validate(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update read state: {exc}") from exc


@router.get(
    "/api/macro/vintage-walk",
    response_model=MacroVintageWalkResponseDTO,
    summary="Point-in-Time Macro Vintage Walk & Subsequent Revision (Chapter 6 & App H)",
)
def get_macro_vintage_walk_endpoint(
    cutoff_date: Optional[str] = Query(None, description="Point-in-time cutoff date YYYY-MM-DD"),
    service: MacroVintageApplicationService = Depends(get_macro_vintage_service),
) -> MacroVintageWalkResponseDTO:
    """Calculates Point-in-Time Macro Vintage As-Of values and subsequent revisions without look-ahead bias."""
    target_cutoff = cutoff_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    walk = service.get_vintage_walk(cutoff_date=target_cutoff)
    rows = [
        MacroAsOfRowDTO(
            series_id=r.series_id,
            observation_date=r.observation_date,
            cutoff_date=r.cutoff_date,
            selected_vintage_date=r.selected_vintage_date,
            asof_value=float(r.asof_value) if r.asof_value is not None else None,
            latest_vintage_date=r.latest_vintage_date,
            latest_value=float(r.latest_value) if r.latest_value is not None else None,
            revision_after_cutoff=float(r.revision_after_cutoff) if r.revision_after_cutoff is not None else None,
            unit=r.unit,
            seasonal_adjustment=r.seasonal_adjustment,
            status=r.status,
            evidence_label=r.evidence_label,
        )
        for r in walk
    ]
    return MacroVintageWalkResponseDTO(
        cutoff_date=target_cutoff,
        rows=rows,
        total_series_count=len(rows),
        status="REVIEW_REQUIRED",
    )
