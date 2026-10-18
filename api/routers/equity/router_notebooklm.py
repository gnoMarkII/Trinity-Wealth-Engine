"""HTTP router for Equity NotebookLM research export endpoints.

Hexagonal Architecture Invariant:
Depends solely on EquityNotebookLMExportService via dependency injection.
Does NOT perform direct filesystem I/O, database queries, or external network calls.
"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_equity_notebooklm_export_service
from api.schemas.equity_notebooklm import (
    EquityNotebookLMExportRequestDTO,
    EquityNotebookLMExportResponseDTO,
    EquityNotebookLMExportStatusDTO,
    equity_export_record_to_dto,
)
from application.equity.notebooklm_export_ports import EquityExportBlockedError
from application.equity.notebooklm_export_service import EquityNotebookLMExportService

router = APIRouter()


@router.post(
    "/{ticker}/notebooklm/exports",
    status_code=202,
    response_model=EquityNotebookLMExportResponseDTO,
    summary="Freeze equity corpus bundle and dispatch to NotebookLM research pipeline",
)
def export_equity_to_notebooklm(
    ticker: str,
    payload: Optional[EquityNotebookLMExportRequestDTO] = None,
    service: EquityNotebookLMExportService = Depends(get_equity_notebooklm_export_service),
) -> EquityNotebookLMExportResponseDTO:
    market = (payload.market if payload and payload.market else "US").strip().upper()
    mode = (payload.mode if payload else "all_retained") or "all_retained"

    try:
        record = service.request_export(ticker=ticker, market=market, mode=mode)
    except EquityExportBlockedError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "equity_export_blocked", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_request", "message": str(exc)},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "equity_export_failed", "message": f"Failed to prepare export: {exc}"},
        ) from exc

    return EquityNotebookLMExportResponseDTO(
        export_id=record.export_id,
        ticker=record.ticker,
        market=record.market,
        job_id=record.job_id,
        state=record.state,
        stage=record.stage,
        message=f"Equity research bundle for {record.ticker} frozen and dispatched to NotebookLM worker",
    )


@router.get(
    "/{ticker}/notebooklm/exports/latest",
    response_model=Optional[EquityNotebookLMExportStatusDTO],
    summary="Get the most recent NotebookLM export status for a ticker",
)
def get_latest_equity_notebooklm_export(
    ticker: str,
    market: str = Query("US", description="Market identifier (e.g. US, TH)"),
    service: EquityNotebookLMExportService = Depends(get_equity_notebooklm_export_service),
) -> Optional[EquityNotebookLMExportStatusDTO]:
    record = service.get_latest_export(ticker=ticker, market=market)
    if record is None:
        return None
    return equity_export_record_to_dto(record)


@router.get(
    "/notebooklm/exports/{export_id}",
    response_model=EquityNotebookLMExportStatusDTO,
    summary="Get export status and manifest source details by export ID",
)
def get_equity_notebooklm_export_by_id(
    export_id: str,
    service: EquityNotebookLMExportService = Depends(get_equity_notebooklm_export_service),
) -> EquityNotebookLMExportStatusDTO:
    record = service.get_export(export_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "equity_export_not_found", "message": f"Export '{export_id}' not found"},
        )
    return equity_export_record_to_dto(record)


@router.post(
    "/notebooklm/exports/{export_id}/retry",
    status_code=202,
    response_model=EquityNotebookLMExportResponseDTO,
    summary="Retry a failed or partial equity export using its existing frozen bundle",
)
def retry_equity_notebooklm_export(
    export_id: str,
    service: EquityNotebookLMExportService = Depends(get_equity_notebooklm_export_service),
) -> EquityNotebookLMExportResponseDTO:
    try:
        record = service.retry_export(export_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "equity_export_not_found", "message": str(exc)},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "retry_failed", "message": str(exc)},
        ) from exc

    return EquityNotebookLMExportResponseDTO(
        export_id=record.export_id,
        ticker=record.ticker,
        market=record.market,
        job_id=record.job_id,
        state=record.state,
        stage=record.stage,
        message=f"Export retry for {record.ticker} dispatched to NotebookLM worker",
    )
