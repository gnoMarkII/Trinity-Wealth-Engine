"""Authenticated endpoints for BBLAM PVD statement analysis, commit, and archive access."""
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from api.auth import get_session_id, require_session
from api.dependencies import get_pvd_sync_service
from api.schemas.pvd_sync import PvdAnalyzeResponse, PvdCommitRequest, PvdCommitResponse
from tools.portfolio.domain.errors import StagedScanExpiredError, StagedScanForbiddenError, StagedScanNotFoundError
from tools.portfolio.services.pvd_sync_service import MAX_PDF_BYTES, PvdSyncService

router = APIRouter(prefix="/api/portfolio/pvd", tags=["Portfolio PVD"], dependencies=[Depends(require_session)])


@router.post("/analyze-upload", response_model=PvdAnalyzeResponse)
async def analyze_pvd_upload(
    file: UploadFile = File(...),
    portfolio_id: str = Form("default"),
    password: Optional[str] = Form(None),
    vesting_percent: str = Form("100"),
    session_id: str = Depends(get_session_id),
    service: PvdSyncService = Depends(get_pvd_sync_service),
) -> PvdAnalyzeResponse:
    if Path(file.filename or "").suffix.lower() != ".pdf":
        raise HTTPException(status_code=422, detail="กรุณาเลือกไฟล์ PDF")
    pdf_bytes = await file.read(MAX_PDF_BYTES + 1)
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="ขนาดไฟล์ PDF ต้องไม่เกิน 10 MB")
    try:
        vesting = Decimal(vesting_percent)
    except InvalidOperation as exc:
        raise HTTPException(status_code=422, detail="Vesting ต้องเป็นตัวเลข") from exc
    try:
        result = await run_in_threadpool(
            service.analyze_upload,
            pdf_bytes,
            portfolio_id=portfolio_id,
            session_id=session_id,
            password=password,
            vesting_percent=vesting,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PvdAnalyzeResponse(**result.__dict__)


@router.post("/commit", response_model=PvdCommitResponse)
def commit_pvd_statement(
    payload: PvdCommitRequest,
    session_id: str = Depends(get_session_id),
    service: PvdSyncService = Depends(get_pvd_sync_service),
) -> PvdCommitResponse:
    try:
        result = service.commit(
            payload.scan_id,
            portfolio_id=payload.portfolio_id,
            session_id=session_id,
            idempotency_key=payload.idempotency_key,
            vesting_percent=payload.vesting_percent,
            selected_item_ids=payload.selected_item_ids,
        )
    except StagedScanExpiredError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except (StagedScanNotFoundError,) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StagedScanForbiddenError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return PvdCommitResponse(status=result.status, message=result.message, archive_id=result.archive_id, state=result.state)


@router.get("/documents/{archive_id}")
def get_pvd_document(
    archive_id: str,
    portfolio_id: str = "default",
    service: PvdSyncService = Depends(get_pvd_sync_service),
) -> Response:
    try:
        data = service.read_document(archive_id, portfolio_id=portfolio_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="pvd-{archive_id[:12]}.pdf"'},
    )
