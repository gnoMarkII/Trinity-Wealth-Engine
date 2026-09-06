"""SCBAM Fund Click Synchronization Router (Hexagonal Driving Adapter)."""
import asyncio
from contextlib import contextmanager
import json
import threading
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse, HTMLResponse

from api.auth import require_session, get_session_id
from api.dependencies import get_portfolio_service
from api.schemas.scb_sync import (
    SCBAMBatchScanRequestDTO,
    SCBAMSingleScanEmailRequestDTO,
    SCBAMEmailListResponseDTO,
    SCBAMEmailMetadataDTO,
    SCBAMStagedItemDTO,
    SCBAMStagedItemFeeDTO,
    SCBAMScanResponseDTO,
    SCBAMCommitRequestDTO,
    SCBAMCommitResponseDTO,
    SCBAMEmailHtmlResponseDTO,
)
from api.schemas.portfolio import ActualPortfolioStateDTO
from tools.portfolio.domain.errors import (
    StagedScanExpiredError,
    StagedScanForbiddenError,
    StagedScanNotFoundError,
    TradeDuplicateError,
    TradeReconciliationError,
)
from tools.portfolio.domain.models import TradeImportItem
from tools.portfolio.service import PortfolioService

router = APIRouter(prefix="/api/portfolio/scb", tags=["SCBAM Sync"], dependencies=[Depends(require_session)])


@contextmanager
def _map_scb_exceptions():
    try:
        yield
    except StagedScanNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except StagedScanExpiredError as e:
        raise HTTPException(status_code=status.HTTP_410_GONE, detail=str(e))
    except StagedScanForbiddenError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except TradeDuplicateError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except TradeReconciliationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"เกิดข้อผิดพลาดในการประมวลผล SCBAM: {e}") from e


def _item_to_dto(it: TradeImportItem) -> SCBAMStagedItemDTO:
    price_str = f"{it.price:f}"
    if "." in price_str:
        price_str = price_str.rstrip("0").rstrip(".")
    return SCBAMStagedItemDTO(
        item_id=it.item_id,
        trade_date=it.trade_date,
        settlement_date=it.settlement_date,
        symbol=it.symbol,
        action=it.action,
        units=f"{it.units:g}",
        price=price_str,
        gross_amount=f"{it.gross_amount:.2f}",
        fees=SCBAMStagedItemFeeDTO(
            commission=f"{it.fees.commission:.2f}",
            vat=f"{it.fees.vat:.2f}",
            other_fees=f"{it.fees.other_fees:.2f}",
            fee_currency=it.fees.fee_currency,
        ),
        net_amount=f"{it.net_amount:.2f}",
        currency=it.currency,
        exchange_rate=f"{it.exchange_rate:.4f}" if it.exchange_rate else None,
        confirmation_no=it.confirmation_no,
        order_id=it.order_id,
        source=it.source,
        fingerprint=it.fingerprint,
        line_index=it.line_index,
        cash_adjusted=it.cash_adjusted,
        asset_type=getattr(it, "asset_type", "Mutual Fund"),
    )


@router.get("/emails", response_model=SCBAMEmailListResponseDTO)
def search_scbam_emails_endpoint(
    query: str = "",
    limit: int = 500,
    service: PortfolioService = Depends(get_portfolio_service),
) -> SCBAMEmailListResponseDTO:
    if not getattr(service, "_scbam_sync_service", None):
        raise HTTPException(status_code=503, detail="SCBAM Sync Service is not available")
    with _map_scb_exceptions():
        items = service._scbam_sync_service.email_source.search_scbam_emails(query=query, limit=limit)
        return SCBAMEmailListResponseDTO(
            emails=[
                SCBAMEmailMetadataDTO(
                    message_id=m.message_id,
                    attachment_id=m.attachment_id,
                    subject=m.subject,
                    sender=m.sender,
                    received_at=m.received_at,
                    filename=m.filename,
                    size_bytes=m.size_bytes,
                )
                for m in items
            ]
        )


@router.post("/scan/email", response_model=SCBAMScanResponseDTO)
def scan_email_endpoint(
    payload: SCBAMSingleScanEmailRequestDTO,
    session_id: str = Depends(get_session_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> SCBAMScanResponseDTO:
    if not getattr(service, "_scbam_sync_service", None):
        raise HTTPException(status_code=503, detail="SCBAM Sync Service is not available")
    with _map_scb_exceptions():
        scan_id, items = service._scbam_sync_service.scan_scbam_email(
            message_id=payload.message_id,
            portfolio_id=payload.portfolio_id,
        )
        dto_items = [_item_to_dto(it) for it in items]
        return SCBAMScanResponseDTO(
            scan_id=scan_id,
            item_count=len(dto_items),
            items=dto_items,
        )


@router.post("/scan/batch-stream")
async def scan_batch_stream_endpoint(
    payload: SCBAMBatchScanRequestDTO,
    session_id: str = Depends(get_session_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> StreamingResponse:
    if not getattr(service, "_scbam_sync_service", None):
        raise HTTPException(status_code=503, detail="SCBAM Sync Service is not available")

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def sync_worker():
            try:
                for raw_line in service._scbam_sync_service.stream_scbam_sync(
                    portfolio_id=payload.portfolio_id,
                    since_date=payload.since_date,
                    limit=payload.limit,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, raw_line)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as e:
                err_event = f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
                loop.call_soon_threadsafe(queue.put_nowait, err_event)
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=sync_worker, daemon=True).start()

        while True:
            try:
                raw_chunk = await asyncio.wait_for(queue.get(), timeout=3.0)
                if raw_chunk is None:
                    break
                yield raw_chunk
            except asyncio.TimeoutError:
                yield ": ping\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/staged/{scan_id}", response_model=SCBAMScanResponseDTO)
def get_staged_endpoint(
    scan_id: str,
    session_id: str = Depends(get_session_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> SCBAMScanResponseDTO:
    if not getattr(service, "_scbam_sync_service", None):
        raise HTTPException(status_code=503, detail="SCBAM Sync Service is not available")
    with _map_scb_exceptions():
        items = service._scbam_sync_service.staging.get_staged_items(scan_id)
        dto_items = [_item_to_dto(it) for it in items]
        return SCBAMScanResponseDTO(
            scan_id=scan_id,
            item_count=len(dto_items),
            items=dto_items,
        )


@router.post("/commit/{scan_id}", response_model=SCBAMCommitResponseDTO)
def commit_staged_endpoint(
    scan_id: str,
    payload: SCBAMCommitRequestDTO,
    session_id: str = Depends(get_session_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> SCBAMCommitResponseDTO:
    if not getattr(service, "_scbam_sync_service", None):
        raise HTTPException(status_code=503, detail="SCBAM Sync Service is not available")
    with _map_scb_exceptions():
        state, imported_count = service._scbam_sync_service.commit_scbam_sync(
            portfolio_id=payload.portfolio_id,
            scan_session_id=scan_id,
            selected_item_ids=payload.selected_item_ids,
        )
        return SCBAMCommitResponseDTO(
            ok=True,
            imported_count=imported_count,
            state=ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True)),
        )


@router.get("/emails/{email_id}/html", response_class=HTMLResponse)
def get_email_html_endpoint(
    email_id: str,
    service: PortfolioService = Depends(get_portfolio_service),
) -> HTMLResponse:
    if not getattr(service, "_scbam_sync_service", None):
        raise HTTPException(status_code=503, detail="SCBAM Sync Service is not available")
    with _map_scb_exceptions():
        html = service._scbam_sync_service.get_email_html(email_id)
        return HTMLResponse(content=html)
