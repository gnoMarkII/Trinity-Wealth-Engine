import asyncio
from contextlib import contextmanager
import json
import threading
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from api.auth import require_session, get_session_id
from api.dependencies import get_portfolio_service
from api.schemas.dime_sync import (
    DimeBatchScanRequestDTO,
    DimeEmailListResponseDTO,
    DimeEmailMetadataDTO,
    DimeScanEmailRequestDTO,
    DimeStagedItemDTO,
    DimeStagedItemFeeDTO,
    DimeScanResponseDTO,
    DimeCommitRequestDTO,
    DimeCommitResponseDTO,
    DimePdfTextPageDTO,
    DimePdfTextResponseDTO,
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


router = APIRouter(prefix="/api/portfolio/dime", tags=["Dime Sync"], dependencies=[Depends(require_session)])


@contextmanager
def _map_dime_exceptions():
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"เกิดข้อผิดพลาดในการประมวลผล: {e}") from e


def _item_to_dto(it: TradeImportItem) -> DimeStagedItemDTO:
    price_str = f"{it.price:f}"
    if "." in price_str:
        price_str = price_str.rstrip("0").rstrip(".")
    return DimeStagedItemDTO(
        item_id=it.item_id,
        trade_date=it.trade_date,
        settlement_date=it.settlement_date,
        symbol=it.symbol,
        action=it.action,
        units=f"{it.units:g}",
        price=price_str,
        gross_amount=f"{it.gross_amount:.2f}",
        fees=DimeStagedItemFeeDTO(
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
    )


@router.get("/emails", response_model=DimeEmailListResponseDTO)
def search_dime_emails_endpoint(
    query: str = "",
    limit: int = 500,
    service: PortfolioService = Depends(get_portfolio_service),
) -> DimeEmailListResponseDTO:
    if not service._dime_sync_service:
        raise HTTPException(status_code=503, detail="Dime Sync Service is not available")
    with _map_dime_exceptions():
        items = service._dime_sync_service.scan_emails(query=query, limit=limit)
        return DimeEmailListResponseDTO(
            emails=[
                DimeEmailMetadataDTO(
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


@router.post("/scan/batch-stream")
async def scan_batch_stream_endpoint(
    payload: DimeBatchScanRequestDTO,
    session_id: str = Depends(get_session_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> StreamingResponse:
    if not service._dime_sync_service:
        raise HTTPException(status_code=503, detail="Dime Sync Service is not available")

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def sync_worker():
            try:
                for event in service._dime_sync_service.stream_batch_sync(
                    password=payload.password,
                    force_rescan=payload.force_rescan,
                    portfolio_id=payload.portfolio_id,
                    session_id=session_id,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"event": "error", "data": {"detail": str(e)}})
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=sync_worker, daemon=True).start()

        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=3.0)
                if item is None:
                    break
                evt_name = item.get("event", "message")
                data_str = json.dumps(item.get("data", {}), ensure_ascii=False)
                yield f"event: {evt_name}\ndata: {data_str}\n\n"
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


@router.post("/scan/email", response_model=DimeScanResponseDTO)
def scan_email_attachment_endpoint(
    payload: DimeScanEmailRequestDTO,
    session_id: str = Depends(get_session_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> DimeScanResponseDTO:
    if not service._dime_sync_service:
        raise HTTPException(status_code=503, detail="Dime Sync Service is not available")
    with _map_dime_exceptions():
        scan_id, items = service._dime_sync_service.parse_and_stage_email(
            message_id=payload.message_id,
            attachment_id=payload.attachment_id,
            password=payload.password,
            session_id=session_id,
        )
        dto_items = [_item_to_dto(it) for it in items]
        return DimeScanResponseDTO(
            scan_id=scan_id,
            item_count=len(dto_items),
            items=dto_items,
        )


@router.post("/scan/upload", response_model=DimeScanResponseDTO)
async def scan_upload_pdf_endpoint(
    pdf_file: UploadFile = File(...),
    password: Optional[str] = Form(None),
    session_id: str = Depends(get_session_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> DimeScanResponseDTO:
    if not service._dime_sync_service:
        raise HTTPException(status_code=503, detail="Dime Sync Service is not available")

    content = await pdf_file.read()
    with _map_dime_exceptions():
        scan_id, items = service._dime_sync_service.parse_and_stage_upload(
            pdf_bytes=content,
            password=password,
            session_id=session_id,
        )
        dto_items = [_item_to_dto(it) for it in items]
        return DimeScanResponseDTO(
            scan_id=scan_id,
            item_count=len(dto_items),
            items=dto_items,
        )


@router.get("/staged/{scan_id}", response_model=DimeScanResponseDTO)
def get_staged_endpoint(
    scan_id: str,
    session_id: str = Depends(get_session_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> DimeScanResponseDTO:
    if not service._dime_sync_service:
        raise HTTPException(status_code=503, detail="Dime Sync Service is not available")
    with _map_dime_exceptions():
        items = service._dime_sync_service.get_staged(scan_id=scan_id, session_id=session_id)
        dto_items = [_item_to_dto(it) for it in items]
        return DimeScanResponseDTO(
            scan_id=scan_id,
            item_count=len(dto_items),
            items=dto_items,
        )


@router.post("/commit/{scan_id}", response_model=DimeCommitResponseDTO)
def commit_staged_endpoint(
    scan_id: str,
    payload: DimeCommitRequestDTO,
    session_id: str = Depends(get_session_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> DimeCommitResponseDTO:
    if not service._dime_sync_service:
        raise HTTPException(status_code=503, detail="Dime Sync Service is not available")
    with _map_dime_exceptions():
        # Get count before commit
        items = service._dime_sync_service.get_staged(scan_id=scan_id, session_id=session_id)
        state = service._dime_sync_service.commit_staged(
            scan_id=scan_id,
            session_id=session_id,
            portfolio_id=payload.portfolio_id,
        )
        return DimeCommitResponseDTO(
            ok=True,
            imported_count=len(items),
            state=ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True)),
        )


@router.get("/pdf")
def get_dime_pdf_endpoint(
    message_id: str,
    attachment_id: str,
    password: Optional[str] = None,
    decrypt: bool = True,
    service: PortfolioService = Depends(get_portfolio_service),
) -> Response:
    if not service._dime_sync_service:
        raise HTTPException(status_code=503, detail="Dime Sync Service is not available")
    with _map_dime_exceptions():
        pdf_bytes, filename = service._dime_sync_service.get_email_pdf(
            message_id=message_id,
            attachment_id=attachment_id,
            password=password,
            decrypt=decrypt,
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Content-Type": "application/pdf",
                "X-Frame-Options": "SAMEORIGIN",
                "Content-Security-Policy": "frame-ancestors 'self' http://localhost:5173 http://localhost:8000 http://127.0.0.1:5173 http://127.0.0.1:8000",
            },
        )


@router.get("/pdf-text", response_model=DimePdfTextResponseDTO)
def get_dime_pdf_text_endpoint(
    message_id: str,
    attachment_id: str,
    password: Optional[str] = None,
    service: PortfolioService = Depends(get_portfolio_service),
) -> DimePdfTextResponseDTO:
    if not service._dime_sync_service:
        raise HTTPException(status_code=503, detail="Dime Sync Service is not available")
    with _map_dime_exceptions():
        data = service._dime_sync_service.get_email_pdf_text(
            message_id=message_id,
            attachment_id=attachment_id,
            password=password,
        )
        return DimePdfTextResponseDTO(
            message_id=data["message_id"],
            attachment_id=data["attachment_id"],
            filename=data["filename"],
            page_count=data["page_count"],
            pages=[
                DimePdfTextPageDTO(page_number=p["page_number"], text=p["text"])
                for p in data.get("pages", [])
            ],
        )


