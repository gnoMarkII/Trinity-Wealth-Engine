"""ARR Gold Trade Confirmation Synchronization Router (Hexagonal Driving Adapter)."""
import asyncio
from contextlib import contextmanager
import json
import threading
from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.auth import require_session, get_session_id
from api.dependencies import get_portfolio_service
from api.schemas.arr_gold_sync import (
    ArrGoldBatchScanRequestDTO,
    ArrGoldEmailListResponseDTO,
    ArrGoldEmailMetadataDTO,
    ArrGoldScanEmailRequestDTO,
    ArrGoldStagedItemDTO,
    ArrGoldStagedItemFeeDTO,
    ArrGoldScanResponseDTO,
    ArrGoldCommitRequestDTO,
    ArrGoldCommitResponseDTO,
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


router = APIRouter(prefix="/api/portfolio/arr-gold", tags=["ARR Gold Sync"], dependencies=[Depends(require_session)])


@contextmanager
def _map_arr_gold_exceptions():
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"เกิดข้อผิดพลาดในการประมวลผล ARR Gold: {e}") from e


def _item_or_dict_to_dto(it: Any) -> ArrGoldStagedItemDTO:
    if isinstance(it, dict):
        fees = it.get("fees") or {}
        if not isinstance(fees, dict):
            fees = getattr(fees, "model_dump", lambda: {})()
        return ArrGoldStagedItemDTO(
            item_id=str(it.get("item_id", "")),
            trade_date=str(it.get("trade_date", "")),
            settlement_date=it.get("settlement_date"),
            symbol=str(it.get("symbol", "GOLD-99.99")),
            action=str(it.get("action", "BUY")),
            units=str(it.get("units", "0")),
            price=str(it.get("price", "0")),
            gross_amount=str(it.get("gross_amount", "0")),
            fees=ArrGoldStagedItemFeeDTO(
                commission=str(fees.get("commission", "0.00")),
                vat=str(fees.get("vat", "0.00")),
                other_fees=str(fees.get("other_fees", "0.00")),
                fee_currency=str(fees.get("fee_currency", "USD")),
            ),
            net_amount=str(it.get("net_amount", "0")),
            currency=str(it.get("currency", "USD")),
            exchange_rate=str(it.get("exchange_rate")) if it.get("exchange_rate") is not None else None,
            confirmation_no=str(it.get("confirmation_no", "")),
            order_id=it.get("order_id"),
            source=str(it.get("source", "ARR_GOLD")),
            fingerprint=str(it.get("fingerprint", "")),
            line_index=int(it.get("line_index", 0)),
            cash_adjusted=bool(it.get("cash_adjusted", False)),
            asset_type=str(it.get("asset_type", "Gold")),
        )

    price_str = f"{it.price:f}"
    if "." in price_str:
        price_str = price_str.rstrip("0").rstrip(".")
    return ArrGoldStagedItemDTO(
        item_id=it.item_id,
        trade_date=it.trade_date,
        settlement_date=it.settlement_date,
        symbol=it.symbol,
        action=it.action,
        units=f"{it.units:g}",
        price=price_str,
        gross_amount=f"{it.gross_amount:.2f}",
        fees=ArrGoldStagedItemFeeDTO(
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
        asset_type=getattr(it, "asset_type", "Gold"),
    )


@router.get("/emails", response_model=ArrGoldEmailListResponseDTO)
def search_arr_gold_emails_endpoint(
    query: str = "",
    limit: int = 50,
    service: PortfolioService = Depends(get_portfolio_service),
) -> ArrGoldEmailListResponseDTO:
    if not getattr(service, "_arr_gold_sync_service", None):
        raise HTTPException(status_code=503, detail="ARR Gold Sync Service is not available")
    with _map_arr_gold_exceptions():
        items = service._arr_gold_sync_service.scan_emails(query=query, limit=limit)
        return ArrGoldEmailListResponseDTO(
            emails=[
                ArrGoldEmailMetadataDTO(
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


@router.post("/scan/email", response_model=ArrGoldScanResponseDTO)
def scan_email_endpoint(
    payload: ArrGoldScanEmailRequestDTO,
    session_id: str = Depends(get_session_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> ArrGoldScanResponseDTO:
    if not getattr(service, "_arr_gold_sync_service", None):
        raise HTTPException(status_code=503, detail="ARR Gold Sync Service is not available")
    with _map_arr_gold_exceptions():
        scan_id, previews, warnings = service._arr_gold_sync_service.scan_and_stage_email(
            message_id=payload.message_id,
            attachment_id=payload.attachment_id or "",
            subject=payload.subject or "",
            session_id=session_id,
        )
        dto_items = [_item_or_dict_to_dto(it) for it in previews]
        return ArrGoldScanResponseDTO(
            scan_id=scan_id,
            item_count=len(dto_items),
            items=dto_items,
            warnings=warnings,
        )


@router.get("/staged/{scan_id}", response_model=ArrGoldScanResponseDTO)
def get_staged_endpoint(
    scan_id: str,
    session_id: str = Depends(get_session_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> ArrGoldScanResponseDTO:
    if not getattr(service, "_arr_gold_sync_service", None):
        raise HTTPException(status_code=503, detail="ARR Gold Sync Service is not available")
    with _map_arr_gold_exceptions():
        items = service._arr_gold_sync_service.get_staged(scan_id=scan_id, session_id=session_id)
        dto_items = [_item_or_dict_to_dto(it) for it in items]
        return ArrGoldScanResponseDTO(
            scan_id=scan_id,
            item_count=len(dto_items),
            items=dto_items,
        )


@router.post("/commit/{scan_id}", response_model=ArrGoldCommitResponseDTO)
def commit_staged_endpoint(
    scan_id: str,
    payload: ArrGoldCommitRequestDTO,
    session_id: str = Depends(get_session_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> ArrGoldCommitResponseDTO:
    if not getattr(service, "_arr_gold_sync_service", None):
        raise HTTPException(status_code=503, detail="ARR Gold Sync Service is not available")
    with _map_arr_gold_exceptions():
        res = service._arr_gold_sync_service.commit_staged(
            scan_id=scan_id,
            portfolio_id=payload.portfolio_id,
            selected_item_ids=payload.selected_item_ids,
            session_id=session_id,
        )
        state = res.get("state")
        return ArrGoldCommitResponseDTO(
            ok=True,
            imported_count=res.get("imported_count", 0),
            state=ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True)),
        )


@router.post("/scan/batch-stream")
async def scan_batch_stream_endpoint(
    payload: ArrGoldBatchScanRequestDTO,
    session_id: str = Depends(get_session_id),
    service: PortfolioService = Depends(get_portfolio_service),
) -> StreamingResponse:
    if not getattr(service, "_arr_gold_sync_service", None):
        raise HTTPException(status_code=503, detail="ARR Gold Sync Service is not available")

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def sync_worker():
            try:
                for sse_chunk in service._arr_gold_sync_service.stream_batch_sync(
                    portfolio_id=payload.portfolio_id,
                    session_id=session_id,
                    force_rescan=payload.force_rescan,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, sse_chunk)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as e:
                err_msg = json.dumps({"message": f"Sync worker failed: {str(e)}"}, ensure_ascii=False)
                loop.call_soon_threadsafe(queue.put_nowait, f"event: error\ndata: {err_msg}\n\n")
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=sync_worker, daemon=True).start()

        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=5.0)
                if item is None:
                    break
                yield item
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
