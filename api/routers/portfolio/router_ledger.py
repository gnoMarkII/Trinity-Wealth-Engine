"""FastAPI Sub-router for Portfolio Transactions and Ledger Replay."""
from typing import Optional
from fastapi import APIRouter, Depends, Response

from api.auth import require_session
from api.dependencies import get_portfolio_service
from tools.portfolio.service import PortfolioService
from api.schemas import (
    ActualPortfolioStateDTO,
    TransactionItemDTO,
    TransactionSummaryDTO,
    TransactionListResponseDTO,
    UpdateTransactionNoteRequestDTO,
    EditTransactionRequestDTO,
)
from api.routers.portfolio.common import handle_portfolio_exceptions

router = APIRouter(dependencies=[Depends(require_session)])


@router.get("/api/portfolio/actual/transactions", response_model=TransactionListResponseDTO)
def get_actual_transactions(
    symbol: Optional[str] = None,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> TransactionListResponseDTO:
    with handle_portfolio_exceptions("Transactions lock timeout"):
        rows = service.get_structured_trades_log(portfolio_id=portfolio_id, symbol=symbol)
        tx_items = [TransactionItemDTO.model_validate(r) for r in rows]

        total_buy_count = sum(1 for t in tx_items if t.action == "BUY")
        total_sell_count = sum(1 for t in tx_items if t.action == "SELL")
        total_buy_thb = sum(t.cost_thb for t in tx_items if t.action == "BUY")
        total_sell_thb = sum(
            (t.cost_thb + (t.realized_pnl_thb or 0.0))
            for t in tx_items
            if t.action == "SELL"
        )
        total_realized_pnl_thb = sum(
            t.realized_pnl_thb for t in tx_items if t.realized_pnl_thb is not None
        )

        summary = TransactionSummaryDTO(
            total_buy_count=total_buy_count,
            total_sell_count=total_sell_count,
            total_buy_thb=total_buy_thb,
            total_sell_thb=total_sell_thb,
            total_realized_pnl_thb=total_realized_pnl_thb,
        )
        return TransactionListResponseDTO(
            portfolio_id=portfolio_id,
            transactions=tx_items,
            summary=summary,
        )


@router.patch("/api/portfolio/actual/transactions/{tx_id}/note", response_model=TransactionItemDTO)
def update_transaction_note_endpoint(
    tx_id: str,
    payload: UpdateTransactionNoteRequestDTO,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> TransactionItemDTO:
    with handle_portfolio_exceptions("Transactions lock timeout"):
        updated = service.update_trade_note(
            tx_id=tx_id, notes=payload.notes, portfolio_id=portfolio_id
        )
        return TransactionItemDTO.model_validate(updated)


@router.put("/api/portfolio/actual/transactions/{tx_id}", response_model=ActualPortfolioStateDTO)
def edit_transaction_endpoint(
    tx_id: str,
    payload: EditTransactionRequestDTO,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions("Transactions lock timeout"):
        state = service.edit_transaction(
            tx_id=tx_id,
            timestamp=payload.timestamp,
            units=payload.units,
            price=payload.price,
            fx_rate=payload.fx_rate,
            notes=payload.notes,
            adjust_cash=payload.adjust_cash,
            portfolio_id=portfolio_id,
        )
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.post("/api/portfolio/actual/transactions/{tx_id}/void", response_model=ActualPortfolioStateDTO)
def void_transaction_endpoint(
    tx_id: str,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions("Transactions lock timeout"):
        state = service.void_transaction(
            tx_id=tx_id,
            portfolio_id=portfolio_id,
        )
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.delete("/api/portfolio/actual/transactions/{tx_id}", response_model=ActualPortfolioStateDTO)
def delete_transaction_endpoint(
    tx_id: str,
    response: Response,
    adjust_cash: bool = True,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    response.headers["X-Deprecation-Warning"] = (
        "DELETE /api/portfolio/actual/transactions/{tx_id} is deprecated and will be removed in v2.0; "
        "use POST /api/portfolio/actual/transactions/{tx_id}/void instead"
    )
    with handle_portfolio_exceptions("Transactions lock timeout"):
        state = service.void_transaction(
            tx_id=tx_id,
            portfolio_id=portfolio_id,
        )
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))
