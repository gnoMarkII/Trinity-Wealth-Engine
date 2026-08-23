"""FastAPI Sub-router for Portfolio Transactions and Ledger Replay."""
import sys
from typing import Optional
from fastapi import APIRouter, Depends

from api.auth import require_session
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


def _get_mod(name: str):
    routes_mod = sys.modules.get("api.routes_portfolio")
    if routes_mod and hasattr(routes_mod, name):
        return getattr(routes_mod, name)
    if name == "portfolio_trading":
        from tools.portfolio import trading as mod
        return mod
    elif name == "portfolio_ledger_replay":
        from tools.portfolio import ledger_replay as mod
        return mod
    return None


@router.get(
    "/api/portfolio/actual/transactions",
    response_model=TransactionListResponseDTO,
)
def get_actual_transactions(
    symbol: Optional[str] = None,
    portfolio_id: str = "default",
) -> TransactionListResponseDTO:
    with handle_portfolio_exceptions("Transactions lock timeout"):
        portfolio_trading = _get_mod("portfolio_trading")
        rows = portfolio_trading.get_structured_trades_log(
            portfolio_id=portfolio_id, symbol=symbol
        )
        tx_items = [TransactionItemDTO.model_validate(r) for r in rows]

        # Calculate summary statistics
        total_buy_count = sum(1 for t in tx_items if t.action == "BUY")
        total_sell_count = sum(1 for t in tx_items if t.action == "SELL")
        total_buy_thb = sum(t.cost_thb for t in tx_items if t.action == "BUY")
        total_sell_thb = sum(t.cost_thb for t in tx_items if t.action == "SELL")
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


@router.patch(
    "/api/portfolio/actual/transactions/{tx_id}/note",
    response_model=TransactionItemDTO,
)
def update_transaction_note_endpoint(
    tx_id: str,
    payload: UpdateTransactionNoteRequestDTO,
    portfolio_id: str = "default",
) -> TransactionItemDTO:
    with handle_portfolio_exceptions("Transactions lock timeout"):
        portfolio_trading = _get_mod("portfolio_trading")
        updated = portfolio_trading.update_trade_note(
            tx_id=tx_id, notes=payload.notes, portfolio_id=portfolio_id
        )
        return TransactionItemDTO.model_validate(updated)


@router.put(
    "/api/portfolio/actual/transactions/{tx_id}",
    response_model=ActualPortfolioStateDTO,
)
def edit_transaction_endpoint(
    tx_id: str,
    payload: EditTransactionRequestDTO,
    portfolio_id: str = "default",
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions("Transactions lock timeout"):
        portfolio_ledger_replay = _get_mod("portfolio_ledger_replay")
        state = portfolio_ledger_replay.edit_transaction(
            tx_id=tx_id,
            timestamp=payload.timestamp,
            units=payload.units,
            price=payload.price,
            fx_rate=payload.fx_rate,
            notes=payload.notes,
            adjust_cash=payload.adjust_cash,
            portfolio_id=portfolio_id,
        )
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.delete(
    "/api/portfolio/actual/transactions/{tx_id}",
    response_model=ActualPortfolioStateDTO,
)
def delete_transaction_endpoint(
    tx_id: str,
    adjust_cash: bool = True,
    portfolio_id: str = "default",
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions("Transactions lock timeout"):
        portfolio_ledger_replay = _get_mod("portfolio_ledger_replay")
        state = portfolio_ledger_replay.delete_transaction(
            tx_id=tx_id,
            adjust_cash=adjust_cash,
            portfolio_id=portfolio_id,
        )
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )
