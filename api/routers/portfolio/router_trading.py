"""FastAPI Sub-router for Trades, Cash Flows, Incomes, FX Rates, and Holding Edits."""
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends

from api.auth import require_session
from api.dependencies import get_portfolio_service
from tools.portfolio.service import PortfolioService
from api.schemas import (
    ActualPortfolioStateDTO,
    TradeRequestDTO,
    CashFlowRequestDTO,
    IncomeRequestDTO,
    EditHoldingRequestDTO,
    FXRateResponseDTO,
    SyncDividendsResponseDTO,
)
from api.routers.portfolio.common import handle_portfolio_exceptions

router = APIRouter(dependencies=[Depends(require_session)])


@router.post("/api/portfolio/actual/trade", response_model=ActualPortfolioStateDTO)
def execute_trade_endpoint(
    payload: TradeRequestDTO,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = service.structured_execute_trade(
            symbol=payload.symbol,
            asset_type=payload.asset_type,
            action=payload.action,
            units=payload.units,
            price=payload.price,
            currency=payload.currency,
            exchange_rate=payload.exchange_rate,
            date=payload.date,
            notes=payload.notes,
            bucket_id=payload.bucket_id,
            portfolio_id=portfolio_id,
        )
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.post("/api/portfolio/actual/cashflow", response_model=ActualPortfolioStateDTO)
def manage_cash_flow_endpoint(
    payload: CashFlowRequestDTO,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = service.structured_manage_cash_flow(
            amount=payload.amount,
            action=payload.action,
            currency=payload.currency,
            exchange_rate=payload.exchange_rate,
            date=payload.date,
            notes=payload.notes,
            portfolio_id=portfolio_id,
        )
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.post("/api/portfolio/actual/income", response_model=ActualPortfolioStateDTO)
def record_income_endpoint(
    payload: IncomeRequestDTO,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = service.structured_record_income(
            income_type=payload.income_type,
            amount_thb=payload.amount_thb,
            source_symbol=payload.source_symbol,
            date=payload.date,
            notes=payload.notes,
            portfolio_id=portfolio_id,
        )
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.put("/api/portfolio/actual/holdings/{symbol}/edit", response_model=ActualPortfolioStateDTO)
def edit_holding_endpoint(
    symbol: str,
    payload: EditHoldingRequestDTO,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = service.structured_edit_holding(
            symbol=symbol,
            units=payload.units,
            avg_cost=payload.avg_cost,
            accumulated_dividend_thb=payload.accumulated_dividend_thb,
            asset_type=payload.asset_type,
            reason=payload.reason,
            bucket_id=payload.bucket_id,
            portfolio_id=portfolio_id,
        )
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.delete("/api/portfolio/actual/holdings/{symbol}", response_model=ActualPortfolioStateDTO)
def remove_holding_endpoint(
    symbol: str,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = service.structured_remove_holding(symbol, portfolio_id=portfolio_id)
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.get("/api/portfolio/actual/fx-rate", response_model=FXRateResponseDTO)
def get_fx_rate_endpoint(
    date: Optional[str] = None,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> FXRateResponseDTO:
    with handle_portfolio_exceptions(f"Get FX rate for date '{date}'"):
        rate, source = service.fetch_fx_rate(date_str=date, portfolio_id=portfolio_id)
        return FXRateResponseDTO(
            date=date or datetime.now(timezone.utc).isoformat()[:10],
            currency_pair="USDTHB",
            rate=rate,
            source=source,
        )


@router.post("/api/portfolio/actual/sync-dividends", response_model=SyncDividendsResponseDTO)
def sync_dividends_endpoint(
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> SyncDividendsResponseDTO:
    with handle_portfolio_exceptions(f"Sync dividends for portfolio '{portfolio_id}'"):
        raw_data = service.sync_dividends_from_history(portfolio_id=portfolio_id)
        return SyncDividendsResponseDTO.model_validate(raw_data)
