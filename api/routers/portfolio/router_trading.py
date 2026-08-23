"""FastAPI Sub-router for Trades, Cash Flows, Incomes, FX Rates, and Holding Edits."""
import sys
from typing import Optional
from fastapi import APIRouter, Depends

from api.auth import require_session
from api.schemas import (
    ActualPortfolioStateDTO,
    TradeRequestDTO,
    CashFlowRequestDTO,
    IncomeRequestDTO,
    EditHoldingRequestDTO,
    FXRateResponseDTO,
    SyncDividendsResponseDTO,
)
from tools.portfolio.models import _now_iso
from api.routers.portfolio.common import handle_portfolio_exceptions

router = APIRouter(dependencies=[Depends(require_session)])


def _get_mod(name: str):
    routes_mod = sys.modules.get("api.routes_portfolio")
    if routes_mod and hasattr(routes_mod, name):
        return getattr(routes_mod, name)
    if name == "portfolio_core":
        from tools.portfolio import core as mod
        return mod
    elif name == "portfolio_trading":
        from tools.portfolio import trading as mod
        return mod
    elif name == "portfolio_prices":
        from tools.portfolio import prices as mod
        return mod
    elif name == "portfolio_dividends":
        from tools.portfolio import dividends as mod
        return mod
    return None


@router.post("/api/portfolio/actual/trade", response_model=ActualPortfolioStateDTO)
def execute_trade_endpoint(
    payload: TradeRequestDTO, portfolio_id: str = "default"
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        portfolio_trading = _get_mod("portfolio_trading")
        state = portfolio_trading.structured_execute_trade(
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
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.post(
    "/api/portfolio/actual/cashflow", response_model=ActualPortfolioStateDTO
)
def manage_cash_flow_endpoint(
    payload: CashFlowRequestDTO, portfolio_id: str = "default"
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        portfolio_trading = _get_mod("portfolio_trading")
        state = portfolio_trading.structured_manage_cash_flow(
            amount=payload.amount,
            action=payload.action,
            currency=payload.currency,
            exchange_rate=payload.exchange_rate,
            date=payload.date,
            notes=payload.notes,
            portfolio_id=portfolio_id,
        )
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.post(
    "/api/portfolio/actual/income", response_model=ActualPortfolioStateDTO
)
def record_income_endpoint(
    payload: IncomeRequestDTO, portfolio_id: str = "default"
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        portfolio_trading = _get_mod("portfolio_trading")
        state = portfolio_trading.structured_record_income(
            income_type=payload.income_type,
            amount_thb=payload.amount_thb,
            source_symbol=payload.source_symbol,
            date=payload.date,
            notes=payload.notes,
            portfolio_id=portfolio_id,
        )
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.put(
    "/api/portfolio/actual/holdings/{symbol}/edit",
    response_model=ActualPortfolioStateDTO,
)
def edit_holding_endpoint(
    symbol: str, payload: EditHoldingRequestDTO, portfolio_id: str = "default"
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        portfolio_trading = _get_mod("portfolio_trading")
        state = portfolio_trading.structured_edit_holding(
            symbol=symbol,
            units=payload.units,
            avg_cost=payload.avg_cost,
            accumulated_dividend_thb=payload.accumulated_dividend_thb,
            asset_type=payload.asset_type,
            reason=payload.reason,
            bucket_id=payload.bucket_id,
            portfolio_id=portfolio_id,
        )
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.delete(
    "/api/portfolio/actual/holdings/{symbol}",
    response_model=ActualPortfolioStateDTO,
)
def remove_holding_endpoint(symbol: str, portfolio_id: str = "default") -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        portfolio_trading = _get_mod("portfolio_trading")
        state = portfolio_trading.structured_remove_holding(symbol, portfolio_id=portfolio_id)
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.get("/api/portfolio/actual/fx-rate", response_model=FXRateResponseDTO)
def get_fx_rate_endpoint(date: Optional[str] = None, portfolio_id: str = "default") -> FXRateResponseDTO:
    with handle_portfolio_exceptions(f"Get FX rate for date '{date}'"):
        portfolio_core = _get_mod("portfolio_core")
        portfolio_prices = _get_mod("portfolio_prices")
        post, state = portfolio_core._load_or_init(portfolio_id=portfolio_id)
        portfolio_fallback = state.fx_rates.get("USDTHB", 36.5)
        rate, source = portfolio_prices.fetch_fx_rate(date_str=date, fallback_rate=portfolio_fallback)
        return FXRateResponseDTO(
            date=date or _now_iso()[:10],
            currency_pair="USDTHB",
            rate=rate,
            source=source,
        )


@router.post("/api/portfolio/actual/sync-dividends", response_model=SyncDividendsResponseDTO)
def sync_dividends_endpoint(portfolio_id: str = "default") -> SyncDividendsResponseDTO:
    with handle_portfolio_exceptions(f"Sync dividends for portfolio '{portfolio_id}'"):
        portfolio_dividends = _get_mod("portfolio_dividends")
        raw_data = portfolio_dividends.sync_dividends_from_history(portfolio_id=portfolio_id)
        return SyncDividendsResponseDTO.model_validate(raw_data)
