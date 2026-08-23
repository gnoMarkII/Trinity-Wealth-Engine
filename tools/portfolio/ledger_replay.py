from typing import Optional
from tools.portfolio import get_default_service
from tools.portfolio.domain.models import PortfolioState
from tools.portfolio.domain.calculations import _replay_symbol_trades

def edit_transaction(
    tx_id: str,
    timestamp: Optional[str] = None,
    units: Optional[float] = None,
    price: Optional[float] = None,
    fx_rate: Optional[float] = None,
    notes: Optional[str] = None,
    adjust_cash: bool = True,
    portfolio_id: str = "default",
) -> PortfolioState:
    return get_default_service().edit_transaction(
        tx_id=tx_id,
        timestamp=timestamp,
        units=units,
        price=price,
        fx_rate=fx_rate,
        notes=notes,
        adjust_cash=adjust_cash,
        portfolio_id=portfolio_id,
    )

def delete_transaction(
    tx_id: str, adjust_cash: bool = True, portfolio_id: str = "default"
) -> PortfolioState:
    return get_default_service().delete_transaction(
        tx_id=tx_id, adjust_cash=adjust_cash, portfolio_id=portfolio_id
    )
