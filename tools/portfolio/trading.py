from typing import Optional, List, Dict, Literal
from tools.portfolio import get_default_service
from tools.portfolio.domain.models import PortfolioState, Holding, _now_iso
from tools.portfolio.domain.constants import (
    CASH_THB_SYMBOL,
    CASH_USD_SYMBOL,
    CASH_SYMBOL,
    _CASH_SYMBOLS,
    _FLOAT_EPS,
    _MONEY_DP,
    _COST_DP,
    _PCT_DP,
    _EDITABLE_HOLDING_FIELDS,
)
from tools.portfolio.domain.calculations import compute_total_cost as _compute_total_cost
from tools.portfolio.adapters.markdown.paths import (
    get_trades_log_filepath as _get_trades_log_filepath,
    _TRADES_LOG_HEADER,
    _LOCK_TIMEOUT,
)
from tools.portfolio.adapters.markdown.repository_adapter import (
    _sanitize_csv_field,
    _read_and_migrate_trade_log_locked,
)

def _migrate_trades_log_if_needed(portfolio_id: str = "default") -> None:
    fpath = _get_trades_log_filepath(portfolio_id)
    if fpath.exists():
        _read_and_migrate_trade_log_locked(fpath)
from tools.portfolio.agent_tools import (
    execute_trade,
    record_income,
    batch_import_holdings,
    manage_cash_flow,
    update_fx_rate,
    edit_holding,
)

from tools.portfolio.prices import fetch_latest_price, fetch_fx_rate

_PRICE_FETCH_TIMEOUT = 6.0

# --- Structured API Functions Delegation ---

def structured_execute_trade(
    symbol: str,
    asset_type: str,
    action: Literal["buy", "sell"],
    units: float,
    price: float,
    currency: Literal["THB", "USD"] = "THB",
    exchange_rate: Optional[float] = None,
    date: Optional[str] = None,
    notes: str = "",
    bucket_id: Optional[str] = None,
    portfolio_id: str = "default",
) -> PortfolioState:
    return get_default_service().structured_execute_trade(
        symbol=symbol,
        asset_type=asset_type,
        action=action,
        units=units,
        price=price,
        currency=currency,
        exchange_rate=exchange_rate,
        date=date,
        notes=notes,
        bucket_id=bucket_id,
        portfolio_id=portfolio_id,
    )

def structured_manage_cash_flow(
    amount: float,
    action: Literal["deposit", "withdraw"],
    currency: Literal["THB", "USD"] = "THB",
    exchange_rate: Optional[float] = None,
    date: Optional[str] = None,
    notes: str = "",
    portfolio_id: str = "default",
) -> PortfolioState:
    return get_default_service().structured_manage_cash_flow(
        amount=amount,
        action=action,
        currency=currency,
        exchange_rate=exchange_rate,
        date=date,
        notes=notes,
        portfolio_id=portfolio_id,
    )

def structured_record_income(
    income_type: Literal["Dividend", "Interest", "Rental", "Other"],
    amount_thb: float,
    source_symbol: Optional[str] = None,
    date: Optional[str] = None,
    notes: str = "",
    portfolio_id: str = "default",
) -> PortfolioState:
    return get_default_service().structured_record_income(
        income_type=income_type,
        amount_thb=amount_thb,
        source_symbol=source_symbol,
        date=date,
        notes=notes,
        portfolio_id=portfolio_id,
    )

def structured_edit_holding(
    symbol: str,
    units: Optional[float] = None,
    avg_cost: Optional[float] = None,
    accumulated_dividend_thb: Optional[float] = None,
    asset_type: Optional[str] = None,
    reason: str = "",
    bucket_id: Optional[str] = None,
    portfolio_id: str = "default",
) -> PortfolioState:
    return get_default_service().structured_edit_holding(
        symbol=symbol,
        units=units,
        avg_cost=avg_cost,
        accumulated_dividend_thb=accumulated_dividend_thb,
        asset_type=asset_type,
        reason=reason,
        bucket_id=bucket_id,
        portfolio_id=portfolio_id,
    )

def structured_remove_holding(symbol: str, portfolio_id: str = "default") -> PortfolioState:
    return get_default_service().structured_remove_holding(symbol=symbol, portfolio_id=portfolio_id)

def get_structured_trades_log(
    portfolio_id: str = "default", symbol: Optional[str] = None
) -> List[Dict]:
    return get_default_service().get_structured_trades_log(portfolio_id=portfolio_id, symbol=symbol)

def update_trade_note(tx_id: str, notes: str, portfolio_id: str = "default") -> Dict:
    return get_default_service().update_trade_note(tx_id=tx_id, notes=notes, portfolio_id=portfolio_id)

from tools.portfolio.adapters.markdown.repository_adapter import _get_portfolio_lock

def _fetch_fx_rate() -> Optional[float]:
    from tools.portfolio.prices import _fetch_fx_rate as prices_fetch_fx
    return prices_fetch_fx()

# --- Compatibility Hooks for tests/conftest.py ---

def _execute_trade_locked(symbol, asset_type, action, units, price, currency="THB", notes="", portfolio_id="default"):
    res_str, _ = get_default_service()._execute_trade_internal(
        symbol=symbol, asset_type=asset_type, action=action, units=units, price=price, currency=currency, notes=notes, portfolio_id=portfolio_id
    )
    return res_str

def _record_income_locked(income_type, amount_thb, source_symbol=None, portfolio_id="default"):
    res_str, _ = get_default_service()._record_income_internal(
        income_type=income_type, amount_thb=amount_thb, source_symbol=source_symbol, portfolio_id=portfolio_id
    )
    return res_str

def _manage_cash_flow_locked(*args, **kwargs):
    currency = kwargs.get("currency", "THB")
    notes = kwargs.get("notes", "")
    portfolio_id = kwargs.get("portfolio_id", "default")

    if "amount" in kwargs and "action" in kwargs:
        amt = float(kwargs["amount"])
        act = kwargs["action"]
    elif len(args) >= 2:
        if isinstance(args[0], (int, float)):
            amt = float(args[0])
            act = args[1]
        else:
            act = args[0]
            amt = float(args[1])
        if len(args) >= 3:
            currency = args[2]
        if len(args) >= 4:
            notes = args[3]
        if len(args) >= 5:
            portfolio_id = args[4]
    else:
        raise ValueError("Invalid arguments for _manage_cash_flow_locked")

    res_str, _ = get_default_service()._manage_cash_flow_internal(
        amount=amt, action=act, currency=currency, portfolio_id=portfolio_id
    )
    return res_str

def _update_fx_rate_locked(rate=None, portfolio_id="default"):
    if rate is not None:
        if rate <= 0:
            raise ValueError("rate ต้องมากกว่า 0")
        new_rate = float(rate)
        source = "manual"
    else:
        new_rate = _fetch_fx_rate()
        if new_rate is None or new_rate <= 0:
            raise ValueError("auto-fetch FX ล้มเหลว กรุณาระบุ rate ด้วยตนเอง")
        source = "yfinance"

    with get_default_service().repo.unit_of_work(portfolio_id) as uow:
        state = uow.load_state()
        old_rate = state.fx_rates.get("USDTHB", 36.5)
        state.fx_rates["USDTHB"] = new_rate
        from tools.portfolio.domain.calculations import recalc_all
        from tools.portfolio.domain.ledger_change import LedgerChange
        recalc_all(state)
        uow.commit(state, LedgerChange(kind="unchanged"))
        return f"[FX {source}] USDTHB: {old_rate:.4f} → {new_rate:.4f}"

def _edit_holding_locked(symbol, units=None, avg_cost=None, accumulated_dividend_thb=None, asset_type=None, reason="", portfolio_id="default"):
    res_str, _ = get_default_service()._edit_holding_internal(
        symbol=symbol, units=units, avg_cost=avg_cost, accumulated_dividend_thb=accumulated_dividend_thb, asset_type=asset_type, reason=reason, portfolio_id=portfolio_id
    )
    return res_str
