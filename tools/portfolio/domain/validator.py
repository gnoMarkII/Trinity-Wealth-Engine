import re
from typing import Literal, Optional
from .constants import _CASH_SYMBOLS, CASH_THB_SYMBOL, CASH_USD_SYMBOL, _FLOAT_EPS
from .models import PortfolioState, Holding
from .errors import InvalidTradeError, InsufficientCashError, HoldingNotFoundError

_PORTFOLIO_ID_RE = re.compile(r"^[a-z0-9_\-]+$")


def validate_portfolio_id(portfolio_id: Optional[str] = "default") -> str:
    """Normalize and validate portfolio_id string format."""
    pid = (portfolio_id or "default").strip().lower()
    if pid == "default":
        return pid
    if not _PORTFOLIO_ID_RE.match(pid):
        raise ValueError(f"portfolio_id ไม่ถูกต้อง — อนุญาตเฉพาะ a-z, 0-9, _, - เท่านั้น (ได้ {portfolio_id!r})")
    return pid


def validate_trade_request(
    symbol: str,
    asset_type: str,
    action: str,
    units: float,
    price: float,
    currency: str = "THB",
) -> None:
    """ตรวจความถูกต้องเบื้องต้นของพารามิเตอร์การเทรด."""
    clean_sym = symbol.strip().upper()
    if clean_sym in _CASH_SYMBOLS:
        raise InvalidTradeError(f"ห้ามเทรด {clean_sym} ผ่าน execute_trade — ให้ใช้ manage_cash_flow สำหรับฝาก/ถอนเงินสด")
    if units <= 0:
        raise InvalidTradeError("units ต้องมากกว่า 0")
    if price <= 0:
        raise InvalidTradeError("price ต้องมากกว่า 0")
    if action not in ("buy", "sell"):
        raise InvalidTradeError("action ต้องเป็น 'buy' หรือ 'sell'")
    if currency not in ("THB", "USD"):
        raise InvalidTradeError("currency ต้องเป็น 'THB' หรือ 'USD'")


def validate_cash_availability(
    state: PortfolioState,
    amount_needed: float,
    currency: Literal["THB", "USD"] = "THB",
) -> None:
    """ตรวจว่ามีเงินสดในสกุลเงินนั้นเพียงพอหรือไม่."""
    cash_sym = CASH_THB_SYMBOL if currency == "THB" else CASH_USD_SYMBOL
    cash_holding = next((h for h in state.holdings if h.symbol == cash_sym), None)
    current_cash = cash_holding.units if cash_holding else 0.0

    if current_cash < amount_needed - _FLOAT_EPS:
        raise InsufficientCashError(
            f"เงินสด {currency} ไม่พอสำหรับการทำรายการ: ต้องการ {amount_needed:,.2f} {currency} "
            f"แต่มีอยู่ {current_cash:,.2f} {currency} (ขาด {amount_needed - current_cash:,.2f} {currency})"
        )


def validate_holding_for_sell(holding: Optional[Holding], sell_units: float) -> None:
    """ตรวจว่ามีสินทรัพย์ที่จะขายเพียงพอหรือไม่."""
    if holding is None or holding.status == "archived" or holding.units <= _FLOAT_EPS:
        raise HoldingNotFoundError("ไม่พบสินทรัพย์นี้ในพอร์ต หรือจำนวนหน่วยเป็น 0")
    if sell_units > holding.units + _FLOAT_EPS:
        raise InvalidTradeError(
            f"จำนวนหน่วยที่จะขาย ({sell_units:,.4f}) มากกว่าที่ถืออยู่ ({holding.units:,.4f})"
        )


def validate_cash_flow_request(action: str, amount: float, currency: str) -> None:
    """ตรวจพารามิเตอร์การฝาก/ถอนเงินสด."""
    if amount <= 0:
        raise ValueError("amount ต้องมากกว่า 0")
    if action not in ("deposit", "withdraw"):
        raise ValueError("action ต้องเป็น 'deposit' หรือ 'withdraw'")
    if currency not in ("THB", "USD"):
        raise ValueError("currency ต้องเป็น 'THB' หรือ 'USD'")
