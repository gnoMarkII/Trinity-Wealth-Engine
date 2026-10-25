"""Real-World Market Lot Rounding and Order Sizing Rules.

Architecture Layer: core/portfolio_advisory/ (Pure Domain)
Enforces Defect D4:
- Thai SET Equities: Board Lot (100-share multiples) and Odd Lot (1..99 shares)
- US Equities: Whole Shares and 4-decimal Fractional Shares (Dime/WealthX)
- Mutual Funds & Gold: Monetary THB amounts without integer lot constraints
- Tax-Locked Vehicles: Zero order sizes for sell/trim actions
"""
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import math
from typing import Literal, Optional

from core.portfolio_advisory.guardrails import safe_decimal
from core.portfolio_advisory.models import LotSizingBreakdown


def calculate_market_lot_sizing(
    market: Literal["TH", "US", "GLOBAL"],
    native_currency: Literal["THB", "USD"],
    target_amount_native: Decimal,
    current_price_native: Decimal,
    fx_rate: Decimal,
    is_tax_locked: bool = False,
) -> LotSizingBreakdown:
    """Calculates realistic exchange-executable lot sizes according to market rules."""
    target_amt = safe_decimal(target_amount_native, quantum="0.01")
    price = safe_decimal(current_price_native, quantum="0.0001")
    fx = safe_decimal(fx_rate, default=Decimal("35.50"), quantum="0.0001")

    if is_tax_locked or target_amt <= Decimal("0") or price <= Decimal("0"):
        return LotSizingBreakdown(
            market=market,
            native_currency=native_currency,
            board_lot_shares=0,
            odd_lot_shares=0,
            whole_shares=0,
            fractional_shares=Decimal("0.0000"),
            amount_thb=Decimal("0.00"),
            amount_native=Decimal("0.00"),
            estimated_execution_price_native=price,
        )

    raw_shares = target_amt / price

    if market == "TH" and native_currency == "THB":
        total_int_shares = int(raw_shares.to_integral_value(rounding=ROUND_DOWN))
        board_lot = (total_int_shares // 100) * 100
        odd_lot = total_int_shares % 100
        amt_thb = safe_decimal(Decimal(total_int_shares) * price, quantum="0.01")

        return LotSizingBreakdown(
            market="TH",
            native_currency="THB",
            board_lot_shares=board_lot,
            odd_lot_shares=odd_lot,
            whole_shares=total_int_shares,
            fractional_shares=safe_decimal(raw_shares, quantum="0.0001"),
            amount_thb=amt_thb,
            amount_native=amt_thb,
            estimated_execution_price_native=price,
        )

    elif market == "US" or native_currency == "USD":
        whole = int(raw_shares.to_integral_value(rounding=ROUND_DOWN))
        fractional = safe_decimal(raw_shares, quantum="0.0001")
        amt_usd = safe_decimal(fractional * price, quantum="0.01")
        amt_thb = safe_decimal(amt_usd * fx, quantum="0.01")

        return LotSizingBreakdown(
            market="US",
            native_currency="USD",
            board_lot_shares=None,
            odd_lot_shares=None,
            whole_shares=whole,
            fractional_shares=fractional,
            amount_thb=amt_thb,
            amount_native=amt_usd,
            estimated_execution_price_native=price,
        )

    else:
        # Mutual Funds, Gold, etc.
        amt_thb = target_amt if native_currency == "THB" else safe_decimal(target_amt * fx, quantum="0.01")
        return LotSizingBreakdown(
            market="GLOBAL",
            native_currency=native_currency,
            board_lot_shares=None,
            odd_lot_shares=None,
            whole_shares=int(raw_shares.to_integral_value(rounding=ROUND_DOWN)),
            fractional_shares=safe_decimal(raw_shares, quantum="0.0001"),
            amount_thb=amt_thb,
            amount_native=target_amt,
            estimated_execution_price_native=price,
        )
