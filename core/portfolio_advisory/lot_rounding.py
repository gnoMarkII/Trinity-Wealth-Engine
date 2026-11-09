"""Real-World Market Lot Rounding and Order Sizing Rules.

Architecture Layer: core/portfolio_advisory/ (Pure Domain)
Enforces Defect D4:
- Thai SET Equities: Board Lot (100-share multiples) and Odd Lot (1..99 shares)
- US Equities: Whole Shares and 4-decimal Fractional Shares (Dime/WealthX)
- Mutual Funds & Gold: Monetary THB amounts without integer lot constraints
- Tax-Locked Vehicles: Zero order sizes for sell/trim actions
"""
from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from typing import Literal, Optional

from core.portfolio_advisory.guardrails import safe_decimal
from core.portfolio_advisory.models import LotSizingBreakdown


def _normalized_fx(fx_rate: Decimal) -> Decimal:
    fx = safe_decimal(fx_rate, default=Decimal("35.50"), quantum="0.0001")
    if fx is None or fx <= Decimal("0"):
        return Decimal("35.50")
    return fx


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
    fx = _normalized_fx(fx_rate)

    if is_tax_locked or target_amt is None or price is None or target_amt <= Decimal("0") or price <= Decimal("0"):
        return LotSizingBreakdown(
            market=market,
            native_currency=native_currency,
            board_lot_shares=0,
            odd_lot_shares=0,
            whole_shares=0,
            fractional_shares=Decimal("0.0000"),
            amount_thb=Decimal("0.00"),
            amount_native=Decimal("0.00"),
            estimated_execution_price_native=price if price is not None else Decimal("0.0000"),
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
            fractional_shares=safe_decimal(Decimal(total_int_shares), quantum="0.0001"),
            amount_thb=amt_thb,
            amount_native=amt_thb,
            estimated_execution_price_native=price,
        )

    elif market == "US" or native_currency == "USD":
        fractional = safe_decimal(raw_shares.quantize(Decimal("0.0001"), rounding=ROUND_DOWN), quantum="0.0001")
        whole = int(fractional.to_integral_value(rounding=ROUND_DOWN))
        amt_usd = safe_decimal(fractional * price, quantum="0.01")
        amt_thb = safe_decimal(amt_usd * fx, quantum="0.01")

        return LotSizingBreakdown(
            market="US" if market == "US" else ("GLOBAL" if market == "GLOBAL" else "US"),
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
        # Mutual Funds, Gold, Bonds in THB monetary units
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


def calculate_fee_inclusive_lot_sizing(
    market: Literal["TH", "US", "GLOBAL"],
    native_currency: Literal["THB", "USD"],
    max_cash_budget_native: Decimal,
    current_price_native: Decimal,
    fx_rate: Decimal,
    adtv_native: Optional[Decimal] = None,
    annualized_volatility_pct: Optional[Decimal] = None,
    is_tax_locked: bool = False,
) -> LotSizingBreakdown:
    """Calculates maximal executable lot size such that order_value + exact_friction <= budget.

    Uses discrete binary search over exact exchange executable units:
    - TH equities: integer shares (1-share step, split into 100-share board + odd lots)
    - US equities: 0.0001-share fractional steps
    - GLOBAL funds/bonds/gold: 0.01 monetary steps

    Guarantees both:
    1. returned.amount_native + estimate_order_friction(...).total_friction_native <= budget
    2. the next executable unit would strictly exceed budget
    """
    from core.portfolio_advisory.friction import estimate_order_friction

    budget = safe_decimal(max_cash_budget_native, quantum="0.01")
    price = safe_decimal(current_price_native, quantum="0.0001")
    fx = _normalized_fx(fx_rate)

    if is_tax_locked or budget is None or price is None or budget <= Decimal("0") or price <= Decimal("0"):
        return calculate_market_lot_sizing(
            market=market,
            native_currency=native_currency,
            target_amount_native=Decimal("0.00"),
            current_price_native=price if price is not None else Decimal("0.0000"),
            fx_rate=fx,
            is_tax_locked=is_tax_locked,
        )

    is_th_stock = market == "TH" and native_currency == "THB"
    is_us_stock = market == "US" or native_currency == "USD"

    def _build_for_ticks(ticks: int) -> LotSizingBreakdown:
        if ticks <= 0:
            if is_th_stock:
                return LotSizingBreakdown(
                    market="TH",
                    native_currency="THB",
                    board_lot_shares=0,
                    odd_lot_shares=0,
                    whole_shares=0,
                    fractional_shares=Decimal("0.0000"),
                    amount_thb=Decimal("0.00"),
                    amount_native=Decimal("0.00"),
                    estimated_execution_price_native=price,
                )
            elif is_us_stock:
                return LotSizingBreakdown(
                    market="GLOBAL" if market == "GLOBAL" else "US",
                    native_currency="USD",
                    board_lot_shares=None,
                    odd_lot_shares=None,
                    whole_shares=0,
                    fractional_shares=Decimal("0.0000"),
                    amount_thb=Decimal("0.00"),
                    amount_native=Decimal("0.00"),
                    estimated_execution_price_native=price,
                )
            else:
                return LotSizingBreakdown(
                    market="GLOBAL",
                    native_currency=native_currency,
                    board_lot_shares=None,
                    odd_lot_shares=None,
                    whole_shares=0,
                    fractional_shares=Decimal("0.0000"),
                    amount_thb=Decimal("0.00"),
                    amount_native=Decimal("0.00"),
                    estimated_execution_price_native=price,
                )

        if is_th_stock:
            board_lot = (ticks // 100) * 100
            odd_lot = ticks % 100
            amt_thb = safe_decimal(Decimal(ticks) * price, quantum="0.01")
            return LotSizingBreakdown(
                market="TH",
                native_currency="THB",
                board_lot_shares=board_lot,
                odd_lot_shares=odd_lot,
                whole_shares=ticks,
                fractional_shares=safe_decimal(Decimal(ticks), quantum="0.0001"),
                amount_thb=amt_thb,
                amount_native=amt_thb,
                estimated_execution_price_native=price,
            )

        if is_us_stock:
            fractional = (Decimal(ticks) / Decimal("10000")).quantize(Decimal("0.0001"))
            whole = int(fractional.to_integral_value(rounding=ROUND_DOWN))
            amt_usd = safe_decimal(fractional * price, quantum="0.01")
            amt_thb = safe_decimal(amt_usd * fx, quantum="0.01")
            return LotSizingBreakdown(
                market="GLOBAL" if market == "GLOBAL" else "US",
                native_currency="USD",
                board_lot_shares=None,
                odd_lot_shares=None,
                whole_shares=whole,
                fractional_shares=fractional,
                amount_thb=amt_thb,
                amount_native=amt_usd,
                estimated_execution_price_native=price,
            )

        amt_native = (Decimal(ticks) / Decimal("100")).quantize(Decimal("0.01"))
        amt_thb = amt_native if native_currency == "THB" else safe_decimal(amt_native * fx, quantum="0.01")
        raw_shares = amt_native / price
        return LotSizingBreakdown(
            market="GLOBAL",
            native_currency=native_currency,
            board_lot_shares=None,
            odd_lot_shares=None,
            whole_shares=int(raw_shares.to_integral_value(rounding=ROUND_DOWN)),
            fractional_shares=safe_decimal(raw_shares, quantum="0.0001"),
            amount_thb=amt_thb,
            amount_native=amt_native,
            estimated_execution_price_native=price,
        )

    def _cost_for_ticks(ticks: int) -> tuple[LotSizingBreakdown, Decimal]:
        sizing = _build_for_ticks(ticks)
        if sizing.amount_native <= Decimal("0"):
            return sizing, Decimal("0.00")
        f_est = estimate_order_friction(
            market=market,
            action_side="BUY",
            order_amount_native=sizing.amount_native,
            adtv_native=adtv_native,
            annualized_volatility_pct=annualized_volatility_pct,
        )
        return sizing, sizing.amount_native + f_est.total_friction_native

    if is_th_stock:
        high_ticks = int((budget / price).to_integral_value(rounding=ROUND_DOWN)) + 1
    elif is_us_stock:
        high_ticks = int(((budget / price) * Decimal("10000")).to_integral_value(rounding=ROUND_DOWN)) + 1
    else:
        high_ticks = int((budget * Decimal("100")).to_integral_value(rounding=ROUND_DOWN))

    lo = 0
    hi = max(0, high_ticks)

    hi_sizing, hi_cost = _cost_for_ticks(hi)
    if hi_cost <= budget:
        return hi_sizing

    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, mid_cost = _cost_for_ticks(mid)
        if mid_cost <= budget:
            lo = mid
        else:
            hi = mid - 1

    best_sizing, _ = _cost_for_ticks(lo)
    return best_sizing
