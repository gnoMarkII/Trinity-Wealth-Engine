"""Execution Friction, Square-Root Market Impact & Rebalance Cost Engine.

Architecture Layer: core/portfolio_advisory/ (Pure Domain)
Strictly adheres to Hexagonal Architecture:
- Pure Python standard library only (dataclasses, decimal, math, typing).
- Zero dependencies on frameworks, databases, or API layers.
- Deterministic math for transaction costs and liquidity constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Literal, Optional

_SQRT_252 = Decimal("252").sqrt()
ImpactStatus = Literal[
    "AVAILABLE",
    "UNAVAILABLE_MISSING_ADTV",
    "UNAVAILABLE_MISSING_VOLATILITY",
    "UNAVAILABLE_MISSING_BOTH",
    "ZERO_ORDER",
]


def _coerce_finite_decimal(val: object, name: str) -> Decimal:
    if isinstance(val, bool) or val is None:
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        dec = val if isinstance(val, Decimal) else Decimal(str(val))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{name} is not a valid Decimal: {val!r}") from exc
    if not dec.is_finite():
        raise ValueError(f"{name} must be finite, got {val!r}")
    return dec


def _coerce_optional_finite_positive(val: object) -> Optional[Decimal]:
    if isinstance(val, bool) or val is None:
        return None
    try:
        dec = val if isinstance(val, Decimal) else Decimal(str(val))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not dec.is_finite() or dec <= Decimal("0"):
        return None
    return dec


@dataclass(frozen=True)
class OrderFrictionEstimate:
    """Estimated transaction friction and market impact breakdown for a prospective order."""
    order_amount_native: Decimal
    explicit_fee_native: Decimal
    explicit_fee_bps: Decimal
    market_impact_bps: Decimal
    total_friction_bps: Decimal
    total_friction_native: Decimal
    participation_rate_pct: Optional[Decimal] = None
    liquidity_warning: Optional[str] = None
    market_impact_status: ImpactStatus = "AVAILABLE"


def estimate_order_friction(
    market: Literal["TH", "US", "GLOBAL"],
    action_side: Literal["BUY", "SELL"],
    order_amount_native: Decimal,
    adtv_native: Optional[Decimal] = None,
    annualized_volatility_pct: Optional[Decimal] = None,
    impact_coeff_y: Decimal = Decimal("0.60"),
    min_commission_thb: Decimal = Decimal("50.0"),
) -> OrderFrictionEstimate:
    """Estimates explicit brokerage fees and non-linear square-root market impact.

    Native monetary amounts (`explicit_fee_native`, unrounded square-root market impact,
    and `total_friction_native`) are authoritative; basis points (`*_bps`) are rounded to
    0.1 bps solely for display so rounded bps never understate actual monetary fees.

    Market Impact Law (Bouchaud et al. / Kyle):
        I(Q) = Y * sigma_daily * sqrt(Q / ADTV)
        where sigma_daily = annualized_volatility / sqrt(252)

    Explicit fees:
        - TH: 15.7 bps commission + clearing/reg fees + 7% VAT = ~16.8 bps (with 50 THB minimum)
        - US: ~10 bps flat / fractional fee + SEC/TAF fees on sell
        - GLOBAL: 0 bps (Funds, Bonds, Cash)
    """
    order_amt = _coerce_finite_decimal(order_amount_native, "order_amount_native")
    coeff_y = _coerce_finite_decimal(impact_coeff_y, "impact_coeff_y")
    min_fee_thb = _coerce_finite_decimal(min_commission_thb, "min_commission_thb")
    if coeff_y < Decimal("0"):
        raise ValueError("impact_coeff_y must be non-negative")
    if min_fee_thb < Decimal("0"):
        raise ValueError("min_commission_thb must be non-negative")

    if order_amt <= Decimal("0"):
        return OrderFrictionEstimate(
            order_amount_native=Decimal("0.00"),
            explicit_fee_native=Decimal("0.00"),
            explicit_fee_bps=Decimal("0.00"),
            market_impact_bps=Decimal("0.00"),
            total_friction_bps=Decimal("0.00"),
            total_friction_native=Decimal("0.00"),
            participation_rate_pct=Decimal("0.00"),
            liquidity_warning=None,
            market_impact_status="ZERO_ORDER",
        )

    # 1. Explicit brokerage fees (authoritative native monetary fee)
    if market == "TH":
        # 0.168% base fee (16.8 bps), minimum 50.00 THB
        base_rate = Decimal("0.00168")
        raw_fee = order_amt * base_rate
        fee_native = max(raw_fee, min_fee_thb).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        raw_fee_bps = (fee_native / order_amt) * Decimal("10000")
        fee_bps = raw_fee_bps.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    elif market == "US":
        # 10 bps base fee; additional ~2 bps on sell side for SEC Section 31 + TAF
        bps = Decimal("10.0") if action_side == "BUY" else Decimal("12.0")
        fee_bps = bps
        fee_native = (order_amt * (bps / Decimal("10000"))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        fee_bps = Decimal("0.0")
        fee_native = Decimal("0.00")

    # 2. Participation rate & Square-Root Market Impact
    adtv_val = _coerce_optional_finite_positive(adtv_native)
    vol_val = _coerce_optional_finite_positive(annualized_volatility_pct)

    impact_bps = Decimal("0.0")
    raw_impact_native = Decimal("0")
    part_rate_pct: Optional[Decimal] = None
    warning: Optional[str] = None

    if adtv_val is not None:
        part_ratio = order_amt / adtv_val
        part_rate_pct = (part_ratio * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        if part_ratio > Decimal("0.10"):
            warning = f"Order participation rate ({part_rate_pct}%) exceeds 10% of 20d ADTV; severe market impact expected."
        elif part_ratio > Decimal("0.05"):
            warning = f"Order participation rate ({part_rate_pct}%) exceeds 5% of 20d ADTV; moderate market impact expected."

        if vol_val is not None:
            daily_vol = vol_val / (_SQRT_252 * Decimal("100"))
            impact_rate = coeff_y * daily_vol * part_ratio.sqrt()
            raw_impact_native = order_amt * impact_rate
            impact_bps = (impact_rate * Decimal("10000")).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            impact_status: ImpactStatus = "AVAILABLE"
        else:
            impact_status = "UNAVAILABLE_MISSING_VOLATILITY"
    elif vol_val is not None:
        impact_status = "UNAVAILABLE_MISSING_ADTV"
    else:
        impact_status = "UNAVAILABLE_MISSING_BOTH"

    # Authoritative monetary total: sum exact native fee + unrounded market impact before 0.01 rounding
    total_friction_native = (fee_native + raw_impact_native).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    # Display bps derived from authoritative unrounded monetary components (never vice-versa)
    total_friction_bps = (((fee_native + raw_impact_native) / order_amt) * Decimal("10000")).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP
    )

    return OrderFrictionEstimate(
        order_amount_native=order_amt.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        explicit_fee_native=fee_native,
        explicit_fee_bps=fee_bps,
        market_impact_bps=impact_bps,
        total_friction_bps=total_friction_bps,
        total_friction_native=total_friction_native,
        participation_rate_pct=part_rate_pct,
        liquidity_warning=warning,
        market_impact_status=impact_status,
    )


def compute_rebalance_turnover_and_cost(
    current_weights_pct: dict[str, Decimal],
    target_weights_pct: dict[str, Decimal],
    one_way_cost_bps: Decimal = Decimal("20.0"),
) -> tuple[Decimal, Decimal]:
    """Calculates one-way rebalancing turnover and total friction drag.

    Turnover = 0.5 * sum(|w_target,i - w_current,i|)
    Rebalance Cost bps = 2 * (Turnover / 100) * one_way_cost_bps
    """
    one_way_bps = _coerce_finite_decimal(one_way_cost_bps, "one_way_cost_bps")
    if one_way_bps < Decimal("0"):
        raise ValueError("one_way_cost_bps must be non-negative")

    all_keys = set(current_weights_pct.keys()) | set(target_weights_pct.keys())
    abs_diff_sum = Decimal("0")

    for k in all_keys:
        w_cur = _coerce_finite_decimal(current_weights_pct.get(k, Decimal("0")), f"current_weights_pct[{k}]")
        w_tgt = _coerce_finite_decimal(target_weights_pct.get(k, Decimal("0")), f"target_weights_pct[{k}]")
        abs_diff_sum += abs(w_tgt - w_cur)

    # One-way turnover percentage (e.g. 15.00 for 15% rebalance)
    one_way_turnover_pct = (abs_diff_sum / Decimal("2")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    # Total round-trip cost in bps of portfolio NAV (computed from unrounded turnover)
    turnover_fraction = (abs_diff_sum / Decimal("2")) / Decimal("100")
    cost_bps = (turnover_fraction * (one_way_bps * Decimal("2"))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

    return one_way_turnover_pct, cost_bps
