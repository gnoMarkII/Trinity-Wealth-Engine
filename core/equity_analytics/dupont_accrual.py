"""Deterministic 3-Factor DuPont Analysis & Sloan Accrual Quality Engine.

Architecture Layer: core/equity_analytics/ (Pure Domain)
Strict Pure Python — no numpy, pandas, pydantic, sqlite3, or API imports.
Enforces:
- 3-Factor DuPont Identity:
    ROE = (Net Income / Revenue) * (Revenue / Total Assets) * (Total Assets / Stockholders Equity)
        = Net Profit Margin * Asset Turnover * Financial Leverage
- Sloan Accrual Ratio (Richard Sloan, 1996, The Accounting Review):
    Accrual = (Net Income - Operating Cash Flow) / Average Total Assets
    - Accrual > +0.10: EARNINGS_MANIPULATION_WARNING (Poor quality, non-cash earnings)
    - Accrual in [-0.10, +0.10]: NEUTRAL
    - Accrual < -0.10: HIGH_QUALITY (Conservative accounting, cash generation exceeds income)
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Literal, Optional, Union

_PUBLISHING_RESIDUAL_TOLERANCE = Decimal("0.0100")


@dataclass(frozen=True)
class DuPont3FactorBreakdown:
    """3-Factor DuPont decomposition of Return on Equity (ROE)."""
    net_profit_margin_pct: Decimal
    asset_turnover: Decimal
    equity_multiplier: Decimal
    roe_pct: Decimal
    closure_error: Decimal
    internal_closure_error: Decimal = Decimal("0.0000")
    published_roe_reconstructed_pct: Decimal = Decimal("0.0000")
    is_published_reconciled: bool = True
    raw_asset_turnover: Optional[Decimal] = None


@dataclass(frozen=True)
class SloanAccrualAnalysis:
    """Sloan Accrual Ratio and Earnings Quality Rating."""
    sloan_accrual_ratio: Decimal
    earnings_quality_rating: Literal["HIGH_QUALITY", "NEUTRAL", "EARNINGS_MANIPULATION_WARNING"]
    net_income: Decimal
    operating_cash_flow: Decimal
    average_total_assets: Decimal
    interpretation: str
    raw_sloan_accrual_ratio: Optional[Decimal] = None
    assets_basis: Literal["two_period_average", "single_period_ending"] = "single_period_ending"


def _require_finite_decimal(val: Union[Decimal, int, float, str], label: str) -> Decimal:
    if isinstance(val, bool):
        raise ValueError(f"{label} must be numeric, not bool")
    if isinstance(val, Decimal):
        dec = val
    else:
        try:
            dec = Decimal(str(val))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"invalid decimal for {label}: {val!r}") from exc
    if not dec.is_finite():
        raise ValueError(f"{label} must be finite, got {val!r}")
    return dec


def compute_dupont_3factor(
    net_income: Decimal,
    revenue: Decimal,
    total_assets: Decimal,
    stockholders_equity: Decimal,
) -> Optional[DuPont3FactorBreakdown]:
    """Computes exact 3-factor DuPont decomposition: Margin * Turnover * Leverage.

    Returns None if any denominator is zero or negative.
    Raises ValueError on non-finite inputs.
    Reports published-factor reconstruction error honestly in `closure_error`
    while keeping `internal_closure_error` separate.
    """
    ni = _require_finite_decimal(net_income, "net_income")
    rev = _require_finite_decimal(revenue, "revenue")
    ta = _require_finite_decimal(total_assets, "total_assets")
    eq = _require_finite_decimal(stockholders_equity, "stockholders_equity")

    if (
        rev <= Decimal("0")
        or ta <= Decimal("0")
        or eq <= Decimal("0")
    ):
        return None

    with localcontext() as ctx:
        ctx.prec = 34
        margin = (ni / rev) * Decimal("100")
        turnover = rev / ta
        leverage = ta / eq
        roe = (ni / eq) * Decimal("100")
        decomposed_roe = margin * turnover * leverage
        internal_err = abs(roe - decomposed_roe)

    pub_margin = margin.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    pub_turnover = turnover.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    pub_leverage = leverage.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    pub_roe = roe.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    pub_reconstructed_roe = (pub_margin * pub_turnover * pub_leverage).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    )
    pub_closure_err = abs(pub_roe - (pub_margin * pub_turnover * pub_leverage)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP
    )

    return DuPont3FactorBreakdown(
        net_profit_margin_pct=pub_margin,
        asset_turnover=pub_turnover,
        equity_multiplier=pub_leverage,
        roe_pct=pub_roe,
        closure_error=pub_closure_err,
        internal_closure_error=internal_err.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP),
        published_roe_reconstructed_pct=pub_reconstructed_roe,
        is_published_reconciled=pub_closure_err <= _PUBLISHING_RESIDUAL_TOLERANCE,
        raw_asset_turnover=turnover,
    )


def compute_sloan_accrual_ratio(
    net_income: Decimal,
    operating_cash_flow: Decimal,
    total_assets_current: Decimal,
    total_assets_prior: Optional[Decimal] = None,
    display_quantum: Decimal = Decimal("0.0001"),
) -> Optional[SloanAccrualAnalysis]:
    """Computes Sloan Accrual Ratio: (Net Income - Operating Cash Flow) / Avg Assets.

    Compares the UNROUNDED canonical ratio against strict thresholds (+0.10 / -0.10)
    before rounding for display so classification never shifts with display precision.
    Returns None if current or supplied prior assets are zero or negative.
    Raises ValueError on non-finite inputs.
    """
    ni = _require_finite_decimal(net_income, "net_income")
    ocf = _require_finite_decimal(operating_cash_flow, "operating_cash_flow")
    ta_cur = _require_finite_decimal(total_assets_current, "total_assets_current")
    ta_prior = (
        _require_finite_decimal(total_assets_prior, "total_assets_prior")
        if total_assets_prior is not None
        else None
    )

    if ta_cur <= Decimal("0"):
        return None
    if ta_prior is not None:
        if ta_prior <= Decimal("0"):
            return None
        avg_assets = (ta_cur + ta_prior) / Decimal("2")
        assets_basis: Literal["two_period_average", "single_period_ending"] = "two_period_average"
    else:
        avg_assets = ta_cur
        assets_basis = "single_period_ending"

    with localcontext() as ctx:
        ctx.prec = 34
        accrual_amount = ni - ocf
        raw_ratio = accrual_amount / avg_assets

    sloan_display = raw_ratio.quantize(display_quantum, rounding=ROUND_HALF_UP)

    if raw_ratio > Decimal("0.10"):
        rating: Literal["HIGH_QUALITY", "NEUTRAL", "EARNINGS_MANIPULATION_WARNING"] = "EARNINGS_MANIPULATION_WARNING"
        desc = (
            f"High accruals ({sloan_display} [raw {raw_ratio:.6f}] > +0.10): "
            "Net income substantially outpaces operating cash flow; "
            "potential aggressive revenue recognition or capitalized expenses."
        )
    elif raw_ratio < Decimal("-0.10"):
        rating = "HIGH_QUALITY"
        desc = (
            f"Conservative accruals ({sloan_display} [raw {raw_ratio:.6f}] < -0.10): "
            "Operating cash flow significantly exceeds net income; "
            "high-quality, cash-backed earnings profile."
        )
    else:
        rating = "NEUTRAL"
        desc = f"Moderate accruals ({sloan_display} [raw {raw_ratio:.6f}]): Cash flow reasonably tracks reported net income."

    return SloanAccrualAnalysis(
        sloan_accrual_ratio=sloan_display,
        earnings_quality_rating=rating,
        net_income=ni,
        operating_cash_flow=ocf,
        average_total_assets=avg_assets,
        interpretation=desc,
        raw_sloan_accrual_ratio=raw_ratio,
        assets_basis=assets_basis,
    )
