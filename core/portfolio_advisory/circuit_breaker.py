"""Portfolio Drawdown Circuit Breakers & Realized Tracking Error.

Architecture Layer: core/portfolio_advisory/ (Pure Domain)
Strict Pure Python — no numpy, pandas, pydantic, sqlite3, or API imports.
Enforces:
- 4-Tier Graduated Drawdown Defense:
    - NORMAL (DD <= 8.0%): Standard operation
    - TIER_1_CAUTION (8.0% < DD <= 12.0%): Trim speculative / low conviction positions (+0.5 hurdle)
    - TIER_2_THROTTLE (12.0% < DD <= 15.0%): Restrict buy programs to Core Tier-1 conviction assets only, raise hurdle by +1.0
    - TIER_3_DEFENSIVE (DD > 15.0%): Emergency capital preservation, halt all new buy additions (+2.0 hurdle)
- Realized Annualized Tracking Error (%):
    TE = std(R_p - R_b) * sqrt(annualization_factor) * 100
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Literal, Optional


class DrawdownPostureTier(str, Enum):
    """Graduated risk posture based on portfolio drawdown severity."""
    NORMAL = "NORMAL"
    TIER_1_CAUTION = "TIER_1_CAUTION"
    TIER_2_THROTTLE = "TIER_2_THROTTLE"
    TIER_3_DEFENSIVE = "TIER_3_DEFENSIVE"


@dataclass(frozen=True)
class PortfolioDrawdownPosture:
    """Evaluated drawdown circuit breaker status and capital deployment restrictions."""
    peak_nav_thb: Decimal
    current_nav_thb: Decimal
    drawdown_pct: Decimal
    posture_tier: DrawdownPostureTier
    action_restrictions: List[str] = field(default_factory=list)
    hurdle_adjustment: Decimal = Decimal("0.0")
    is_peak_available: bool = True
    status: Literal["AVAILABLE", "UNAVAILABLE_UNKNOWN_PEAK"] = "AVAILABLE"


def _require_finite_decimal(val: Decimal | int | float | str, label: str) -> Decimal:
    if isinstance(val, bool):
        raise ValueError(f"{label} must be numeric, not bool")
    if isinstance(val, Decimal):
        d = val
    else:
        try:
            d = Decimal(str(val))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"invalid decimal for {label}: {val!r}") from exc
    if not d.is_finite():
        raise ValueError(f"{label} must be finite, got {val!r}")
    return d


def evaluate_drawdown_circuit_breaker(
    current_nav_thb: Decimal,
    peak_nav_thb: Optional[Decimal],
    tier_1_threshold_pct: Decimal = Decimal("8.0"),
    tier_2_threshold_pct: Decimal = Decimal("12.0"),
    tier_3_threshold_pct: Decimal = Decimal("15.0"),
) -> PortfolioDrawdownPosture:
    """Evaluates the unrounded drawdown from historical high-water mark against 4 graduated risk tiers."""
    cur_nav = _require_finite_decimal(current_nav_thb, "current_nav_thb")
    if cur_nav < Decimal("0"):
        raise ValueError(f"current_nav_thb must be non-negative, got {cur_nav}")

    t1 = _require_finite_decimal(tier_1_threshold_pct, "tier_1_threshold_pct")
    t2 = _require_finite_decimal(tier_2_threshold_pct, "tier_2_threshold_pct")
    t3 = _require_finite_decimal(tier_3_threshold_pct, "tier_3_threshold_pct")
    if not (Decimal("0") < t1 < t2 < t3 <= Decimal("100")):
        raise ValueError(
            f"Thresholds must satisfy 0 < tier_1 ({t1}) < tier_2 ({t2}) < tier_3 ({t3}) <= 100"
        )

    if peak_nav_thb is None:
        return PortfolioDrawdownPosture(
            peak_nav_thb=Decimal("0.00"),
            current_nav_thb=cur_nav.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            drawdown_pct=Decimal("0.00"),
            posture_tier=DrawdownPostureTier.NORMAL,
            action_restrictions=["High-water mark peak NAV is unavailable; drawdown defense inactive."],
            hurdle_adjustment=Decimal("0.0"),
            is_peak_available=False,
            status="UNAVAILABLE_UNKNOWN_PEAK",
        )

    peak_nav = _require_finite_decimal(peak_nav_thb, "peak_nav_thb")
    if peak_nav <= Decimal("0"):
        return PortfolioDrawdownPosture(
            peak_nav_thb=Decimal("0.00"),
            current_nav_thb=cur_nav.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            drawdown_pct=Decimal("0.00"),
            posture_tier=DrawdownPostureTier.NORMAL,
            action_restrictions=["High-water mark peak NAV <= 0 or unavailable; drawdown defense inactive."],
            hurdle_adjustment=Decimal("0.0"),
            is_peak_available=False,
            status="UNAVAILABLE_UNKNOWN_PEAK",
        )

    if cur_nav >= peak_nav:
        return PortfolioDrawdownPosture(
            peak_nav_thb=peak_nav.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            current_nav_thb=cur_nav.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            drawdown_pct=Decimal("0.00"),
            posture_tier=DrawdownPostureTier.NORMAL,
            action_restrictions=[],
            hurdle_adjustment=Decimal("0.0"),
            is_peak_available=True,
            status="AVAILABLE",
        )

    # Unrounded canonical DD% = (Peak - Current) / Peak * 100; round only for display
    raw_dd = ((peak_nav - cur_nav) / peak_nav) * Decimal("100")
    dd_pct = raw_dd.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    restrictions: List[str] = []
    hurdle_adj = Decimal("0.0")

    if raw_dd > t3:
        tier = DrawdownPostureTier.TIER_3_DEFENSIVE
        hurdle_adj = Decimal("2.0")
        restrictions.extend([
            f"Drawdown {raw_dd:.4f}% (display {dd_pct:.2f}%) breached Tier 3 threshold ({t3}%); EMERGENCY DEFENSIVE MODE.",
            "All new buy orders halted.",
            "Speculative and high-beta holdings quarantined for immediate review.",
            "Cash and cash-equivalent vehicle preservation prioritized.",
        ])
    elif raw_dd > t2:
        tier = DrawdownPostureTier.TIER_2_THROTTLE
        hurdle_adj = Decimal("1.0")
        restrictions.extend([
            f"Drawdown {raw_dd:.4f}% (display {dd_pct:.2f}%) breached Tier 2 threshold ({t2}%); CAPITAL THROTTLE ACTIVE.",
            "Buy programs restricted to Core Tier-1 conviction assets only.",
            "Hurdle rate increased by +1.0 point.",
            "Rebalance turnover capped.",
        ])
    elif raw_dd > t1:
        tier = DrawdownPostureTier.TIER_1_CAUTION
        hurdle_adj = Decimal("0.5")
        restrictions.extend([
            f"Drawdown {raw_dd:.4f}% (display {dd_pct:.2f}%) breached Tier 1 threshold ({t1}%); CAUTION ACTIVE.",
            "Trim low-conviction or stop-loss breaching positions.",
            "Watchlist additions require additional margin of safety.",
        ])
    else:
        tier = DrawdownPostureTier.NORMAL
        hurdle_adj = Decimal("0.0")

    return PortfolioDrawdownPosture(
        peak_nav_thb=peak_nav.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        current_nav_thb=cur_nav.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        drawdown_pct=dd_pct,
        posture_tier=tier,
        action_restrictions=restrictions,
        hurdle_adjustment=hurdle_adj,
        is_peak_available=True,
        status="AVAILABLE",
    )


def compute_realized_tracking_error(
    portfolio_returns: List[Decimal],
    benchmark_returns: List[Decimal],
    annualization_factor: Decimal = Decimal("252.0"),
    min_observations: int = 10,
) -> Optional[Decimal]:
    """Computes realized annualized tracking error (%) from aligned return series using pure Decimal.

    TE = std(R_p - R_b) * sqrt(annualization_factor) * 100
    """
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError(
            f"Return series length mismatch: portfolio={len(portfolio_returns)} vs benchmark={len(benchmark_returns)}"
        )
    if isinstance(min_observations, bool) or min_observations < 2:
        raise ValueError(f"min_observations must be >= 2, got {min_observations}")

    ann_factor = _require_finite_decimal(annualization_factor, "annualization_factor")
    if ann_factor <= Decimal("0"):
        raise ValueError(f"annualization_factor must be positive, got {ann_factor}")

    n = len(portfolio_returns)
    if n < min_observations:
        return None

    excess_returns: List[Decimal] = [
        _require_finite_decimal(portfolio_returns[i], f"portfolio_returns[{i}]")
        - _require_finite_decimal(benchmark_returns[i], f"benchmark_returns[{i}]")
        for i in range(n)
    ]

    n_dec = Decimal(str(n))
    mean_excess = sum(excess_returns, Decimal("0")) / n_dec
    variance = sum(((e - mean_excess) ** 2 for e in excess_returns), Decimal("0")) / Decimal(str(n - 1))
    if variance <= Decimal("0"):
        return Decimal("0.00")

    daily_std = variance.sqrt()
    annualized_te = daily_std * ann_factor.sqrt() * Decimal("100")

    return annualized_te.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
