"""Factor Exposure & Portfolio Beta Analytics Engine.

Architecture Rule (Layer 1 - Core):
Pure domain models and deterministic math. Strictly standard library only.
Implements Chapter 8 of 'Claude Code for Hedge Funds':
- Signed position package (Long positive, Short negative)
- Deterministic Gross Market Value (sum of absolute values) and Net Market Value (sum of signed values)
- Strict hard-fail on missing or blank price (never silently zero or guess)
- Availability cutoff filter for return observations
- One-factor descriptive beta via sample covariance / sample variance (ddof=1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple


class MissingPriceError(ValueError):
    """Raised when an active position is missing an approved as-of price."""

    def __init__(self, symbol: str, message: Optional[str] = None):
        self.symbol = symbol
        msg = message or f"Position '{symbol}' is missing an approved price. Silently imputing zero or guessing is strictly prohibited."
        super().__init__(msg)


class InsufficientObservationsError(ValueError):
    """Raised when there are fewer than 2 eligible return observations for sample covariance."""
    pass


class ZeroVarianceError(ValueError):
    """Raised when the factor/benchmark variance is zero."""
    pass


class PositionSide(str, Enum):
    LONG = "Long"
    SHORT = "Short"


@dataclass(frozen=True)
class PositionHolding:
    """An individual position holding with signed quantity and price.

    Rule (Ch. 8):
    The sign is carried in shares and market value.
    Long positions have positive shares. Short positions have negative shares.
    """
    symbol: str
    shares: Decimal
    price: Optional[Decimal]
    side: PositionSide = PositionSide.LONG

    def __post_init__(self) -> None:
        # Enforce consistency between shares sign and side if side is explicitly set
        if self.side == PositionSide.SHORT and self.shares > 0:
            object.__setattr__(self, "shares", -abs(self.shares))
        elif self.side == PositionSide.LONG and self.shares < 0:
            object.__setattr__(self, "shares", abs(self.shares))

    @property
    def market_value(self) -> Decimal:
        """Returns signed market value = shares * price. Raises MissingPriceError if price is None."""
        if self.price is None:
            raise MissingPriceError(self.symbol)
        return self.shares * self.price


@dataclass
class PortfolioExposureSnapshot:
    """Deterministic Gross and Net Portfolio Exposure Snapshot (Chapter 8)."""
    gross_market_value: Decimal
    net_market_value: Decimal
    long_market_value: Decimal
    short_market_value: Decimal
    long_exposure_pct: Decimal       # Long / Gross * 100
    short_exposure_pct: Decimal      # |Short| / Gross * 100
    net_exposure_pct: Decimal        # Net / Gross * 100
    eligible_positions_count: int
    status: str = "REVIEW_REQUIRED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gross_market_value": float(self.gross_market_value),
            "net_market_value": float(self.net_market_value),
            "long_market_value": float(self.long_market_value),
            "short_market_value": float(self.short_market_value),
            "long_exposure_pct": float(self.long_exposure_pct),
            "short_exposure_pct": float(self.short_exposure_pct),
            "net_exposure_pct": float(self.net_exposure_pct),
            "eligible_positions_count": self.eligible_positions_count,
            "status": self.status,
        }


@dataclass(frozen=True)
class ReturnObservation:
    """A single return observation pair (portfolio return, factor/benchmark return)."""
    date: str
    portfolio_return: Decimal
    factor_return: Decimal
    available_at: Optional[datetime] = None


@dataclass
class FactorBetaResult:
    """One-factor descriptive beta calculation result (Chapter 8)."""
    beta: Decimal
    sample_covariance: Decimal
    factor_variance: Decimal
    portfolio_mean: Decimal
    factor_mean: Decimal
    eligible_observations_count: int
    excluded_observations_count: int
    benchmark_name: str
    status: str = "REVIEW_REQUIRED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beta": float(self.beta),
            "sample_covariance": float(self.sample_covariance),
            "factor_variance": float(self.factor_variance),
            "portfolio_mean": float(self.portfolio_mean),
            "factor_mean": float(self.factor_mean),
            "eligible_observations_count": self.eligible_observations_count,
            "excluded_observations_count": self.excluded_observations_count,
            "benchmark_name": self.benchmark_name,
            "status": self.status,
        }


def calculate_portfolio_exposure(
    positions: List[PositionHolding],
    fail_on_missing_price: bool = True,
) -> PortfolioExposureSnapshot:
    """Calculates deterministic Gross, Net, Long, and Short market values across positions.

    Chapter 8 Invariants:
    - market_value = shares * price
    - gross_market_value = sum(|market_value|)
    - net_market_value = sum(market_value)
    - If any price is missing or invalid, raises MissingPriceError naming the position.
    """
    if not positions:
        zero = Decimal("0.0")
        return PortfolioExposureSnapshot(
            gross_market_value=zero,
            net_market_value=zero,
            long_market_value=zero,
            short_market_value=zero,
            long_exposure_pct=zero,
            short_exposure_pct=zero,
            net_exposure_pct=zero,
            eligible_positions_count=0,
            status="REVIEW_REQUIRED",
        )

    long_mv = Decimal("0.0")
    short_mv = Decimal("0.0")

    for pos in positions:
        if pos.price is None or pos.price <= Decimal("0.0"):
            if fail_on_missing_price:
                raise MissingPriceError(pos.symbol)
            continue

        mv = pos.shares * pos.price
        if mv >= Decimal("0.0"):
            long_mv += mv
        else:
            short_mv += mv  # short_mv is negative

    gross_mv = long_mv + abs(short_mv)
    net_mv = long_mv + short_mv

    if gross_mv > Decimal("0.0"):
        long_pct = ((long_mv / gross_mv) * Decimal("100.0")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        short_pct = ((abs(short_mv) / gross_mv) * Decimal("100.0")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        net_pct = ((net_mv / gross_mv) * Decimal("100.0")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        long_pct = Decimal("0.0")
        short_pct = Decimal("0.0")
        net_pct = Decimal("0.0")

    return PortfolioExposureSnapshot(
        gross_market_value=gross_mv,
        net_market_value=net_mv,
        long_market_value=long_mv,
        short_market_value=short_mv,
        long_exposure_pct=long_pct,
        short_exposure_pct=short_pct,
        net_exposure_pct=net_pct,
        eligible_positions_count=len(positions),
        status="REVIEW_REQUIRED",
    )


def calculate_descriptive_factor_beta(
    observations: List[ReturnObservation],
    cutoff: Optional[datetime] = None,
    benchmark_name: str = "SPY",
    tolerance: Decimal = Decimal("1e-9"),
) -> FactorBetaResult:
    """Calculates one-factor descriptive beta using sample covariance and sample variance (ddof=1).

    Chapter 8 Invariants:
    - Excludes observations with available_at > cutoff.
    - sample covariance(Rp, Rm) / sample variance(Rm)
    - ddof = 1 for both covariance and variance.
    - Tolerance 1e-9.
    """
    eligible: List[ReturnObservation] = []
    excluded_count = 0

    for obs in observations:
        if cutoff is not None and obs.available_at is not None:
            # If observation was published/available after cutoff, exclude it
            if obs.available_at > cutoff:
                excluded_count += 1
                continue
        eligible.append(obs)

    n = len(eligible)
    if n < 2:
        raise InsufficientObservationsError(
            f"Expected at least 2 eligible return observations for sample variance, got {n}."
        )

    # 1. Compute means
    dec_n = Decimal(str(n))
    mean_p = sum((obs.portfolio_return for obs in eligible), Decimal("0.0")) / dec_n
    mean_m = sum((obs.factor_return for obs in eligible), Decimal("0.0")) / dec_n

    # 2. Compute sample covariance & sample variance (ddof = 1)
    ddof = Decimal(str(n - 1))
    cov_sum = sum(
        (obs.portfolio_return - mean_p) * (obs.factor_return - mean_m)
        for obs in eligible
    )
    sample_cov = cov_sum / ddof

    var_sum = sum(
        (obs.factor_return - mean_m) ** 2
        for obs in eligible
    )
    sample_var = var_sum / ddof

    if abs(sample_var) < tolerance:
        raise ZeroVarianceError(
            f"Factor variance ({sample_var}) is within tolerance ({tolerance}) of zero; cannot compute beta."
        )

    beta = sample_cov / sample_var

    return FactorBetaResult(
        beta=beta,
        sample_covariance=sample_cov,
        factor_variance=sample_var,
        portfolio_mean=mean_p,
        factor_mean=mean_m,
        eligible_observations_count=n,
        excluded_observations_count=excluded_count,
        benchmark_name=benchmark_name,
        status="REVIEW_REQUIRED",
    )

