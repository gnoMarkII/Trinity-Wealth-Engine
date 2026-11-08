"""Unit tests for Factor Exposure & Portfolio Beta Analytics Engine (Chapter 8).

Verifies the exact synthetic fixtures, invariants, and failure diagnoses from
'Claude Code for Hedge Funds' Chapter 8:
- ALP, BET, GAM, DEL portfolio totals (Gross 16,000, Net 6,000, Long 11,000, Short -5,000)
- Hard-fail on missing price (DEL)
- Availability cutoff filter (excluding post-cutoff 6th observation)
- Exact beta = 1.6 via sample covariance / sample variance (ddof=1, tolerance 1e-9)
"""
from datetime import datetime, timezone
from decimal import Decimal
import pytest

from core.equity_analytics.factor_exposure import (
    calculate_descriptive_factor_beta,
    calculate_portfolio_exposure,
    FactorBetaResult,
    InsufficientObservationsError,
    MissingPriceError,
    PortfolioExposureSnapshot,
    PositionHolding,
    PositionSide,
    ReturnObservation,
    ZeroVarianceError,
)


def test_chapter8_synthetic_positions_exposure():
    """Reproduces the exact four-name package from Chapter 8, page 56-59.

    ALP: 100 Long @ 50  -> +5,000
    BET: 200 Long @ 30  -> +6,000
    GAM: -150 Short @ 20 -> -3,000
    DEL: -50 Short @ 40  -> -2,000

    Expected:
    Long: +11,000
    Short: -5,000
    Gross: 16,000
    Net: 6,000
    """
    positions = [
        PositionHolding(symbol="ALP", shares=Decimal("100"), price=Decimal("50.0"), side=PositionSide.LONG),
        PositionHolding(symbol="BET", shares=Decimal("200"), price=Decimal("30.0"), side=PositionSide.LONG),
        PositionHolding(symbol="GAM", shares=Decimal("-150"), price=Decimal("20.0"), side=PositionSide.SHORT),
        PositionHolding(symbol="DEL", shares=Decimal("-50"), price=Decimal("40.0"), side=PositionSide.SHORT),
    ]

    snapshot = calculate_portfolio_exposure(positions)

    assert snapshot.long_market_value == Decimal("11000.0")
    assert snapshot.short_market_value == Decimal("-5000.0")
    assert snapshot.gross_market_value == Decimal("16000.0")
    assert snapshot.net_market_value == Decimal("6000.0")
    assert snapshot.eligible_positions_count == 4
    assert snapshot.status == "REVIEW_REQUIRED"
    # Long pct = 11000 / 16000 = 68.75%
    assert snapshot.long_exposure_pct == Decimal("68.75")
    # Short pct = 5000 / 16000 = 31.25%
    assert snapshot.short_exposure_pct == Decimal("31.25")
    # Net pct = 6000 / 16000 = 37.50%
    assert snapshot.net_exposure_pct == Decimal("37.50")


def test_chapter8_hard_fail_missing_price():
    """Reproduces Chapter 8 missing price policy: DEL has a blank/missing price.

    The accepted response is an exception naming DEL. It is not zero or guessing.
    """
    positions = [
        PositionHolding(symbol="ALP", shares=Decimal("100"), price=Decimal("50.0"), side=PositionSide.LONG),
        PositionHolding(symbol="BET", shares=Decimal("200"), price=Decimal("30.0"), side=PositionSide.LONG),
        PositionHolding(symbol="GAM", shares=Decimal("-150"), price=Decimal("20.0"), side=PositionSide.SHORT),
        PositionHolding(symbol="DEL", shares=Decimal("-50"), price=None, side=PositionSide.SHORT),
    ]

    with pytest.raises(MissingPriceError) as exc_info:
        calculate_portfolio_exposure(positions, fail_on_missing_price=True)

    assert exc_info.value.symbol == "DEL"
    assert "DEL" in str(exc_info.value)


def test_short_sign_preserved_never_flipped():
    """Diagnoses failure: Short sign flipped causing Net to equal Gross (16,000).

    Verifies that short market value remains strictly negative and net subtracts it.
    """
    positions = [
        PositionHolding(symbol="ALP", shares=Decimal("100"), price=Decimal("50.0"), side=PositionSide.LONG),
        PositionHolding(symbol="DEL", shares=Decimal("-50"), price=Decimal("40.0"), side=PositionSide.SHORT),
    ]
    snapshot = calculate_portfolio_exposure(positions)
    assert snapshot.long_market_value == Decimal("5000.0")
    assert snapshot.short_market_value == Decimal("-2000.0")
    assert snapshot.gross_market_value == Decimal("7000.0")
    assert snapshot.net_market_value == Decimal("3000.0")


def test_chapter8_factor_beta_reproduction():
    """Reproduces Chapter 8 factor exposure beta = 1.6 with availability cutoff.

    Formula: Rp = 1.6 * Rm + 0.002
    Five eligible observations on or before cutoff (2026-08-06 13:30:00 UTC).
    One observation published after cutoff (2026-08-07 12:00:00 UTC) excluded.
    Expected beta = 1.6 with tolerance 1e-9.
    """
    cutoff = datetime(2026, 8, 6, 13, 30, 0, tzinfo=timezone.utc)

    # 5 eligible factor returns
    # Rp = 1.6 * Rm + 0.002
    factor_returns = [
        Decimal("0.010"),
        Decimal("-0.005"),
        Decimal("0.020"),
        Decimal("0.000"),
        Decimal("-0.015"),
    ]

    observations = [
        ReturnObservation(
            date=f"2026-08-0{i+1}",
            factor_return=rm,
            portfolio_return=(Decimal("1.6") * rm + Decimal("0.002")),
            available_at=datetime(2026, 8, i+1, 10, 0, 0, tzinfo=timezone.utc),
        )
        for i, rm in enumerate(factor_returns)
    ]

    # Add 6th observation available AFTER cutoff
    post_cutoff_obs = ReturnObservation(
        date="2026-08-06",
        factor_return=Decimal("0.050"),
        portfolio_return=(Decimal("1.6") * Decimal("0.050") + Decimal("0.002")),
        available_at=datetime(2026, 8, 7, 12, 0, 0, tzinfo=timezone.utc),  # noon UTC Aug 7 > cutoff
    )
    observations.append(post_cutoff_obs)

    result = calculate_descriptive_factor_beta(
        observations=observations,
        cutoff=cutoff,
        benchmark_name="SPY",
        tolerance=Decimal("1e-9"),
    )

    assert result.eligible_observations_count == 5
    assert result.excluded_observations_count == 1
    assert result.status == "REVIEW_REQUIRED"
    # Expected beta is exactly 1.6
    diff = abs(result.beta - Decimal("1.6"))
    assert diff < Decimal("1e-9"), f"Beta {result.beta} differs from 1.6 by {diff}"


def test_insufficient_observations_guard():
    """Guards against single observation or empty list."""
    with pytest.raises(InsufficientObservationsError):
        calculate_descriptive_factor_beta([])

    single = [
        ReturnObservation(date="2026-08-01", factor_return=Decimal("0.01"), portfolio_return=Decimal("0.01"))
    ]
    with pytest.raises(InsufficientObservationsError):
        calculate_descriptive_factor_beta(single)


def test_zero_variance_guard():
    """Guards against flat factor return causing division by zero."""
    flat_obs = [
        ReturnObservation(date="2026-08-01", factor_return=Decimal("0.01"), portfolio_return=Decimal("0.01")),
        ReturnObservation(date="2026-08-02", factor_return=Decimal("0.01"), portfolio_return=Decimal("0.02")),
    ]
    with pytest.raises(ZeroVarianceError):
        calculate_descriptive_factor_beta(flat_obs)

