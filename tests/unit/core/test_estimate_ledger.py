"""Unit tests for Institutional Two-Sided Estimate Ledger (Chapter 2, 5 & App H)."""
from decimal import Decimal
import pytest

from core.equity_analytics.estimate_ledger import (
    EstimateRecord,
    EstimateType,
    RevisionBridgeResult,
    RevisionDriver,
    VariantPerceptionSnapshot,
    append_estimate_revision,
    calculate_variant_perception,
)


def test_chapter5_synthetic_estimate_revision_and_bridge():
    """Book Chapter 5 (p. 39-40):
    Prior Q3 analyst revenue estimate EST-0001 is 1,050.
    New Q3 analyst revenue estimate EST-0002 is 1,090 (+40).
    Drivers: Volume +25, Pricing +10, FX +5 (Sum = +40).
    Prior row must remain byte-for-byte unchanged.
    """
    baseline = EstimateRecord(
        estimate_id="EST-0001",
        ticker="SYN-NSTAR",
        period="2026Q3",
        metric_name="revenue",
        estimate_type=EstimateType.FUND_THESIS,
        value=Decimal("1050.0"),
        unit="USD_M",
        as_of_utc="2026-08-06T13:30:00Z",
        reason_code="baseline",
        owner="analyst_a",
        approval_status="APPROVED",
    )
    ledger = [baseline]

    revised = EstimateRecord(
        estimate_id="EST-0002",
        ticker="SYN-NSTAR",
        period="2026Q3",
        metric_name="revenue",
        estimate_type=EstimateType.FUND_THESIS,
        value=Decimal("1090.0"),
        unit="USD_M",
        as_of_utc="2026-08-06T21:00:00Z",
        reason_code="post_earnings_revision",
        prior_record_id="EST-0001",
        owner="analyst_a",
        approval_status="REVIEW_REQUIRED",
    )
    drivers = [
        RevisionDriver("Volume", Decimal("25.0")),
        RevisionDriver("Pricing", Decimal("10.0")),
        RevisionDriver("FX", Decimal("5.0")),
    ]

    new_ledger, bridge = append_estimate_revision(ledger, revised, drivers)

    # 1. Assert prior row byte-for-byte unchanged
    assert len(new_ledger) == 2
    assert new_ledger[0] == baseline
    assert new_ledger[0].value == Decimal("1050.0")
    assert new_ledger[1].value == Decimal("1090.0")

    # 2. Assert bridge arithmetic
    assert bridge.estimate_delta == Decimal("40.0")
    assert bridge.drivers_sum == Decimal("40.0")
    assert bridge.is_mathematically_balanced is True
    assert bridge.status == "REVIEW_REQUIRED"


def test_revision_unbalanced_drivers_detected():
    """Book Chapter 5 (p. 41 planted failure):
    Drivers total 35 while estimate delta is 40 -> is_mathematically_balanced is False.
    """
    baseline = EstimateRecord(
        estimate_id="EST-0001",
        ticker="GOOGL",
        period="2026Q3",
        metric_name="revenue",
        estimate_type=EstimateType.FUND_THESIS,
        value=Decimal("1000.0"),
        prior_record_id=None,
    )
    revised = EstimateRecord(
        estimate_id="EST-0002",
        ticker="GOOGL",
        period="2026Q3",
        metric_name="revenue",
        estimate_type=EstimateType.FUND_THESIS,
        value=Decimal("1040.0"),
        prior_record_id="EST-0001",
    )
    drivers = [RevisionDriver("Volume", Decimal("35.0"))] # Misses 5.0

    _, bridge = append_estimate_revision([baseline], revised, drivers)
    assert bridge.is_mathematically_balanced is False
    assert bridge.estimate_delta == Decimal("40.0")
    assert bridge.drivers_sum == Decimal("35.0")


def test_variant_perception_calculation_alpha_confirmed():
    """Verify three-way surprise reconciliation:
    Consensus = 86400, Fund Thesis = 87500 (Variant Spread = +1100), Reported = 88268.
    Street Surprise = +1868 (+2.16% Beat)
    Variant Realization = +768 (+0.88% Alpha Realized)
    Verdict = ALPHA_CONFIRMED
    """
    res = calculate_variant_perception(
        ticker="GOOGL",
        period="2026Q2",
        metric_name="revenue",
        reported=Decimal("88268.00"),
        consensus=Decimal("86400.00"),
        fund_thesis=Decimal("87500.00"),
        unit="USD_M",
    )
    assert res.street_surprise == Decimal("1868.00")
    assert res.street_surprise_pct == Decimal("2.16")
    assert res.variant_spread == Decimal("1100.00")
    assert res.variant_spread_pct == Decimal("1.27")
    assert res.variant_realization == Decimal("768.00")
    assert res.variant_realization_pct == Decimal("0.88")
    assert res.alpha_verdict == "ALPHA_CONFIRMED"

