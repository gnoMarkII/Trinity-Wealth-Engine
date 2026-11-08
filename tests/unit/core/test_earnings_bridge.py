"""Unit tests for deterministic EarningsBridgeEngine (Chapter 5 & Appendix C)."""
from decimal import Decimal
import pytest

from core.equity_analytics.earnings_bridge import (
    DriverCategory,
    EarningsBridge,
    EarningsBridgeEngine,
    EarningsDriver,
    EarningsSurprise,
)


def test_compute_surprise_beat():
    surp = EarningsBridgeEngine.compute_surprise(
        reported=Decimal("150.00"),
        consensus=Decimal("140.00"),
        metric_name="revenue",
        unit="USD_M",
    )
    assert surp.metric_name == "revenue"
    assert surp.reported == Decimal("150.00")
    assert surp.consensus == Decimal("140.00")
    assert surp.delta == Decimal("10.00")
    # (10 / 140) * 100 = 7.14%
    assert surp.surprise_pct == Decimal("7.14")
    assert surp.direction == "BEAT"


def test_compute_surprise_miss():
    surp = EarningsBridgeEngine.compute_surprise(
        reported=Decimal("95.00"),
        consensus=Decimal("100.00"),
        metric_name="eps",
        unit="USD",
    )
    assert surp.delta == Decimal("-5.00")
    assert surp.surprise_pct == Decimal("-5.00")
    assert surp.direction == "MISS"


def test_compute_surprise_inline():
    surp = EarningsBridgeEngine.compute_surprise(
        reported=Decimal("100.50"),
        consensus=Decimal("100.00"),
        metric_name="revenue",
    )
    assert surp.surprise_pct == Decimal("0.50")
    assert surp.direction == "INLINE"


def test_build_earnings_bridge_balanced_confirmation():
    # Revenue consensus 1000, reported 1050 -> delta +50
    # Drivers: Volume +30, Price/Mix +15, FX +5 -> sum = 50 (balanced!)
    drivers = [
        EarningsDriver(
            name="Cloud Volume Expansion",
            category=DriverCategory.VOLUME,
            impact_value=Decimal("30.00"),
            unit="USD_M",
            narrative="Enterprise client workload migration acceleration",
        ),
        EarningsDriver(
            name="Contract Renewal Price Increase",
            category=DriverCategory.PRICE_MIX,
            impact_value=Decimal("15.00"),
            unit="USD_M",
            narrative="Annual 5% price uplift on renewal cohort",
        ),
        EarningsDriver(
            name="EUR Tailwinds",
            category=DriverCategory.FX,
            impact_value=Decimal("5.00"),
            unit="USD_M",
            narrative="Favorable FX movement",
        ),
    ]

    bridge = EarningsBridgeEngine.build_earnings_bridge(
        ticker="GOOGL",
        period="Q3-2026",
        as_of_date="2026-11-04",
        revenue_reported=Decimal("1050.00"),
        revenue_consensus=Decimal("1000.00"),
        eps_reported=Decimal("2.10"),
        eps_consensus=Decimal("1.95"),
        revenue_unit="USD_M",
        eps_unit="USD",
        ebit_margin_reported=Decimal("32.5"),
        ebit_margin_consensus=Decimal("31.0"),
        revenue_drivers=drivers,
    )

    assert bridge.is_mathematically_balanced is True
    assert bridge.unallocated_delta == Decimal("0.00")
    assert bridge.sum_drivers == Decimal("50.00")
    assert bridge.thesis_reconciliation_verdict == "THESIS_CONFIRMED"
    assert "confirming active thesis" in bridge.thesis_reconciliation_narrative


def test_build_earnings_bridge_unbalanced_warning():
    # Revenue delta +50, but drivers only sum to +30 -> unbalanced!
    drivers = [
        EarningsDriver(
            name="Cloud Volume",
            category=DriverCategory.VOLUME,
            impact_value=Decimal("30.00"),
            unit="USD_M",
        ),
    ]

    bridge = EarningsBridgeEngine.build_earnings_bridge(
        ticker="NFLX",
        period="Q3-2026",
        as_of_date="2026-11-04",
        revenue_reported=Decimal("1050.00"),
        revenue_consensus=Decimal("1000.00"),
        eps_reported=Decimal("2.10"),
        eps_consensus=Decimal("2.00"),
        revenue_drivers=drivers,
    )

    assert bridge.is_mathematically_balanced is False
    assert bridge.unallocated_delta == Decimal("20.00")
    assert bridge.thesis_reconciliation_verdict == "THESIS_UNDER_PRESSURE"
    assert "mathematically unbalanced" in bridge.thesis_reconciliation_narrative


def test_build_earnings_bridge_negative_volume_flag():
    # Headline beat (+50), but core volume contracted (-20), masked by Price/Mix (+70)
    drivers = [
        EarningsDriver(
            name="Organic Unit Volume Decline",
            category=DriverCategory.VOLUME,
            impact_value=Decimal("-20.00"),
            unit="USD_M",
        ),
        EarningsDriver(
            name="Aggressive Price Hike",
            category=DriverCategory.PRICE_MIX,
            impact_value=Decimal("70.00"),
            unit="USD_M",
        ),
    ]

    bridge = EarningsBridgeEngine.build_earnings_bridge(
        ticker="PG",
        period="Q3-2026",
        as_of_date="2026-11-04",
        revenue_reported=Decimal("1050.00"),
        revenue_consensus=Decimal("1000.00"),
        eps_reported=Decimal("1.50"),
        eps_consensus=Decimal("1.40"),
        revenue_drivers=drivers,
    )

    assert bridge.is_mathematically_balanced is True
    assert bridge.thesis_reconciliation_verdict == "THESIS_UNDER_PRESSURE"
    assert "underlying core volume contraction" in bridge.thesis_reconciliation_narrative


def test_build_earnings_bridge_double_miss():
    bridge = EarningsBridgeEngine.build_earnings_bridge(
        ticker="INTC",
        period="Q3-2026",
        as_of_date="2026-11-04",
        revenue_reported=Decimal("900.00"),
        revenue_consensus=Decimal("1000.00"),
        eps_reported=Decimal("0.80"),
        eps_consensus=Decimal("1.10"),
    )

    assert bridge.thesis_reconciliation_verdict == "THESIS_IMPAIRED"
    assert "Double miss" in bridge.thesis_reconciliation_narrative


def test_earnings_bridge_serialization():
    drivers = [
        EarningsDriver(
            name="Cloud Volume",
            category=DriverCategory.VOLUME,
            impact_value=Decimal("10.00"),
            unit="USD_M",
        ),
    ]
    bridge = EarningsBridgeEngine.build_earnings_bridge(
        ticker="MSFT",
        period="Q1-2027",
        as_of_date="2026-11-04",
        revenue_reported=Decimal("1010.00"),
        revenue_consensus=Decimal("1000.00"),
        eps_reported=Decimal("3.00"),
        eps_consensus=Decimal("2.90"),
        revenue_drivers=drivers,
    )
    d = bridge.to_dict()
    assert d["ticker"] == "MSFT"
    assert d["is_mathematically_balanced"] is True

    restored = EarningsBridge.from_dict(d)
    assert restored.ticker == bridge.ticker
    assert restored.revenue_surprise.delta == bridge.revenue_surprise.delta
    assert restored.revenue_drivers[0].name == "Cloud Volume"

