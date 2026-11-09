"""Unit tests for Phase 1: Brinson-Fachler Attribution & Carino Multi-Period Linking.

Architecture Layer: tests/unit/
Enforces Decimal exactness, weight validation, and Carino geometric closure.
"""
from decimal import Decimal
import pytest

from core.portfolio.attribution import (
    BucketAttributionInput,
    BucketAttributionRow,
    BrinsonAttributionReport,
    compute_single_period_brinson,
    link_multi_period_brinson_carino,
)


def test_single_period_brinson_exact_closure():
    """Verify single-period attribution closes with reconciliation error < 1e-8."""
    buckets = [
        BucketAttributionInput(
            bucket_id="equity_tech",
            bucket_name="Tech Growth",
            portfolio_weight=Decimal("0.50"),
            benchmark_weight=Decimal("0.40"),
            portfolio_return=Decimal("0.10"),  # +10%
            benchmark_return=Decimal("0.08"),  # +8%
        ),
        BucketAttributionInput(
            bucket_id="equity_defensive",
            bucket_name="Defensive Value",
            portfolio_weight=Decimal("0.35"),
            benchmark_weight=Decimal("0.50"),
            portfolio_return=Decimal("0.04"),  # +4%
            benchmark_return=Decimal("0.02"),  # +2%
        ),
        BucketAttributionInput(
            bucket_id="__CASH__",
            bucket_name="Cash Reserves",
            portfolio_weight=Decimal("0.15"),
            benchmark_weight=Decimal("0.10"),
            portfolio_return=Decimal("0.005"), # +0.5%
            benchmark_return=Decimal("0.005"), # +0.5%
        ),
    ]

    report = compute_single_period_brinson(
        buckets=buckets,
        portfolio_cost_rate=Decimal("0.001"),   # 10 bps fee
        benchmark_cost_rate=Decimal("0.000"),
    )

    # 1. Total weights verified
    assert report.is_reconciled is True
    assert abs(report.reconciliation_error) < Decimal("1e-8")

    # 2. Portfolio gross = 0.50*0.10 + 0.35*0.04 + 0.15*0.005 = 0.05 + 0.014 + 0.00075 = 0.06475
    expected_rp_gross = Decimal("0.50") * Decimal("0.10") + Decimal("0.35") * Decimal("0.04") + Decimal("0.15") * Decimal("0.005")
    assert report.portfolio_gross_return == expected_rp_gross

    # 3. Benchmark gross = 0.40*0.08 + 0.50*0.02 + 0.10*0.005 = 0.032 + 0.010 + 0.0005 = 0.0425
    expected_rb_gross = Decimal("0.40") * Decimal("0.08") + Decimal("0.50") * Decimal("0.02") + Decimal("0.10") * Decimal("0.005")
    assert report.benchmark_gross_return == expected_rb_gross

    # 4. Check cost effect
    assert report.cost_effect == Decimal("-0.001")
    assert report.net_active_return == (expected_rp_gross - Decimal("0.001")) - expected_rb_gross

    # 5. Check cash drag is computed for the cash bucket
    cash_row = next(r for r in report.rows if r.bucket_id == "__CASH__")
    assert report.cash_drag_effect == cash_row.total_active_contribution
    assert len(report.rows) == 3


def test_brinson_weight_validation_error():
    """Verify weight deviation beyond tolerance triggers ValueError."""
    bad_buckets = [
        BucketAttributionInput(
            bucket_id="b1",
            bucket_name="Bucket 1",
            portfolio_weight=Decimal("0.70"),
            benchmark_weight=Decimal("0.50"),
            portfolio_return=Decimal("0.05"),
            benchmark_return=Decimal("0.03"),
        ),
        BucketAttributionInput(
            bucket_id="b2",
            bucket_name="Bucket 2",
            portfolio_weight=Decimal("0.20"),  # Sum = 0.90 != 1.0
            benchmark_weight=Decimal("0.50"),
            portfolio_return=Decimal("0.02"),
            benchmark_return=Decimal("0.02"),
        ),
    ]

    with pytest.raises(ValueError, match="Portfolio weights must sum to 1.0"):
        compute_single_period_brinson(bad_buckets)


def test_carino_multi_period_linking_closure():
    """Verify Carino geometric linking across 3 periods closes with error < 1e-8."""
    # Period 1
    p1 = compute_single_period_brinson([
        BucketAttributionInput("equity", "Equity", Decimal("0.70"), Decimal("0.60"), Decimal("0.05"), Decimal("0.03")),
        BucketAttributionInput("cash", "Cash", Decimal("0.30"), Decimal("0.40"), Decimal("0.002"), Decimal("0.002")),
    ], portfolio_cost_rate=Decimal("0.0005"))

    # Period 2
    p2 = compute_single_period_brinson([
        BucketAttributionInput("equity", "Equity", Decimal("0.65"), Decimal("0.60"), Decimal("-0.02"), Decimal("-0.03")),
        BucketAttributionInput("cash", "Cash", Decimal("0.35"), Decimal("0.40"), Decimal("0.002"), Decimal("0.002")),
    ], portfolio_cost_rate=Decimal("0.0005"))

    # Period 3
    p3 = compute_single_period_brinson([
        BucketAttributionInput("equity", "Equity", Decimal("0.75"), Decimal("0.60"), Decimal("0.08"), Decimal("0.06")),
        BucketAttributionInput("cash", "Cash", Decimal("0.25"), Decimal("0.40"), Decimal("0.002"), Decimal("0.002")),
    ], portfolio_cost_rate=Decimal("0.0005"))

    linked = link_multi_period_brinson_carino([p1, p2, p3])

    # 1. Closure verified
    assert linked.is_reconciled is True
    assert abs(linked.reconciliation_error) < Decimal("1e-8")

    # 2. Compounded returns match
    r1_net = p1.portfolio_gross_return - p1.portfolio_cost_rate
    r2_net = p2.portfolio_gross_return - p2.portfolio_cost_rate
    r3_net = p3.portfolio_gross_return - p3.portfolio_cost_rate
    expected_rp_net = ((Decimal("1.0") + r1_net) * (Decimal("1.0") + r2_net) * (Decimal("1.0") + r3_net)) - Decimal("1.0")

    b1_net = p1.benchmark_gross_return - p1.benchmark_cost_rate
    b2_net = p2.benchmark_gross_return - p2.benchmark_cost_rate
    b3_net = p3.benchmark_gross_return - p3.benchmark_cost_rate
    expected_rb_net = ((Decimal("1.0") + b1_net) * (Decimal("1.0") + b2_net) * (Decimal("1.0") + b3_net)) - Decimal("1.0")

    expected_ra_net = expected_rp_net - expected_rb_net
    assert abs(linked.net_active_return - expected_ra_net) < Decimal("1e-8")

    # 3. All rows present
    assert len(linked.rows) == 2


def test_carino_single_period_identity():
    """Verify linking a single period returns the identical report."""
    p1 = compute_single_period_brinson([
        BucketAttributionInput("core", "Core", Decimal("1.0"), Decimal("1.0"), Decimal("0.04"), Decimal("0.02")),
    ])
    linked = link_multi_period_brinson_carino([p1])
    assert linked == p1


def test_r15_weight_tolerance_normalization_and_nested_carino_consistency():
    """R15: Accepted approximate weights reconcile cleanly; nested Carino linking matches direct linking."""
    # 1. Portfolio weight sum 0.9995 within default tolerance 0.001 is consistently normalized
    approx = compute_single_period_brinson([
        BucketAttributionInput("a", "A", Decimal("0.4995"), Decimal("0.5000"), Decimal("0.10"), Decimal("0.08")),
        BucketAttributionInput("b", "B", Decimal("0.5000"), Decimal("0.5000"), Decimal("0.02"), Decimal("0.02")),
    ])
    assert approx.is_reconciled is True
    assert abs(approx.reconciliation_error) < Decimal("1e-8")

    # Outside default tolerance still raises
    with pytest.raises(ValueError, match="Portfolio weights must sum to 1.0"):
        compute_single_period_brinson([
            BucketAttributionInput("a", "A", Decimal("0.4980"), Decimal("0.5000"), Decimal("0.10"), Decimal("0.08")),
            BucketAttributionInput("b", "B", Decimal("0.5000"), Decimal("0.5000"), Decimal("0.02"), Decimal("0.02")),
        ])

    # 2. Exact R15 multi-period cost-drag & associativity reproduction
    p1 = compute_single_period_brinson(
        [BucketAttributionInput("a", "A", Decimal("1.0"), Decimal("1.0"), Decimal("0.10"), Decimal("0.00"))],
        portfolio_cost_rate=Decimal("0.01"),
    )
    p2 = compute_single_period_brinson(
        [BucketAttributionInput("a", "A", Decimal("1.0"), Decimal("1.0"), Decimal("0.10"), Decimal("0.00"))],
        portfolio_cost_rate=Decimal("0.01"),
    )
    p3 = compute_single_period_brinson(
        [BucketAttributionInput("a", "A", Decimal("1.0"), Decimal("1.0"), Decimal("0.10"), Decimal("0.00"))],
        portfolio_cost_rate=Decimal("0.01"),
    )

    l12 = link_multi_period_brinson_carino([p1, p2])
    assert abs(l12.net_active_return - Decimal("0.1881")) < Decimal("1e-8")
    # Effective gross - effective cost matches compounded net return
    assert abs((l12.portfolio_gross_return - l12.portfolio_cost_rate) - l12.portfolio_net_return) < Decimal("1e-8")
    assert l12.nominal_portfolio_cost_rate_sum == Decimal("0.02")

    direct_123 = link_multi_period_brinson_carino([p1, p2, p3])
    nested_12_3 = link_multi_period_brinson_carino([l12, p3])
    assert abs(direct_123.net_active_return - Decimal("0.295029")) < Decimal("1e-7")
    assert abs(nested_12_3.net_active_return - direct_123.net_active_return) < Decimal("1e-8")
    assert abs(nested_12_3.portfolio_gross_return - direct_123.portfolio_gross_return) < Decimal("1e-8")
    assert nested_12_3.is_reconciled is True

    # 3. Total-loss period (-100%) rejected before log domain crash
    total_loss = compute_single_period_brinson([
        BucketAttributionInput("a", "A", Decimal("1.0"), Decimal("1.0"), Decimal("-1.00"), Decimal("0.02")),
    ])
    with pytest.raises(ValueError, match="<= -100%"):
        link_multi_period_brinson_carino([total_loss, p1])
