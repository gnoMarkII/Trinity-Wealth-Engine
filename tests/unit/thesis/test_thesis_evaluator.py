"""Unit tests for Pure Domain ThesisEvaluator."""
from datetime import datetime
from decimal import Decimal
import pytest

from core.thesis.evaluator import ThesisEvaluator
from core.thesis.models import (
    CatalystEvent,
    FalsifierRule,
    ThesisClaimRecord,
    ThesisExceptionRow,
)


@pytest.fixture
def sample_claim():
    falsifier1 = FalsifierRule(
        falsifier_id="FAL-MARGIN",
        metric_name="gross_margin",
        operator="<",
        threshold_value=Decimal("70.0"),
        unit="PERCENT",
        invalidation_impact="BREAKS_BUY_CASE",
    )
    falsifier2 = FalsifierRule(
        falsifier_id="FAL-DEBT",
        metric_name="net_debt_ebitda",
        operator=">",
        threshold_value=Decimal("3.0"),
        unit="RATIO",
        invalidation_impact="FORCES_PORTFOLIO_TRIM",
    )
    catalyst1 = CatalystEvent(
        catalyst_id="CAT-EARNINGS",
        event_name="Q3 Earnings Release",
        expected_date_str="2026-11-15",
        confidence_window_days=5,
        source_locator="SEC 8-K",
        status="PENDING",
    )
    return ThesisClaimRecord.create(
        claim_id="TH-NVDA-01",
        ticker="NVDA",
        version=1,
        as_of_utc="2026-11-01T00:00:00Z",
        claim_statement="NVDA thesis based on gross margin >= 70% and net debt <= 3.0x",
        falsifiers=[falsifier1, falsifier2],
        linked_catalysts=[catalyst1],
    )


def test_falsifier_breach_less_than(sample_claim):
    # Observed margin = 68.5% which is < 70.0% -> BREACH
    exceptions = ThesisEvaluator.evaluate_claim(
        claim=sample_claim,
        observed_metrics={"gross_margin": Decimal("68.5")},
        as_of_date_str="2026-11-02",
        source_citations={"gross_margin": "SEC 10-Q p.12"},
    )
    breaches = [row for row in exceptions if row.exception_type == "FALSIFIER_BREACHED"]
    assert len(breaches) == 1
    exc = breaches[0]
    assert exc.metric_name == "gross_margin"
    assert exc.observed_value == "68.5 PERCENT"
    assert exc.delta_impact == "BREAKS_BUY_CASE"
    assert exc.human_action == "REVIEW_REQUIRED"
    assert exc.source_citation == "SEC 10-Q p.12"


def test_falsifier_no_breach(sample_claim):
    # Observed margin = 72.0% (>= 70.0%) and net debt = 1.5x (<= 3.0x) -> NO BREACH
    exceptions = ThesisEvaluator.evaluate_claim(
        claim=sample_claim,
        observed_metrics={
            "gross_margin": Decimal("72.0"),
            "net_debt_ebitda": Decimal("1.5"),
        },
        as_of_date_str="2026-11-02",
    )
    assert len(exceptions) == 0


def test_catalyst_date_changed(sample_claim):
    # Earnings postponed from 2026-11-15 to 2026-11-25
    exceptions = ThesisEvaluator.evaluate_claim(
        claim=sample_claim,
        observed_metrics={},
        as_of_date_str="2026-11-02",
        observed_events={
            "CAT-EARNINGS": {
                "scheduled_date": "2026-11-25",
                "source_locator": "IR Press Release Update",
            }
        },
    )
    changes = [row for row in exceptions if row.exception_type == "DATE_CHANGED"]
    assert len(changes) == 1
    exc = changes[0]
    assert exc.expected_value == "2026-11-15"
    assert exc.observed_value == "2026-11-25"
    assert exc.source_citation == "IR Press Release Update"
    assert exc.human_action == "REVIEW_REQUIRED"


def test_catalyst_stale_detection(sample_claim):
    # Catalyst expected 2026-11-15 with window 5 days (deadline 2026-11-20)
    # Today is 2026-11-25 (5 days overdue past deadline)
    exceptions = ThesisEvaluator.evaluate_claim(
        claim=sample_claim,
        observed_metrics={},
        as_of_date_str="2026-11-25",
    )
    stale = [row for row in exceptions if row.exception_type == "STALE"]
    assert len(stale) == 1
    exc = stale[0]
    assert "Overdue by 10 days" in exc.observed_value
    assert exc.human_action == "REVIEW_REQUIRED"


def test_catalyst_not_stale_within_window(sample_claim):
    # Catalyst expected 2026-11-15 with window 5 days (deadline 2026-11-20)
    # Today is 2026-11-18 (within window) -> NOT STALE
    exceptions = ThesisEvaluator.evaluate_claim(
        claim=sample_claim,
        observed_metrics={},
        as_of_date_str="2026-11-18",
    )
    assert not [row for row in exceptions if row.exception_type in {"DATE_CHANGED", "STALE", "NEW_SOURCE"}]
    assert all(row.exception_type == "NOT_EVALUABLE" for row in exceptions)


def test_multiple_breaches_combined(sample_claim):
    # Both falsifiers breached + catalyst date changed
    exceptions = ThesisEvaluator.evaluate_claim(
        claim=sample_claim,
        observed_metrics={
            "gross_margin": Decimal("65.0"),
            "net_debt_ebitda": Decimal("3.8"),
        },
        as_of_date_str="2026-11-02",
        observed_events={
            "CAT-EARNINGS": {"scheduled_date": "2026-12-01"}
        },
    )
    assert len(exceptions) == 3
    types = [e.exception_type for e in exceptions]
    assert types.count("FALSIFIER_BREACHED") == 2
    assert types.count("DATE_CHANGED") == 1
    for e in exceptions:
        assert e.human_action == "REVIEW_REQUIRED"


def test_evaluator_operator_coverage():
    claim = ThesisClaimRecord.create(
        claim_id="TH-OP",
        ticker="TEST",
        version=1,
        as_of_utc="2026-11-01T00:00:00Z",
        claim_statement="Operator test",
    )
    # Test operators: <=, >=, ==, !=, pct_drop_exceeds
    r_lte = FalsifierRule("F1", "metric", "<=", Decimal("10.0"))
    assert ThesisEvaluator.evaluate_falsifier(claim, r_lte, Decimal("10.0"), "now") is not None
    assert ThesisEvaluator.evaluate_falsifier(claim, r_lte, Decimal("10.1"), "now") is None

    r_gte = FalsifierRule("F2", "metric", ">=", Decimal("10.0"))
    assert ThesisEvaluator.evaluate_falsifier(claim, r_gte, Decimal("10.0"), "now") is not None
    assert ThesisEvaluator.evaluate_falsifier(claim, r_gte, Decimal("9.9"), "now") is None

    r_eq = FalsifierRule("F3", "metric", "==", Decimal("10.0"))
    assert ThesisEvaluator.evaluate_falsifier(claim, r_eq, Decimal("10.0"), "now") is not None
    assert ThesisEvaluator.evaluate_falsifier(claim, r_eq, Decimal("10.1"), "now") is None

    r_neq = FalsifierRule("F4", "metric", "!=", Decimal("10.0"))
    assert ThesisEvaluator.evaluate_falsifier(claim, r_neq, Decimal("9.9"), "now") is not None
    assert ThesisEvaluator.evaluate_falsifier(claim, r_neq, Decimal("10.0"), "now") is None

    r_drop = FalsifierRule("F5", "metric", "pct_drop_exceeds", Decimal("20.0"))
    assert ThesisEvaluator.evaluate_falsifier(claim, r_drop, Decimal("25.0"), "now") is not None
    assert ThesisEvaluator.evaluate_falsifier(claim, r_drop, Decimal("15.0"), "now") is None

def test_missing_required_falsifier_metric_is_not_evaluable():
    from core.thesis.evaluator import ThesisEvaluator
    from core.thesis.models import FalsifierRule, ThesisClaimRecord
    from core.thesis.evidence_taxonomy import EvidenceTier
    from decimal import Decimal

    rule = FalsifierRule(
        falsifier_id="FAL-GROWTH",
        metric_name="revenue_growth",
        operator="<",
        threshold_value=Decimal("5"),
        unit="PERCENT",
    )
    claim = ThesisClaimRecord.create(
        claim_id="TH-ACME-v1",
        ticker="ACME",
        version=1,
        as_of_utc="2026-11-06T00:00:00Z",
        claim_statement="Revenue growth stays above the floor.",
        falsifiers=[rule],
        evidence_tier=EvidenceTier.INFERRED,
    )
    rows = ThesisEvaluator.evaluate_claim(
        claim=claim,
        observed_metrics={},
        as_of_date_str="2026-11-06",
        detected_at_utc="2026-11-06T12:00:00Z",
    )
    assert len(rows) == 1
    assert rows[0].exception_type == "NOT_EVALUABLE"
    assert rows[0].human_action == "REVIEW_REQUIRED"
    assert rows[0].status == "PENDING"
