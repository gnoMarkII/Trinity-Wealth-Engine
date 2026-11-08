"""Unit tests for Anti-Citation-Laundering & Claim Evidence Validator.

Hexagonal Architecture Rule: Pure domain testing using standard library fixtures.
"""
from decimal import Decimal
import pytest

from core.thesis.evidence_taxonomy import EvidenceTier
from core.thesis.claim_validator import (
    ClaimRecord,
    ClaimValidationResult,
    validate_claim_evidence,
    batch_validate_claims,
)


def test_documented_claim_verified_exact_match():
    """Documented claim matching primary source disclosure passes validation."""
    claim = ClaimRecord(
        claim_id="CLM-NVDA-REV-01",
        ticker="NVDA",
        statement="NVIDIA reported Q2 FY2026 revenue of $30.04B, driven by Hopper/Blackwell compute.",
        metric_name="revenue_usd_billions",
        claimed_value=Decimal("30.04"),
        evidence_tier=EvidenceTier.DOCUMENTED,
        source_locator="SEC 10-Q Q2-2026 p.18 Item 1",
        accession_number="0001045810-26-000045",
    )

    primary_facts = {
        "revenue_usd_billions": Decimal("30.04"),
        "gross_margin_pct": Decimal("75.1"),
    }

    result = validate_claim_evidence(claim, primary_facts)
    assert result.is_valid is True
    assert result.effective_tier == EvidenceTier.DOCUMENTED
    assert result.status == "VERIFIED"
    assert result.delta_bps == Decimal("0.00")
    assert "matches primary source" in result.explanation


def test_anti_citation_laundering_detected_and_demoted():
    """AI hallucinates or misquotes a number while providing a real SEC link.
    
    The validator detects the mismatch, demotes to UNVERIFIED, and flags CITATION_MISMATCH.
    """
    claim = ClaimRecord(
        claim_id="CLM-NVDA-MARGIN-01",
        ticker="NVDA",
        statement="NVIDIA gross margin expanded to 82.5% in Q2 (citing SEC 10-Q).",
        metric_name="gross_margin_pct",
        claimed_value=Decimal("82.5"),  # Hallucinated or wrong quarter
        evidence_tier=EvidenceTier.DOCUMENTED,
        source_locator="SEC 10-Q Q2-2026 p.18 Item 1",
        tolerance_bps=10,  # 0.10% tolerance
    )

    # Actual number in 10-Q is 75.1%
    primary_facts = {
        "revenue_usd_billions": Decimal("30.04"),
        "gross_margin_pct": Decimal("75.1"),
    }

    result = validate_claim_evidence(claim, primary_facts)
    assert result.is_valid is False
    assert result.effective_tier == EvidenceTier.UNVERIFIED
    assert result.status == "CITATION_MISMATCH"
    assert result.delta_bps is not None
    assert result.delta_bps > Decimal("10")
    assert "Anti-Citation Laundering VIOLATION" in result.explanation


def test_documented_claim_within_rounding_tolerance():
    """Claim within 10 bps rounding tolerance passes validation."""
    claim = ClaimRecord(
        claim_id="CLM-AAPL-MARGIN-01",
        ticker="AAPL",
        statement="Apple gross margin reached 46.3% (reported 46.26%).",
        metric_name="gross_margin_pct",
        claimed_value=Decimal("46.3"),
        evidence_tier=EvidenceTier.DOCUMENTED,
        source_locator="Apple Inc Form 10-Q Q3 p.22",
        tolerance_bps=15,
    )

    primary_facts = {
        "gross_margin_pct": Decimal("46.26"),
    }

    result = validate_claim_evidence(claim, primary_facts)
    assert result.is_valid is True
    assert result.effective_tier == EvidenceTier.DOCUMENTED
    assert result.status == "VERIFIED"


def test_documented_claim_missing_locator():
    """Documented claim without source locator fails immediately."""
    claim = ClaimRecord(
        claim_id="CLM-BAD-01",
        ticker="NVDA",
        statement="Revenue was huge",
        metric_name="revenue",
        claimed_value=Decimal("30.0"),
        evidence_tier=EvidenceTier.DOCUMENTED,
        source_locator="",  # Missing
    )
    result = validate_claim_evidence(claim, {"revenue": Decimal("30.0")})
    assert result.is_valid is False
    assert result.effective_tier == EvidenceTier.UNVERIFIED
    assert result.status == "UNVERIFIED_SOURCE"


def test_documented_claim_missing_primary_fact():
    """Claim metric absent from primary facts extracted from source is demoted."""
    claim = ClaimRecord(
        claim_id="CLM-MISS-01",
        ticker="NVDA",
        statement="Datacenter AI revenue reached 26.3B",
        metric_name="datacenter_ai_revenue",
        claimed_value=Decimal("26.3"),
        evidence_tier=EvidenceTier.DOCUMENTED,
        source_locator="SEC 10-Q p.19",
    )
    result = validate_claim_evidence(claim, {"total_revenue": Decimal("30.04")})
    assert result.is_valid is False
    assert result.effective_tier == EvidenceTier.UNVERIFIED
    assert result.status == "MISSING_PRIMARY_FACT"


def test_directly_tested_claim_verified_against_engine_output():
    """Directly tested claim against Python execution engine."""
    claim = ClaimRecord(
        claim_id="CLM-TEST-01",
        ticker="SPY",
        statement="Portfolio beta against SPY is 1.050",
        metric_name="portfolio_beta",
        claimed_value=Decimal("1.050"),
        evidence_tier=EvidenceTier.DIRECTLY_TESTED,
        source_locator="Python Sample Beta (N=20)",
    )
    engine_facts = {"portfolio_beta": Decimal("1.050")}
    result = validate_claim_evidence(claim, engine_facts)
    assert result.is_valid is True
    assert result.effective_tier == EvidenceTier.DIRECTLY_TESTED
    assert result.status == "VERIFIED"


def test_directly_tested_claim_math_mismatch():
    """Directly tested claim where claimed number doesn't match engine math."""
    claim = ClaimRecord(
        claim_id="CLM-TEST-02",
        ticker="SPY",
        statement="Portfolio beta is 0.850",
        metric_name="portfolio_beta",
        claimed_value=Decimal("0.850"),
        evidence_tier=EvidenceTier.DIRECTLY_TESTED,
        source_locator="Python Sample Beta (N=20)",
    )
    engine_facts = {"portfolio_beta": Decimal("1.050")}
    result = validate_claim_evidence(claim, engine_facts)
    assert result.is_valid is False
    assert result.effective_tier == EvidenceTier.UNVERIFIED
    assert result.status == "TEST_MATH_MISMATCH"


def test_inferred_claim_opinion_preservation():
    """Analytical hypothesis is recognized as INFERRED, not rejected as fraud."""
    claim = ClaimRecord(
        claim_id="CLM-INF-01",
        ticker="NVDA",
        statement="Blackwell shipment ramp may face packaging bottlenecks in Q3.",
        evidence_tier=EvidenceTier.INFERRED,
        source_locator="AI Analyst Rationale",
    )
    result = validate_claim_evidence(claim, {})
    assert result.is_valid is True
    assert result.effective_tier == EvidenceTier.INFERRED
    assert result.status == "INFERRED_OPINION"


def test_batch_validate_claims():
    """Batch validation flags batch validity and reports all items."""
    claims = [
        ClaimRecord(
            claim_id="CLM-01",
            ticker="NVDA",
            statement="Revenue 30.04B",
            metric_name="revenue",
            claimed_value=Decimal("30.04"),
            evidence_tier=EvidenceTier.DOCUMENTED,
            source_locator="SEC 10-Q",
        ),
        ClaimRecord(
            claim_id="CLM-02",
            ticker="NVDA",
            statement="Margin 90.0%",
            metric_name="margin",
            claimed_value=Decimal("90.0"),  # Wrong!
            evidence_tier=EvidenceTier.DOCUMENTED,
            source_locator="SEC 10-Q",
        ),
    ]
    facts = {
        "revenue": Decimal("30.04"),
        "margin": Decimal("75.1"),
    }
    results, all_valid = batch_validate_claims(claims, facts)
    assert len(results) == 2
    assert all_valid is False
    assert results[0].is_valid is True
    assert results[1].is_valid is False
    assert results[1].status == "CITATION_MISMATCH"

