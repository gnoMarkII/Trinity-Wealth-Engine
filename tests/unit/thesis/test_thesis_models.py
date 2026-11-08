"""Unit tests for Structured Thesis & As-Of Research State Domain Models."""
from decimal import Decimal
import pytest
from dataclasses import FrozenInstanceError

from core.thesis.evidence_taxonomy import EvidenceProvenance, EvidenceTier
from core.thesis.models import (
    CatalystEvent,
    FalsifierRule,
    ThesisClaimRecord,
    compute_thesis_checksum,
)


def test_falsifier_rule_creation_and_serialization():
    rule = FalsifierRule(
        falsifier_id="FAL-NVDA-DC-REV",
        metric_name="datacenter_revenue",
        operator="<",
        threshold_value=Decimal("35000000000"),
        unit="USD",
        period_basis="QUARTERLY",
        invalidation_impact="BREAKS_BUY_CASE",
        narrative_explanation="Revenue below 35B indicates slowing hyperscaler Capex.",
    )
    d = rule.to_dict()
    assert d["falsifier_id"] == "FAL-NVDA-DC-REV"
    assert d["threshold_value"] == "35000000000"

    reconstructed = FalsifierRule.from_dict(d)
    assert reconstructed.falsifier_id == rule.falsifier_id
    assert reconstructed.threshold_value == Decimal("35000000000")
    assert reconstructed == rule


def test_catalyst_event_creation_and_serialization():
    cat = CatalystEvent(
        catalyst_id="CAT-001",
        event_name="Q3 Earnings Release",
        expected_date_str="2026-11-20",
        confidence_window_days=7,
        source_locator="SEC 8-K / IR Calendar",
    )
    d = cat.to_dict()
    assert d["expected_date_str"] == "2026-11-20"
    reconstructed = CatalystEvent.from_dict(d)
    assert reconstructed == cat


def test_thesis_claim_record_checksum_and_integrity():
    falsifier = FalsifierRule(
        falsifier_id="FAL-01",
        metric_name="gross_margin",
        operator="<",
        threshold_value=Decimal("70.0"),
        unit="PERCENT",
    )
    catalyst = CatalystEvent(
        catalyst_id="CAT-01",
        event_name="Earnings Call",
        expected_date_str="2026-11-15",
    )
    claim = ThesisClaimRecord.create(
        claim_id="TH-NVDA-001",
        ticker="NVDA",
        version=1,
        as_of_utc="2026-11-02T12:00:00Z",
        claim_statement="NVDA maintains >70% gross margins through Blackwell ramp.",
        falsifiers=[falsifier],
        linked_catalysts=[catalyst],
        evidence_tier=EvidenceTier.DOCUMENTED,
    )

    assert claim.checksum_sha256 != ""
    assert claim.verify_integrity() is True

    # Tampered record must fail integrity
    tampered_claim = ThesisClaimRecord(
        claim_id=claim.claim_id,
        ticker=claim.ticker,
        version=claim.version,
        as_of_utc=claim.as_of_utc,
        claim_statement="TAMPERED STATEMENT",
        falsifiers=claim.falsifiers,
        linked_catalysts=claim.linked_catalysts,
        status=claim.status,
        evidence_tier=claim.evidence_tier,
        author_role=claim.author_role,
        prior_claim_id=claim.prior_claim_id,
        checksum_sha256=claim.checksum_sha256,
    )
    assert tampered_claim.verify_integrity() is False


def test_thesis_claim_immutability():
    claim = ThesisClaimRecord.create(
        claim_id="TH-001",
        ticker="AAPL",
        version=1,
        as_of_utc="2026-11-02T12:00:00Z",
        claim_statement="Services growth > 12%",
    )
    with pytest.raises(FrozenInstanceError):
        claim.status = "REVISED"  # type: ignore


def test_evidence_provenance_badge_formatting():
    prov_doc = EvidenceProvenance(
        tier=EvidenceTier.DOCUMENTED,
        source_locator="SEC 10-Q 2026-Q2 p.34",
    )
    assert prov_doc.to_badge_tag() == "[DOCUMENTED: SEC 10-Q 2026-Q2 p.34]"

    prov_test = EvidenceProvenance(
        tier=EvidenceTier.DIRECTLY_TESTED,
        source_locator="Python Sandbox DCF",
    )
    assert prov_test.to_badge_tag() == "[TESTED: Python Sandbox DCF]"

    prov_inf = EvidenceProvenance(
        tier=EvidenceTier.INFERRED,
        source_locator="Analyst Synthesis",
    )
    assert prov_inf.to_badge_tag() == "[INFERRED: Analyst Synthesis]"

    prov_unv = EvidenceProvenance(
        tier=EvidenceTier.UNVERIFIED,
        source_locator="Twitter rumor",
    )
    assert prov_unv.to_badge_tag() == "[UNVERIFIED: Twitter rumor]"

