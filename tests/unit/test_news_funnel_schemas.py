"""Unit tests สำหรับ schemas/news_funnel_schemas.py"""
import pytest
from schemas.news_funnel_schemas import (
    MacroImpactTriageResult,
)


def test_triage_is_high_impact_computed_field():
    # Case 1: macro score >= 7 -> True
    triage1 = MacroImpactTriageResult(
        macro_impact_score=7,
        asset_impact_score=5,
        primary_tags=["inflation"],
        extracted_tickers=["NVDA"],
        extracted_themes=["policy"],
        triage_reasoning="High macro impact"
    )
    assert triage1.is_high_impact is True

    # Case 2: asset score >= 7 -> True
    triage2 = MacroImpactTriageResult(
        macro_impact_score=4,
        asset_impact_score=8,
        primary_tags=["earnings"],
        extracted_tickers=["AAPL"],
        extracted_themes=["earnings"],
        triage_reasoning="High asset impact"
    )
    assert triage2.is_high_impact is True

    # Case 3: both < 7 -> False
    triage3 = MacroImpactTriageResult(
        macro_impact_score=6,
        asset_impact_score=6,
        primary_tags=["general"],
        extracted_tickers=[],
        extracted_themes=["growth"],
        triage_reasoning="Moderate impact"
    )
    assert triage3.is_high_impact is False

    # Check model_dump includes computed field
    dump = triage1.model_dump()
    assert dump["is_high_impact"] is True


def test_strip_wikilink_alias_and_whitespace():
    from schemas.news_funnel_schemas import strip_wikilink
    assert strip_wikilink("[[AAPL]]") == "AAPL"
    assert strip_wikilink("[[AAPL|Apple Inc.]]") == "AAPL"
    assert strip_wikilink("  [[AAPL | Apple Inc.]]  ") == "AAPL"
    assert strip_wikilink("AAPL | Apple Inc.") == "AAPL"
    assert strip_wikilink("AAPL") == "AAPL"


def test_extracted_links_validator():
    triage = MacroImpactTriageResult(
        macro_impact_score=8,
        asset_impact_score=8,
        primary_tags=["tech"],
        extracted_tickers=["[[AAPL|Apple Inc.]]", "  [[NVDA]]  "],
        extracted_themes=["[[policy|Monetary Policy]]", "inflation"]
    )
    assert triage.extracted_tickers == ["AAPL", "NVDA"]
    assert triage.extracted_themes == ["policy", "inflation"]
