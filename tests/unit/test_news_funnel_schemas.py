"""Unit tests สำหรับ schemas/news_funnel_schemas.py"""
import pytest
from schemas.news_funnel_schemas import (
    NewsFunnelRawItem,
    ClusteredNewsEvent,
    MacroImpactTriageResult,
    MacroThemeDigest,
    DailyFunnelReport,
)


def test_news_funnel_raw_item():
    item = NewsFunnelRawItem(
        title="Fed Signals Steady Rates",
        summary="Service inflation remains high",
        link="https://example.com/fed",
        source="Investing.com",
        published_at="2026-07-13T08:00:00Z"
    )
    assert item.title == "Fed Signals Steady Rates"
    assert item.summary == "Service inflation remains high"
    assert item.source == "Investing.com"


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


def test_daily_funnel_report():
    theme = MacroThemeDigest(
        theme_title="Fed Policy & Inflation",
        key_takeaways=["Service inflation sticky"],
        linked_assets=["[[NVDA]]", "[[US Treasury]]"],
        linked_themes=["[[Monetary Policy]]", "[[Inflation]]"],
        policy_implications="Hold cash buffer"
    )
    report = DailyFunnelReport(
        report_title="Macro Themes Digest - 2026-07-13 Morning",
        report_date="2026-07-13",
        batch_period="morning_12h",
        approved_by="scheduled_auto",
        themes=[theme],
        total_events_analyzed=15,
        high_impact_event_ids=["evt-1"]
    )
    assert len(report.themes) == 1
    assert report.themes[0].linked_assets == ["[[NVDA]]", "[[US Treasury]]"]
    assert report.approved_by == "scheduled_auto"


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
