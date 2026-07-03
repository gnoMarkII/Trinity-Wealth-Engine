import pytest
from core.nlp_utils import calculate_freshness, calculate_event_confidence, select_representative_news
from schemas.macro_schemas import MacroTheme, ThemeCategory
from pydantic import ValidationError
from datetime import datetime, timedelta, timezone

def test_calculate_freshness():
    # Test valid decay paths
    score, reason = calculate_freshness(0, ThemeCategory.POLICY)
    assert score == 1.0
    
    score, reason = calculate_freshness(24, ThemeCategory.RISK_SENTIMENT)
    assert 0.0 < score < 1.0
    
    # Very old should approach 0
    score, reason = calculate_freshness(9000, ThemeCategory.POLICY)
    assert score >= 0.0
    assert score <= 0.1

def test_calculate_event_confidence():
    assert calculate_event_confidence(1) == 0.39
    assert calculate_event_confidence(2) == 0.61
    assert calculate_event_confidence(5) == 1.0
    assert calculate_event_confidence(10) == 1.0

def test_macro_theme_validation():
    # Valid theme
    theme = MacroTheme(
        category=ThemeCategory.POLICY,
        theme_title="Test",
        deduplicated_summary="Test summary",
        age_hours=10,
        sources_count=3,
        market_impact_score=0.5,
        asset_impacts={"equity": "bullish"}
    )
    
    assert theme.event_confidence == 0.77  # (from 3 sources)
    assert theme.freshness_score > 0.0
    assert "equity" in theme.investment_conviction_contribution
    
    # Test computed conviction
    # conviction = bullish(1.0) * impact(0.5) * confidence(0.77) * freshness
    expected_conviction = 1.0 * 0.5 * 0.77 * theme.freshness_score
    assert abs(theme.investment_conviction_contribution["equity"] - expected_conviction) < 0.001

def test_schema_out_of_bounds():
    # market_impact_score > 1.0 should fail
    with pytest.raises(ValidationError):
        MacroTheme(
            category=ThemeCategory.POLICY,
            theme_title="Test",
            deduplicated_summary="Test summary",
            age_hours=10,
            sources_count=3,
            market_impact_score=5.0
        )
    
    # age_hours < 0 should fail
    with pytest.raises(ValidationError):
        MacroTheme(
            category=ThemeCategory.POLICY,
            theme_title="Test",
            deduplicated_summary="Test summary",
            age_hours=-5,
            sources_count=3
        )

def test_parse_failed_timestamp_handling():
    theme = MacroTheme(
        category=ThemeCategory.POLICY,
        theme_title="Test",
        deduplicated_summary="Test summary",
        age_hours=9999,
        sources_count=1
    )
    assert theme.freshness_score == 0.0
    assert theme.freshness_reason == "Missing or unparseable timestamp"

def test_select_representative_news():
    now = datetime.now(timezone.utc)
    cluster = [
        {"source": "random blog", "title": "Old News", "published_at": now - timedelta(days=2)},
        {"source": "reuters", "title": "Tier 1 News", "published_at": now - timedelta(days=1)},
        {"source": "bloomberg", "title": "Tier 1 Recent", "published_at": now},
        {"source": "wsj", "title": "Tier 1 Missing Date", "published_at": None}
    ]
    
    rep = select_representative_news(cluster)
    
    # Should pick 'bloomberg' because it is tier 1 and the most recent
    assert rep["source"] == "bloomberg"
    assert rep["title"] == "Tier 1 Recent"
    assert rep["sources_count"] == 4
