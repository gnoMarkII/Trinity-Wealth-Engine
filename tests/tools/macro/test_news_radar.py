from datetime import datetime, timezone
import pytest
from unittest.mock import patch, MagicMock

from tools.macro.news_radar import (
    FEEDS,
    _parse_entry_published_at,
    get_news_candidates,
    generate_news_radar_daily,
)


def test_feeds_structure():
    """ตรวจสอบว่า Feeds ทั้งหมดมี schema ที่ถูกต้อง และไม่มี feed ที่ตายแล้ว (investing.com 285)"""
    assert len(FEEDS) >= 4
    for feed in FEEDS:
        assert "name" in feed and feed["name"]
        assert "url" in feed and feed["url"].startswith("http")
        assert "investing.com/rss/news_285.rss" not in feed["url"]


def test_parse_entry_published_at():
    """ตรวจสอบการแปลง published_parsed, updated_parsed และ pub_date_str"""
    # 1. published_parsed
    entry_parsed = MagicMock()
    entry_parsed.published_parsed = (2026, 9, 28, 10, 30, 0, 0, 271, 0)
    dt = _parse_entry_published_at(entry_parsed)
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 9
    assert dt.day == 28

    # 2. updated_parsed fallback
    entry_updated = MagicMock(spec=["updated_parsed"])
    entry_updated.updated_parsed = (2026, 9, 28, 12, 0, 0, 0, 271, 0)
    dt2 = _parse_entry_published_at(entry_updated)
    assert dt2 is not None
    assert dt2.hour == 12

    # 3. String pub_date
    entry_str = MagicMock(spec=["published"])
    entry_str.published = "Mon, 28 Sep 2026 04:04:16 GMT"
    dt3 = _parse_entry_published_at(entry_str)
    assert dt3 is not None
    assert dt3.year == 2026

    # 4. None / unparseable
    entry_empty = MagicMock(spec=[])
    assert _parse_entry_published_at(entry_empty) is None


from datetime import timedelta

def test_get_news_candidates_freshness_first_sorting(monkeypatch):
    """ตรวจสอบว่าการจัดลำดับให้ความสำคัญกับข่าวสด (<24 ชม.) ก่อนข่าวเก่า"""
    now = datetime.now(timezone.utc)
    t_fresh = now - timedelta(hours=2)
    t_recent = now - timedelta(hours=30)
    t_stale = now - timedelta(hours=70)

    # Fake entries across different ages
    fake_entries = [
        MagicMock(
            title="Stale News 70h ago",
            link="http://example.com/stale",
            published_parsed=t_stale.utctimetuple(),
            summary="Stale summary",
            source=None,
        ),
        MagicMock(
            title="Recent News 30h ago",
            link="http://example.com/recent",
            published_parsed=t_recent.utctimetuple(),
            summary="Recent summary",
            source=None,
        ),
        MagicMock(
            title="Fresh News 2h ago",
            link="http://example.com/fresh",
            published_parsed=t_fresh.utctimetuple(),
            summary="Fresh summary",
            source=None,
        ),
    ]

    monkeypatch.setattr("tools.macro.news_radar._fetch_feed_entries", lambda feed: fake_entries if "CNBC" in feed["name"] else [])
    monkeypatch.setattr("tools.macro.news_radar._is_url_fetched", lambda link: False)

    candidates = get_news_candidates(max_items=10)
    assert len(candidates) == 3
    # Fresh (<24h) must be first
    assert candidates[0]["title"] == "Fresh News 2h ago"
    assert candidates[0]["age_hours"] in (1, 2, 3)
    # Recent (24-48h) must be second
    assert candidates[1]["title"] == "Recent News 30h ago"
    # Stale (>48h) must be third
    assert candidates[2]["title"] == "Stale News 70h ago"


def test_generate_news_radar_daily_markdown(monkeypatch):
    """ตรวจสอบว่า generate_news_radar_daily ผลิต Markdown ตารางสรุปได้อย่างถูกต้อง"""
    mock_candidates = [
        {
            "title": "Fed Rate Decision Expected",
            "summary": "Policy meeting underway",
            "link": "https://example.com/fed",
            "source": "CNBC",
            "sources_count": 2,
            "age_hours": 3,
            "freshness_reason": "Fresh (3h)",
            "is_stale": False,
            "is_fetched": False,
        }
    ]
    monkeypatch.setattr("tools.macro.news_radar.get_news_candidates", lambda max_items=25: mock_candidates)

    report = generate_news_radar_daily.invoke({})
    assert "📡 Macro News Radar" in report
    assert "Fed Rate Decision Expected" in report
    assert "https://example.com/fed" in report
    assert "[ ]" in report
