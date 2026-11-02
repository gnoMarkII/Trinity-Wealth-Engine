"""API Integration tests for Daily and Weekly Digest endpoints."""
import pytest
from unittest.mock import MagicMock
from api.dependencies import get_digest_service
from api.main import app


@pytest.fixture
def mock_digest_service():
    mock = MagicMock()
    mock.get_candidates.return_value = {
        "last_fetched_at": "2026-10-25T08:00:00Z",
        "news_candidates": [
            {
                "candidate_id": "cand_1",
                "title": "Fed holds interest rates steady",
                "summary": "Central bank keeps policy rate unchanged.",
                "source": "Bloomberg",
                "links": ["https://bloomberg.com/news/1"],
                "sources_count": 1,
                "published_at": "2026-10-25T07:00:00Z",
                "age_hours": 1,
                "freshness_tier": "ultra_fresh",
                "freshness_label": "1h ago",
                "is_stale": False,
                "matched_tickers": ["THB"],
                "portfolio_badges": [],
            }
        ],
        "youtube_candidates": [
            {
                "video_id": "yt_vid_1",
                "title": "Market Wrap Today",
                "channel": "Money Channel",
                "link": "https://youtube.com/watch?v=yt_vid_1",
                "published_at": "2026-10-25T06:00:00Z",
                "age_hours": 2,
                "freshness_tier": "ultra_fresh",
                "freshness_label": "2h ago",
                "thumbnail": "",
                "is_fetched": False,
                "cached_summary_snippet": "Bullish view on energy",
                "cached_tickers": ["PTTEP"],
            }
        ],
        "stale_dropped_count": 3,
    }

    mock.get_daily_digest.return_value = {
        "has_digest": True,
        "digest": {
            "digest_id": "daily-2026-10-25-morning",
            "date": "2026-10-25",
            "period": "morning",
            "generated_at": "2026-10-25T08:00:00Z",
            "market_mood": "risk_on",
            "market_mood_summary": "Global markets rally as bond yields stabilize.",
            "transmission_chain": "US Yields Fall -> Thai Baht Strengthens -> Inflows to SET Large Caps",
            "top_portfolio_focus": "DELTA and CPALL poised to benefit from liquidity expansion.",
            "selected_video_ids": ["yt_vid_1"],
            "items": [
                {
                    "source_candidate_id": "cand_1",
                    "thai_headline": "เฟดคงดอกเบี้ยตามคาด ตลาดหุ้นโลกขานรับเชิงบวก",
                    "what_happened": "เฟดคงอัตราดอกเบี้ยนโยบาย",
                    "why_it_matters": "ช่วยหนุนสภาพคล่องในตลาดเกิดใหม่",
                    "thesis_signal": "thesis_reinforcer",
                    "impact_type": "direct_holding",
                    "affected_tickers": ["DELTA"],
                    "affected_sectors": ["Technology"],
                    "macro_impact_score": 8,
                    "asset_impact_score": 7,
                    "time_horizon": "short_term",
                    "youtube_corroboration": "Money Channel ระบุว่าหุ้นเทคฯ จะได้ประโยชน์",
                    "source": "Bloomberg",
                    "link": "https://bloomberg.com/news/1",
                    "links": ["https://bloomberg.com/news/1"],
                    "published_at": "2026-10-25T07:00:00Z",
                    "age_hours": 1,
                    "freshness_tier": "ultra_fresh",
                    "freshness_label": "1h ago",
                    "portfolio_badges": [
                        {
                            "symbol": "DELTA",
                            "company_name": "Delta Electronics",
                            "impact_type": "direct_holding",
                            "weight_pct": 5.0,
                            "sector": "Technology",
                        }
                    ],
                    "is_read": False,
                    "is_new": True,
                }
            ],
            "youtube_highlights": [
                {
                    "video_id": "yt_vid_1",
                    "thai_title": "วิเคราะห์ภาพรวมตลาดหลังเฟด",
                    "channel": "Money Channel",
                    "key_takeaways": ["ดอกเบี้ยผ่านจุดสูงสุดแล้ว", "ให้เน้นหุ้นปันผลและเทคฯ"],
                    "mentioned_tickers": ["DELTA"],
                    "alignment_with_news": "confirmed",
                    "alignment_note": "มุมมองสอดคล้องกับข่าว",
                }
            ],
        },
        "snapshot_age_hours": 1,
        "is_snapshot_stale": False,
        "fresh_pending_count": 0,
        "available_dates": ["2026-10-25", "2026-10-24"],
    }

    mock.synthesize_daily.return_value = mock.get_daily_digest.return_value

    mock.get_weekly_digest.return_value = {
        "has_digest": True,
        "digest": {
            "digest_id": "weekly-2026-10-25",
            "from_date": "2026-10-18",
            "to_date": "2026-10-25",
            "generated_at": "2026-10-25T08:00:00Z",
            "executive_themes": ["Disinflation trend continues", "Tech momentum sustained"],
            "portfolio_health_summary": "All portfolio companies are tracking their respective thesis.",
            "portfolio_matrix": [
                {
                    "ticker": "DELTA",
                    "thesis_status": "strong_hold",
                    "thesis_status_label": "Strong Hold",
                    "weekly_summary": "Benefited from tech tailwinds and AI demand.",
                }
            ],
            "consensus_narratives": ["Soft landing likely"],
            "contrarian_and_pitches": ["Watch out for regional bank credit spreads"],
            "week_ahead_checklist": ["Check US Non-farm Payrolls on Friday"],
            "youtube_highlights": [],
        },
        "available_dates": ["2026-10-25"],
    }

    mock.synthesize_weekly.return_value = mock.get_weekly_digest.return_value

    mock.update_read_state.return_value = {
        "ok": True,
        "read_item_ids": ["cand_1"],
    }

    return mock


def test_get_digest_candidates(authed_client, mock_digest_service):
    app.dependency_overrides[get_digest_service] = lambda: mock_digest_service
    try:
        resp = authed_client.get("/api/macro/digest/candidates?days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["news_candidates"]) == 1
        assert data["news_candidates"][0]["candidate_id"] == "cand_1"
        assert len(data["youtube_candidates"]) == 1
        assert data["youtube_candidates"][0]["video_id"] == "yt_vid_1"
        assert data["stale_dropped_count"] == 3
    finally:
        app.dependency_overrides.pop(get_digest_service, None)


def test_get_daily_digest(authed_client, mock_digest_service):
    app.dependency_overrides[get_digest_service] = lambda: mock_digest_service
    try:
        resp = authed_client.get("/api/macro/digest/daily")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_digest"] is True
        assert data["available_dates"] == ["2026-10-25", "2026-10-24"]
        assert data["digest"]["market_mood"] == "risk_on"
        assert len(data["digest"]["items"]) == 1
        assert data["digest"]["items"][0]["thai_headline"].startswith("เฟดคงดอกเบี้ย")
    finally:
        app.dependency_overrides.pop(get_digest_service, None)


def test_synthesize_daily_digest(authed_client, mock_digest_service):
    app.dependency_overrides[get_digest_service] = lambda: mock_digest_service
    try:
        payload = {
            "selected_video_ids": ["yt_vid_1"],
            "excluded_news_ids": [],
            "custom_youtube_urls": [],
            "custom_news_items": [],
        }
        resp = authed_client.post("/api/macro/digest/daily/synthesize", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_digest"] is True
        mock_digest_service.synthesize_daily.assert_called_once()
    finally:
        app.dependency_overrides.pop(get_digest_service, None)


def test_get_and_synthesize_weekly_digest(authed_client, mock_digest_service):
    app.dependency_overrides[get_digest_service] = lambda: mock_digest_service
    try:
        # GET
        resp_get = authed_client.get("/api/macro/digest/weekly")
        assert resp_get.status_code == 200
        data_get = resp_get.json()
        assert data_get["has_digest"] is True
        assert data_get["available_dates"] == ["2026-10-25"]
        assert len(data_get["digest"]["portfolio_matrix"]) == 1

        # POST synthesize
        payload = {
            "days": 7,
            "selected_video_ids": ["yt_vid_1"],
            "custom_youtube_urls": [],
        }
        resp_post = authed_client.post("/api/macro/digest/weekly/synthesize", json=payload)
        assert resp_post.status_code == 200
        data_post = resp_post.json()
        assert data_post["has_digest"] is True
    finally:
        app.dependency_overrides.pop(get_digest_service, None)


def test_update_read_state(authed_client, mock_digest_service):
    app.dependency_overrides[get_digest_service] = lambda: mock_digest_service
    try:
        payload = {
            "item_ids": ["cand_1"],
            "mark_as_read": True,
        }
        resp = authed_client.post("/api/macro/digest/read-state", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "cand_1" in data["read_item_ids"]
    finally:
        app.dependency_overrides.pop(get_digest_service, None)
