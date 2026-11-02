"""Unit tests for Core News Digest deterministic engine."""
from datetime import datetime, timedelta, timezone
import pytest

from core.news_digest.engine import (
    evaluate_freshness,
    filter_fresh_news_candidates,
    match_tickers_and_portfolio,
    build_one_shot_daily_prompt,
    build_one_shot_weekly_prompt,
)
from core.news_digest.models import (
    PreLLMNewsCandidate,
    PortfolioHoldingContext,
)


def test_evaluate_freshness_none():
    now_utc = datetime.now(timezone.utc)
    age, tier, label, is_stale = evaluate_freshness(None, now_utc=now_utc)
    assert age == 9999
    assert tier == "stale"
    assert is_stale is True


def test_evaluate_freshness_ultra_fresh():
    now_utc = datetime.now(timezone.utc)
    two_hours_ago = (now_utc - timedelta(hours=2)).isoformat()
    age, tier, label, is_stale = evaluate_freshness(two_hours_ago, now_utc=now_utc)
    assert age == 2
    assert tier == "ultra_fresh"
    assert is_stale is False
    assert "2 ชม." in label or "สดใหม่" in label


def test_evaluate_freshness_fresh_today():
    now_utc = datetime.now(timezone.utc)
    ten_hours_ago = (now_utc - timedelta(hours=10)).isoformat()
    age, tier, label, is_stale = evaluate_freshness(ten_hours_ago, now_utc=now_utc)
    assert age == 10
    assert tier == "fresh_today"
    assert is_stale is False
    assert "10 ชม." in label or "วันนี้" in label


def test_evaluate_freshness_stale_over_36_hours():
    now_utc = datetime.now(timezone.utc)
    forty_hours_ago = (now_utc - timedelta(hours=40)).isoformat()
    age, tier, label, is_stale = evaluate_freshness(forty_hours_ago, now_utc=now_utc, max_age_hours=36)
    assert age == 40
    assert tier == "stale"
    assert is_stale is True


def test_filter_fresh_news_candidates():
    now_utc = datetime.now(timezone.utc)
    cand_fresh = {
        "candidate_id": "c1",
        "title": "NVIDIA Q3 Earnings beat expectations",
        "summary": "Record revenue",
        "source": "Bloomberg",
        "published_at": (now_utc - timedelta(hours=4)).isoformat(),
    }
    cand_stale = {
        "candidate_id": "c2",
        "title": "Old news from last week",
        "summary": "Outdated news",
        "source": "Reuters",
        "published_at": (now_utc - timedelta(hours=72)).isoformat(),
    }
    cand_no_date = {
        "candidate_id": "c3",
        "title": "News without date",
        "summary": "No timestamp",
        "source": "Unknown",
        "published_at": None,
    }

    fresh, dropped = filter_fresh_news_candidates(
        [cand_fresh, cand_stale, cand_no_date],
        now_utc=now_utc,
        max_age_hours=36,
    )
    assert len(fresh) == 1
    assert fresh[0]["candidate_id"] == "c1"
    assert dropped == 2


def test_match_tickers_and_portfolio():
    holdings_map = {
        "NVDA": PortfolioHoldingContext(
            symbol="NVDA",
            company_name="NVIDIA Corporation",
            impact_type="direct_holding",
            weight_pct=8.5,
            unrealized_pnl_pct=15.0,
            sector="Technology",
        ),
        "DELTA": PortfolioHoldingContext(
            symbol="DELTA",
            company_name="Delta Electronics Thailand",
            impact_type="watchlist",
            weight_pct=0.0,
            sector="Technology",
        ),
    }

    text = "NVDA and DELTA announce new AI data center partnership with CPALL."
    matched, badges = match_tickers_and_portfolio(
        text=text,
        explicit_tickers=["CPALL"],
        holdings_map=holdings_map,
    )

    assert "NVDA" in matched
    assert "DELTA" in matched
    assert "CPALL" in matched

    badge_syms = [b.symbol for b in badges]
    assert "NVDA" in badge_syms
    assert "DELTA" in badge_syms
    nvda_badge = next(b for b in badges if b.symbol == "NVDA")
    assert nvda_badge.impact_type == "direct_holding"
    assert nvda_badge.weight_pct == 8.5

    # Test Thai adjacent text and stopword suppression
    thai_text = "รายงานพิเศษ: หุ้นDELTAพุ่งแรงรับกระแสAI แต่ตลาดเตือน It is on track for all players"
    matched_thai, _ = match_tickers_and_portfolio(
        text=thai_text,
        explicit_tickers=[],
        holdings_map=holdings_map,
    )
    assert "DELTA" in matched_thai
    assert "IT" not in matched_thai
    assert "ON" not in matched_thai
    assert "ALL" not in matched_thai

    # Test explicit cashtag $ON matching
    cashtag_text = "Semiconductor rally: $ON reports earnings beating expectations"
    matched_cashtag, _ = match_tickers_and_portfolio(
        text=cashtag_text,
        explicit_tickers=[],
        holdings_map={"ON": PortfolioHoldingContext(symbol="ON", impact_type="direct_holding")},
    )
    assert "ON" in matched_cashtag


def test_build_one_shot_daily_prompt():
    prompt = build_one_shot_daily_prompt(
        candidates=[
            PreLLMNewsCandidate(
                candidate_id="n1",
                title="NVDA hits all-time high",
                summary="AI demand surges",
                source="CNBC",
            )
        ],
        youtube_docs=[
            {
                "video_id": "yt123",
                "title": "Fed Interest Rate Discussion",
                "channel": "Money Coach",
                "transcript": "Interest rates may pause here.",
            }
        ],
        portfolio_summary="Total AUM: 1,000,000 THB | Direct Holdings: NVDA (10%)",
    )

    assert "ข้อมูลพอร์ตและ WATCHLIST" in prompt
    assert "NVDA" in prompt
    assert "คลิป YOUTUBE" in prompt
    assert "yt123" in prompt
    assert "One-Shot Mega-Batch" in prompt or "Chief Investment Officer" in prompt


def test_digest_store_auto_pruning():
    from tools.macro.digest_store import _prune_digest_store

    state = {
        "daily_history": {f"2026-01-{i:02d}": {"date": f"2026-01-{i:02d}"} for i in range(1, 71)},
        "weekly_history": {f"2026-W{i:02d}": {"digest_id": f"2026-W{i:02d}"} for i in range(1, 36)},
        "ticker_intelligence_index": {
            "DELTA": {
                "news_items": [{"id": f"news_{j}"} for j in range(40)],
                "youtube_mentions": [{"video_id": f"vid_{j}"} for j in range(25)],
            }
        },
        "read_item_ids": [f"read_{k}" for k in range(1200)],
    }

    _prune_digest_store(state)

    # Assert 60 daily records kept
    assert len(state["daily_history"]) == 60
    assert "2026-01-70" in state["daily_history"]
    assert "2026-01-01" not in state["daily_history"]

    # Assert 26 weekly records kept
    assert len(state["weekly_history"]) == 26
    assert "2026-W35" in state["weekly_history"]
    assert "2026-W01" not in state["weekly_history"]

    # Assert ticker news items capped at 25 and youtube mentions at 15
    delta_entry = state["ticker_intelligence_index"]["DELTA"]
    assert len(delta_entry["news_items"]) == 25
    assert len(delta_entry["youtube_mentions"]) == 15

    # Assert read_item_ids capped at 1000
    assert len(state["read_item_ids"]) == 1000
    assert state["read_item_ids"][-1] == "read_1199"
    assert state["read_item_ids"][0] == "read_200"

