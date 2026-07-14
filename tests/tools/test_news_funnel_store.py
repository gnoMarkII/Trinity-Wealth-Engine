"""Unit tests สำหรับ tools/macro/news_funnel_store.py"""
import os
import tempfile
import pytest

from tools.macro.news_funnel_store import (
    load_store,
    save_store,
    is_title_or_url_processed,
    save_triage_events,
    get_pending_high_impact_events,
    mark_events_synthesized,
)


@pytest.fixture
def temp_store(tmp_path):
    store_file = tmp_path / "test_state.json"
    return str(store_file)


def test_load_initial_store(temp_store):
    state = load_store(store_path=temp_store)
    assert state["schema_version"] == 1
    assert state["processed_urls"] == []
    assert state["pending_events"] == []


def test_save_and_load_store(temp_store):
    state = load_store(store_path=temp_store)
    state["processed_urls"].append("https://example.com/a")
    save_store(state, store_path=temp_store)

    reloaded = load_store(store_path=temp_store)
    assert reloaded["schema_version"] == 1
    assert "https://example.com/a" in reloaded["processed_urls"]


def test_is_title_or_url_processed(temp_store):
    state = load_store(store_path=temp_store)
    state["processed_urls"].append("https://example.com/news1")
    state["processed_titles"].append("Federal Reserve keeps rates unchanged amid inflation concerns")
    save_store(state, store_path=temp_store)

    assert is_title_or_url_processed("Other Title", "https://example.com/news1", store_path=temp_store) is True
    # Exact title match
    assert is_title_or_url_processed("Federal Reserve keeps rates unchanged amid inflation concerns", "https://example.com/news2", store_path=temp_store) is True
    # Similar title match (Jaccard similarity >= threshold)
    assert is_title_or_url_processed("Federal Reserve keeps rates unchanged amid inflation", "https://example.com/news2", store_path=temp_store) is True
    # Unrelated title
    assert is_title_or_url_processed("Tech stocks surge on strong earnings reports", "https://example.com/news3", store_path=temp_store) is False


def test_triage_events_and_pending_high_impact(temp_store):
    events = [
        {
            "event_id": "e1",
            "canonical_title": "High Impact Macro Event",
            "comprehensive_summary": "Summary 1",
            "macro_impact_score": 8,
            "asset_impact_score": 5,
            "is_high_impact": True,
            "links": ["https://example.com/e1"],
        },
        {
            "event_id": "e2",
            "canonical_title": "Low Impact Event",
            "comprehensive_summary": "Summary 2",
            "macro_impact_score": 4,
            "asset_impact_score": 3,
            "is_high_impact": False,
            "links": ["https://example.com/e2"],
        },
    ]

    save_triage_events(events, store_path=temp_store)

    pending = get_pending_high_impact_events(store_path=temp_store)
    assert len(pending) == 1
    assert pending[0]["event_id"] == "e1"

    # Verify deduplication of processed URLs and titles
    assert is_title_or_url_processed("High Impact Macro Event", "https://example.com/e1", store_path=temp_store) is True


def test_mark_events_synthesized(temp_store):
    events = [
        {
            "event_id": "e1",
            "canonical_title": "Event 1",
            "macro_impact_score": 8,
            "asset_impact_score": 7,
            "is_high_impact": True,
        }
    ]
    save_triage_events(events, store_path=temp_store)

    mark_events_synthesized(["e1"], store_path=temp_store)

    pending = get_pending_high_impact_events(store_path=temp_store)
    assert len(pending) == 0

    state = load_store(store_path=temp_store)
    assert any(ev["event_id"] == "e1" and ev["status"] == "synthesized" for ev in state["pending_events"])
