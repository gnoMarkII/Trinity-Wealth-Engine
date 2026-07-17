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
    update_events_status,
    get_filtered_or_rejected_events,
    save_raw_candidates,
    get_raw_candidates,
    remove_processed_raw_candidates,
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


def test_update_events_status_synthesized(temp_store):
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

    update_events_status(synthesized_ids=["e1"], store_path=temp_store)

    pending = get_pending_high_impact_events(store_path=temp_store)
    assert len(pending) == 0

    state = load_store(store_path=temp_store)
    assert any(ev["event_id"] == "e1" and ev["status"] == "synthesized" for ev in state["pending_events"])


def test_update_events_status_rejected(temp_store):
    events = [
        {
            "event_id": "e2",
            "canonical_title": "Event 2",
            "macro_impact_score": 8,
            "asset_impact_score": 7,
            "is_high_impact": True,
        }
    ]
    save_triage_events(events, store_path=temp_store)

    update_events_status(rejected_ids=["e2"], store_path=temp_store)

    pending = get_pending_high_impact_events(store_path=temp_store)
    assert len(pending) == 0

    state = load_store(store_path=temp_store)
    assert any(ev["event_id"] == "e2" and ev["status"] == "rejected" and "rejected_at" in ev for ev in state["pending_events"])


def test_update_events_status_skipped_error(temp_store):
    events = [
        {
            "event_id": "e3",
            "canonical_title": "Event 3",
            "macro_impact_score": 8,
            "asset_impact_score": 7,
            "is_high_impact": True,
        }
    ]
    save_triage_events(events, store_path=temp_store)

    update_events_status(
        skipped_error_ids=["e3"],
        error_msgs={"e3": "Paywall detected"},
        store_path=temp_store,
    )

    state = load_store(store_path=temp_store)
    ev3 = next(ev for ev in state["pending_events"] if ev["event_id"] == "e3")
    assert ev3["status"] == "skipped_error"
    assert ev3["error_msg"] == "Paywall detected"
    assert "skipped_at" in ev3


def test_get_filtered_or_rejected_events(temp_store):
    events = [
        {
            "event_id": "high_pending",
            "canonical_title": "High Impact Pending",
            "macro_impact_score": 8,
            "asset_impact_score": 7,
            "is_high_impact": True,
            "ingested_at": "2026-07-17T10:00:00",
        },
        {
            "event_id": "low_pending",
            "canonical_title": "Low Impact Pending",
            "macro_impact_score": 4,
            "asset_impact_score": 3,
            "is_high_impact": False,
            "ingested_at": "2026-07-17T11:00:00",
        },
        {
            "event_id": "rej_ev",
            "canonical_title": "Rejected Event",
            "macro_impact_score": 8,
            "asset_impact_score": 8,
            "is_high_impact": True,
            "ingested_at": "2026-07-17T12:00:00",
        },
    ]
    save_triage_events(events, store_path=temp_store)
    update_events_status(rejected_ids=["rej_ev"], store_path=temp_store)

    filtered = get_filtered_or_rejected_events(store_path=temp_store)
    ids = [f["event_id"] for f in filtered]
    assert "rej_ev" in ids
    assert "low_pending" in ids
    assert "high_pending" not in ids
    # Check sorting descending by ingested_at
    assert filtered[0]["event_id"] == "rej_ev"
    assert filtered[1]["event_id"] == "low_pending"


def test_store_raw_candidates_and_pruning(temp_store):
    items = [
        {"title": "Raw News 1", "link": "https://example.com/raw1"},
        {"title": "Raw News 2", "link": "https://example.com/raw2"},
    ]
    save_raw_candidates(items, store_path=temp_store)

    raw = get_raw_candidates(store_path=temp_store)
    assert len(raw) == 2
    assert raw[0]["title"] == "Raw News 1"
    assert "fetched_at" in raw[0]

    # test is_title_or_url_processed with include_raw=True vs include_raw=False
    assert is_title_or_url_processed("Raw News 1", "https://example.com/raw1", store_path=temp_store, include_raw=True) is True
    assert is_title_or_url_processed("Raw News 1", "https://example.com/raw1", store_path=temp_store, include_raw=False) is False

    # test pruning older than 48 hours and hard cap 500
    from datetime import datetime, timedelta
    old_iso = (datetime.now() - timedelta(hours=50)).isoformat()
    state = load_store(store_path=temp_store)
    state["raw_candidates"].append({"title": "Very Old News", "link": "https://example.com/old", "fetched_at": old_iso})
    # add 505 items to exceed hard cap
    for i in range(505):
        state["raw_candidates"].append({"title": f"Bulk {i}", "link": f"https://example.com/bulk/{i}", "fetched_at": datetime.now().isoformat()})
    save_store(state, store_path=temp_store)

    state_after = load_store(store_path=temp_store)
    assert len(state_after["raw_candidates"]) == 500
    assert not any(item["title"] == "Very Old News" for item in state_after["raw_candidates"])


def test_store_raw_candidates_remove_by_identity(temp_store):
    items = [
        {"title": "Keep Me", "link": "https://example.com/keep"},
        {"title": "Remove Me By URL", "link": "https://example.com/remove-url"},
        {"title": "Remove Me By Title", "link": "https://example.com/remove-title"},
    ]
    save_raw_candidates(items, store_path=temp_store)

    # remove by identity
    processed_urls = {"https://example.com/remove-url"}
    processed_titles = {"Remove Me By Title"}
    remove_processed_raw_candidates(processed_urls, processed_titles, store_path=temp_store)

    raw_after = get_raw_candidates(store_path=temp_store)
    assert len(raw_after) == 1
    assert raw_after[0]["title"] == "Keep Me"


