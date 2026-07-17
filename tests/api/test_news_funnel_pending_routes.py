import pytest
from fastapi.testclient import TestClient
from api.main import app
from tools.macro import news_funnel_store


def test_get_and_delete_news_funnel_pending(authed_client, monkeypatch):
    mock_events = [
        {
            "event_id": "ev-101",
            "canonical_title": "Test High Impact News 1",
            "comprehensive_summary": "Summary 1",
            "macro_impact_score": 8,
            "asset_impact_score": 7,
            "extracted_tickers": ["THB"],
            "extracted_themes": ["macro"],
            "links": ["http://example.com/1"],
            "status": "pending_synthesis",
            "is_high_impact": True,
        },
        {
            "event_id": "ev-102",
            "canonical_title": "Test High Impact News 2",
            "comprehensive_summary": "Summary 2",
            "macro_impact_score": 9,
            "asset_impact_score": 8,
            "extracted_tickers": ["GOLD"],
            "extracted_themes": ["inflation"],
            "links": ["http://example.com/2"],
            "status": "pending_synthesis",
            "is_high_impact": True,
        },
    ]

    monkeypatch.setattr(news_funnel_store, "get_pending_high_impact_events", lambda store_path=None: mock_events)

    rejected_list = []
    def mock_update_status(rejected_ids=None, synthesized_ids=None, store_path=None):
        if rejected_ids:
            rejected_list.extend(rejected_ids)
            for ev in mock_events[:]:
                if ev["event_id"] in rejected_ids:
                    mock_events.remove(ev)

    monkeypatch.setattr(news_funnel_store, "update_events_status", mock_update_status)
    monkeypatch.setattr("api.news_funnel_cards.upsert_news_funnel_card", lambda period, pending: None)

    # 1. GET pending items
    response = authed_client.get("/api/macro/news_funnel/pending")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["event_id"] == "ev-101"
    assert data[1]["event_id"] == "ev-102"

    # 2. DELETE (Reject) ev-101
    del_resp = authed_client.delete("/api/macro/news_funnel/pending/ev-101")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["ok"] is True
    assert del_data["remaining_count"] == 1
    assert rejected_list == ["ev-101"]

    # 3. GET again should show remaining
    response2 = authed_client.get("/api/macro/news_funnel/pending")
    assert response2.status_code == 200
    assert len(response2.json()) == 1
    assert response2.json()[0]["event_id"] == "ev-102"


def test_get_news_funnel_filtered(authed_client, monkeypatch):
    mock_filtered = [
        {
            "event_id": "ev-rej-1",
            "canonical_title": "Rejected Low Impact",
            "comprehensive_summary": "Summary rej",
            "macro_impact_score": 5,
            "asset_impact_score": 4,
            "status": "rejected",
            "triage_reasoning": "Score below threshold",
            "ingested_at": "2026-07-16T10:00:00",
        },
        {
            "event_id": "ev-err-1",
            "canonical_title": "Skipped Error Item",
            "comprehensive_summary": "Summary err",
            "macro_impact_score": 8,
            "asset_impact_score": 7,
            "status": "skipped_error",
            "error_msg": "Paywalled",
            "ingested_at": "2026-07-16T11:00:00",
        },
    ]
    monkeypatch.setattr(news_funnel_store, "get_filtered_or_rejected_events", lambda store_path=None: mock_filtered)

    response = authed_client.get("/api/macro/news_funnel/filtered")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["event_id"] == "ev-rej-1"
    assert data[0]["status"] == "rejected"
    assert data[0]["triage_reasoning"] == "Score below threshold"
    assert data[1]["event_id"] == "ev-err-1"
    assert data[1]["status"] == "skipped_error"
    assert data[1]["error_msg"] == "Paywalled"
