"""Unit tests สำหรับ agents/news_funnel_flow.py"""
from langchain_core.messages import AIMessage
from agents.news_funnel_flow import (
    build_news_funnel_graph,
    gate_node,
    load_pending_node,
    synthesize_node,
)
from tools.macro.news_funnel_store import save_triage_events


def test_load_pending_node(tmp_path):
    store_file = str(tmp_path / "state.json")
    events = [
        {
            "event_id": "ev-1",
            "canonical_title": "Inflation rises",
            "macro_impact_score": 8,
            "asset_impact_score": 6,
            "is_high_impact": True,
        }
    ]
    save_triage_events(events, store_path=store_file)

    state = {"store_path": store_file}
    out = load_pending_node(state)
    assert len(out["candidates"]) == 1
    assert out["candidates"][0]["event_id"] == "ev-1"
    assert isinstance(out["messages"][0], AIMessage)


def test_synthesize_node(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_NEWS_FUNNEL_LLM", "true")
    store_file = str(tmp_path / "state.json")
    vault_dir = str(tmp_path / "vault")
    events = [
        {
            "event_id": "ev-1",
            "canonical_title": "Inflation rises",
            "comprehensive_summary": "Summary text",
            "macro_impact_score": 8,
            "asset_impact_score": 6,
            "is_high_impact": True,
            "extracted_tickers": ["NVDA"],
            "extracted_themes": ["inflation"],
        }
    ]
    save_triage_events(events, store_path=store_file)

    state = {
        "period": "morning",
        "approved_event_ids": ["ev-1"],
        "store_path": store_file,
        "vault_root": vault_dir,
    }
    out = synthesize_node(state)
    assert "✓ สังเคราะห์ธีมเศรษฐกิจสำเร็จ" in out["result_summary"]


def test_build_news_funnel_graph():
    graph = build_news_funnel_graph()
    assert graph is not None


def test_gate_node_fail_closed(monkeypatch):
    monkeypatch.setattr("agents.news_funnel_flow.interrupt", lambda payload: {"approved_event_ids": None})
    cmd = gate_node({"candidates": []})
    assert cmd.update["approved_event_ids"] == []

    monkeypatch.setattr("agents.news_funnel_flow.interrupt", lambda payload: {})
    cmd2 = gate_node({"candidates": []})
    assert cmd2.update["approved_event_ids"] == []
