import os
from unittest.mock import MagicMock, patch
import json
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.manager_agent import build_graph, AgentState
from schemas.micro_quant_schemas import MicroQuantOutput, QuantSignals, AtomicMarketSnapshot


def test_equity_synthesizer_llm_failure_saves_partial_revision(monkeypatch):
    """Simulate LLM connection refusal during synthesis and verify quant output is preserved."""
    monkeypatch.setenv("GOOGLE_API_KEY", "mock-google-api-key-for-test")
    monkeypatch.setenv("GEMINI_API_KEY", "mock-google-api-key-for-test")
    monkeypatch.setenv("OPENAI_API_KEY", "mock-openai-api-key-for-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-openrouter-api-key-for-test")
    snapshot = AtomicMarketSnapshot(
        analysis_price=172.78,
        analysis_price_as_of="2026-08-27",
        price_source="ohlcv_close",
        latest_ohlcv_close=172.78,
        latest_ohlcv_date="2026-08-27",
        shares_outstanding=733713653,
        market_cap=126771044965.34,
        price_sync_status="synced",
        retrieved_at="2026-08-27T10:00:00Z",
    )

    quant_signals = QuantSignals(
        ticker="FTNT",
        market="US",
        currency="USD",
        company_name="Fortinet, Inc.",
        as_of_date="2026-08-27",
        evaluated_at="2026-08-27T10:00:00Z",
        atomic_market_snapshot=snapshot,
    )

    state: AgentState = {
        "messages": [HumanMessage(content="วิเคราะห์หุ้น FTNT", name="manager")],
        "route_meta": {"source": "manager", "target": "equity_synthesizer"},
        "task_queue": [],
        "replan_count": 0,
        "turn_id": "test_turn_123",
        "equity_quant_score": quant_signals.model_dump(mode="json"),
        "equity_quant_raw": quant_signals.model_dump_json(),
        "equity_narrative_raw": None,
        "equity_narrative_context": None,
        "equity_save_to_vault": False,
        "equity_output": None,
        "equity_news_raw": None,
        "quant_raw": None,
        "quant_score": None,
        "narrative_raw": None,
        "narrative_context": None,
    }

    graph = build_graph()
    nodes = graph.nodes
    equity_synthesizer_node = nodes["equity_synthesizer"]

    # When get_llm or invoke fails with ConnectionRefusedError
    with patch("agents.manager_agent.get_llm", side_effect=ConnectionRefusedError("LLM server connection refused (simulated)")):
        cmd = equity_synthesizer_node.invoke(state)
        assert cmd.goto == "post_equity_intel"
        update = cmd.update
        assert "equity_output" in update
        assert update["equity_output"]["ticker"] == "FTNT"
        assert "LLM_UNAVAILABLE" in update["equity_output"]["narrative_analysis"]
        assert "ตัวเลขรีเฟรชแล้ว" in update["equity_output"]["base_case_summary"]
