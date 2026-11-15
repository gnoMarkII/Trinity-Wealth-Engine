"""Unit tests for GeminiStreamEngine prompt guidelines, intent classification, and ticker resolution."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage
import pytest

from tools.hermes_bridge.gemini_stream_engine import (
    GeminiStreamEngine,
    SYSTEM_PROMPT_TEMPLATE,
    classify_hermes_intent,
)
from tools.market.ticker_resolver import find_company_alias_matches, resolve_equity_ticker


def _classify_is_run_order(user_message: str) -> bool:
    """Delegate directly to production classify_hermes_intent()."""
    return classify_hermes_intent(user_message)["is_run_order"]


def test_system_prompt_emphasizes_read_mode_for_thesis():
    """Verify that SYSTEM_PROMPT_TEMPLATE explicitly instructs reading existing thesis/reports first."""
    assert "get_equity_promises_and_reviews" in SYSTEM_PROMPT_TEMPLATE
    assert "get_equity_report_summary" in SYSTEM_PROMPT_TEMPLATE
    assert "trigger_deep_research_job" in SYSTEM_PROMPT_TEMPLATE
    # Verify Read & Review Mode instruction is present
    assert "Read & Review Mode" in SYSTEM_PROMPT_TEMPLATE
    # Verify Thesis check is mapped to read mode
    assert "ตรวจ Thesis" in SYSTEM_PROMPT_TEMPLATE
    assert "ห้ามสั่งรันงานวิจัยใหม่เด็ดขาด" in SYSTEM_PROMPT_TEMPLATE


def test_system_prompt_proactive_self_evolving_guidelines():
    """Verify that SYSTEM_PROMPT_TEMPLATE instructs proactive calculation, skill synthesis, and memory evolution."""
    assert "Proactive Sandbox Calculation" in SYSTEM_PROMPT_TEMPLATE
    assert "run_sandbox_calculation" in SYSTEM_PROMPT_TEMPLATE
    assert "Decimal" in SYSTEM_PROMPT_TEMPLATE
    assert "ห้ามคิดเลขในใจเด็ดขาด" in SYSTEM_PROMPT_TEMPLATE
    assert "Autonomous Skill Extraction" in SYSTEM_PROMPT_TEMPLATE
    assert "save_sandbox_skill" in SYSTEM_PROMPT_TEMPLATE
    assert "_quarantine" in SYSTEM_PROMPT_TEMPLATE
    assert "Autonomous Memory Hygiene" in SYSTEM_PROMPT_TEMPLATE
    assert "update_advisor_memory" in SYSTEM_PROMPT_TEMPLATE


@pytest.mark.parametrize(
    "query",
    [
        "ตรวจ Thesis ของ NFLX หน่อย",
        "ตรวจ Thesis ของ NFLX หน่อย ว่าเปลี่ยนไปรึยัง",
        "ตรวจสุขภาพหุ้น NFLX",
        "ขอดูบทวิเคราะห์ NFLX",
        "สรุปสถานะ NFLX ล่าสุดให้หน่อย",
        "NFLX มีการผิดสัญญาของผู้บริหารไหม",
        "สถานะของ NFLX ตอนนี้เป็นยังไงบ้าง",
        "ดูรายงานเดิมของ NFLX หน่อย",
        "อธิบายบทวิเคราะห์หุ้น INTEL",
        "ช่วยอธิบายบทวิเคราะห์ INTEL ให้หน่อย",
        "อธิบายรายงานหุ้น Google หน่อย",
        "ช่วยอ่านบทวิเคราะห์ Nvidia ที่วิเคราะห์ไว้แล้วให้ที",
    ],
)
def test_thesis_and_review_queries_are_not_run_orders(query: str):
    """User queries asking to explain/read/check thesis or existing reports must NOT trigger background research runs."""
    intent = classify_hermes_intent(query)
    assert intent["is_run_order"] is False
    assert intent["is_read_intent"] is True


@pytest.mark.parametrize(
    "raw_query,expected_ticker",
    [
        ("INTEL", "INTC"),
        ("Intel", "INTC"),
        ("Intel Corporation", "INTC"),
        ("Google", "GOOGL"),
        ("Alphabet", "GOOGL"),
        ("nvidia", "NVDA"),
        ("AAPL", "AAPL"),
        ("ptt", "PTT.BK"),
    ],
)
def test_resolve_equity_ticker_maps_company_aliases(raw_query: str, expected_ticker: str):
    assert resolve_equity_ticker(raw_query) == expected_ticker


def test_resolve_equity_ticker_uses_dynamic_known_records():
    records = [{"ticker": "INTC", "company_name": "Intel Corporation"}]
    assert resolve_equity_ticker("Intel Corp", known_records=records) == "INTC"
    assert find_company_alias_matches("อธิบายบทวิเคราะห์หุ้น INTEL") == ["INTC"]


def test_stream_reply_blocks_trigger_deep_research_on_explain_query(tmp_path: Path, monkeypatch):
    """Even if the LLM erroneously invokes trigger_deep_research_job on 'อธิบายบทวิเคราะห์หุ้น INTEL',
    GeminiStreamEngine must deny it and read existing report instead without dispatching any job.
    """
    monkeypatch.setattr("tools.hermes_bridge.gemini_stream_engine.get_model_name", lambda *a, **kw: "gemini-2.5-flash")
    monkeypatch.setattr("tools.hermes_bridge.gemini_stream_engine.detect_provider", lambda *a, **kw: "google")

    engine = GeminiStreamEngine(sandbox_root=tmp_path)

    dispatched_calls = []
    report_calls = []

    class _FakeDispatchTool:
        name = "trigger_deep_research_job"

        def invoke(self, args):
            dispatched_calls.append(args)
            return "Job ID: should-never-run"

    class _FakeReportTool:
        name = "get_equity_report_summary"

        def invoke(self, args):
            report_calls.append(args)
            return '{"ticker": "INTC", "company_name": "Intel Corporation", "composite_score": 71.5}'

    engine._tool_map["trigger_deep_research_job"] = _FakeDispatchTool()
    engine._tool_map["get_equity_report_summary"] = _FakeReportTool()

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "trigger_deep_research_job",
                    "args": {"ticker": "INTEL", "research_type": "equity_intel"},
                    "id": "call_bad_1",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_equity_report_summary",
                    "args": {"ticker": "INTEL"},
                    "id": "call_read_2",
                }
            ],
        ),
        AIMessage(content="นี่คือคำอธิบายบทวิเคราะห์หุ้น INTC (Intel Corporation) จากรายงานล่าสุดครับ"),
    ]
    call_idx = 0

    class _FakeBoundLlm:
        async def ainvoke(self, msgs):
            nonlocal call_idx
            resp = responses[call_idx]
            call_idx += 1
            return resp

    class _FakeLlm:
        def bind_tools(self, tools):
            return _FakeBoundLlm()

    monkeypatch.setattr("tools.hermes_bridge.gemini_stream_engine.get_llm", lambda **kw: _FakeLlm())

    async def _collect():
        chunks = []
        async for ch in engine.stream_reply("sess-1", "อธิบายบทวิเคราะห์หุ้น INTEL"):
            chunks.append(ch)
        return "".join(chunks)

    reply_text = asyncio.run(_collect())
    assert "INTC" in reply_text
    assert dispatched_calls == []
    assert report_calls == [{"ticker": "INTEL"}]
    assert engine.pop_dispatched_jobs("sess-1") == []


@pytest.mark.parametrize(
    "query",
    [
        "สั่งรันงานวิจัย NFLX ใหม่",
        "วิเคราะห์ NFLX ใหม่ตั้งแต่ต้น",
        "รัน macro ใหม่",
        "ช่วยสั่งรันงานวิจัยอัปเดตภาวะเศรษฐกิจมหภาค",
        "สั่งรัน equity_intel สำหรับ AAPL",
        "รันรายงานเศรษฐกิจใหม่",
    ],
)
def test_explicit_run_commands_are_classified_as_run_orders(query: str):
    """Explicit dispatch commands must be correctly classified as run orders."""
    assert _classify_is_run_order(query) is True
