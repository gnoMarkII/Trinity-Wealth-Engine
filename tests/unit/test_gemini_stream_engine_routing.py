"""Unit tests for GeminiStreamEngine prompt guidelines and intent classification."""
from __future__ import annotations

import pytest
from tools.hermes_bridge.gemini_stream_engine import SYSTEM_PROMPT_TEMPLATE


def _classify_is_run_order(user_message: str) -> bool:
    """Mirrors the intent classification rule in GeminiStreamEngine."""
    user_msg_lower = user_message.lower()
    is_read_intent = any(
        w in user_msg_lower
        for w in [
            "ตรวจ",
            "ขอดู",
            "ดู",
            "สรุป",
            "สถานะ",
            "thesis",
            "สัญญา",
            "เดิม",
            "ล่าสุด",
            "เป็นยังไง",
            "เปลี่ยนไปรึยัง",
            "เปลี่ยนไหม",
        ]
    )
    is_explicit_run = (
        any(
            w in user_msg_lower
            for w in [
                "สั่งรัน",
                "รันงานวิจัย",
                "รันรายงาน",
                "ดึงข้อมูลใหม่",
                "สดใหม่",
                "ข้อมูลสด",
                "ตั้งแต่ต้น",
                "macro_intel",
                "equity_intel",
            ]
        )
        or ("รัน" in user_msg_lower and "ใหม่" in user_msg_lower)
        or ("วิเคราะห์" in user_msg_lower and "ใหม่" in user_msg_lower)
    )
    return is_explicit_run and not is_read_intent


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
    ],
)
def test_thesis_and_review_queries_are_not_run_orders(query: str):
    """User queries asking to check thesis or read existing reports must NOT trigger background research runs."""
    assert _classify_is_run_order(query) is False


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
