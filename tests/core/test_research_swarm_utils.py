"""Tests for research_swarm_flow pure helpers and deterministic dispatcher — no LLM calls"""
from unittest.mock import MagicMock

import pytest

from agents.research_swarm_flow import (
    _has_entity_frontmatter,
    _msg_role,
    classify_research_tasks,
    generate_manager_summary,
)
from langchain_core.messages import HumanMessage, AIMessage


class TestHasEntityFrontmatter:
    def test_valid_frontmatter(self):
        text = "---\nentity_type: article_note\ntitle: Test\n---\n\n# Body"
        assert _has_entity_frontmatter(text)

    def test_missing_entity_type(self):
        text = "---\ntitle: Test\n---\n\n# Body"
        assert not _has_entity_frontmatter(text)

    def test_no_frontmatter_at_all(self):
        assert not _has_entity_frontmatter("Just plain text")

    def test_markdown_horizontal_rule_not_mistaken(self):
        # "---" as HR in middle of text must not match
        text = "Some content\n\n---\n\nMore content"
        assert not _has_entity_frontmatter(text)

    def test_horizontal_rule_after_whitespace_not_mistaken(self):
        text = "\n\n---\n\nMore content without entity_type"
        assert not _has_entity_frontmatter(text)

    def test_entity_type_in_first_30_lines(self):
        lines = ["---"] + [f"field_{i}: val" for i in range(28)] + ["entity_type: book_note", "---"]
        assert _has_entity_frontmatter("\n".join(lines))

    def test_entity_type_beyond_30_lines_not_matched(self):
        lines = ["---"] + [f"field_{i}: val" for i in range(35)] + ["entity_type: article_note", "---"]
        assert not _has_entity_frontmatter("\n".join(lines))

    def test_empty_string(self):
        assert not _has_entity_frontmatter("")

    def test_only_dashes(self):
        assert not _has_entity_frontmatter("---")

    def test_leading_whitespace_ignored(self):
        text = "   \n---\nentity_type: holding\n---"
        assert _has_entity_frontmatter(text)



class TestMsgRole:
    def test_human_message_returns_human(self):
        assert _msg_role(HumanMessage(content="hi")) == "human"

    def test_ai_message_returns_assistant(self):
        assert _msg_role(AIMessage(content="response")) == "assistant"

    def test_other_object_returns_assistant(self):
        class FakeMsg:
            content = "data"
        assert _msg_role(FakeMsg()) == "assistant"


class TestClassifyResearchTasks:
    def test_empty_input_returns_response_text(self):
        decision = classify_research_tasks("")
        assert len(decision.tasks) == 0
        assert "ขอโทษครับ" in decision.response_text

    def test_macro_general_query(self):
        decision = classify_research_tasks("วิเคราะห์สภาวะเศรษฐกิจโลกและเงินเฟ้อสหรัฐ")
        assert len(decision.tasks) == 1
        assert decision.tasks[0].target == "macro_intel"
        assert decision.tasks[0].save_to_vault is True

    def test_equity_with_spaced_ticker(self):
        decision = classify_research_tasks("ช่วยวิเคราะห์หุ้น AAPL แบบละเอียด")
        assert len(decision.tasks) == 1
        assert decision.tasks[0].target == "equity_intel"
        assert decision.tasks[0].save_to_vault is True

    def test_equity_with_thai_glued_ticker(self):
        # ทดสอบ edge case ภาษาไทยติดกับตัวย่อภาษาอังกฤษ (Unicode word boundary \b กับดัก)
        decision = classify_research_tasks("วิเคราะห์NVDA")
        assert len(decision.tasks) == 1
        assert decision.tasks[0].target == "equity_intel"

        decision2 = classify_research_tasks("ดูAAPLหน่อย")
        assert len(decision2.tasks) == 1
        assert decision2.tasks[0].target == "equity_intel"

    def test_equity_no_save_flag(self):
        decision = classify_research_tasks("วิเคราะห์ TSLA ดูเฉยๆ ไม่ต้องเซฟ")
        assert len(decision.tasks) == 1
        assert decision.tasks[0].target == "equity_intel"
        assert decision.tasks[0].save_to_vault is False

    def test_archivist_query(self):
        decision = classify_research_tasks("ค้นหาข้อมูลใน vault เกี่ยวกับ semiconductor")
        assert len(decision.tasks) == 1
        assert decision.tasks[0].target == "archivist"

    def test_non_ticker_upper_tokens_do_not_trigger_equity(self):
        decision = classify_research_tasks("รายงาน GDP และ CPI ล่าสุดจาก FED และ BOT")
        assert len(decision.tasks) == 1
        assert decision.tasks[0].target == "macro_intel"

    def test_hyphenated_and_foreign_tickers(self):
        decision_hyphen = classify_research_tasks("วิเคราะห์หุ้น BRK-B")
        assert len(decision_hyphen.tasks) == 1
        assert decision_hyphen.tasks[0].target == "equity_intel"

        decision_thai = classify_research_tasks("วิเคราะห์หุ้น PTT.BK")
        assert len(decision_thai.tasks) == 1
        assert decision_thai.tasks[0].target == "equity_intel"

        decision_foreign = classify_research_tasks("วิเคราะห์ตัว 7203.T")
        assert len(decision_foreign.tasks) == 1
        assert decision_foreign.tasks[0].target == "equity_intel"

    def test_multi_ticker_query_generates_multiple_tasks(self):
        decision = classify_research_tasks("วิเคราะห์หุ้น AAPL และ NVDA")
        assert len(decision.tasks) == 2
        assert decision.tasks[0].target == "equity_intel"
        assert "AAPL" in decision.tasks[0].instruction
        assert decision.tasks[1].target == "equity_intel"
        assert "NVDA" in decision.tasks[1].instruction

    def test_multi_ticker_case_insensitive_with_no_save(self):
        decision = classify_research_tasks("วิเคราะห์หุ้น aapl และ nvda ดูเฉยๆ ไม่ต้องเซฟ")
        assert len(decision.tasks) == 2
        assert all(t.target == "equity_intel" for t in decision.tasks)
        assert all(t.save_to_vault is False for t in decision.tasks)

    def test_macro_indices_do_not_produce_false_positive_tickers(self):
        decision_sp = classify_research_tasks("วิเคราะห์ S&P 500 และ U.S. Treasury Yield")
        assert len(decision_sp.tasks) == 1
        assert decision_sp.tasks[0].target == "macro_intel"

        decision_vix = classify_research_tasks("ดูดัชนี VIX กับ DXY วันนี้หน่อย")
        assert len(decision_vix.tasks) == 1
        assert decision_vix.tasks[0].target == "macro_intel"

    def test_general_stock_market_phrase_routes_to_macro(self):
        decision = classify_research_tasks("วิเคราะห์ภาวะตลาดหุ้นสหรัฐและตลาดหุ้นไทยวันนี้")
        assert len(decision.tasks) == 1
        assert decision.tasks[0].target == "macro_intel"

    def test_explicit_equity_command_words_do_not_become_fake_tickers(self):
        decision_dive = classify_research_tasks("วิเคราะห์หุ้น AAPL deep dive")
        assert len(decision_dive.tasks) == 1
        assert decision_dive.tasks[0].target == "equity_intel"
        assert "AAPL" in decision_dive.tasks[0].instruction

        decision_stock = classify_research_tasks("Analyze stock AAPL")
        assert len(decision_stock.tasks) == 1
        assert decision_stock.tasks[0].target == "equity_intel"

        decision_vs = classify_research_tasks("วิเคราะห์หุ้น NVDA vs AMD")
        assert len(decision_vs.tasks) == 2
        assert decision_vs.tasks[0].instruction == "วิเคราะห์หุ้น NVDA"
        assert decision_vs.tasks[1].instruction == "วิเคราะห์หุ้น AMD"

    def test_single_letter_explicit_ticker(self):
        decision_v = classify_research_tasks("วิเคราะห์หุ้น V")
        assert len(decision_v.tasks) == 1
        assert decision_v.tasks[0].target == "equity_intel"

    def test_archivist_ingestion_and_urls(self):
        decision_yt = classify_research_tasks("สรุป YouTube นี้ให้หน่อย https://youtu.be/ABCDE")
        assert len(decision_yt.tasks) == 1
        assert decision_yt.tasks[0].target == "archivist"

        decision_digest = classify_research_tasks("สร้าง Weekly Digest ของ YouTube ให้หน่อย")
        assert len(decision_digest.tasks) == 1
        assert decision_digest.tasks[0].target == "archivist"

        decision_pdf = classify_research_tasks("สรุปบทความวิจัย https://example.com/macro_report.pdf")
        assert len(decision_pdf.tasks) == 1
        assert decision_pdf.tasks[0].target == "archivist"


def test_canonicalize_macro_report_direction_handles_empty_sidecar_dict():
    from agents.pipelines.macro_helpers import canonicalize_macro_report_direction
    mock_direction = MagicMock()
    mock_direction.observable_registry = {"test_obs": MagicMock()}

    validated, report, sector_analysis = canonicalize_macro_report_direction(
        direction=mock_direction,
        sidecar_result=(None, {}),
        sector_analysis_data=None,
    )
    # Must cleanly return mock_direction without raising ValidationError
    assert validated is mock_direction
    assert report == {}
    assert sector_analysis is None


def test_generate_manager_summary_uses_manager_model(monkeypatch):
    model = MagicMock()
    model.invoke.return_value = AIMessage(content="# สรุปจากผู้จัดการ")
    monkeypatch.setattr("agents.research_swarm_flow.get_llm", lambda **kwargs: model)

    summary = generate_manager_summary("วิเคราะห์พอร์ต", [("researcher", "พบความเสี่ยง"), ("equity_intel", "งบการเงินปกติ")])

    assert summary == "# สรุปจากผู้จัดการ"
    model.invoke.assert_called_once()
