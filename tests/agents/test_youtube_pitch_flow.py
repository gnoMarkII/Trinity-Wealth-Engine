"""Unit tests สำหรับ agents/youtube_pitch_flow.py"""
from unittest.mock import MagicMock, patch
import pytest

from agents.youtube_pitch_flow import (
    YouTubePitchState,
    build_youtube_pitch_graph,
    fetch_topics_node,
    gate_node,
    generate_pitches_node,
    synthesize_notebooklm_node,
)
from schemas.youtube_pitch_schemas import YouTubeContentPitchBatch, YouTubeContentPitchItem


@patch("agents.youtube_pitch_flow.fetch_news_for_pitching")
def test_fetch_topics_node(mock_fetch):
    mock_fetch.return_value = (
        [{"event_id": "ev-1", "canonical_title": "ข่าวตลาดหุ้น"}],
        "macro_baselines_str",
        True,  # is_fallback triggered
    )
    state: YouTubePitchState = {"instruction": "[from_date=2026-07-01, to_date=2026-07-18] หาไอเดีย"}
    res = fetch_topics_node(state)

    assert res["from_date"] == "2026-07-01"
    assert res["to_date"] == "2026-07-18"
    assert len(res["news_candidates"]) == 1
    assert res["macro_baselines"] == "macro_baselines_str"
    # ต้องมีข้อความเตือน Layer 2 fallback และข้อความสรุปจำนวนข่าว
    assert len(res["messages"]) == 2
    assert "คำเตือน" in res["messages"][0].content


@patch("agents.youtube_pitch_flow.generate_youtube_pitches")
def test_generate_pitches_node(mock_gen):
    item = YouTubeContentPitchItem(
        pitch_id="p-1",
        working_titles=["1", "2", "3"],
        target_audience="Target",
        core_hook="Hook",
        key_questions_to_answer=["Q1", "Q2", "Q3"],
        research_hypotheses=["H1", "H2"],
        source_event_ids=["ev-1"],
        source_links=["http://test.com"],
        source_titles=["ข่าว"],
        recommended_format="15m",
        estimated_impact="Impact",
    )
    mock_gen.return_value = YouTubeContentPitchBatch(
        pitches=[item],
        date_range_summary="summary",
        total_source_events=1,
    )

    state: YouTubePitchState = {"news_candidates": [{"event_id": "ev-1"}], "instruction": "test"}
    res = generate_pitches_node(state)

    assert len(res["pitches"]) == 1
    assert res["pitches"][0]["pitch_id"] == "p-1"
    assert "สร้างไอเดียคลิป" in res["messages"][0].content


@patch("agents.youtube_pitch_flow.interrupt")
def test_gate_node(mock_interrupt):
    mock_interrupt.return_value = {"approved_pitch_ids": ["p-1"]}
    state: YouTubePitchState = {"pitches": [{"pitch_id": "p-1"}]}
    cmd = gate_node(state)

    assert cmd.goto == "synthesize_notebooklm"
    assert cmd.update["approved_pitch_ids"] == ["p-1"]


def test_synthesize_notebooklm_node_zero_approval():
    # ทดสอบ Zero-file Protection กรณีไม่อนุมัติเลย
    state: YouTubePitchState = {"approved_pitch_ids": [], "pitches": [{"pitch_id": "p-1"}]}
    res = synthesize_notebooklm_node(state)

    assert "อนุมัติ 0 รายการ" in res["result_summary"]
    assert "ไม่สร้างไฟล์" in res["result_summary"]


@patch("agents.youtube_pitch_flow.save_notebooklm_source")
@patch("agents.youtube_pitch_flow.synthesize_notebooklm_source")
def test_synthesize_notebooklm_node_success(mock_synth, mock_save):
    mock_synth.return_value = "# Briefing Book"
    mock_save.return_value = "c:/vault/30_Knowledge_Base/NotebookLM_Sources/2026-07-18_test.md"

    item_dict = {
        "pitch_id": "p-1",
        "working_titles": ["หัวข้อหลัก", "หัวข้อรอง", "หัวข้อสาม"],
        "target_audience": "Target",
        "core_hook": "Hook",
        "key_questions_to_answer": ["Q1", "Q2", "Q3"],
        "research_hypotheses": ["H1", "H2"],
        "source_event_ids": ["ev-1"],
        "source_links": ["http://test.com"],
        "source_titles": ["ข่าว"],
        "recommended_format": "15m",
        "estimated_impact": "Impact",
    }
    state: YouTubePitchState = {
        "approved_pitch_ids": ["p-1"],
        "pitches": [item_dict],
        "news_candidates": [{"event_id": "ev-1"}],
        "macro_baselines": "macro",
    }
    res = synthesize_notebooklm_node(state)

    assert "บันทึก Briefing Book สำเร็จ" in res["result_summary"]
    mock_synth.assert_called_once()
    mock_save.assert_called_once()


def test_build_youtube_pitch_graph():
    graph = build_youtube_pitch_graph()
    assert graph is not None
