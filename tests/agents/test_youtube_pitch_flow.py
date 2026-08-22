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
    res = generate_pitches_node(state, {})

    assert len(res["pitches"]) == 1
    assert res["pitches"][0]["pitch_id"] == "p-1"
    assert "สร้างไอเดียคลิป" in res["messages"][0].content


@patch("agents.youtube_pitch_flow.interrupt")
def test_gate_node(mock_interrupt):
    mock_interrupt.return_value = {"approved_pitch_ids": ["p-1"]}
    state: YouTubePitchState = {"pitches": [{"pitch_id": "p-1", "source_readiness": "ready"}]}
    cmd = gate_node(state, {})

    assert cmd.goto == "synthesize_notebooklm"
    assert cmd.update["approved_pitch_ids"] == ["p-1"]


def test_synthesize_notebooklm_node_zero_approval():
    # ทดสอบ Zero-file Protection กรณีไม่อนุมัติเลย
    state: YouTubePitchState = {"approved_pitch_ids": [], "pitches": [{"pitch_id": "p-1"}]}
    res = synthesize_notebooklm_node(state, {})

    assert "อนุมัติ 0 รายการ" in res["result_summary"]
    assert "ไม่สร้างไฟล์" in res["result_summary"]


@patch("tools.content.briefing_artifacts.save_briefing_artifact")
@patch("agents.youtube_pitch_flow.synthesize_notebooklm_source")
def test_synthesize_notebooklm_node_success(mock_synth, mock_save):
    class DummyResult:
        content = "# Briefing Book"
        quality_report = None
        override_audit = None
    mock_synth.return_value = DummyResult()
    mock_art = type('MockArt', (), {'path': 'c:/vault/30_Knowledge_Base/NotebookLM_Sources/2026-07-18_test.md'})()
    mock_save.return_value = mock_art

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
    res = synthesize_notebooklm_node(state, {})

    print("RES:", res["result_summary"])
    assert "บันทึก Briefing Book สำเร็จ" in res["result_summary"]
    mock_synth.assert_called_once()
    mock_save.assert_called_once()


@patch("agents.youtube_pitch_flow.synthesize_notebooklm_source", side_effect=ValueError("quality gate failed"))
def test_synthesize_notebooklm_node_raises_when_every_approved_pitch_fails(mock_synth):
    state: YouTubePitchState = {
        "approved_pitch_ids": ["p-1"],
        "pitches": [{
            "pitch_id": "p-1", "working_titles": ["one", "two", "three"],
            "target_audience": "Target", "core_hook": "Hook",
            "key_questions_to_answer": ["Q1", "Q2", "Q3"], "research_hypotheses": ["H1", "H2"],
            "source_event_ids": ["ev-1"], "source_links": ["https://example.test"],
            "source_titles": ["Source"], "recommended_format": "15m", "estimated_impact": "High",
        }],
        "news_candidates": [{"event_id": "ev-1"}],
    }
    with pytest.raises(RuntimeError, match="every approved pitch"):
        synthesize_notebooklm_node(state, {})
    mock_synth.assert_called_once()


@patch("agents.youtube_pitch_flow.synthesize_notebooklm_source")
def test_synthesize_notebooklm_node_marks_partial_failure(mock_synth, monkeypatch):
    mock_art = type('MockArt', (), {'path': 'C:/vault/saved_pitch.md'})()
    monkeypatch.setattr("tools.content.briefing_artifacts.save_briefing_artifact", lambda *args, **kwargs: mock_art)
    class DummyResult:
        content = "# briefing"
        quality_report = None
        override_audit = None
    mock_synth.side_effect = [DummyResult(), ValueError("quality gate failed")]
    base = {
        "working_titles": ["one", "two", "three"], "target_audience": "Target", "core_hook": "Hook",
        "key_questions_to_answer": ["Q1", "Q2", "Q3"], "research_hypotheses": ["H1", "H2"],
        "source_event_ids": ["ev-1"], "source_links": ["https://example.test"],
        "source_titles": ["Source"], "recommended_format": "15m", "estimated_impact": "High",
    }
    state: YouTubePitchState = {
        "approved_pitch_ids": ["p-1", "p-2"],
        "pitches": [{**base, "pitch_id": "p-1"}, {**base, "pitch_id": "p-2"}],
        "news_candidates": [{"event_id": "ev-1"}],
    }
    result = synthesize_notebooklm_node(state, {})

    assert result["synthesis_status"] == "done_with_errors"
    assert len(result["synthesis_failures"]) == 1


def test_build_youtube_pitch_graph():
    graph = build_youtube_pitch_graph()
    assert graph is not None


@patch("agents.youtube_pitch_flow.interrupt", return_value={"action": "refresh_sources"})
def test_gate_node_routes_a_source_refresh_request_back_to_generation(_mock_interrupt):
    cmd = gate_node({"pitches": [{"pitch_id": "p-1", "source_readiness": "blocked"}]}, {})

    assert cmd.goto == "generate_pitches"
    assert cmd.update["force_provenance_refresh"] is True


@patch("agents.youtube_pitch_flow.prepare_verified_candidate_pool")
@patch("agents.youtube_pitch_flow.assess_pitch_source_readiness")
@patch("agents.youtube_pitch_flow.generate_youtube_pitches")
def test_generate_pitches_regenerates_from_verified_pool_when_initial_drafts_are_unselectable(
    mock_generate, mock_readiness, mock_prepare
):
    initial = YouTubeContentPitchItem(
        pitch_id="initial", working_titles=["A", "B", "C"], target_audience="Target", core_hook="Hook",
        key_questions_to_answer=["Q1", "Q2", "Q3"], research_hypotheses=["H1", "H2"], source_event_ids=["raw-1"],
        source_links=["https://raw.example/1"], source_titles=["Raw"], recommended_format="15m", estimated_impact="High",
    )
    regenerated = initial.model_copy(update={"pitch_id": "verified", "source_event_ids": ["verified-1"]})
    mock_generate.side_effect = [
        YouTubeContentPitchBatch(pitches=[initial], date_range_summary="", total_source_events=1),
        YouTubeContentPitchBatch(pitches=[regenerated], date_range_summary="", total_source_events=1),
    ]
    mock_readiness.side_effect = [
        ("blocked", ["metadata_missing"], ["METADATA_MISSING"], [{"event_id": "raw-1"}]),
        ("ready", [], [], [{"event_id": "verified-1"}]),
    ]
    refreshed = [{"event_id": "verified-1", "links": ["https://verified.example/1"], "publisher": "Verified", "published_at": "2026-07-23", "verification_status": "verified"}]
    mock_prepare.return_value = (refreshed, refreshed, {"verified_candidates": 1, "refreshed_candidates": 1, "status_counts": {"verified": 1}})

    result = generate_pitches_node({"news_candidates": [{"event_id": "raw-1"}], "instruction": "test"}, {})

    assert mock_generate.call_count == 2
    assert mock_generate.call_args_list[1].kwargs["candidates"] == refreshed
    assert result["pitches"][0]["pitch_id"] == "verified"
    assert result["pitches"][0]["source_readiness"] == "ready"
    assert result["approval_block_reason"] is None


@patch("agents.youtube_pitch_flow.prepare_verified_candidate_pool")
@patch("agents.youtube_pitch_flow.assess_pitch_source_readiness")
@patch("agents.youtube_pitch_flow.generate_youtube_pitches")
def test_generate_pitches_routes_to_error_when_no_verified_source_can_be_selected(
    mock_generate, mock_readiness, mock_prepare
):
    from agents.youtube_pitch_flow import _route_after_pitch_generation, provenance_unavailable_node

    item = YouTubeContentPitchItem(
        pitch_id="blocked", working_titles=["A", "B", "C"], target_audience="Target", core_hook="Hook",
        key_questions_to_answer=["Q1", "Q2", "Q3"], research_hypotheses=["H1", "H2"], source_event_ids=["raw-1"],
        source_links=["https://raw.example/1"], source_titles=["Raw"], recommended_format="15m", estimated_impact="High",
    )
    mock_generate.return_value = YouTubeContentPitchBatch(pitches=[item], date_range_summary="", total_source_events=1)
    mock_readiness.return_value = ("blocked", ["macro error"], ["MACRO_COVERAGE_FAILURE"], [{"event_id": "raw-1"}])
    mock_prepare.return_value = (
        [{"event_id": "raw-1"}], [],
        {"verified_candidates": 0, "refreshed_candidates": 1, "independent_publishers": 0, "status_counts": {"metadata_missing": 1}},
    )

    result = generate_pitches_node({"news_candidates": [{"event_id": "raw-1"}], "instruction": "test"}, {})

    assert result["approval_block_reason"] == "no_selectable_pitch"
    assert _route_after_pitch_generation(result) == "provenance_unavailable"
    with pytest.raises(ValueError, match="No selectable YouTube Pitch"):
        provenance_unavailable_node(result)


def test_persist_parking_lot_node_success(tmp_path, monkeypatch):
    from agents.youtube_pitch_flow import persist_parking_lot_node
    from api import state_db

    vault_dir = tmp_path / "vault"
    db_path = str(tmp_path / "state.db")
    monkeypatch.setattr("agents.youtube_pitch_flow.VAULT_PATH", str(vault_dir))
    monkeypatch.setenv("WEBUI_STATE_DB_PATH", db_path)



    state: YouTubePitchState = {
        "pitches": [{
            "pitch_id": "p-1",
            "parking_lot_ideas": ["ไอเดียที่ 1", "ไอเดียที่ 2"],
        }],
        "synthesized_pitch_ids": ["p-1"],
        "synthesis_status": "done",
    }
    config = {"configurable": {"job_id": "job-success-123"}}

    res = persist_parking_lot_node(state, config)
    assert res["synthesis_status"] == "done"
    assert len(res["synthesis_warnings"]) == 0
    assert len(res["messages"]) == 1

    # Kanban cards in DB
    conn = state_db.get_connection(db_path)
    cards = state_db.list_kanban_cards(conn)
    assert len(cards) == 2


def test_persist_parking_lot_node_db_failure_fallback(tmp_path, monkeypatch):
    from agents.youtube_pitch_flow import persist_parking_lot_node
    import api.state_db

    vault_dir = tmp_path / "vault"
    monkeypatch.setattr("agents.youtube_pitch_flow.VAULT_PATH", str(vault_dir))

    # Mock DB failure
    def mock_db_error(*args, **kwargs):
        raise RuntimeError("SQLite database is locked or corrupted")

    monkeypatch.setattr("api.state_db.create_parking_lot_cards_atomic", mock_db_error)

    state: YouTubePitchState = {
        "pitches": [{
            "pitch_id": "p-fail",
            "parking_lot_ideas": ["ไอเดียที่ 1", "ไอเดียที่ 2"],
        }],
        "synthesized_pitch_ids": ["p-fail"],
        "synthesis_status": "done",
    }
    config = {"configurable": {"job_id": "job-fail-456"}}

    res = persist_parking_lot_node(state, config)
    assert res["synthesis_status"] == "done_with_warnings"
    assert len(res["synthesis_warnings"]) == 1
    assert "SQLite" in res["synthesis_warnings"][0]

    # Check Vault outbox file exists and is pending
    outbox_file = vault_dir / "NotebookLM_Sources" / "outbox" / "parking_job-fail-456_p-fail.json"
    assert outbox_file.exists()

