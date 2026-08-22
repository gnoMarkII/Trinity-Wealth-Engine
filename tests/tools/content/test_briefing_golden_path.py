import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from schemas.briefing_book_schemas import (
    InvestigativeBriefingBookDraft,
    ResearchQualityReport,
)
from schemas.youtube_pitch_schemas import YouTubeContentPitchItem
from tests.fixtures.briefing_fixtures import (
    make_valid_evidence_bundle,
    make_valid_briefing_draft,
    make_valid_macro_observation,
)
from tools.content.briefing_renderer import (
    render_briefing_book,
    append_data_gap_notes,
    prepend_unverified_draft_banner,
)
from tools.content.briefing_quality import validate_briefing_book_quality
from tools.content.briefing_artifacts import save_briefing_artifact
from schemas.briefing_book_schemas import PublishableBriefingResult


@pytest.fixture
def mock_pitch():
    return YouTubeContentPitchItem(
        pitch_id="golden-path-123",
        working_titles=["Title 1", "Title 2", "Title 3"],
        target_audience="All",
        core_hook="Hook",
        key_questions_to_answer=["Q1", "Q2", "Q3"],
        research_hypotheses=["H1", "H2"],
        source_event_ids=["src-1", "src-2"],
        source_links=[],
        source_titles=[],
        recommended_format="10m",
        estimated_impact="High",
        investigation_mode="mixed"
    )


def test_golden_path_macro_mode_success(mock_pitch, tmp_path):
    # Setup
    mock_pitch.investigation_mode = "macro"
    bundle = make_valid_evidence_bundle(mode="macro")
    
    # Needs inflation, rates, and energy
    bundle.macro_snapshot.observations.append(make_valid_macro_observation("energy"))
    
    draft = make_valid_briefing_draft()
    
    # Act - Renderer
    rendered = render_briefing_book(draft, bundle)

    # Act - Quality Gate
    report = validate_briefing_book_quality(bundle, draft, rendered)

    # Expected
    assert report.score == 100
    assert report.publishable is True
    assert report.status == "pass"
    
    # Act - Persistence
    result = PublishableBriefingResult(
        content=rendered.content,
        draft=draft,
        quality_report=report,
        evidence_bundle=bundle,
    )
    
    saved_artifact = save_briefing_artifact(result, title="Golden Macro", vault_root=tmp_path, date_str="2026-07-24")
    saved_path = saved_artifact.path
        
    assert Path(saved_path).exists()
    # Test should ensure sidecar and index are also created correctly (to be implemented)


def test_golden_path_macro_mode_missing_rates_fails(mock_pitch):
    mock_pitch.investigation_mode = "macro"
    bundle = make_valid_evidence_bundle(mode="macro")
    
    # Remove rates
    bundle.macro_snapshot.observations = [obs for obs in bundle.macro_snapshot.observations if obs.category != "rates"]
    
    draft = make_valid_briefing_draft()
    rendered = render_briefing_book(draft, bundle)
    report = validate_briefing_book_quality(bundle, draft, rendered)

    # Expected: fails with macro code
    assert report.publishable is False
    assert report.score < 100
    assert any("rates" in issue.description.lower() or issue.code == "MACRO_MISSING_RATES" for issue in report.issues)


def test_golden_path_stock_mode_success(mock_pitch, tmp_path):
    mock_pitch.investigation_mode = "stock"
    bundle = make_valid_evidence_bundle(mode="stock")
    
    draft = make_valid_briefing_draft()
    rendered = render_briefing_book(draft, bundle)
    report = validate_briefing_book_quality(bundle, draft, rendered)
    
    assert report.score == 100
    assert report.publishable is True


def test_golden_path_stock_mode_missing_eligible_asset(mock_pitch):
    import pytest
    from unittest.mock import patch
    from tools.content.youtube_pitcher import synthesize_notebooklm_source
    mock_pitch.investigation_mode = "stock"
    mock_pitch.target_symbols = []
    mock_pitch.source_event_ids = ["EV1"]
    events = [{"event_id": "EV1", "symbols": [], "independence_key": "x", "source_url": "x"}]
    with patch("tools.content.provenance_enrichment.assess_pitch_source_readiness", return_value=("ready", [], [], events)):
        with pytest.raises(ValueError, match="No eligible asset found for Stock Mode"):
            synthesize_notebooklm_source(mock_pitch, source_events=events)


def test_golden_path_mixed_mode_success(mock_pitch, tmp_path):
    mock_pitch.investigation_mode = "mixed"
    bundle = make_valid_evidence_bundle(mode="mixed")
    
    draft = make_valid_briefing_draft()
    rendered = render_briefing_book(draft, bundle)
    report = validate_briefing_book_quality(bundle, draft, rendered)
    
    assert report.score == 100
    assert report.publishable is True


def test_golden_path_mixed_mode_financial_unavailable(mock_pitch):
    mock_pitch.investigation_mode = "mixed"
    bundle = make_valid_evidence_bundle(mode="mixed")
    
    # Make financial unavailable
    bundle.financial_snapshots[0].status = "unavailable"
    
    draft = make_valid_briefing_draft()
    rendered = render_briefing_book(draft, bundle)
    report = validate_briefing_book_quality(bundle, draft, rendered)
    
    assert report.publishable is False
    # Data gaps are now bypassable for Unverified Draft fallback
    has_non_bypassable = any(issue.severity == "blocker" and not issue.bypassable for issue in report.issues)
    assert has_non_bypassable is False


def test_golden_path_draft_metadata_missing_allowlist(mock_pitch):
    mock_pitch.investigation_mode = "mixed"
    bundle = make_valid_evidence_bundle(mode="mixed")

    # Metadata missing and make it unverified so it doesn't trigger INCOMPLETE_CANONICAL blocker
    bundle.sources[0].published_at = None
    bundle.sources[0].verification_status = "unverified"
    # We must also change the classification of E01 (which uses src-1) to something other than verified_fact
    bundle.evidence_items[0].classification = "speculation"

    draft = make_valid_briefing_draft()
    rendered = render_briefing_book(draft, bundle)
    report = validate_briefing_book_quality(bundle, draft, rendered)

    # It should not be publishable, but should have a bypassable issue
    assert report.publishable is False
    has_bypassable = any(issue.bypassable for issue in report.issues)
    assert has_bypassable is True


def test_golden_path_draft_visual_defect_rejects(mock_pitch):
    mock_pitch.investigation_mode = "mixed"
    bundle = make_valid_evidence_bundle(mode="mixed")
    
    draft = make_valid_briefing_draft()
    # Induce visual defect (e.g. empty visual directives)
    draft.visual_directives = []

    rendered = render_briefing_book(draft, bundle)
    report = validate_briefing_book_quality(bundle, draft, rendered)
    
    assert report.publishable is False
    has_non_bypassable = any(issue.severity == "blocker" and not issue.bypassable for issue in report.issues)
    assert has_non_bypassable is True


def test_append_data_gap_notes_no_gaps_leaves_content_unchanged():
    content = "# Title\n\nBody"
    assert append_data_gap_notes(content, numeric_warnings=[], macro_unavailable_reasons=[]) == content


def test_append_data_gap_notes_numeric_warning_is_visible_in_markdown():
    content = "# Title\n\nBody"
    result = append_data_gap_notes(
        content,
        numeric_warnings=["Narrative cites E14 but its numeric value (175.88) is not found nearby"],
        macro_unavailable_reasons=[],
    )
    assert content in result
    assert "## Data Gaps" in result
    assert "E14" in result
    assert "175.88" in result


def test_append_data_gap_notes_macro_unavailable_is_visible_in_markdown():
    content = "# Title\n\nBody"
    result = append_data_gap_notes(
        content,
        numeric_warnings=[],
        macro_unavailable_reasons=["FRED API timeout"],
    )
    assert "## Data Gaps" in result
    assert "FRED API timeout" in result


def test_prepend_unverified_draft_banner_appears_right_after_title():
    content = "# Title\n\nBody"
    result = prepend_unverified_draft_banner(
        content,
        reason="User accepted incomplete provenance",
        issues=["ไม่พบสำนักข่าวจากหน้าแหล่งข้อมูล"],
    )
    assert result.startswith("# Title")
    assert "Unverified Draft" in result
    assert "User accepted incomplete provenance" in result
    assert "ไม่พบสำนักข่าวจากหน้าแหล่งข้อมูล" in result
    assert result.index("Unverified Draft") < result.index("Body")


def test_render_briefing_book_with_audio_directive():
    draft = make_valid_briefing_draft()
    bundle = make_valid_evidence_bundle(mode="macro")
    bundle.macro_snapshot.observations.append(make_valid_macro_observation("energy"))
    draft.audio_overview_directive = "เน้นความกระชับ ดำเนินเรื่องแบบเดินหน้าทางเดียว ไม่พูดวนซ้ำ"

    rendered = render_briefing_book(draft, bundle)
    assert "## NotebookLM Audio Directive" in rendered.content
    assert "เน้นความกระชับ ดำเนินเรื่องแบบเดินหน้าทางเดียว ไม่พูดวนซ้ำ" in rendered.content
    assert "NotebookLM Audio Directive" in rendered.section_names


def test_render_briefing_book_without_audio_directive():
    draft = make_valid_briefing_draft()
    bundle = make_valid_evidence_bundle(mode="macro")
    bundle.macro_snapshot.observations.append(make_valid_macro_observation("energy"))
    draft.audio_overview_directive = None

    rendered = render_briefing_book(draft, bundle)
    assert "## NotebookLM Audio Directive" not in rendered.content
    assert "NotebookLM Audio Directive" not in rendered.section_names


def test_script_loopback_warning_advisory():
    draft = make_valid_briefing_draft()
    bundle = make_valid_evidence_bundle(mode="macro")
    bundle.macro_snapshot.observations.append(make_valid_macro_observation("energy"))
    
    # Inject near-identical sentence into Act I and Act II
    draft.act1_script = draft.act1_script + "\nอัตราเงินเฟ้อที่พุ่งสูงขึ้นกำลังกดดันการเติบโตทางเศรษฐกิจอย่างรุนแรง"
    draft.act2_script = draft.act2_script + "\nอัตราเงินเฟ้อที่พุ่งสูงขึ้นกำลังกดดันการเติบโตทางเศรษฐกิจอย่างรุนแรง"

    rendered = render_briefing_book(draft, bundle)
    report = validate_briefing_book_quality(bundle, draft, rendered)

    assert report.publishable is True
    assert len(report.hard_blockers) == 0
    assert any(issue.code == "SCRIPT_LOOPBACK_WARNING" for issue in report.issues)

