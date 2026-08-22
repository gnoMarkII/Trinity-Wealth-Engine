"""Unit tests สำหรับ tools/content/youtube_pitcher.py"""
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from schemas.youtube_pitch_schemas import YouTubeContentPitchBatch, YouTubeContentPitchItem
from tools.content.youtube_pitcher import (
    fetch_news_for_pitching,
    generate_youtube_pitches,
    parse_date_filters_from_instruction,
    synthesize_notebooklm_source,
)
from tools.content.briefing_artifacts import save_briefing_artifact


def test_parse_date_filters_from_instruction():
    # 1. แท็กครบ from_date และ to_date
    res1 = parse_date_filters_from_instruction("[from_date=2026-07-01, to_date=2026-07-18, lookback_days=17]")
    assert res1["from_date"] == "2026-07-01"
    assert res1["to_date"] == "2026-07-18"
    assert res1["lookback_days"] == 17

    # 2. คำสั่งข้อความภาษาไทย "ย้อนหลัง 14 วัน"
    res2 = parse_date_filters_from_instruction("หาไอเดียทำคลิป YouTube เจาะลึกข่าว ย้อนหลัง 14 วัน กรุณาจัดเต็ม")
    assert res2["lookback_days"] == 14

    # 3. คำสั่งว่าง
    res3 = parse_date_filters_from_instruction("")
    assert res3["lookback_days"] == 7
    assert res3["from_date"] is None


@patch("tools.content.youtube_pitcher.load_store")
@patch("tools.content.youtube_pitcher.get_macro_baselines")
def test_fetch_news_for_pitching_layer1_and_fallback(mock_baselines, mock_load, tmp_path, monkeypatch):
    monkeypatch.setattr("tools.content.youtube_pitcher.VAULT_PATH", tmp_path)
    mock_baselines.invoke.return_value = '{"macro": "ok"}'

    now_iso = datetime.now().isoformat()
    mock_load.return_value = {
        "pending_events": [
            {
                "event_id": "ev-1",
                "canonical_title": "ข่าวธนาคารกลางปรับดอกเบี้ย",
                "comprehensive_summary": "สรุปข่าวชั้น 1",
                "links": ["http://test.com/1"],
                "ingested_at": now_iso,
            }
        ]
    }

    # สร้างไฟล์ใน Layer 2 (30_Knowledge_Base/News/*.md)
    layer2_dir = tmp_path / "30_Knowledge_Base" / "News"
    layer2_dir.mkdir(parents=True)
    md_file = layer2_dir / "2026-07-01_ข่าวเก่า.md"
    md_file.write_text(
        "---\n"
        "title: ข่าวเก่าใน Layer 2 ที่ Store ตัดทิ้งไปแล้ว\n"
        f"date: {(datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')}\n"
        "---\n"
        "# สรุปเนื้อหาข่าวเก่าในอดีต\n"
        "รายละเอียดเนื้อหาเต็ม... https://test.com/layer2\n",
        encoding="utf-8",
    )

    # ทดสอบดึงข่าวโดยกำหนด lookback_days=14 (ย้อนหลังเกิน 7 วัน -> Trigger Layer 2 Fallback)
    candidates, macro_str, is_fallback = fetch_news_for_pitching(lookback_days=14)
    assert is_fallback is True
    assert len(candidates) >= 1
    assert any(c.get("event_id") == "ev-1" for c in candidates)
    assert any("ข่าวเก่า" in c.get("canonical_title", "") for c in candidates)
    assert macro_str == '{"macro": "ok"}'


@patch("tools.content.youtube_pitcher.invoke_structured_llm")
def test_generate_youtube_pitches_success(mock_invoke):
    mock_item = YouTubeContentPitchItem(
        pitch_id="uuid-test",
        working_titles=["คำถามเจาะลึก?", "วิเคราะห์สมมติฐาน", "เตือนภัยตลาด"],
        target_audience="นักลงทุนไทย",
        core_thesis="ใจความสำคัญหลักหนึ่งประโยคที่ยาวเกิน 15 ตัวอักษรแน่นอน",
        primary_anchor_event_id="ev-1",
        primary_anchor_title="ข่าวธนาคารกลาง",
        key_questions_to_answer=["ข้อ 1", "ข้อ 2", "ข้อ 3"],
        research_hypotheses=["สมมติฐาน 1", "สมมติฐาน 2"],
        source_event_ids=["ev-1"],
        source_links=["http://test.com/1"],
        source_titles=["ข่าวธนาคารกลาง"],
        recommended_format="Deep Dive 15m",
        estimated_impact="Impact",
        counter_intuitive_lead="เบาะแสสำคัญค้านสายตา: ตลาดหุ้นเติบโตแต่กระแสเงินสดติดลบ",
        analogy_generator="คำเปรียบเปรย: เหมือนรถที่วิ่งด้วยความเร็วสูงแต่เชื้อเพลิงกำลังจะหมด",
        audience_takeaway="เก็บเงินสดสำรอง 6 เดือนก่อนตัดสินใจลงทุนเพิ่ม",
        thumbnail_concept="ภาพกราฟตลาดหุ้นพุ่งขึ้นแต่กระเป๋าเงินโล่ง",
    )

    mock_invoke.return_value = YouTubeContentPitchBatch(
        pitches=[mock_item],
        date_range_summary="7 วัน",
        total_source_events=1,
    )

    batch = generate_youtube_pitches(
        candidates=[{"event_id": "ev-1", "canonical_title": "ข่าวธนาคารกลาง", "links": ["http://test.com/1"]}],
        max_pitches=3,
    )
    assert len(batch.pitches) == 1
    assert batch.pitches[0].working_titles[0] == "คำถามเจาะลึก?"


@patch("tools.content.youtube_pitcher.invoke_structured_llm")
def test_generate_youtube_pitches_retry_and_lenient_fallback(mock_invoke):
    # จำลอง LLM ยกเว้น error (เช่น validation fail ซ้ำ) เพื่อยืนยัน heuristic fallback
    mock_invoke.side_effect = Exception("Mock LLM validation error")

    batch = generate_youtube_pitches(
        candidates=[{"event_id": "ev-1", "canonical_title": "ข่าวตลาดหุ้นผันผวน", "links": ["http://test.com/1"]}],
        max_pitches=3,
    )
    assert len(batch.pitches) == 1
    assert "เจาะลึก:" in batch.pitches[0].working_titles[0]
    assert len(batch.pitches[0].working_titles) == 3


@patch("tools.content.briefing_quality.validate_briefing_book_quality")
@patch("tools.content.provenance_enrichment.assess_pitch_source_readiness")
@patch("tools.content.youtube_pitcher.invoke_structured_llm")
def test_synthesize_notebooklm_source(mock_invoke, mock_assess, mock_validate):
    mock_assess.return_value = ("ready", [], [], [{"event_id": "ev-1", "title": "ข่าว 1"}])

    from schemas.briefing_book_schemas import ResearchQualityReport, InvestigativeBriefingBookDraft
    mock_validate.return_value = ResearchQualityReport(score=100, status="pass", publishable=True, issues=[], advisories=[])

    mock_draft = InvestigativeBriefingBookDraft(
        title="Mock Title",
        executive_summary="📑 สรุปผู้บริหารและแหล่งอ้างอิง",
        causality_scenarios=[
            {"scenario_id": "S1", "name": "Base", "description": "Base", "probability_pct": 50, "trigger_conditions": [], "falsification_triggers": [], "evidence_ids": [], "threshold_basis": ""}
        ],
        asset_impacts=[
            {"symbol_or_name": "AAPL", "impact_type": "direct_upside", "reasoning": "Good", "risk_factors": [], "invalidation_conditions": [], "evidence_ids": []}
        ],
        bull_case="Bull",
        bear_case="Bear",
        falsification_triggers=["F1"],
        act1_script="A1",
        act2_script="A2",
        act3_script="A3",
        visual_directives=[],
        notebooklm_prompts=[],
    )
    mock_invoke.return_value = mock_draft

    pitch = YouTubeContentPitchItem(
        pitch_id="uuid-test",
        working_titles=["1", "2", "3"],
        target_audience="คนดู",
        core_hook="Hook",
        key_questions_to_answer=["Q1", "Q2", "Q3"],
        research_hypotheses=["H1", "H2"],
        source_event_ids=["ev-1"],
        source_links=["http://test.com/1"],
        source_titles=["ข่าว 1"],
        recommended_format="15m",
        estimated_impact="Impact",
    )

    res = synthesize_notebooklm_source(pitch, source_events=[{"event_id": "ev-1", "title": "ข่าว 1"}])
    assert "📑 สรุปผู้บริหารและแหล่งอ้างอิง" in res.content


def test_save_briefing_artifact_with_thai_filename(tmp_path):
    content = "# Briefing Book เนื้อหาเต็ม"
    title = "วิเคราะห์หุ้นเทคไทยและโลก ปี 2026"

    from schemas.briefing_book_schemas import PublishableBriefingResult, ResearchQualityReport
    from tests.fixtures.briefing_fixtures import make_valid_briefing_draft, make_valid_evidence_bundle
    synthesis = PublishableBriefingResult(
        content=content,
        draft=make_valid_briefing_draft(),
        quality_report=ResearchQualityReport(score=100, status="pass", publishable=True),
        evidence_bundle=make_valid_evidence_bundle()
    )

    saved_artifact = save_briefing_artifact(synthesis, title, vault_root=tmp_path, date_str="2026-07-18")
    saved_path = saved_artifact.path

    assert saved_path.exists()
    assert "30_Knowledge_Base" in str(saved_path) and "NotebookLM_Sources" in str(saved_path)
    assert "วิเคราะห์หุ้นเทคไทยและโลก" in saved_path.name
    assert "2026-07-18_" in saved_path.name
    assert "_verified.md" in saved_path.name
    assert saved_path.read_text(encoding="utf-8") == content

    # ทดสอบ collision (_2) (Now tests idempotency)
    saved_path_2 = save_briefing_artifact(synthesis, title, vault_root=tmp_path, date_str="2026-07-18").path
    assert saved_path_2.exists()
    assert saved_path_2 == saved_path


@patch("tools.content.youtube_pitcher.load_store")
@patch("tools.content.youtube_pitcher.get_macro_baselines")
def test_fetch_news_for_pitching_with_youtube_summaries_always_include(mock_baselines, mock_load, tmp_path, monkeypatch):
    monkeypatch.setattr("tools.content.youtube_pitcher.VAULT_PATH", tmp_path)
    mock_baselines.invoke.return_value = '{"macro": "ok"}'
    mock_load.return_value = {"pending_events": []}

    from datetime import datetime, timedelta, timezone
    recent_date = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")

    yt_dir = tmp_path / "30_Knowledge_Base" / "YouTube_Summaries"
    yt_dir.mkdir(parents=True)
    md_file = yt_dir / f"{recent_date}_youtube_summary.md"
    md_file.write_text(
        "---\n"
        f"title: YouTube Insight abc12345678 {recent_date}\n"
        "entity_type: youtube_insight\n"
        "channel: Pi Securities\n"
        f"date: {recent_date}\n"
        "source_url: https://www.youtube.com/watch?v=abc12345678\n"
        "---\n"
        "# Title\n"
        "## ใจความสำคัญ\n"
        "- ตลาดหุ้นไทยฟื้นตัวอย่างแข็งแกร่งด้วยแรงหนุนจากหุ้น Defensive\n"
        "## แนวคิดการลงทุน\n"
        "- เน้นสะสมหุ้นปันผลและกลุ่มโรงพยาบาลเพื่อรับมือความผันผวน\n",
        encoding="utf-8",
    )

    candidates, macro_str, is_fallback = fetch_news_for_pitching(lookback_days=30)
    assert len(candidates) == 1
    c = candidates[0]
    assert c["source_layer"] == "layer2_youtube"
    assert "[YouTube Guru View - Pi Securities] ตลาดหุ้นไทยฟื้นตัวอย่างแข็งแกร่ง" in c["canonical_title"]
    assert "[ใจความสำคัญ]" in c["comprehensive_summary"]
    assert "[แนวคิดการลงทุน]" in c["comprehensive_summary"]
    assert c["links"] == ["https://www.youtube.com/watch?v=abc12345678"]


@patch("tools.content.youtube_pitcher.invoke_structured_llm")
def test_generate_pitches_internal_quota_and_truncation(mock_invoke):
    from tools.content.youtube_pitcher import _generate_pitches_internal

    mock_invoke.return_value = YouTubeContentPitchBatch(pitches=[], date_range_summary="30 วัน", total_source_events=20)

    # สร้าง candidates 15 news items + 10 yt items (โดยให้ long summary ที่ยาว 600 chars เพื่อเช็ค truncation)
    long_summary = "ก" * 600
    candidates = []
    for i in range(15):
        candidates.append({
            "event_id": f"news-{i}",
            "canonical_title": f"News {i}",
            "comprehensive_summary": long_summary,
            "source_layer": "layer1_store",
            "ingested_at": f"2026-07-15T10:{i:02d}:00",
        })
    for j in range(10):
        candidates.append({
            "event_id": f"yt-{j}",
            "canonical_title": f"YT {j}",
            "comprehensive_summary": long_summary,
            "source_layer": "layer2_youtube",
            "ingested_at": f"2026-07-16T10:{j:02d}:00",
        })

    _generate_pitches_internal(candidates, max_pitches=3, instruction="", date_summary="30 วัน")

    mock_invoke.assert_called_once()
    prompt_lines = mock_invoke.call_args[1]["prompt_lines"] if "prompt_lines" in mock_invoke.call_args[1] else mock_invoke.call_args[0][2]
    prompt_text = "\n".join(prompt_lines)

    # ต้องมี news 12 รายการ และ yt 8 รายการ (รวม 20 รายการ)
    assert "จำนวนข้อมูลและบทวิเคราะห์ที่คัดกรองมาทั้งหมด: 20 รายการ (จากทั้งหมด 25 รายการในคลัง)" in prompt_text
    # News truncation: ข่าวล่าสุด (news-14 ถึง news-3) ต้องอยู่ครบ และโดนตัดที่ 250 chars
    assert "news-14" in prompt_text
    assert "news-0" not in prompt_text
    # YT truncation: yt ล่าสุด (yt-9 ถึง yt-2) ต้องอยู่ครบ และโดนตัดที่ 550 chars
    assert "yt-9" in prompt_text
    assert "yt-0" not in prompt_text
    # เช็คความยาวสรุปใน prompt (โดย split ตาม block ของ candidate [news- หรือ [yt-)
    for block in prompt_text.split("\n["):
        if block.startswith("news-") and "   สรุป: " in block:
            summary_part = block.split("   สรุป: ")[1].split("\n")[0]
            assert len(summary_part) <= 250
        elif block.startswith("yt-") and "   สรุป: " in block:
            summary_part = block.split("   สรุป: ")[1].split("\n")[0]
            assert len(summary_part) == 550

def test_visual_markers_validation_rules():
    from tools.content.briefing_quality import validate_briefing_book_quality

    bundle = MagicMock()
    bundle.sources = []
    bundle.evidence_items = []

    source = MagicMock()
    source.source_id = "S1"
    source.source_type = "news"
    source.verification_status = "verified"
    source.published_at = "2026-07-01"
    source.independence_key = "K1"
    source.publisher = "Pub1"
    source.url = "http"
    source.ingested_at = "2026-07-01"

    source2 = MagicMock()
    source2.source_id = "S2"
    source2.source_type = "news"
    source2.verification_status = "verified"
    source2.published_at = "2026-07-01"
    source2.independence_key = "K2"
    source2.publisher = "Pub2"
    source2.url = "http"
    source2.ingested_at = "2026-07-01"

    bundle.sources = [source, source2]

    ev1 = MagicMock()
    ev1.evidence_id = "E01"
    ev1.source_ids = ["S1"]
    ev1.classification = "verified_fact"
    ev1.claim = "X"
    ev1.metric_name = None
    ev1.value = None
    ev1.observed_at = "2026-07-01"

    ev2 = MagicMock()
    ev2.evidence_id = "E02"
    ev2.source_ids = ["S2"]
    ev2.classification = "verified_fact"
    ev2.claim = "Y"
    ev2.metric_name = None
    ev2.value = None
    ev2.observed_at = "2026-07-01"

    bundle.evidence_items = [ev1, ev2]
    bundle.investigation_mode = "narrative"
    bundle.macro_snapshot = None
    bundle.financial_snapshots = []

    draft = MagicMock()
    draft.executive_summary = "[E01] [S1]"
    draft.causality_scenarios = []
    draft.notebooklm_prompts = []
    draft.asset_impacts = []

    d1 = MagicMock()
    d1.visual_id = "V01"
    d1.act = "Act I"
    d1.series_keys = ["Key1"]
    d1.date_range = "2026"
    d1.sources = ["S1"]
    d1.evidence_ids = ["E01"]
    d1.historical_comparison = False

    d2 = MagicMock()
    d2.visual_id = "V02"
    d2.act = "Act II"
    d2.series_keys = ["Key2"]
    d2.date_range = "2026"
    d2.sources = ["S2"]
    d2.evidence_ids = ["E02"]
    d2.historical_comparison = False

    d3 = MagicMock()
    d3.visual_id = "V03"
    d3.act = "Act III"
    d3.series_keys = ["Key3"]
    d3.date_range = "2026"
    d3.sources = ["S1", "S2"]
    d3.evidence_ids = ["E01", "E02"]
    d3.historical_comparison = False

    draft.visual_directives = [d1, d2, d3]

    from tools.content.briefing_renderer import RenderedBriefing
    from schemas.briefing_book_schemas import RenderedVisualMarker
    v1_act1 = RenderedVisualMarker(id="V01", act="Act I", evidence_ids=["E01"])
    v2_act2 = RenderedVisualMarker(id="V02", act="Act II", evidence_ids=["E02"])
    v3_act3 = RenderedVisualMarker(id="V03", act="Act III", evidence_ids=["E01", "E02"])

    md_missing = RenderedBriefing(content="", section_names=["Act I", "Act II", "Act III"], visual_markers=[v2_act2, v3_act3], cited_evidence_ids=[], cited_source_ids=[])
    report1 = validate_briefing_book_quality(bundle, draft, md_missing)
    assert report1.status == "fail"

    md_dup = RenderedBriefing(content="", section_names=["Act I", "Act II", "Act III"], visual_markers=[v1_act1, v1_act1, v2_act2, v3_act3], cited_evidence_ids=[], cited_source_ids=[])
    report2 = validate_briefing_book_quality(bundle, draft, md_dup)
    assert any(i.code == "VISUAL_DUPLICATED" for i in report2.issues)

    v1_act2 = RenderedVisualMarker(id="V01", act="Act II", evidence_ids=["E01"])
    md_wrong_act = RenderedBriefing(content="", section_names=["Act I", "Act II", "Act III"], visual_markers=[v1_act2, v2_act2, v3_act3], cited_evidence_ids=[], cited_source_ids=[])
    report3 = validate_briefing_book_quality(bundle, draft, md_wrong_act)
    assert any(i.code == "VISUAL_WRONG_ACT" for i in report3.issues)

    v1_mismatch = RenderedVisualMarker(id="V01", act="Act I", evidence_ids=["E02"])
    md_mismatch = RenderedBriefing(content="", section_names=["Act I", "Act II", "Act III"], visual_markers=[v1_mismatch, v2_act2, v3_act3], cited_evidence_ids=[], cited_source_ids=[])
    report4 = validate_briefing_book_quality(bundle, draft, md_mismatch)
    assert any(i.code == "VISUAL_EVIDENCE_MISMATCH" for i in report4.issues)

    md_good = RenderedBriefing(content="", section_names=["Act I", "Act II", "Act III"], visual_markers=[v1_act1, v2_act2, v3_act3], cited_evidence_ids=[], cited_source_ids=[])
    report5 = validate_briefing_book_quality(bundle, draft, md_good)
    assert not any(i.code in ("VISUAL_MISSING", "VISUAL_DUPLICATED", "VISUAL_WRONG_ACT", "VISUAL_EVIDENCE_MISMATCH") for i in report5.issues)


def test_numeric_grounding_accepts_comma_formatted_numbers():
    """Regression test: a citation next to a comma-grouped number like
    '24,975.82' must not be flagged as ungrounded. The old regex
    (`\\d+(?:\\.\\d+)?`) split on the comma into '24' and '975.82', neither of
    which matched the evidence value, producing a false-positive
    NUMERIC_GROUNDING_WARNING even though the LLM had narrated the number
    correctly."""
    from tools.content.briefing_quality import validate_briefing_book_quality
    from tools.content.briefing_renderer import RenderedBriefing

    bundle = MagicMock()
    source = MagicMock()
    source.source_id = "S1"
    source.source_type = "news"
    source.verification_status = "verified"
    source.published_at = "2026-07-01"
    source.independence_key = "K1"
    source.publisher = "Pub1"
    source.url = "http"
    source.ingested_at = "2026-07-01"
    bundle.sources = [source]

    ev1 = MagicMock()
    ev1.evidence_id = "E01"
    ev1.source_ids = ["S1"]
    ev1.classification = "verified_fact"
    ev1.claim = "Nasdaq value"
    ev1.metric_name = "Nasdaq"
    ev1.value = 24975.82
    ev1.observed_at = "2026-07-01"
    bundle.evidence_items = [ev1]
    bundle.investigation_mode = "macro"
    bundle.macro_snapshot = None
    bundle.financial_snapshots = []

    draft = MagicMock()
    draft.executive_summary = "ดัชนี Nasdaq Composite อยู่ที่ 24,975.82 จุด [E01]"
    draft.bull_case = ""
    draft.bear_case = ""
    draft.act1_script = ""
    draft.act2_script = ""
    draft.act3_script = ""
    draft.causality_scenarios = []
    draft.asset_impacts = []
    draft.notebooklm_prompts = []
    draft.visual_directives = []

    rendered = RenderedBriefing(content="", section_names=[], visual_markers=[], cited_evidence_ids=set(), cited_source_ids=set())

    report = validate_briefing_book_quality(bundle, draft, rendered)
    assert not any(i.code == "NUMERIC_GROUNDING_WARNING" for i in report.issues)


@patch("tools.content.youtube_pitcher.invoke_structured_llm")
@patch("tools.content.youtube_pitcher.normalize_visual_directives")
@patch("tools.content.briefing_artifacts.save_briefing_artifact")
@patch("tools.content.briefing_quality.validate_briefing_book_quality")
@patch("tools.content.briefing_renderer.render_briefing_book")
@patch("tools.content.provenance_enrichment.assess_pitch_source_readiness")
@patch("tools.content.briefing_evidence.build_briefing_evidence")
def test_synthesize_notebooklm_source_presentation_style_prompt_branching(
    mock_build, mock_readiness, mock_render, mock_validate, mock_save, mock_normalize, mock_invoke
):
    from schemas.briefing_book_schemas import InvestigativeBriefingBookDraft
    
    mock_draft = InvestigativeBriefingBookDraft(
        title="test", executive_summary="test", act1_script="test",
        act2_script="test", act3_script="test", causality_scenarios=[],
        asset_impacts=[], bull_case="bull", bear_case="bear",
        falsification_triggers=[], notebooklm_prompts=[], visual_directives=[]
    )
    mock_invoke.return_value = mock_draft
    mock_normalize.return_value = mock_draft
    mock_render.return_value = MagicMock()
    mock_validate.return_value = MagicMock(status="pass", blockers=[], issues=[])

    mock_build.return_value = MagicMock(sources=[], evidence_items=[])
    source_events = [{"event_id": "e1", "canonical_title": "t1"}]
    mock_readiness.return_value = ("ready", [], [], source_events)
    mock_render.side_effect = Exception("Stop Here!")
    
    import pytest
    # 1. Test "narrative"
    pitch_narrative = YouTubeContentPitchItem(
        pitch_id="p1", working_titles=["1","2","3"], target_audience="a", core_hook="h",
        key_questions_to_answer=["1","2","3"], research_hypotheses=["1","2"],
        source_event_ids=["e1"], source_links=[], source_titles=["t1"], recommended_format="f",
        estimated_impact="i", presentation_style="narrative"
    )
    with pytest.raises(Exception, match="Stop Here!"):
        synthesize_notebooklm_source(pitch_narrative, source_events)
    call_args_narrative = mock_invoke.call_args[1]["prompt_lines"]
    assert any("บทความเชิงลึก (Narrative Deep Dive)" in line for line in call_args_narrative)
    assert not any("บทสัมภาษณ์ (Interview Q&A)" in line for line in call_args_narrative)
    
    # 2. Test "interview_qa"
    pitch_qa = YouTubeContentPitchItem(
        pitch_id="p2", working_titles=["1","2","3"], target_audience="a", core_hook="h",
        key_questions_to_answer=["1","2","3"], research_hypotheses=["1","2"],
        source_event_ids=["e1"], source_links=[], source_titles=["t1"], recommended_format="f",
        estimated_impact="i", presentation_style="interview_qa"
    )
    with pytest.raises(Exception, match="Stop Here!"):
        synthesize_notebooklm_source(pitch_qa, source_events)
    call_args_qa = mock_invoke.call_args[1]["prompt_lines"]
    assert any("บทสัมภาษณ์ (Interview Q&A)" in line for line in call_args_qa)
    assert not any("บทความเชิงลึก (Narrative Deep Dive)" in line for line in call_args_qa)


@patch("tools.content.youtube_pitcher.invoke_structured_llm")
@patch("tools.content.youtube_pitcher.normalize_visual_directives")
@patch("tools.content.briefing_artifacts.save_briefing_artifact")
@patch("tools.content.briefing_quality.validate_briefing_book_quality")
@patch("tools.content.briefing_renderer.render_briefing_book")
@patch("tools.content.provenance_enrichment.assess_pitch_source_readiness")
@patch("tools.content.briefing_evidence.build_briefing_evidence")
def test_synthesize_notebooklm_source_pitch_info_includes_audience_takeaway(
    mock_build, mock_readiness, mock_render, mock_validate, mock_save, mock_normalize, mock_invoke
):
    """audience_takeaway ต้องถูกใส่เข้า pitch_info ที่ป้อนให้ LLM สร้าง Briefing Book (WIIFM)"""
    from schemas.briefing_book_schemas import InvestigativeBriefingBookDraft

    mock_draft = InvestigativeBriefingBookDraft(
        title="test", executive_summary="test", act1_script="test",
        act2_script="test", act3_script="test", causality_scenarios=[],
        asset_impacts=[], bull_case="bull", bear_case="bear",
        falsification_triggers=[], notebooklm_prompts=[], visual_directives=[]
    )
    mock_invoke.return_value = mock_draft
    mock_normalize.return_value = mock_draft
    mock_render.return_value = MagicMock()
    mock_validate.return_value = MagicMock(status="pass", blockers=[], issues=[])
    mock_build.return_value = MagicMock(sources=[], evidence_items=[])
    source_events = [{"event_id": "e1", "canonical_title": "t1"}]
    mock_readiness.return_value = ("ready", [], [], source_events)
    mock_render.side_effect = Exception("Stop Here!")

    pitch = YouTubeContentPitchItem(
        pitch_id="p3", working_titles=["1", "2", "3"], target_audience="a", core_hook="h",
        key_questions_to_answer=["1", "2", "3"], research_hypotheses=["1", "2"],
        source_event_ids=["e1"], source_links=[], source_titles=["t1"], recommended_format="f",
        estimated_impact="i", audience_takeaway="เก็บเงินสดสำรอง 6 เดือนก่อนตัดสินใจลงทุนเพิ่ม",
    )
    with pytest.raises(Exception, match="Stop Here!"):
        synthesize_notebooklm_source(pitch, source_events)
    prompt_lines = mock_invoke.call_args[1]["prompt_lines"]
    assert any("Audience Takeaway: เก็บเงินสดสำรอง" in line for line in prompt_lines)


@patch("tools.content.youtube_pitcher.invoke_structured_llm")
@patch("tools.content.youtube_pitcher.normalize_visual_directives")
@patch("tools.content.briefing_artifacts.save_briefing_artifact")
@patch("tools.content.briefing_quality.validate_briefing_book_quality")
@patch("tools.content.briefing_renderer.render_briefing_book")
@patch("tools.content.provenance_enrichment.assess_pitch_source_readiness")
@patch("tools.content.briefing_evidence.build_briefing_evidence")
def test_synthesize_notebooklm_source_prompt_requires_grounded_citations(
    mock_build, mock_readiness, mock_render, mock_validate, mock_save, mock_normalize, mock_invoke
):
    """Regression test for the Act III bug where an evidence ID was cited on a
    purely qualitative claim ('อาจได้รับผลกระทบหนัก [E14]') with no number nearby —
    the prompt must explicitly forbid that pattern."""
    from schemas.briefing_book_schemas import InvestigativeBriefingBookDraft

    mock_draft = InvestigativeBriefingBookDraft(
        title="test", executive_summary="test", act1_script="test",
        act2_script="test", act3_script="test", causality_scenarios=[],
        asset_impacts=[], bull_case="bull", bear_case="bear",
        falsification_triggers=[], notebooklm_prompts=[], visual_directives=[]
    )
    mock_invoke.return_value = mock_draft
    mock_normalize.return_value = mock_draft
    mock_build.return_value = MagicMock(sources=[], evidence_items=[])
    source_events = [{"event_id": "e1", "canonical_title": "t1"}]
    mock_readiness.return_value = ("ready", [], [], source_events)
    mock_render.side_effect = Exception("Stop Here!")

    import pytest
    pitch = YouTubeContentPitchItem(
        pitch_id="p1", working_titles=["1", "2", "3"], target_audience="a", core_hook="h",
        key_questions_to_answer=["1", "2", "3"], research_hypotheses=["1", "2"],
        source_event_ids=["e1"], source_links=[], source_titles=["t1"], recommended_format="f",
        estimated_impact="i",
    )
    with pytest.raises(Exception, match="Stop Here!"):
        synthesize_notebooklm_source(pitch, source_events)
    prompt_lines = mock_invoke.call_args[1]["prompt_lines"]
    prompt_text = "\n".join(prompt_lines)
    assert "ประโยคเดียวกันหรือประโยคที่อยู่ติดกันเสมอ" in prompt_text
    assert "ห้ามอ้างอิง Evidence ID ต่อท้ายข้อความเชิงคุณภาพโดยไม่ระบุตัวเลขประกอบเด็ดขาด" in prompt_text


@patch("tools.content.youtube_pitcher.invoke_structured_llm")
@patch("tools.content.youtube_pitcher.normalize_visual_directives")
@patch("tools.content.briefing_quality.validate_briefing_book_quality")
@patch("tools.content.briefing_renderer.render_briefing_book")
@patch("tools.content.provenance_enrichment.assess_pitch_source_readiness")
@patch("tools.content.briefing_evidence.build_briefing_evidence")
def test_synthesize_notebooklm_source_unverified_draft_never_publishable(
    mock_build, mock_readiness, mock_render, mock_validate, mock_normalize, mock_invoke
):
    """Regression test for #AG-33: a pitch that only reaches the Unverified
    Draft path because of a bypassable SINGLE_INDEPENDENT_SOURCE cap can still
    score >= 80 with no blocker in the content-level gate, which makes
    validate_briefing_book_quality report publishable=True. Constructing
    UnverifiedBriefingDraftResult with that report used to raise
    'Draft result must not be publishable' and crash the whole job."""
    from schemas.briefing_book_schemas import (
        InvestigativeBriefingBookDraft,
        ResearchQualityReport,
        QualityIssueRecord,
        UnverifiedBriefingDraftResult,
        BriefingEvidenceBundle,
    )

    mock_draft = InvestigativeBriefingBookDraft(
        title="test", executive_summary="test", act1_script="test",
        act2_script="test", act3_script="test", causality_scenarios=[],
        asset_impacts=[], bull_case="bull", bear_case="bear",
        falsification_triggers=[], notebooklm_prompts=[], visual_directives=[]
    )
    mock_invoke.return_value = mock_draft
    mock_normalize.return_value = mock_draft
    mock_build.return_value = BriefingEvidenceBundle(pitch_id="p-002-oil-crisis")
    mock_render.return_value = MagicMock(content="# Test Title\n\nBody")

    # Reproduces the exact shape from #AG-33: a bypassable "cap" issue,
    # score capped but still >= 80, no blockers -> publishable computes True.
    mock_validate.return_value = ResearchQualityReport(
        score=85,
        status="degraded",
        publishable=True,
        issues=[
            QualityIssueRecord(
                code="SINGLE_INDEPENDENT_SOURCE",
                category="provenance",
                severity="cap",
                description="Only one independent core source group is available",
                bypassable=True,
            )
        ],
    )

    source_events = [{"event_id": "e1", "canonical_title": "t1"}]
    # SINGLE_INDEPENDENT_SOURCE is allowlisted for the Unverified Draft bypass.
    mock_readiness.return_value = ("blocked", ["ต้องมีอย่างน้อย 2 กลุ่มแหล่งข่าวอิสระ"], ["SINGLE_INDEPENDENT_SOURCE"], source_events)

    pitch = YouTubeContentPitchItem(
        pitch_id="p-002-oil-crisis", working_titles=["1", "2", "3"], target_audience="a", core_hook="h",
        key_questions_to_answer=["1", "2", "3"], research_hypotheses=["1", "2"],
        source_event_ids=["e1"], source_links=[], source_titles=["t1"], recommended_format="f",
        estimated_impact="i", source_readiness_issues=["ต้องมีอย่างน้อย 2 กลุ่มแหล่งข่าวอิสระ"],
    )
    override_audit = {
        "job_id": "job-1", "thread_id": "thread-1", "pitch_id": "p-002-oil-crisis",
        "policy_version": "unverified-draft-v1", "reason": "User accepted single-source provenance",
        "server_timestamp": "2026-07-25T00:00:00", "token_hash": "abc123",
        "source_readiness_snapshot": ["SINGLE_INDEPENDENT_SOURCE"],
    }

    result = synthesize_notebooklm_source(
        pitch, source_events, output_mode="unverified_draft", override_audit=override_audit
    )

    assert isinstance(result, UnverifiedBriefingDraftResult)
    assert result.quality_report.publishable is False
    assert "Unverified Draft" in result.content

@patch("tools.content.youtube_pitcher.invoke_structured_llm")
@patch("tools.content.youtube_pitcher.normalize_visual_directives")
@patch("tools.content.briefing_quality.validate_briefing_book_quality")
@patch("tools.content.briefing_renderer.render_briefing_book")
@patch("tools.content.provenance_enrichment.assess_pitch_source_readiness")
@patch("tools.content.briefing_evidence.build_briefing_evidence")
def test_synthesize_notebooklm_source_unverified_draft_financial_provider_bypass(
    mock_build, mock_readiness, mock_render, mock_validate, mock_normalize, mock_invoke
):
    from schemas.briefing_book_schemas import (
        InvestigativeBriefingBookDraft,
        ResearchQualityReport,
        QualityIssueRecord,
        UnverifiedBriefingDraftResult,
        BriefingEvidenceBundle,
    )

    mock_draft = InvestigativeBriefingBookDraft(
        title="test", executive_summary="test", act1_script="test",
        act2_script="test", act3_script="test", causality_scenarios=[],
        asset_impacts=[], bull_case="bull", bear_case="bear",
        falsification_triggers=[], notebooklm_prompts=[], visual_directives=[]
    )
    mock_invoke.return_value = mock_draft
    mock_normalize.return_value = mock_draft
    mock_build.return_value = BriefingEvidenceBundle(pitch_id="p-002")
    mock_render.return_value = MagicMock(content="# Test Title\n\nBody")

    mock_validate.return_value = ResearchQualityReport(
        score=75,
        status="degraded",
        publishable=True,
        issues=[
            QualityIssueRecord(
                code="FINANCIAL_PROVIDER_UNAVAILABLE",
                category="financial",
                severity="blocker",
                description="No usable financial statement was returned",
                bypassable=True,
            )
        ],
    )

    source_events = [{"event_id": "e1", "canonical_title": "t1"}]
    mock_readiness.return_value = ("ready", [], [], source_events)

    pitch = YouTubeContentPitchItem(
        pitch_id="p-002", working_titles=["1", "2", "3"], target_audience="a", core_hook="h",
        key_questions_to_answer=["1", "2", "3"], research_hypotheses=["1", "2"],
        source_event_ids=["e1"], source_links=[], source_titles=["t1"], recommended_format="f",
        estimated_impact="i", source_readiness_issues=[],
    )

    result = synthesize_notebooklm_source(
        pitch, source_events, output_mode="unverified_draft",
        override_audit={
            "job_id": "job-1", "thread_id": "thread-1", "pitch_id": "p-00x",
            "policy_version": "test", "reason": "test",
            "server_timestamp": "2026-07-25T00:00:00", "token_hash": "hash",
            "source_readiness_snapshot": ["SINGLE_INDEPENDENT_SOURCE"]
        }
    )

    assert isinstance(result, UnverifiedBriefingDraftResult)
    assert result.quality_report.publishable is False
    assert "No usable financial statement was returned" in result.content

@patch("tools.content.youtube_pitcher.invoke_structured_llm")
@patch("tools.content.youtube_pitcher.normalize_visual_directives")
@patch("tools.content.briefing_quality.validate_briefing_book_quality")
@patch("tools.content.briefing_renderer.render_briefing_book")
@patch("tools.content.provenance_enrichment.assess_pitch_source_readiness")
@patch("tools.content.briefing_evidence.build_briefing_evidence")
def test_synthesize_notebooklm_source_unverified_draft_missing_macro_bypass(
    mock_build, mock_readiness, mock_render, mock_validate, mock_normalize, mock_invoke
):
    from schemas.briefing_book_schemas import (
        InvestigativeBriefingBookDraft,
        ResearchQualityReport,
        QualityIssueRecord,
        UnverifiedBriefingDraftResult,
        BriefingEvidenceBundle,
    )

    mock_draft = InvestigativeBriefingBookDraft(
        title="test", executive_summary="test", act1_script="test",
        act2_script="test", act3_script="test", causality_scenarios=[],
        asset_impacts=[], bull_case="bull", bear_case="bear",
        falsification_triggers=[], notebooklm_prompts=[], visual_directives=[]
    )
    mock_invoke.return_value = mock_draft
    mock_normalize.return_value = mock_draft
    mock_build.return_value = BriefingEvidenceBundle(pitch_id="p-003")
    mock_render.return_value = MagicMock(content="# Test Title\n\nBody")

    mock_validate.return_value = ResearchQualityReport(
        score=70,
        status="degraded",
        publishable=True,
        issues=[
            QualityIssueRecord(
                code="MISSING_MACRO_SNAPSHOT",
                category="macro",
                severity="cap",
                description="Macro/mixed briefing has no provider macro snapshot",
                bypassable=True,
            )
        ],
    )

    source_events = [{"event_id": "e1", "canonical_title": "t1"}]
    mock_readiness.return_value = ("ready", [], [], source_events)

    pitch = YouTubeContentPitchItem(
        pitch_id="p-003", working_titles=["1", "2", "3"], target_audience="a", core_hook="h",
        key_questions_to_answer=["1", "2", "3"], research_hypotheses=["1", "2"],
        source_event_ids=["e1"], source_links=[], source_titles=["t1"], recommended_format="f",
        estimated_impact="i", source_readiness_issues=[],
    )

    result = synthesize_notebooklm_source(
        pitch, source_events, output_mode="unverified_draft",
        override_audit={
            "job_id": "job-1", "thread_id": "thread-1", "pitch_id": "p-00x",
            "policy_version": "test", "reason": "test",
            "server_timestamp": "2026-07-25T00:00:00", "token_hash": "hash",
            "source_readiness_snapshot": ["SINGLE_INDEPENDENT_SOURCE"]
        }
    )

    assert isinstance(result, UnverifiedBriefingDraftResult)
    assert result.quality_report.publishable is False
    assert "Macro/mixed briefing has no provider macro snapshot" in result.content

@patch("tools.content.youtube_pitcher.invoke_structured_llm")
@patch("tools.content.youtube_pitcher.normalize_visual_directives")
@patch("tools.content.briefing_quality.validate_briefing_book_quality")
@patch("tools.content.briefing_renderer.render_briefing_book")
@patch("tools.content.provenance_enrichment.assess_pitch_source_readiness")
@patch("tools.content.briefing_evidence.build_briefing_evidence")
def test_synthesize_notebooklm_source_unverified_draft_stale_macro_bypass(
    mock_build, mock_readiness, mock_render, mock_validate, mock_normalize, mock_invoke
):
    from schemas.briefing_book_schemas import (
        InvestigativeBriefingBookDraft,
        ResearchQualityReport,
        QualityIssueRecord,
        UnverifiedBriefingDraftResult,
        BriefingEvidenceBundle,
    )

    mock_draft = InvestigativeBriefingBookDraft(
        title="test", executive_summary="test", act1_script="test",
        act2_script="test", act3_script="test", causality_scenarios=[],
        asset_impacts=[], bull_case="bull", bear_case="bear",
        falsification_triggers=[], notebooklm_prompts=[], visual_directives=[]
    )
    mock_invoke.return_value = mock_draft
    mock_normalize.return_value = mock_draft
    mock_build.return_value = BriefingEvidenceBundle(pitch_id="p-004")
    mock_render.return_value = MagicMock(content="# Test Title\n\nBody")

    mock_validate.return_value = ResearchQualityReport(
        score=85,
        status="degraded",
        publishable=True,
        issues=[
            QualityIssueRecord(
                code="STALE_MACRO_SNAPSHOT",
                category="macro",
                severity="cap",
                description="All macro observations are stale",
                bypassable=True,
            )
        ],
    )

    source_events = [{"event_id": "e1", "canonical_title": "t1"}]
    mock_readiness.return_value = ("ready", [], [], source_events)

    pitch = YouTubeContentPitchItem(
        pitch_id="p-004", working_titles=["1", "2", "3"], target_audience="a", core_hook="h",
        key_questions_to_answer=["1", "2", "3"], research_hypotheses=["1", "2"],
        source_event_ids=["e1"], source_links=[], source_titles=["t1"], recommended_format="f",
        estimated_impact="i", source_readiness_issues=[],
    )

    result = synthesize_notebooklm_source(
        pitch, source_events, output_mode="unverified_draft",
        override_audit={
            "job_id": "job-1", "thread_id": "thread-1", "pitch_id": "p-00x",
            "policy_version": "test", "reason": "test",
            "server_timestamp": "2026-07-25T00:00:00", "token_hash": "hash",
            "source_readiness_snapshot": ["SINGLE_INDEPENDENT_SOURCE"]
        }
    )

    assert isinstance(result, UnverifiedBriefingDraftResult)
    assert result.quality_report.publishable is False
    assert "All macro observations are stale" in result.content


def test_create_parking_lot_cards_atomic(tmp_path):
    from api import state_db
    db_path = str(tmp_path / "state.db")

    ideas = ["ไอเดียที่ 1  ", "ไอเดียที่ 2", "  ไอเดียที่ 1"]
    count = state_db.create_parking_lot_cards_atomic(ideas=ideas, source_pitch_id="p-100", db_path=db_path)
    assert count == 2

    # Second insert with same ideas -> ignored, rowcount=0, count=0
    count_again = state_db.create_parking_lot_cards_atomic(ideas=ideas, source_pitch_id="p-100", db_path=db_path)
    assert count_again == 0

    conn = state_db.get_connection(db_path)
    cards = state_db.list_kanban_cards(conn)
    assert len(cards) == 2
    assert cards[0]["display_seq"] == 1
    assert cards[1]["display_seq"] == 2
    assert cards[0]["column_name"] == "backlog"


def test_create_parking_lot_cards_atomic_concurrent(tmp_path):
    import concurrent.futures
    from api import state_db
    db_path = str(tmp_path / "concurrent_state.db")

    def insert_batch(batch_idx):
        ideas = [f"ไอเดียกลุ่ม {batch_idx} รายการ {i}" for i in range(5)]
        return state_db.create_parking_lot_cards_atomic(ideas=ideas, source_pitch_id=f"p-{batch_idx}", db_path=db_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(insert_batch, i) for i in range(4)]
        results = [f.result() for f in futures]

    assert sum(results) == 20
    conn = state_db.get_connection(db_path)
    cards = state_db.list_kanban_cards(conn)
    assert len(cards) == 20
    seqs = [c["display_seq"] for c in cards]
    assert sorted(seqs) == list(range(1, 21))


def test_financial_snapshot_reference_coercion():
    from tools.market.financial_autopsy import FinancialAutopsyPeriod, FinancialAutopsySnapshot
    from tools.content.youtube_pitcher import _financial_snapshot_reference
    from schemas.briefing_book_schemas import FinancialAutopsySnapshotRef, FinancialAutopsyPeriodRecord

    period = FinancialAutopsyPeriod(
        fiscal_period_end="2026-06-30",
        free_cash_flow=1000000.0,
        operating_cash_flow=1200000.0,
        capital_expenditure=200000.0,
        total_debt=500000.0,
        total_revenue=5000000.0,
        net_income=800000.0,
        ebit=900000.0,
        income_before_tax=850000.0,
    )
    snap = FinancialAutopsySnapshot(
        ticker="WDC",
        provider_symbol="WDC",
        market="US",
        currency="USD",
        retrieval_timestamp="2026-08-17T00:00:00",
        periods=[period],
        market_cap=25000000000.0,
        health_summary="Strong cash flow",
    )

    ref = _financial_snapshot_reference(snap)
    assert isinstance(ref, FinancialAutopsySnapshotRef)
    assert ref.symbol == "WDC"
    assert len(ref.periods) == 1
    assert isinstance(ref.periods[0], FinancialAutopsyPeriodRecord)
    assert ref.periods[0].fiscal_period_end == "2026-06-30"
    assert ref.periods[0].free_cash_flow == 1000000.0


def test_normalize_draft_evidence_references():
    from schemas.briefing_book_schemas import (
        InvestigativeBriefingBookDraft,
        BriefingEvidenceBundle,
        EvidenceItem,
        ScenarioRecord,
        AssetImpactRecord,
        VisualEvidenceDirective,
    )
    from tools.content.youtube_pitcher import normalize_draft_evidence_references

    bundle = BriefingEvidenceBundle(
        pitch_id="p-1",
        evidence_items=[

            EvidenceItem(
                evidence_id="E01",
                claim="Fact 1",
                source_ids=["S01"],
                classification="verified_fact",
            ),
            EvidenceItem(
                evidence_id="E02",
                claim="Fact 2",
                source_ids=["S02"],
                classification="verified_fact",
                metric_name="TTD:Revenue",
            ),
        ]
    )


    draft = InvestigativeBriefingBookDraft(
        title="Title",
        executive_summary="Summary",
        bull_case="Bull",
        bear_case="Bear",
        falsification_triggers=["F1"],
        act1_script="Act 1",
        act2_script="Act 2",
        act3_script="Act 3",
        causality_scenarios=[
            ScenarioRecord(
                scenario_id="SC01",
                name="Base",
                description="Desc",
                probability_pct=100.0,
                trigger_conditions=["T1"],
                falsification_triggers=["F1"],
                evidence_ids=["[E01]", "E2", "S01"],  # formatting variations
            )
        ],
        asset_impacts=[
            AssetImpactRecord(
                symbol_or_name="TTD",
                impact_type="direct_upside",
                reasoning="Reason",
                risk_factors=["R1"],
                invalidation_conditions=["I1"],
                evidence_ids=["S_FIN_TTD", "UNKNOWN"],  # invalid / source
            )
        ],
        visual_directives=[
            VisualEvidenceDirective(
                visual_id="V01",
                act="Act I",
                title="Title",
                chart_type="Type",
                description="Desc",
                annotation="Annotation",
                evidence_ids=["[E02]"],
                series_keys=["KEY"],
                date_range="2026",
                sources=["S01"],
            )

        ],
        notebooklm_prompts=[],
    )


    normalized = normalize_draft_evidence_references(draft, bundle)
    # Scenarios should be cleaned to valid E01, E02
    assert normalized.causality_scenarios[0].evidence_ids == ["E01", "E02"]
    # Asset impacts for TTD should resolve to E02
    assert "E02" in normalized.asset_impacts[0].evidence_ids
    # Visual directive should be cleaned of brackets
    assert normalized.visual_directives[0].evidence_ids == ["E02"]


def test_candidate_topic_ranking_prioritization():
    from tools.content.youtube_pitcher import _extract_topic_terms, _score_candidate_relevance

    instruction = "ไอเดียต่อยอดจาก YouTube Pitch (p-004): ผลกระทบของ Applied Materials ต่อดัชนี Nasdaq [lookback_days=7]"
    terms, clean_query = _extract_topic_terms(instruction)
    assert "Applied" in terms
    assert "Materials" in terms
    assert "Nasdaq" in terms

    matching_cand = {
        "canonical_title": "ตลาดหุ้นร่วงหลังยอดค้าปลีกเซอร์ไพรส์ ด้าน Applied Materials ดิ่งหนักจากผลประกอบการ",
        "summary": "Applied Materials รายงานผลประกอบการ",
        "tags": ["AMAT", "Technology"],
        "target_symbols": ["AMAT"],
        "ingested_at": "2026-08-10T10:00:00",
    }
    unrelated_cand = {
        "canonical_title": "ราคาน้ำมันดิบ Brent พุ่งสูงขึ้น",
        "summary": "ความตึงเครียดในตะวันออกกลางดันราคาน้ำมัน",
        "tags": ["Energy"],
        "target_symbols": ["OIL"],
        "ingested_at": "2026-08-17T12:00:00",  # Newer date
    }

    match_score = _score_candidate_relevance(matching_cand, terms, clean_query)
    unrelated_score = _score_candidate_relevance(unrelated_cand, terms, clean_query)

    assert match_score > unrelated_score
    assert match_score >= 30.0




