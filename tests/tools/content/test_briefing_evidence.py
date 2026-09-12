"""Unit tests สำหรับ tools/content/briefing_evidence.py"""
import json
from pathlib import Path

from schemas.briefing_book_schemas import (
    BriefingEvidenceBundle,
    SourceRecord,
    EvidenceItem,
    InvestigativeBriefingBookDraft,
)
from schemas.youtube_pitch_schemas import YouTubeContentPitchItem
from tools.content.briefing_evidence import build_briefing_evidence, _get_independence_key
from tools.content.youtube_pitcher import (
    normalize_visual_directives,
    select_financial_autopsy_assets,
)
from tools.content.briefing_renderer import render_briefing_book
from tools.content.briefing_artifacts import save_briefing_artifact
from tools.content.briefing_quality import validate_briefing_book_quality
from types import SimpleNamespace
import pytest


def test_independence_key_deduplication():
    assert _get_independence_key("FINNOMENA", "finnomena.com") == "finnomena_group"
    assert _get_independence_key("FINNOMENA Official", "youtube.com") == "finnomena_group"
    assert _get_independence_key("Reuters", "reuters.com") == "reuters_group"


def test_build_briefing_evidence():
    pitch = YouTubeContentPitchItem(
        pitch_id="pitch-01",
        working_titles=["1", "2", "3"],
        target_audience="นักลงทุน",
        core_hook="Hook",
        key_questions_to_answer=["Q1", "Q2", "Q3"],
        research_hypotheses=["H1", "H2"],
        source_event_ids=["ev-1"],
        source_links=["https://youtube.com/watch?v=123"],
        source_titles=["ข่าวทดสอบ"],
        recommended_format="15m",
        estimated_impact="High",
    )

    matched = [
        {
            "event_id": "ev-1",
            "original_title": "วิเคราะห์วิกฤตน้ำมันดิบโลก",
            "canonical_title": "วิเคราะห์วิกฤตน้ำมันดิบโลก",
            "publisher": "FINNOMENA",
            "channel": "FINNOMENA",
            "links": ["https://youtube.com/watch?v=123"],
            "source_layer": "layer2_youtube",
            "ingested_at": "2026-07-21T10:00:00",
            "published_at": "2026-07-20",
            "verification_status": "verified",
            "comprehensive_summary": "ราคาน้ำมันปรับตัวขึ้นจากความตึงเครียดในตะวันออกกลาง คาดการณ์ดอกเบี้ยทรงตัว",
            "key_metrics": "Brent $85/bbl",
        }
    ]

    bundle = build_briefing_evidence(pitch, matched)

    assert isinstance(bundle, BriefingEvidenceBundle)
    assert len(bundle.sources) == 1
    assert bundle.sources[0].source_id == "S01"
    assert bundle.sources[0].independence_key == "finnomena_group"
    assert bundle.sources[0].verification_status == "verified"

    assert len(bundle.evidence_items) >= 1
    assert bundle.evidence_items[0].source_ids == ["S01"]
    assert bundle.evidence_items[0].classification in ("source_reported_fact", "consensus", "hypothesis", "verified_fact")
    metric_evidence = next(item for item in bundle.evidence_items if item.metric_name == "Brent $85/bbl")
    assert metric_evidence.observed_at is None
    assert metric_evidence.reported_at == "2026-07-20"
    assert metric_evidence.time_semantics == "reported"


def test_quality_gate_rejects_empty_briefing_bundle():
    draft = SimpleNamespace(
        executive_summary="", bull_case="", bear_case="", act1_script="", act2_script="", act3_script="",
        causality_scenarios=[], asset_impacts=[], visual_directives=[], notebooklm_prompts=[],
    )
    from tools.content.briefing_renderer import RenderedBriefing
    empty_rendered = RenderedBriefing(
        content="",
        section_names=[],
        visual_markers=[],
        cited_evidence_ids=[],
        cited_source_ids=[]
    )
    report = validate_briefing_book_quality(BriefingEvidenceBundle(pitch_id="empty"), draft, empty_rendered)
    assert report.score < 100
    assert report.issues



def test_all_unverified_core_sources_are_a_hard_failure():
    bundle = BriefingEvidenceBundle(
        pitch_id="oil-war-regression",
        investigation_mode="macro",
        sources=[
            SourceRecord(
                source_id="S01", original_title="Unverified oil claim", publisher="Unverified",
                host="youtube.com", published_at=None, ingested_at="2026-07-23T10:00:00",
                url="https://youtube.com/watch?v=1", source_type="creator_commentary",
                independence_key="youtube.com_group", verification_status="unverified",
            ),
            SourceRecord(
                source_id="S02", original_title="Another unverified oil claim", publisher="Unverified",
                host="youtube.com", published_at=None, ingested_at="2026-07-23T10:00:00",
                url="https://youtube.com/watch?v=2", source_type="creator_commentary",
                independence_key="youtube.com_group", verification_status="unverified",
            ),
        ],
        evidence_items=[
            EvidenceItem(
                evidence_id="E01", claim="Brent traded at 88.28 without a stated timestamp",
                classification="source_reported_fact", value=88.28, source_ids=["S01"],
                confidence="low", metric_name="Brent", period_type="instant",
            ),
            EvidenceItem(
                evidence_id="E02", claim="Brent traded at 92.54 without a stated timestamp",
                classification="source_reported_fact", value=92.54, source_ids=["S02"],
                confidence="low", metric_name="Brent", period_type="instant",
            ),
        ],
    )
    draft = SimpleNamespace(
        executive_summary="Oil claim [E01]", bull_case="[E01]", bear_case="[E02]",
        act1_script="Act I", act2_script="Act II", act3_script="Act III",
        causality_scenarios=[], asset_impacts=[], visual_directives=[], notebooklm_prompts=[],
    )

    from tools.content.briefing_renderer import RenderedBriefing
    empty_rendered = RenderedBriefing(
        content="",
        section_names=[],
        visual_markers=[],
        cited_evidence_ids=[],
        cited_source_ids=[]
    )
    report = validate_briefing_book_quality(bundle, draft, empty_rendered)

    assert report.status == "fail"
    assert report.publishable is False
    assert any("All core sources are unverified" in item.description for item in report.issues)
    assert any("Conflicting metric values" in item.description for item in report.issues)


def test_renderer_owns_each_visual_marker_once():
    directive = {
        "visual_id": "V01", "act": "Act I", "title": "Oil futures", "chart_type": "line",
        "date_range": "2026-07-20 to 2026-07-23", "series_keys": ["BZ=F"],
        "sources": ["Yahoo Finance"], "annotation": "Move", "evidence_ids": ["E01"],
        "data_mode": "provider_series"
    }
    draft = InvestigativeBriefingBookDraft(
        title="Valid Metadata Test",
        executive_summary="[E01]",
        causality_scenarios=[
            {"scenario_id": "S1", "name": "Base", "description": "Base", "probability_pct": 50, "trigger_conditions": [], "falsification_triggers": [], "evidence_ids": [], "threshold_basis": ""}
        ],
        asset_impacts=[
            {"symbol_or_name": "AAPL", "impact_type": "direct_upside", "reasoning": "Good", "risk_factors": [], "invalidation_conditions": [], "evidence_ids": []}
        ],
        bull_case="Bull",
        bear_case="Bear",
        falsification_triggers=["F1"],
        act1_script="[VISUAL_EVIDENCE id=V01 evidence=E01]",
        act2_script="[VISUAL_EVIDENCE id=V02 evidence=E02]",
        act3_script="[VISUAL_EVIDENCE id=V03 evidence=E03]",
        visual_directives=[directive], notebooklm_prompts=[],
    )
    bundle = BriefingEvidenceBundle(
        pitch_id="marker-test",
        evidence_items=[EvidenceItem(
            evidence_id="E01", claim="Oil futures observation", classification="source_reported_fact",
            source_ids=["S01"], confidence="medium",
        )],
        sources=[SourceRecord(
            source_id="S01", original_title="Source", publisher="Reuters", host="reuters.com",
            published_at="2026-07-23", ingested_at="2026-07-23T10:00:00", url="https://reuters.com/x",
            source_type="wire_service", independence_key="reuters_group", verification_status="verified",
        )],
    )

    rendered = render_briefing_book(draft, bundle)

    assert rendered.content.count("[VISUAL_EVIDENCE id=V01 evidence=E01]") == 1
    assert "evidence=E99" not in rendered.content


def test_visual_normalizer_turns_generic_event_measure_into_evidence_table():
    directive = SimpleNamespace(
        visual_id="V03", act="Act III", title="War cost", chart_type="Bar",
        series_keys=["war cost"], date_range="2026-07-23", sources=["Evidence ledger"],
        annotation="Cost estimate", evidence_ids=["E01"],
    )
    draft = SimpleNamespace(visual_directives=[directive])

    normalize_visual_directives(draft)

    assert directive.data_mode == "evidence_table"
    assert directive.series_keys == ["EVIDENCE_TABLE"]


def test_visual_normalizer_enforces_declared_evidence_table_contract():
    """Regression for AG-20: an LLM declared evidence_table with chart series.

    The normalizer must canonicalize the declaration before the quality gate,
    rather than failing V02/V03 after both briefing-book generation attempts.
    """
    directives = [
        SimpleNamespace(
            visual_id="V02", act="Act II", title="Oil transmission", chart_type="Flow",
            series_keys=["BZ=F", "US CPI"], date_range="2026-07-23",
            sources=["Evidence ledger"], annotation="Transmission path", evidence_ids=["E01"],
            data_mode="evidence_table",
        ),
        SimpleNamespace(
            visual_id="V03", act="Act III", title="Portfolio consequences", chart_type="Bar",
            series_keys=["XLE", "^GSPC"], date_range="2026-07-23",
            sources=["Evidence ledger"], annotation="Scenario impact", evidence_ids=["E02"],
            data_mode="evidence_table",
        ),
    ]

    normalize_visual_directives(SimpleNamespace(visual_directives=directives))

    assert all(directive.data_mode == "evidence_table" for directive in directives)
    assert all(directive.series_keys == ["EVIDENCE_TABLE"] for directive in directives)


def test_financial_autopsy_ignores_incidental_ticker():
    pitch = YouTubeContentPitchItem(
        pitch_id="oil-story", working_titles=["Oil shock: supply risk", "2", "3"],
        target_audience="Investors", core_hook="Oil supply", key_questions_to_answer=["Q1", "Q2", "Q3"],
        research_hypotheses=["Supply disruption", "Demand response"], source_event_ids=["ev-1"],
        source_links=["https://example.com/oil"], source_titles=["Oil story"],
        recommended_format="15m", estimated_impact="High", investigation_mode="mixed",
    )

    assets = select_financial_autopsy_assets(pitch, [{
        "event_id": "ev-1", "title": "Oil story", "extracted_tickers": ["GOOG"],
    }])

    assert assets == []


def test_save_verifies_persisted_content_and_quality_hash(tmp_path, monkeypatch):
    from schemas.briefing_book_schemas import ResearchQualityReport
    import tools.content.youtube_pitcher as pitcher

    monkeypatch.setattr("tools.archivist.indexer._index_upsert", lambda *args, **kwargs: None)
    monkeypatch.setattr("tools.archivist.indexer.flush_index_if_dirty", lambda *args, **kwargs: None)

    content = "# AG-20 verified briefing"
    report = ResearchQualityReport(
        score=100,
        status="pass",
        publishable=True,
        issues=[],
        advisories=[],
    )

    from schemas.briefing_book_schemas import PublishableBriefingResult
    from tests.fixtures.briefing_fixtures import make_valid_briefing_draft, make_valid_evidence_bundle
    synthesis = PublishableBriefingResult(
        content=content,
        draft=make_valid_briefing_draft(),
        quality_report=report,
        evidence_bundle=make_valid_evidence_bundle()
    )

    saved = save_briefing_artifact(
        synthesis,
        "AG-20",
        vault_root=tmp_path,
        date_str="2026-07-23",
    ).path
    sidecar = saved.with_suffix(".quality.json")

    from tools.archivist.parser import _strip_frontmatter
    assert _strip_frontmatter(saved.read_text(encoding="utf-8")) == content
    saved_report = json.loads(sidecar.read_text(encoding="utf-8"))
    import hashlib
    assert saved_report["content_sha256"] == hashlib.sha256(content.encode("utf-8")).hexdigest()
