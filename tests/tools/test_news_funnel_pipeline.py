"""Unit tests สำหรับ tools/macro/news_funnel.py"""
import os
import pytest

from tools.macro.news_funnel import (
    THEME_DISPLAY_NAME_MAP,
    canonicalize_ticker_names,
    ensure_concept_stubs_exist,
    run_news_funnel_ingest,
    run_news_funnel_synthesize,
)
from unittest.mock import patch
from schemas.news_funnel_schemas import MacroImpactTriageResult, MacroThemeDigest, TriageBatchResult, ThemeSynthesisBatchResult
from tools.macro.news_funnel_store import load_store, prune_old_events, save_store, save_triage_events


@pytest.fixture
def test_paths(tmp_path):
    store_file = tmp_path / "state.json"
    vault_dir = tmp_path / "vault"
    return str(store_file), str(vault_dir)


def test_theme_display_name_map_and_tickers():
    assert THEME_DISPLAY_NAME_MAP["policy"] == "Monetary Policy"
    assert THEME_DISPLAY_NAME_MAP["risk_sentiment"] == "Market Risk Sentiment"

    tickers = ["NVIDIA", "[[APPLE]]", "PTT", "[[Gold]]"]
    canon = canonicalize_ticker_names(tickers)
    assert canon == ["NVDA", "AAPL", "PTT", "Gold"]


def test_ensure_concept_stubs(test_paths):
    _, vault_dir = test_paths
    created = ensure_concept_stubs_exist(["Gold", "[[US Treasury]]"], vault_root=vault_dir)
    assert len(created) == 2
    for path in created:
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "entity_type: concept_stub" in content


def test_ingest_pipeline(test_paths, monkeypatch):
    monkeypatch.setenv("MOCK_NEWS_FUNNEL_LLM", "true")
    store_file, _ = test_paths
    candidates = [
        {
            "title": "Fed Signals Steady Rates Amid Service Inflation",
            "summary": "Officials urge patience on rate cuts",
            "link": "https://example.com/fed1",
            "source": "Bloomberg",
        },
        {
            "title": "NVIDIA Reports Strong AI Chip Demand",
            "summary": "Data centers continue expanding",
            "link": "https://example.com/nvda1",
            "source": "Reuters",
        },
    ]

    res = run_news_funnel_ingest(candidates=candidates, store_path=store_file)
    assert res["status"] == "success"
    assert res["ingested_count"] == 2
    assert res["high_impact_count"] == 2

    # Second ingest of same candidates should deduplicate
    res2 = run_news_funnel_ingest(candidates=candidates, store_path=store_file)
    assert res2["ingested_count"] == 0


def test_synthesize_zero_pending_protection(test_paths):
    store_file, vault_dir = test_paths
    # Do not ingest anything -> pending = 0
    res = run_news_funnel_synthesize(
        period="morning",
        store_path=store_file,
        vault_root=vault_dir,
        custom_date="2026-07-13",
    )

    assert res["status"] == "no_pending_events"
    assert res["synthesized"] == 0
    # Must NOT create any markdown digest file
    themes_dir = os.path.join(vault_dir, "30_Knowledge_Base", "News", "Themes")
    assert not os.path.exists(themes_dir) or os.listdir(themes_dir) == []


def test_synthesize_with_events_and_hitl_filter(test_paths, monkeypatch):
    monkeypatch.setenv("MOCK_NEWS_FUNNEL_LLM", "true")
    store_file, vault_dir = test_paths
    events = [
        {
            "event_id": "ev-100",
            "canonical_title": "Fed rate decision and Inflation risk",
            "comprehensive_summary": "Service inflation sticky",
            "macro_impact_score": 8,
            "asset_impact_score": 6,
            "is_high_impact": True,
            "extracted_tickers": ["NVDA", "Gold"],
            "extracted_themes": ["policy", "inflation"],
        },
        {
            "event_id": "ev-200",
            "canonical_title": "Oil supply disruption geopolitics",
            "comprehensive_summary": "Crude oil rallies",
            "macro_impact_score": 7,
            "asset_impact_score": 7,
            "is_high_impact": True,
            "extracted_tickers": ["PTT"],
            "extracted_themes": ["geopolitics"],
        },
    ]
    save_triage_events(events, store_path=store_file)

    # Synthesize only ev-100 via approved_event_ids
    res = run_news_funnel_synthesize(
        period="morning",
        approved_event_ids=["ev-100"],
        store_path=store_file,
        vault_root=vault_dir,
        custom_date="2026-07-13",
    )

    assert res["status"] == "success"
    assert res["synthesized"] == 1
    file_path = res["file_path"]
    assert os.path.exists(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "Macro Themes Digest - 2026-07-13 Morning" in content
        assert "[[NVDA]]" in content
        assert "[[Monetary Policy]]" in content
        assert "[[Inflation]]" in content

    # Check state store that ev-100 is synthesized and ev-200 is still pending
    state = load_store(store_path=store_file)
    assert any(ev["event_id"] == "ev-100" and ev["status"] == "synthesized" for ev in state["pending_events"])
    assert any(ev["event_id"] == "ev-200" and ev["status"] == "pending_synthesis" for ev in state["pending_events"])


def test_overwrite_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_NEWS_FUNNEL_LLM", "true")
    store_file = str(tmp_path / "test_state.json")
    vault_dir = str(tmp_path / "vault")
    events = [
        {
            "event_id": "ev-A1",
            "canonical_title": "Title A1",
            "comprehensive_summary": "Sum 1",
            "macro_impact_score": 8,
            "asset_impact_score": 5,
            "is_high_impact": True,
            "extracted_tickers": ["NVDA"],
            "extracted_themes": ["policy"],
        },
        {
            "event_id": "ev-A2",
            "canonical_title": "Title A2",
            "comprehensive_summary": "Sum 2",
            "macro_impact_score": 8,
            "asset_impact_score": 5,
            "is_high_impact": True,
            "extracted_tickers": ["AAPL"],
            "extracted_themes": ["policy"],
        },
    ]
    save_triage_events(events, store_path=store_file)

    res1 = run_news_funnel_synthesize(
        period="morning",
        approved_event_ids=["ev-A1"],
        store_path=store_file,
        vault_root=vault_dir,
        custom_date="2026-07-13",
    )
    assert res1["status"] == "success"
    file1 = res1["file_path"]
    assert file1.endswith("2026-07-13_Morning_MacroThemes.md")

    res2 = run_news_funnel_synthesize(
        period="morning",
        approved_event_ids=["ev-A2"],
        store_path=store_file,
        vault_root=vault_dir,
        custom_date="2026-07-13",
    )
    assert res2["status"] == "success"
    file2 = res2["file_path"]
    assert file2.endswith("2026-07-13_Morning_MacroThemes_2.md")
    assert file1 != file2


def test_clustering_before_triage(tmp_path, monkeypatch):
    store_file = str(tmp_path / "test_state.json")
    monkeypatch.setenv("MOCK_NEWS_FUNNEL_LLM", "true")
    similar_candidates = [
        {"title": "Fed rate decision update", "link": "https://n.test/1", "source": "Reuters", "summary": "Fed rate"},
        {"title": "Fed rate decision update", "link": "https://n.test/2", "source": "Bloomberg", "summary": "Fed rate"},
        {"title": "Fed rate decision update", "link": "https://n.test/3", "source": "WSJ", "summary": "Fed rate"},
    ]
    res = run_news_funnel_ingest(candidates=similar_candidates, store_path=store_file)
    assert res["status"] == "success"
    # ควรถูกรวมกลุ่มเหลือ 1 cluster event ที่มี source_count = 3
    assert res["ingested_count"] == 1


def test_llm_triage_and_synthesis_mocked(tmp_path, monkeypatch):
    store_file = str(tmp_path / "test_state.json")
    vault_dir = str(tmp_path / "vault")
    monkeypatch.setenv("MOCK_NEWS_FUNNEL_LLM", "false")

    mock_triage_res = TriageBatchResult(results=[
        MacroImpactTriageResult(
            macro_impact_score=8,
            asset_impact_score=9,
            primary_tags=["macro"],
            extracted_tickers=["NVIDIA"],
            extracted_themes=["policy"],
            triage_reasoning="Mock LLM high impact",
        )
    ])

    with patch("core.llm_factory.get_llm") as mock_get_llm:
        mock_llm = mock_get_llm.return_value
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_triage_res
        res = run_news_funnel_ingest(
            candidates=[{"title": "Central Bank Policy", "link": "https://test.ex/cb", "summary": "Rate decision"}],
            store_path=store_file,
        )
        assert res["status"] == "success"
        assert res["high_impact_count"] == 1

    mock_synth_res = ThemeSynthesisBatchResult(themes=[
        MacroThemeDigest(
            theme_title="Synthesized Policy Theme",
            key_takeaways=["Key takeaway 1"],
            linked_assets=["NVIDIA", "PTT"],
            linked_themes=["policy"],
            policy_implications="Watch rates closely",
        )
    ])

    with patch("core.llm_factory.get_llm") as mock_get_llm:
        mock_llm = mock_get_llm.return_value
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_synth_res
        s_res = run_news_funnel_synthesize(
            period="morning",
            store_path=store_file,
            vault_root=vault_dir,
            custom_date="2026-07-13",
        )
        assert s_res["status"] == "success"
        assert len(s_res["report"]["themes"]) == 1
        # ตรวจสอบ canonicalization
        assert set(s_res["report"]["themes"][0]["linked_assets"]) == {"[[NVDA]]", "[[PTT]]"}


def test_store_retention_prune(tmp_path):
    from datetime import datetime, timedelta
    store_file = str(tmp_path / "test_prune_state.json")
    old_iso = (datetime.now() - timedelta(days=10)).isoformat()
    new_iso = datetime.now().isoformat()

    state = {
        "schema_version": 1,
        "pending_events": [
            {"event_id": "old-synth", "status": "synthesized", "ingested_at": old_iso},
            {"event_id": "new-synth", "status": "synthesized", "ingested_at": new_iso},
            {"event_id": "old-pending", "status": "pending_synthesis", "ingested_at": old_iso},
        ],
        "processed_urls": [f"url-{i}" for i in range(2500)],
        "processed_titles": [],
    }
    save_store(state, store_path=store_file)
    loaded = load_store(store_path=store_file)
    ids = [ev["event_id"] for ev in loaded["pending_events"]]
    assert "old-synth" not in ids
    assert "old-pending" not in ids
    assert "new-synth" in ids

    assert len(loaded["processed_urls"]) == 2000
    assert loaded["processed_urls"][0] == "url-500"
    assert loaded["processed_urls"][-1] == "url-2499"


def test_clean_and_truncate_summary_html():
    from tools.macro.news_funnel import _clean_and_truncate_summary
    html_text = "<p>The Federal Reserve decided to hold interest rates steady today amid ongoing concerns about sticky inflation in the services sector and robust employment data that exceeded market expectations.</p><div class='ad'>Ad text here</div>"
    cleaned = _clean_and_truncate_summary(html_text, max_len=60)
    assert "<p>" not in cleaned
    assert "</div>" not in cleaned
    assert len(cleaned) <= 65
    assert cleaned.endswith("...")


def test_theme_canonicalization_with_brackets():
    from tools.macro.news_funnel import _deterministic_synthesize_themes
    events = [
        {
            "event_id": "ev-1",
            "canonical_title": "Fed rate meeting",
            "comprehensive_summary": "Summary",
            "extracted_tickers": ["[[NVDA]]"],
            "extracted_themes": ["[[policy]]", "[[geopolitics|Geopolitics]]"],
            "macro_impact_score": 8,
            "asset_impact_score": 5,
        }
    ]
    themes = _deterministic_synthesize_themes(events)
    assert len(themes) > 0
    theme_links = themes[0].linked_themes
    assert "[[Monetary Policy]]" in theme_links
    assert "[[Geopolitics & Commodities]]" in theme_links
    assert "[[[[Monetary Policy]]]]" not in theme_links
    assert "[[[[policy]]]]" not in theme_links


def test_unified_cli_runner(monkeypatch, tmp_path):
    from scripts import run_news_funnel, run_news_funnel_auto
    store_file = str(tmp_path / "cli_test_state.json")
    
    # Test running via unified runner with --mode ingest
    monkeypatch.setattr("sys.argv", ["run_news_funnel.py", "--mode", "ingest", "--store-path", store_file])
    run_news_funnel.main()
    
    # Verify store was initialized
    assert os.path.exists(store_file)
    
    # Test running via deprecated runner triggers warning and runs unified main
    monkeypatch.setattr("sys.argv", ["run_news_funnel_auto.py", "--mode", "ingest", "--store-path", store_file])
    with pytest.deprecated_call():
        run_news_funnel_auto.main()

