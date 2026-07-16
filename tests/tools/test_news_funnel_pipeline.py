"""Unit tests สำหรับ tools/macro/news_funnel.py"""
import os
import pytest

from tools.macro.news_funnel import (
    canonicalize_ticker_names,
    ensure_concept_stubs_exist,
    run_news_funnel_ingest,
    run_news_funnel_synthesize,
)
from unittest.mock import patch
from schemas.news_funnel_schemas import MacroImpactTriageResult, TriageBatchResult
from tools.macro.news_funnel_store import load_store, prune_old_events, save_store, save_triage_events


@pytest.fixture
def test_paths(tmp_path):
    store_file = tmp_path / "state.json"
    vault_dir = tmp_path / "vault"
    return str(store_file), str(vault_dir)


def test_canonicalize_ticker_names():
    tickers = ["NVIDIA", "[[APPLE]]", "PTT", "[[Gold]]"]
    canon = canonicalize_ticker_names(tickers)
    assert canon == ["NVDA", "AAPL", "PTT", "Gold"]


def test_ensure_concept_stubs(test_paths):
    _, vault_dir = test_paths
    # Create old News/Concepts folder and a stub inside it to verify migration
    old_dir = os.path.join(vault_dir, "30_Knowledge_Base", "News", "Concepts")
    os.makedirs(old_dir, exist_ok=True)
    old_stub_path = os.path.join(old_dir, "OldStub.md")
    with open(old_stub_path, "w", encoding="utf-8") as f:
        f.write("---\ntitle: \"OldStub\"\n---\n# OldStub")

    created = ensure_concept_stubs_exist(["Gold", "[[US Treasury]]"], vault_root=vault_dir)
    assert len(created) == 2
    for path in created:
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "entity_type: concept_stub" in content

    # Verify old stub migrated to new Concepts folder
    new_stub_path = os.path.join(vault_dir, "30_Knowledge_Base", "Concepts", "OldStub.md")
    assert os.path.exists(new_stub_path)
    # Verify old folder News/Concepts is deleted
    assert not os.path.exists(old_dir)


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
    # Create old News/Themes dir to test legacy cleanup
    old_themes_dir = os.path.join(vault_dir, "30_Knowledge_Base", "News", "Themes")
    os.makedirs(old_themes_dir, exist_ok=True)

    # Do not ingest anything -> pending = 0
    res = run_news_funnel_synthesize(
        period="morning",
        store_path=store_file,
        vault_root=vault_dir,
        custom_date="2026-07-13",
    )

    assert res["status"] == "no_pending_events"
    assert res["published_count"] == 0
    assert not os.path.exists(old_themes_dir)


def test_synthesize_with_events_and_hitl_filter(test_paths, monkeypatch):
    monkeypatch.setenv("MOCK_NEWS_FUNNEL_LLM", "true")
    store_file, vault_dir = test_paths
    # Create legacy News/Themes dir to test cleanup during synthesis
    old_themes_dir = os.path.join(vault_dir, "30_Knowledge_Base", "News", "Themes")
    os.makedirs(old_themes_dir, exist_ok=True)

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
    assert res["published_count"] == 1
    assert len(res["created_files"]) == 1
    file_path = res["created_files"][0]
    assert os.path.exists(file_path)
    assert not os.path.exists(old_themes_dir)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "entity_type: article_note" in content
        assert "Fed rate decision and Inflation risk" in content
        assert "> **Macro Impact:** 8/10 | **Asset Impact:** 6/10" in content
        assert "## ใจความสำคัญ\nService inflation sticky" in content
        assert "## หุ้นและสินทรัพย์ที่เกี่ยวข้อง\n- NVDA, Gold" in content
        assert "## ธีมเศรษฐกิจที่เกี่ยวข้อง\n- policy, inflation" in content

    # Check state store that ev-100 is synthesized and ev-200 is rejected per HITL selection
    state = load_store(store_path=store_file)
    assert any(ev["event_id"] == "ev-100" and ev["status"] == "synthesized" for ev in state["pending_events"])
    assert any(ev["event_id"] == "ev-200" and ev["status"] == "rejected" for ev in state["pending_events"])


def test_overwrite_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_NEWS_FUNNEL_LLM", "true")
    store_file = str(tmp_path / "test_state.json")
    vault_dir = str(tmp_path / "vault")
    events = [
        {
            "event_id": "ev-A1",
            "canonical_title": "Same Title",
            "comprehensive_summary": "Sum 1",
            "macro_impact_score": 8,
            "asset_impact_score": 5,
            "is_high_impact": True,
            "extracted_tickers": ["NVDA"],
            "extracted_themes": ["policy"],
        },
        {
            "event_id": "ev-A2",
            "canonical_title": "Same Title",
            "comprehensive_summary": "Sum 2",
            "macro_impact_score": 8,
            "asset_impact_score": 5,
            "is_high_impact": True,
            "extracted_tickers": ["AAPL"],
            "extracted_themes": ["policy"],
        },
    ]
    save_triage_events(events, store_path=store_file)

    res = run_news_funnel_synthesize(
        period="morning",
        store_path=store_file,
        vault_root=vault_dir,
        custom_date="2026-07-13",
        allow_autonomous=True,
    )
    assert res["status"] == "success"
    assert len(res["created_files"]) == 2
    file1 = res["created_files"][0]
    file2 = res["created_files"][1]
    assert "2026-07-13_Same Title.md" in file1 or "2026-07-13_Same Title_2.md" in file1
    assert "2026-07-13_Same Title.md" in file2 or "2026-07-13_Same Title_2.md" in file2
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

    s_res = run_news_funnel_synthesize(
        period="morning",
        store_path=store_file,
        vault_root=vault_dir,
        custom_date="2026-07-13",
        allow_autonomous=True,
    )
    assert s_res["status"] == "success"
    assert s_res["published_count"] == 1
    assert len(s_res["created_files"]) == 1


def test_synthesize_requires_kanban_approval_and_creates_card(tmp_path, monkeypatch):
    store_file = str(tmp_path / "test_state.json")
    vault_dir = str(tmp_path / "vault")
    monkeypatch.setenv("MOCK_NEWS_FUNNEL_LLM", "true")
    monkeypatch.setenv("WEBUI_STATE_DB_PATH", str(tmp_path / "webui_state.sqlite"))

    events = [
        {
            "event_id": "ev-kanban-1",
            "canonical_title": "Kanban Test Event",
            "macro_impact_score": 8,
            "asset_impact_score": 7,
            "is_high_impact": True,
        }
    ]
    save_triage_events(events, store_path=store_file)

    res = run_news_funnel_synthesize(
        period="morning",
        store_path=store_file,
        vault_root=vault_dir,
    )
    assert res["status"] == "require_kanban_approval"
    assert res["published_count"] == 0
    assert len(res["created_files"]) == 0
    assert "Strict HITL Enforced" in res["message"]
    # ชั้น tools ไม่แตะ state_db แล้ว — คืน pending_events + period ให้ caller เป็นคน upsert การ์ด
    assert res["period"] == "morning"
    assert len(res["pending_events"]) == 1

    from api.news_funnel_cards import upsert_news_funnel_card
    upsert_news_funnel_card(res["period"], res["pending_events"])

    from contextlib import closing
    from api import state_db
    with closing(state_db.get_connection()) as conn:
        cards = state_db.list_kanban_cards(conn)
        news_card = next((c for c in cards if c["flow"] == "news_funnel"), None)
        assert news_card is not None
        assert "### 📰 รายการข่าว High-Impact ที่รอการสังเคราะห์" in news_card["prompt"]
        assert "Kanban Test Event" in news_card["prompt"]

    # upsert ซ้ำด้วย period เดิมต้องอัปเดตการ์ดเดิม ไม่สร้างการ์ดใหม่
    upsert_news_funnel_card(res["period"], res["pending_events"])
    with closing(state_db.get_connection()) as conn:
        cards = state_db.list_kanban_cards(conn)
        assert sum(1 for c in cards if c["flow"] == "news_funnel") == 1


def test_ingest_llm_failure_falls_back_to_tagged_heuristic(tmp_path, monkeypatch):
    """LLM triage ล้มเหลวทั้ง batch และรายตัว → ต้อง ingest ด้วย heuristic ที่ถูก tag ไม่ใช่ starve หรือปนกับคะแนน LLM"""
    store_file = str(tmp_path / "test_state.json")
    monkeypatch.setenv("MOCK_NEWS_FUNNEL_LLM", "false")

    with patch("core.llm_factory.get_llm", side_effect=RuntimeError("LLM outage")):
        res = run_news_funnel_ingest(
            candidates=[{"title": "Fed rate decision surprise", "link": "https://test.ex/outage", "summary": "Rates"}],
            store_path=store_file,
        )

    assert res["status"] == "success"
    assert res["ingested_count"] == 1

    state = load_store(store_path=store_file)
    ev = state["pending_events"][0]
    assert ev["triage_source"] == "heuristic_fallback"

    # การ์ด Kanban ต้องมีคำเตือน heuristic ให้ผู้รีวิวเห็น
    from tools.macro.news_funnel import format_news_funnel_card_prompt
    prompt = format_news_funnel_card_prompt("morning", [ev])
    assert "heuristic fallback" in prompt


def test_synthesize_with_approval_rejects_unselected(tmp_path, monkeypatch):
    store_file = str(tmp_path / "test_state.json")
    vault_dir = str(tmp_path / "vault")
    monkeypatch.setenv("MOCK_NEWS_FUNNEL_LLM", "true")

    events = [
        {
            "event_id": "ev-sel-1",
            "canonical_title": "Selected Event",
            "macro_impact_score": 8,
            "asset_impact_score": 7,
            "is_high_impact": True,
        },
        {
            "event_id": "ev-rej-1",
            "canonical_title": "Rejected Event",
            "macro_impact_score": 8,
            "asset_impact_score": 7,
            "is_high_impact": True,
        },
    ]
    save_triage_events(events, store_path=store_file)

    res = run_news_funnel_synthesize(
        period="morning",
        approved_event_ids=["ev-sel-1"],
        store_path=store_file,
        vault_root=vault_dir,
    )
    assert res["status"] == "success"
    assert res["published_count"] == 1
    assert res["rejected_count"] == 1

    state = load_store(store_path=store_file)
    assert any(ev["event_id"] == "ev-sel-1" and ev["status"] == "synthesized" for ev in state["pending_events"])
    assert any(ev["event_id"] == "ev-rej-1" and ev["status"] == "rejected" for ev in state["pending_events"])



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


def test_unified_cli_runner(monkeypatch, tmp_path):
    from cli import run_news_funnel
    from scripts import run_news_funnel_auto
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


def test_synthesize_empty_approval_graceful_skip(tmp_path):
    store_file = str(tmp_path / "test_state.json")
    vault_dir = str(tmp_path / "vault")
    events = [
        {
            "event_id": "ev1",
            "canonical_title": "Title 1",
            "comprehensive_summary": "Sum 1",
            "macro_impact_score": 8,
            "asset_impact_score": 5,
            "is_high_impact": True,
            "status": "pending_synthesis",
        }
    ]
    save_triage_events(events, store_path=store_file)

    res = run_news_funnel_synthesize(
        period="morning",
        approved_event_ids=[],
        store_path=store_file,
        vault_root=vault_dir,
    )
    assert res["status"] == "no_approved_events"
    assert res["rejected_count"] == 0
    assert len(res["created_files"]) == 0

    state = load_store(store_file)
    assert len(state["pending_events"]) == 1
    assert state["pending_events"][0]["status"] == "pending_synthesis"


def test_synthesize_toctou_rejection_protection(tmp_path):
    store_file = str(tmp_path / "test_state.json")
    vault_dir = str(tmp_path / "vault")
    events = [
        {"event_id": "ev-old-1", "canonical_title": "Old 1", "comprehensive_summary": "Sum 1", "macro_impact_score": 8, "is_high_impact": True, "status": "pending_synthesis"},
        {"event_id": "ev-old-2", "canonical_title": "Old 2", "comprehensive_summary": "Sum 2", "macro_impact_score": 8, "is_high_impact": True, "status": "pending_synthesis"},
        {"event_id": "ev-new-toctou", "canonical_title": "New TOCTOU", "comprehensive_summary": "Sum 3", "macro_impact_score": 9, "is_high_impact": True, "status": "pending_synthesis"},
    ]
    save_triage_events(events, store_path=store_file)

    # สมมติว่าตอนผู้ใช้เปิดหน้าจอ มีแค่ ev-old-1 และ ev-old-2 (candidate_event_ids) และผู้ใช้อนุมัติแค่ ev-old-1
    # ข่าวที่เพิ่งเข้ามาใหม่ ev-new-toctou จะต้องไม่ถูกปฏิเสธ (rejected) เพราะไม่อยู่ใน candidate_event_ids ของรอบนั้น
    res = run_news_funnel_synthesize(
        period="morning",
        approved_event_ids=["ev-old-1"],
        candidate_event_ids=["ev-old-1", "ev-old-2"],
        store_path=store_file,
        vault_root=vault_dir,
    )
    assert res["status"] == "success"
    assert res["published_count"] == 1
    assert res["rejected_count"] == 1  # เฉพาะ ev-old-2

    state = load_store(store_file)
    statuses = {e["event_id"]: e["status"] for e in state["pending_events"]}
    assert statuses["ev-old-1"] == "synthesized"
    assert statuses["ev-old-2"] == "rejected"
    assert statuses["ev-new-toctou"] == "pending_synthesis"


