import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from fastapi.testclient import TestClient

from api.main import app
from api.compatibility.equity import _is_agent_generated

client = TestClient(app)


@pytest.fixture(autouse=True)
def override_require_session():
    from api.auth import require_session
    app.dependency_overrides[require_session] = lambda: {"user_id": "mock_user"}
    yield
    app.dependency_overrides = {}


def test_is_agent_generated_filtering():
    # 1. JSON file
    assert _is_agent_generated(Path("AAPL_Latest_News.json"), "{}") is True

    # 2. Frontmatter entity_type variations
    fm_company_news = "---\nentity_type: Company_News\n---\nSome content"
    assert _is_agent_generated(Path("custom_note.md"), fm_company_news) is True

    fm_equity_analysis = "---\nentity_type: equity analysis\n---\nSome content"
    assert _is_agent_generated(Path("custom_note.md"), fm_equity_analysis) is True

    fm_generated_by = "---\ngenerated_by: news_agent\n---\nSome content"
    assert _is_agent_generated(Path("custom_note.md"), fm_generated_by) is True

    # 3. Filename patterns with space and underscore
    assert _is_agent_generated(Path("AAPL Latest News 2026-07-04.md"), "plain content") is True
    assert _is_agent_generated(Path("AAPL_Latest_News_2026-07-04.md"), "plain content") is True
    assert _is_agent_generated(Path("AAPL Equity Analysis 2026-07-04.md"), "plain content") is True
    assert _is_agent_generated(Path("AAPL_Equity_Analysis_2026-07-04.md"), "plain content") is True

    # 4. User note (should NOT be marked as agent-generated)
    user_note = "---\ntype: personal_thesis\n---\n# My AAPL Thesis\nI like AAPL"
    assert _is_agent_generated(Path("AAPL Valuation Notes.md"), user_note) is False


def test_get_equity_notes_integration(tmp_path, monkeypatch):
    import tools.archivist.core as archivist_core

    vault_path = tmp_path / "ObsidianVault"
    vault_path.mkdir()

    monkeypatch.setattr(archivist_core, "VAULT_PATH", vault_path)

    # 1. Setup Watchlist item
    wl_dir = vault_path / "20_Portfolio_Management" / "Current_Holdings" / "WatchlistItems"
    wl_dir.mkdir(parents=True)
    wl_file = wl_dir / "AAPL.md"
    wl_file.write_text("# Watchlist Thesis AAPL\nGreat moat and ecosystem.", encoding="utf-8")

    # 2. Setup Stocks dir notes
    stocks_dir = vault_path / "30_Knowledge_Base" / "Stocks" / "AAPL"
    stocks_dir.mkdir(parents=True)

    # User note
    stock_note = stocks_dir / "My_AAPL_Thesis.md"
    stock_note.write_text("# AAPL Deep Dive\nLong term holder.", encoding="utf-8")

    # ใช้วันที่ปัจจุบัน (ไม่ hardcode) — endpoint กรองด้วย cutoff "days=3 ล่าสุด" นับจากวันที่รันจริง
    # ถ้า hardcode วันที่ในอดีตไว้ เทสต์จะ fail เองเมื่อเวลาผ่านไปเกิน 3 วัน (calendar drift)
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Analysis/News note in Stocks dir (now included as a markdown note)
    agent_note = stocks_dir / f"AAPL_Latest_News_{today_str}.md"
    agent_note.write_text("---\nentity_type: Company_News\n---\nAgent news summary.", encoding="utf-8")

    # 1. Setup News dir with wikilink
    news_dir = vault_path / "30_Knowledge_Base" / "News"
    news_dir.mkdir(parents=True)
    news_note = news_dir / f"{today_str}_Apple_Earnings.md"
    news_note.write_text("Report on [[AAPL]] earnings growth.", encoding="utf-8")

    # 2. Setup YouTube_Summaries dir with hashtag
    yt_dir = vault_path / "30_Knowledge_Base" / "YouTube_Summaries"
    yt_dir.mkdir(parents=True)
    yt_note = yt_dir / f"{today_str}_Morning_Brief.md"
    yt_note.write_text("Discussion about #AAPL market outlook.", encoding="utf-8")

    # Request GET /api/equity/AAPL/notes (default days=3)
    response = client.get("/api/equity/AAPL/notes")
    assert response.status_code == 200
    data = response.json()

    assert data["ticker"] == "AAPL"
    assert data["total_count"] == 2  # News Note + YouTube Note
    items = data["items"]
    matched_types = [item["matched_by"] for item in items]
    assert "news" in matched_types
    assert "youtube" in matched_types

    titles = [item["title"] for item in items]
    assert f"{today_str}_Apple_Earnings" in titles
    assert f"{today_str}_Morning_Brief" in titles

    # Request GET /api/equity/AAPL/notes?days=0 (all time)
    response_all = client.get("/api/equity/AAPL/notes?days=0")
    assert response_all.status_code == 200
    assert response_all.json()["total_count"] == 2


def test_get_equity_note_content_and_path_traversal(tmp_path, monkeypatch):
    import tools.archivist.core as archivist_core

    vault_path = tmp_path / "ObsidianVault"
    vault_path.mkdir()

    monkeypatch.setattr(archivist_core, "VAULT_PATH", vault_path)

    # Valid note
    note_file = vault_path / "30_Knowledge_Base" / "Stocks" / "AAPL" / "Thesis.md"
    note_file.parent.mkdir(parents=True)
    note_file.write_text("Detailed content of AAPL note.", encoding="utf-8")

    rel_path = "30_Knowledge_Base/Stocks/AAPL/Thesis.md"

    # Test valid fetch
    response = client.get(f"/api/equity/notes/content?rel_path={rel_path}")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Thesis"
    assert data["content"] == "Detailed content of AAPL note."

    # Test path traversal attempts
    traversal_paths = [
        "../outside.md",
        "30_Knowledge_Base/../../outside.md",
        "/etc/passwd",
        "\\windows\\system32",
    ]
    for bad_path in traversal_paths:
        res = client.get(f"/api/equity/notes/content?rel_path={bad_path}")
        assert res.status_code in (400, 403)

    # Test non-existent file
    res = client.get("/api/equity/notes/content?rel_path=30_Knowledge_Base/Stocks/AAPL/NonExistent.md")
    assert res.status_code == 404

    # Test non-md file
    json_file = vault_path / "test.json"
    json_file.write_text("{}", encoding="utf-8")
    res = client.get("/api/equity/notes/content?rel_path=test.json")
    assert res.status_code == 400
