"""Unit tests for NotebookLM Worker Discord integration and idempotency"""
import json
from pathlib import Path
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from api import state_db
from api import notebooklm_worker
from tools.content.notebooklm.models import NotebookLMRunResult


@pytest.fixture
def mock_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_state.sqlite")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    state_db.init_schema(conn)
    conn.close()

    def _get_conn(p=None):
        c = sqlite3.connect(p or db_path)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr("api.state_db.get_connection", _get_conn)
    return db_path




def test_extract_briefing_metadata(tmp_path):
    briefing = tmp_path / "2026-08-18_ทดสอบ_หัวข้อ_วิเคราะห์.md"
    content = """# 📑 ทดสอบหัวข้อวิเคราะห์พอดแคสต์

**สรุปภาพรวม:** นี่คือเนื้อหาบทสรุปของผู้บริหารที่ต้องการแสดงบน Discord

---

## Act I: บทนำ
เนื้อหาบทที่หนึ่ง...
"""
    briefing.write_text(content, encoding="utf-8")
    title, summary, source_ref = notebooklm_worker._extract_briefing_metadata(briefing, content)

    assert title == "ทดสอบหัวข้อวิเคราะห์พอดแคสต์"
    assert "**สรุปภาพรวม:** นี่คือเนื้อหาบทสรุปของผู้บริหารที่ต้องการแสดงบน Discord" in summary
    assert "Act I" not in summary
    assert source_ref == "NotebookLM_Sources/2026-08-18_ทดสอบ_หัวข้อ_วิเคราะห์.md"


def test_worker_sends_discord_when_toggle_enabled(mock_db, tmp_path, monkeypatch):
    briefing = tmp_path / "2026-08-18_test.md"
    briefing.write_text("# Test Title\nSummary text\n## Act I", encoding="utf-8")
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"audio-bytes")

    # Create job & card with discord_notify = 1
    conn = state_db.get_connection()
    state_db.create_kanban_card(conn, "card-1", "Test Card", "executing", flow="notebooklm")
    state_db.toggle_kanban_card_discord(conn, "card-1", True)
    state_db.create_job(conn, job_id="job-1", thread_id="thread-1", card_id="card-1", idempotency_key="key-1", instruction=str(briefing), flow="notebooklm")
    conn.close()

    mock_result = NotebookLMRunResult(
        notebook_id="nb-1",
        audio_path=audio,
        status="completed",
        content_hash="hash12345678",
        manifest_path=tmp_path / "manifest.json",
    )

    monkeypatch.setattr("api.notebooklm_worker.run_notebooklm_post_production_pipeline", AsyncMock(return_value=mock_result))
    
    mock_delivery = MagicMock(status="sent", message="ok")
    mock_send = MagicMock(return_value=mock_delivery)
    monkeypatch.setattr("core.discord_notifier.send_notebooklm_audio_discord", mock_send)

    notebooklm_worker.notebooklm_run_fn("job-1", "thread-1", str(briefing))

    mock_send.assert_called_once()
    call_kwargs = mock_send.call_args.kwargs
    assert call_kwargs["title"] == "Test Title"
    assert call_kwargs["source_ref"] == f"NotebookLM_Sources/{briefing.name}"

    # Verify marked in DB
    conn = state_db.get_connection()
    card = state_db.get_kanban_card(conn, "card-1")
    sent_events = json.loads(card["discord_sent_events"])
    assert "notebooklm:hash12345678" in sent_events
    conn.close()


def test_worker_skips_discord_when_toggle_disabled(mock_db, tmp_path, monkeypatch):
    briefing = tmp_path / "test_disabled.md"
    briefing.write_text("# Test Title", encoding="utf-8")
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"audio-bytes")

    conn = state_db.get_connection()
    state_db.create_kanban_card(conn, "card-2", "Test Card", "executing", flow="notebooklm")
    state_db.toggle_kanban_card_discord(conn, "card-2", False)
    state_db.create_job(conn, job_id="job-2", thread_id="thread-2", card_id="card-2", idempotency_key="key-2", instruction=str(briefing), flow="notebooklm")
    conn.close()

    mock_result = NotebookLMRunResult(
        notebook_id="nb-1",
        audio_path=audio,
        status="completed",
        content_hash="hash12345678",
        manifest_path=tmp_path / "manifest.json",
    )
    monkeypatch.setattr("api.notebooklm_worker.run_notebooklm_post_production_pipeline", AsyncMock(return_value=mock_result))
    mock_send = MagicMock()
    monkeypatch.setattr("core.discord_notifier.send_notebooklm_audio_discord", mock_send)

    notebooklm_worker.notebooklm_run_fn("job-2", "thread-2", str(briefing))

    mock_send.assert_not_called()


def test_worker_idempotency_skips_duplicate_delivery(mock_db, tmp_path, monkeypatch):
    briefing = tmp_path / "test_idempotent.md"
    briefing.write_text("# Test Title", encoding="utf-8")
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"audio-bytes")

    conn = state_db.get_connection()
    state_db.create_kanban_card(conn, "card-3", "Test Card", "executing", flow="notebooklm")
    state_db.toggle_kanban_card_discord(conn, "card-3", True)
    state_db.mark_discord_events_sent(conn, "card-3", ["notebooklm:hash12345678"])
    state_db.create_job(conn, job_id="job-3", thread_id="thread-3", card_id="card-3", idempotency_key="key-3", instruction=str(briefing), flow="notebooklm")
    conn.close()

    mock_result = NotebookLMRunResult(
        notebook_id="nb-1",
        audio_path=audio,
        status="completed",
        content_hash="hash12345678",  # Already sent hash
        manifest_path=tmp_path / "manifest.json",
    )
    monkeypatch.setattr("api.notebooklm_worker.run_notebooklm_post_production_pipeline", AsyncMock(return_value=mock_result))
    mock_send = MagicMock()
    monkeypatch.setattr("core.discord_notifier.send_notebooklm_audio_discord", mock_send)

    notebooklm_worker.notebooklm_run_fn("job-3", "thread-3", str(briefing))

    mock_send.assert_not_called()


def test_worker_discord_error_does_not_crash_job(mock_db, tmp_path, monkeypatch):
    briefing = tmp_path / "test_error.md"
    briefing.write_text("# Test Title", encoding="utf-8")
    audio = tmp_path / "test.m4a"
    audio.write_bytes(b"audio-bytes")

    conn = state_db.get_connection()
    state_db.create_kanban_card(conn, "card-4", "Test Card", "executing", flow="notebooklm")
    state_db.toggle_kanban_card_discord(conn, "card-4", True)
    state_db.create_job(conn, job_id="job-4", thread_id="thread-4", card_id="card-4", idempotency_key="key-4", instruction=str(briefing), flow="notebooklm")
    conn.close()

    mock_result = NotebookLMRunResult(
        notebook_id="nb-1",
        audio_path=audio,
        status="completed",
        content_hash="hash12345678",
        manifest_path=tmp_path / "manifest.json",
    )
    monkeypatch.setattr("api.notebooklm_worker.run_notebooklm_post_production_pipeline", AsyncMock(return_value=mock_result))
    # Notifier raises unexpected exception
    monkeypatch.setattr("core.discord_notifier.send_notebooklm_audio_discord", MagicMock(side_effect=RuntimeError("Discord API down")))

    # Must complete cleanly without raising
    notebooklm_worker.notebooklm_run_fn("job-4", "thread-4", str(briefing))



