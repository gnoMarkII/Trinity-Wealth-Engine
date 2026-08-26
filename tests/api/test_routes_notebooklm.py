"""Unit tests สำหรับ api/routes_notebooklm.py — การ์ด flow="notebooklm" สร้างเองผ่าน Kanban ปกติ
เลือกไฟล์ Briefing Book ทีหลังผ่าน available-sources + generate(briefing_file_path)
"""
import uuid
from contextlib import closing
from pathlib import Path
import time

import pytest

from api import routes_notebooklm, state_db
import api.dependencies as dependencies
from tools.content.notebooklm import manifest as manifest_mod
from tools.content.notebooklm.models import PreflightError


@pytest.fixture(autouse=True)
def _isolate_sources_dir(tmp_path, monkeypatch):
    sources_dir = tmp_path / "NotebookLM_Sources"
    sources_dir.mkdir()
    monkeypatch.setattr(routes_notebooklm, "NOTEBOOKLM_SOURCES_DIR", sources_dir.resolve())
    monkeypatch.setattr(dependencies, "NOTEBOOKLM_SOURCES_DIR", sources_dir.resolve())
    monkeypatch.setattr(manifest_mod, "MANIFEST_DIR", tmp_path / "notebooklm_runs")
    monkeypatch.setattr(routes_notebooklm, "check_binary_available", lambda: None)
    monkeypatch.setattr(dependencies, "check_binary_available", lambda: None)
    return sources_dir


def _write_source(sources_dir: Path, filename: str, content: str = "# Briefing\n\nเนื้อหาทดสอบ") -> Path:
    p = sources_dir / filename
    p.write_text(content, encoding="utf-8")
    return p


def _create_card(*, prompt: str | None = None, flow: str = "notebooklm", title: str = "ทดสอบ") -> str:
    card_id = str(uuid.uuid4())
    with closing(state_db.get_connection()) as conn:
        state_db.create_kanban_card(conn, card_id, title, flow=flow, prompt=prompt)
    return card_id


# ── GET /api/notebooklm/available-sources ───────────────────────────────

def test_available_sources_parses_old_format_filename_as_verified(authed_client, _isolate_sources_dir):
    _write_source(_isolate_sources_dir, "2026-07-19_ฟองสบู่ AI แตกจริงหรือแค่พักฐาน.md")

    r = authed_client.get("/api/notebooklm/available-sources")
    assert r.status_code == 200
    [item] = r.json()
    assert item["title"] == "ฟองสบู่ AI แตกจริงหรือแค่พักฐาน"
    assert item["is_verified"] is True


def test_available_sources_parses_new_format_unverified_filename(authed_client, _isolate_sources_dir):
    _write_source(_isolate_sources_dir, "2026-07-24_หัวข้อร่าง_pitch-xyz_rev2_deadbeef_unverified.md")

    r = authed_client.get("/api/notebooklm/available-sources")
    [item] = r.json()
    assert item["title"] == "หัวข้อร่าง_pitch-xyz"
    assert item["is_verified"] is False


def test_available_sources_empty_when_dir_missing(authed_client):
    r = authed_client.get("/api/notebooklm/available-sources")
    assert r.json() == []


def test_available_sources_requires_auth(client):
    r = client.get("/api/notebooklm/available-sources")
    assert r.status_code == 401


# ── POST /api/notebooklm/generate ───────────────────────────────────────

def test_generate_returns_404_for_unknown_card(authed_client):
    r = authed_client.post("/api/notebooklm/generate", json={"card_id": "does-not-exist"})
    assert r.status_code == 404


def test_generate_rejects_non_notebooklm_flow_card(authed_client, _isolate_sources_dir):
    p = _write_source(_isolate_sources_dir, "2026-07-19_ไฟล์.md")
    card_id = _create_card(prompt=str(p), flow="manager")

    r = authed_client.post("/api/notebooklm/generate", json={"card_id": card_id})
    assert r.status_code == 400


def test_generate_requires_briefing_file_path_when_card_has_no_prompt_yet(authed_client):
    card_id = _create_card(prompt=None)

    r = authed_client.post("/api/notebooklm/generate", json={"card_id": card_id})
    assert r.status_code == 400
    assert "เลือกไฟล์" in r.json()["detail"]


def test_generate_rejects_when_source_file_outside_sources_dir(authed_client, tmp_path):
    outside_file = tmp_path / "secret.env"
    outside_file.write_text("API_KEY=xxx", encoding="utf-8")
    card_id = _create_card(prompt=None)

    r = authed_client.post(
        "/api/notebooklm/generate", json={"card_id": card_id, "briefing_file_path": str(outside_file)}
    )
    assert r.status_code == 400


def test_generate_rejects_when_source_file_missing(authed_client, _isolate_sources_dir):
    missing = _isolate_sources_dir / "does_not_exist.md"
    card_id = _create_card(prompt=None)

    r = authed_client.post("/api/notebooklm/generate", json={"card_id": card_id, "briefing_file_path": str(missing)})
    assert r.status_code == 400


def test_generate_returns_503_when_binary_not_available(authed_client, _isolate_sources_dir, monkeypatch):
    p = _write_source(_isolate_sources_dir, "2026-07-26_ไม่มี_binary.md")
    card_id = _create_card(prompt=str(p))

    def _raise_preflight():
        raise PreflightError("ไม่พบคำสั่ง 'notebooklm-mcp'")

    monkeypatch.setattr(routes_notebooklm, "check_binary_available", _raise_preflight)
    monkeypatch.setattr(dependencies, "check_binary_available", _raise_preflight)

    r = authed_client.post("/api/notebooklm/generate", json={"card_id": card_id})
    assert r.status_code == 503
    assert "notebooklm-mcp" in r.json()["detail"]


def test_generate_first_time_selection_persists_prompt_and_is_verified(authed_client, _isolate_sources_dir):
    p = _write_source(_isolate_sources_dir, "2026-07-27_หัวข้อร่าง_pitch-x_rev1_a1b2c3d4_unverified.md")
    card_id = _create_card(prompt=None)

    r = authed_client.post("/api/notebooklm/generate", json={"card_id": card_id, "briefing_file_path": str(p)})
    assert r.status_code == 200

    with closing(state_db.get_connection()) as conn:
        card = state_db.get_kanban_card(conn, card_id)
    assert card["prompt"] == str(p.resolve())
    assert bool(card["is_verified"]) is False


def test_generate_dispatches_moves_card_to_executing_and_returns_job_id(authed_client, _isolate_sources_dir):
    p = _write_source(_isolate_sources_dir, "2026-07-27_สร้างงานใหม่.md")
    card_id = _create_card(prompt=str(p))

    r = authed_client.post("/api/notebooklm/generate", json={"card_id": card_id})
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"]

    # fake run_fn (no-op, patched ใน conftest) จบเร็วมาก — worker อาจไล่การ์ดจาก executing ไป
    # done ได้ก่อน assert นี้จะรัน ของจริงใช้เวลาเป็นนาทีจึงไม่ racy แบบนี้
    # Poll briefly because the queue worker and the request use separate DB
    # connections; the invariant is that dispatch must leave the card out of
    # backlog, not that the worker has reached a particular terminal state.
    card = None
    for _ in range(20):
        with closing(state_db.get_connection()) as conn:
            card = state_db.get_kanban_card(conn, card_id)
        if card and card["column_name"] in ("executing", "done"):
            break
        time.sleep(0.05)
    # สำคัญคือย้ายออกจาก backlog แล้ว
    assert card["column_name"] in ("executing", "done")
    assert card["job_id"] == body["job_id"]


def test_generate_retry_ignores_briefing_file_path_uses_existing_prompt(authed_client, _isolate_sources_dir):
    """การ์ดที่เลือกไฟล์ไปแล้วครั้งหนึ่ง ต้องใช้ path เดิมเสมอ ไม่สลับไฟล์กลางคันแม้ client จะส่ง
    briefing_file_path อื่นมาด้วยความผิดพลาด — กันบั๊ก "retry แล้วเปลี่ยนไฟล์ไปเงียบๆ" """
    p = _write_source(_isolate_sources_dir, "2026-07-27_ไฟล์เดิม.md")
    other = _write_source(_isolate_sources_dir, "2026-07-27_ไฟล์อื่น.md")
    card_id = _create_card(prompt=str(p))

    authed_client.post("/api/notebooklm/generate", json={"card_id": card_id, "briefing_file_path": str(other)})

    with closing(state_db.get_connection()) as conn:
        card = state_db.get_kanban_card(conn, card_id)
    assert card["prompt"] == str(p.resolve())


def test_generate_is_idempotent_for_same_card(authed_client, _isolate_sources_dir):
    """dispatch() idempotency logic เองมีเทสต์ละเอียดที่ tests/api/test_jobs_queue.py — ที่นี่
    แค่ยืนยันว่า route ส่ง instruction/card_id/flow เข้า dispatch() ถูกต้องพอที่ idempotency key
    จะ match งานที่ seed ไว้ล่วงหน้า (seed ตรงๆ กัน race กับ worker thread ที่จบเร็วเกินจะจับจังหวะ)
    """
    p = _write_source(_isolate_sources_dir, "2026-07-28_กดซ้ำ.md")
    card_id = _create_card(prompt=str(p))
    resolved_path = str(p.resolve())
    existing_job_id = str(uuid.uuid4())
    idempotency_key = f"notebooklm:{card_id}:{resolved_path}:both"

    with closing(state_db.get_connection()) as conn:
        state_db.create_job(
            conn, existing_job_id, str(uuid.uuid4()), card_id,
            idempotency_key, resolved_path,
            status="queued", flow="notebooklm",
        )

    r = authed_client.post("/api/notebooklm/generate", json={"card_id": card_id})
    assert r.json()["job_id"] == existing_job_id


def test_generate_requires_auth(client, _isolate_sources_dir):
    p = _write_source(_isolate_sources_dir, "2026-07-19_ไฟล์.md")
    card_id = _create_card(prompt=str(p))
    r = client.post("/api/notebooklm/generate", json={"card_id": card_id})
    assert r.status_code == 401


# ── GET /api/notebooklm/status/{job_id} ─────────────────────────────────

def test_status_returns_404_for_unknown_job(authed_client):
    r = authed_client.get("/api/notebooklm/status/does-not-exist")
    assert r.status_code == 404


def test_status_merges_manifest_detail(authed_client, _isolate_sources_dir):
    p = _write_source(_isolate_sources_dir, "2026-07-30_มีรายละเอียด.md")
    content_hash = manifest_mod.compute_content_hash(p)
    m = manifest_mod.new_manifest(content_hash=content_hash, briefing_path=p)
    m.notebook_id = "nb-999"
    m.status = "completed"
    m.audio_path = "/some/path/audio.mp3"
    manifest_mod.save_manifest(m)

    card_id = _create_card(prompt=str(p))
    generate_resp = authed_client.post("/api/notebooklm/generate", json={"card_id": card_id})
    job_id = generate_resp.json()["job_id"]

    status_resp = authed_client.get(f"/api/notebooklm/status/{job_id}")
    body = status_resp.json()
    assert body["notebook_id"] == "nb-999"
    assert body["audio_path"] == "/some/path/audio.mp3"


def test_status_does_not_crash_when_source_file_deleted_after_dispatch(authed_client, _isolate_sources_dir):
    p = _write_source(_isolate_sources_dir, "2026-07-31_ถูกลบทีหลัง.md")
    card_id = _create_card(prompt=str(p))
    generate_resp = authed_client.post("/api/notebooklm/generate", json={"card_id": card_id})
    job_id = generate_resp.json()["job_id"]

    p.unlink()

    status_resp = authed_client.get(f"/api/notebooklm/status/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["audio_path"] is None


def test_status_requires_auth(client):
    r = client.get("/api/notebooklm/status/anything")
    assert r.status_code == 401
