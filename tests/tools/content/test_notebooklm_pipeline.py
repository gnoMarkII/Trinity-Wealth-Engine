"""Unit tests สำหรับ tools/content/notebooklm/ — mock adapter.call_tool ทั้งหมด ไม่แตะ notebooklm-mcp จริง

Integration test (ใช้ quota/บัญชีจริง) อยู่แยกในไฟล์ที่ mark @pytest.mark.integration เท่านั้น
"""
import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from schemas.briefing_book_schemas import NotebookLMPromptRecord
from tools.content.notebooklm import adapter, pipeline
from tools.content.notebooklm import manifest as manifest_mod
from tools.content.notebooklm.models import (
    ConfirmationRequiredError,
    PreflightError,
    StudioTerminalError,
    StudioTimeoutError,
)


def _prompt(prompt_type: str, question: str = "คำถามทดสอบ", output_format: str = "ข้อความสั้น", prompt_id: str = "P01") -> NotebookLMPromptRecord:
    return NotebookLMPromptRecord(
        prompt_id=prompt_id, prompt_type=prompt_type,
        question_or_prompt=question, expected_output_format=output_format,
    )


# ── Fake adapter harness ────────────────────────────────────────────────

class FakeAdapter:
    """แทน adapter.call_tool ด้วย response ที่ config ไว้ล่วงหน้าต่อ tool name

    response ต่อ tool name รับได้ 3 แบบ: dict คงที่, list (pop ทีละตัวตามลำดับ call),
    callable(arguments)->dict/Exception (สำหรับ logic ที่ต้องแยกตาม arguments เช่น
    studio_status ที่ถูกเรียกทั้งตอน poll และตอนขอ include_details)
    """

    def __init__(self, responses: dict):
        self._responses = responses
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, session, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        if tool_name not in self._responses:
            raise AssertionError(f"unexpected call_tool: {tool_name} args={arguments}")
        resp = self._responses[tool_name]
        if callable(resp):
            resp = resp(arguments)
        elif isinstance(resp, list):
            assert resp, f"response list exhausted for {tool_name}"
            resp = resp.pop(0)
        if inspect.isawaitable(resp):
            # รองรับ handler แบบ async def สำหรับจำลอง call ที่ค้าง/ดีเลย์จริง (asyncio.sleep)
            # ต่างจาก sync callable ทั่วไปที่ตอบกลับทันที
            resp = await resp
        if isinstance(resp, Exception):
            raise resp
        return resp

    def call_count(self, tool_name: str) -> int:
        return sum(1 for name, _ in self.calls if name == tool_name)


@asynccontextmanager
async def _fake_open_session():
    yield SimpleNamespace()


async def _noop_check_auth(session):
    return None


def _install_fake_adapter(monkeypatch, responses: dict) -> FakeAdapter:
    fake = FakeAdapter(responses)
    monkeypatch.setattr(adapter, "call_tool", fake.call_tool)
    monkeypatch.setattr(adapter, "open_session", _fake_open_session)
    monkeypatch.setattr(adapter, "check_auth", _noop_check_auth)
    monkeypatch.setattr(adapter, "check_binary_available", lambda: None)
    return fake


def _fake_download(payload: bytes = b"fake-audio-bytes"):
    def handler(arguments: dict) -> dict:
        out = Path(arguments["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        return {}
    return handler


def _studio_status_handler(poll_sequence: list[dict], detail_response: dict):
    """แยก response ตามว่าเป็น poll call (studio_status ธรรมดา) หรือ download-detail call
    (studio_status include_details=True) — เพราะ pipeline เรียก tool เดียวกันคนละจุดประสงค์
    """
    state = {"n": 0}

    def handler(arguments: dict) -> dict:
        if arguments.get("include_details"):
            return detail_response
        idx = state["n"]
        state["n"] += 1
        return poll_sequence[idx]

    return handler


def _studio_status_hang_once_then(poll_sequence: list[dict], detail_response: dict):
    """เหมือน _studio_status_handler แต่ poll call แรกสุด "ค้าง" ไม่มีวันคืนค่า (จำลอง MCP call
    ที่แฮงค์จริง) — ใช้ทดสอบว่า asyncio.wait_for ใน _poll_studio_status ตัด call ที่ค้างออกได้จริง
    """
    state = {"n": 0}

    async def handler(arguments: dict) -> dict:
        if arguments.get("include_details"):
            return detail_response
        idx = state["n"]
        state["n"] += 1
        if idx == 0:
            await asyncio.sleep(999)  # ไม่มีวันถึง — ต้องถูก asyncio.wait_for ตัดก่อนเสมอ
            return {"status": "ไม่ควรมีใครเห็นค่านี้"}
        return poll_sequence[idx - 1]

    return handler


def _happy_responses(**overrides) -> dict:
    base = {
        "server_info": {"auth_status": "configured"},
        "notebook_create": {"notebook_id": "nb-1"},
        "source_add": {"source_id": "src-1"},
        "source_describe": {"status": "success"},
        "studio_create": {"artifact_id": "art-1"},
        "studio_status": _studio_status_handler(
            poll_sequence=[{"status": "generating", "artifact_id": "art-1"}, {"status": "completed", "artifact_id": "art-1"}],
            detail_response={"status": "completed", "url": "https://example.com/audio.mp3"},
        ),
        "download_artifact": _fake_download(),
    }
    base.update(overrides)
    return base


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest_mod, "MANIFEST_DIR", tmp_path / "notebooklm_runs")
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path / "audio_out")


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch):
    monkeypatch.setattr(pipeline, "_POLL_INITIAL_INTERVAL", 0.001)
    monkeypatch.setattr(pipeline, "_POLL_MAX_INTERVAL", 0.001)
    monkeypatch.setattr(pipeline, "_STUDIO_STATUS_CALL_TIMEOUT", 0.01)


@pytest.fixture
def briefing_file(tmp_path) -> Path:
    p = tmp_path / "test_briefing.md"
    p.write_text("# Test Briefing\n\nเนื้อหาทดสอบสำหรับ NotebookLM pipeline", encoding="utf-8")
    return p


def _seed_manifest(content_hash: str, briefing_path: Path, **fields) -> manifest_mod.NotebookLMManifest:
    m = manifest_mod.new_manifest(content_hash=content_hash, briefing_path=briefing_path)
    for k, v in fields.items():
        setattr(m, k, v)
    manifest_mod.save_manifest(m)
    return m


# ── Happy path ───────────────────────────────────────────────────────────

async def test_happy_path_full_workflow(monkeypatch, briefing_file):
    fake = _install_fake_adapter(monkeypatch, _happy_responses())

    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True,
    )

    assert result.status == "completed"
    assert result.notebook_id == "nb-1"
    assert result.source_id == "src-1"
    assert result.artifact_id == "art-1"
    assert result.audio_path is not None
    assert result.audio_path.exists()
    assert fake.call_count("notebook_create") == 1
    assert fake.call_count("source_add") == 1
    assert fake.call_count("studio_create") == 1
    assert fake.call_count("research_start") == 0  # with_research=False (default)


# ── Confirmation guard — ต้องครอบทุก resume path (จุดที่ v4 เคยพลาด) ────────

async def test_confirmation_guard_blocks_fresh_run(monkeypatch, briefing_file):
    fake = _install_fake_adapter(monkeypatch, _happy_responses())

    with pytest.raises(ConfirmationRequiredError):
        await pipeline.run_notebooklm_post_production_pipeline(
            briefing_file, confirm_generation=False,
        )
    assert fake.call_count("studio_create") == 0


async def test_confirmation_guard_blocks_resume_from_source_added(monkeypatch, briefing_file):
    content_hash = manifest_mod.compute_content_hash(briefing_file)
    _seed_manifest(content_hash, briefing_file, notebook_id="nb-1", source_id="src-1", status="source_added")
    fake = _install_fake_adapter(monkeypatch, _happy_responses())

    with pytest.raises(ConfirmationRequiredError):
        await pipeline.run_notebooklm_post_production_pipeline(
            briefing_file, confirm_generation=False,
        )
    assert fake.call_count("notebook_create") == 0
    assert fake.call_count("source_add") == 0
    assert fake.call_count("studio_create") == 0


async def test_confirmation_guard_blocks_resume_from_research_done(monkeypatch, briefing_file):
    content_hash = manifest_mod.compute_content_hash(briefing_file)
    _seed_manifest(
        content_hash, briefing_file,
        notebook_id="nb-1", source_id="src-1", research_task_id="task-1",
        research_completed=True, status="research_done",
    )
    fake = _install_fake_adapter(monkeypatch, _happy_responses())

    with pytest.raises(ConfirmationRequiredError):
        await pipeline.run_notebooklm_post_production_pipeline(
            briefing_file, confirm_generation=False, with_research=True, research_query="q",
        )
    assert fake.call_count("research_start") == 0
    assert fake.call_count("studio_create") == 0


# ── Resume dispatching ───────────────────────────────────────────────────

async def test_resume_from_notebook_created_skips_notebook_create(monkeypatch, briefing_file):
    content_hash = manifest_mod.compute_content_hash(briefing_file)
    _seed_manifest(content_hash, briefing_file, notebook_id="nb-1", status="notebook_created")
    fake = _install_fake_adapter(monkeypatch, _happy_responses())

    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True,
    )
    assert result.status == "completed"
    assert fake.call_count("notebook_create") == 0
    assert fake.call_count("source_add") == 1


async def test_resume_from_audio_generating_skips_studio_create(monkeypatch, briefing_file):
    content_hash = manifest_mod.compute_content_hash(briefing_file)
    _seed_manifest(
        content_hash, briefing_file,
        notebook_id="nb-1", source_id="src-1", artifact_id="art-1", status="audio_generating",
    )
    fake = _install_fake_adapter(monkeypatch, _happy_responses())

    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True,
    )
    assert result.status == "completed"
    assert fake.call_count("notebook_create") == 0
    assert fake.call_count("source_add") == 0
    assert fake.call_count("studio_create") == 0  # ไม่สร้าง audio job ซ้ำ


async def test_idempotent_when_already_completed(monkeypatch, briefing_file, tmp_path):
    content_hash = manifest_mod.compute_content_hash(briefing_file)
    audio_path = tmp_path / "existing_audio.mp3"
    audio_path.write_bytes(b"already-downloaded")
    _seed_manifest(
        content_hash, briefing_file,
        notebook_id="nb-1", source_id="src-1", artifact_id="art-1",
        audio_path=str(audio_path), status="completed",
    )
    fake = _install_fake_adapter(monkeypatch, {})  # ห้ามมี call ใดๆ เกิดขึ้นเลย

    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=False,
    )
    assert result.status == "completed"
    assert result.audio_path == audio_path
    assert fake.calls == []


# ── Studio terminal failure -> rollback ที่ถูกต้อง (ไม่ลืมว่า research เสร็จไปแล้ว) ──

async def test_studio_terminal_failure_rolls_back_to_source_added(monkeypatch, briefing_file):
    fake = _install_fake_adapter(monkeypatch, _happy_responses(
        studio_status=lambda args: {"status": "failed", "artifact_id": "art-1"},
        # recovery-download ต้องล้มเหลวด้วย ไม่งั้น pipeline จะกู้คืนสำเร็จแทนที่จะ rollback
        # (ดู test_studio_terminal_failure_recovers_via_direct_download สำหรับกรณีกู้คืนสำเร็จ)
        download_artifact=RuntimeError("recovery download ก็ล้มเหลวเหมือนกัน"),
    ))

    with pytest.raises(StudioTerminalError):
        await pipeline.run_notebooklm_post_production_pipeline(
            briefing_file, confirm_generation=True,
        )

    content_hash = manifest_mod.compute_content_hash(briefing_file)
    saved = manifest_mod.load_manifest(manifest_mod.manifest_path_for(content_hash))
    assert saved.status == "source_added"
    assert saved.artifact_id is None


async def test_studio_terminal_failure_rolls_back_to_research_done_not_source_added(monkeypatch, briefing_file):
    content_hash = manifest_mod.compute_content_hash(briefing_file)
    _seed_manifest(
        content_hash, briefing_file,
        notebook_id="nb-1", source_id="src-1", research_task_id="task-1",
        research_completed=True, status="research_done",
    )
    fake = _install_fake_adapter(monkeypatch, _happy_responses(
        studio_status=lambda args: {"status": "failed", "artifact_id": "art-1"},
        download_artifact=RuntimeError("recovery download ก็ล้มเหลวเหมือนกัน"),
    ))

    with pytest.raises(StudioTerminalError):
        await pipeline.run_notebooklm_post_production_pipeline(
            briefing_file, confirm_generation=True, with_research=True, research_query="q",
        )

    saved = manifest_mod.load_manifest(manifest_mod.manifest_path_for(content_hash))
    assert saved.status == "research_done"  # ไม่ใช่ source_added — ต้องจำว่า research เสร็จแล้ว
    assert saved.artifact_id is None

    # resume ครั้งถัดไปต้อง "ไม่" เรียก research ซ้ำ เพราะ research_task_id ยังอยู่ + status=research_done
    fake2 = _install_fake_adapter(monkeypatch, _happy_responses())
    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True, with_research=True, research_query="q",
    )
    assert result.status == "completed"
    assert fake2.call_count("research_start") == 0


# ── studio_create ไม่คืน artifact_id (เจอจริง #AG-49 รอบสอง) ──────────────────────
# ถ้าปล่อยผ่านเงียบๆ manifest จะเข้าสถานะ audio_generating ทั้งที่ artifact_id เป็น None แล้ว
# _poll_studio_status จะพังด้วย error ที่เข้าใจผิดว่า "เช็คสถานะไม่ได้" ทั้งที่ไม่เคยได้ id มาแต่แรก

async def test_generate_audio_raises_when_studio_create_returns_no_artifact_id(monkeypatch, briefing_file):
    _install_fake_adapter(monkeypatch, _happy_responses(
        studio_create={"status": "error", "error": "Could not retrieve studio status."},
    ))

    with pytest.raises(StudioTerminalError, match="artifact_id"):
        await pipeline.run_notebooklm_post_production_pipeline(
            briefing_file, confirm_generation=True,
        )

    content_hash = manifest_mod.compute_content_hash(briefing_file)
    saved = manifest_mod.load_manifest(manifest_mod.manifest_path_for(content_hash))
    assert saved.status == "source_added"  # ไม่ถูกเลื่อนไป audio_generating เลย
    assert saved.artifact_id is None


async def test_resume_after_missing_artifact_id_retries_studio_create_only(monkeypatch, briefing_file):
    _install_fake_adapter(monkeypatch, _happy_responses(studio_create={"error": "boom"}))
    with pytest.raises(StudioTerminalError):
        await pipeline.run_notebooklm_post_production_pipeline(briefing_file, confirm_generation=True)

    fake2 = _install_fake_adapter(monkeypatch, _happy_responses())
    result = await pipeline.run_notebooklm_post_production_pipeline(briefing_file, confirm_generation=True)

    assert result.status == "completed"
    assert fake2.call_count("notebook_create") == 0  # ไม่สร้าง notebook/source ซ้ำ
    assert fake2.call_count("source_add") == 0
    assert fake2.call_count("studio_create") == 1


# ── Recovery: studio_status พังทั้งที่ Audio สร้างเสร็จจริงแล้ว (เจอจริง #AG-49) ──────
# studio_status ตอบ status="error" พร้อมข้อความ "Could not retrieve studio status" — หมายถึง
# "เช็คสถานะไม่ได้" ไม่ใช่ "generation ล้มเหลว" — ต้องลอง download ตรงๆ ก่อน rollback+regenerate

async def test_studio_terminal_failure_recovers_via_direct_download(monkeypatch, briefing_file):
    fake = _install_fake_adapter(monkeypatch, _happy_responses(
        studio_status=lambda args: {"status": "error", "error": "Could not retrieve studio status."},
    ))

    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True,
    )

    assert result.status == "completed"
    assert result.audio_path is not None
    assert fake.call_count("studio_create") == 1  # ไม่ได้ regenerate ใหม่ ใช้ artifact_id เดิม


async def test_recovery_download_falls_back_to_default_extension_when_status_check_fails(monkeypatch, briefing_file):
    """_download_audio เองก็เรียก studio_status(include_details=True) เพื่อเดานามสกุลไฟล์ —
    ถ้า call นั้นพังแบบเดียวกันด้วย ต้อง fallback เป็นนามสกุล default แทนที่จะ abort ทั้งการดาวน์โหลด
    """
    def _studio_status(arguments: dict) -> dict:
        return {"status": "error", "error": "Could not retrieve studio status."}

    fake = _install_fake_adapter(monkeypatch, _happy_responses(studio_status=_studio_status))

    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True,
    )

    assert result.status == "completed"
    assert result.audio_path.suffix == ".m4a"  # fallback default เพราะเช็คนามสกุลจริงไม่ได้


async def test_studio_terminal_failure_recovery_fails_falls_back_to_rollback(monkeypatch, briefing_file):
    """ถ้าทั้ง status-check และ recovery-download ล้มเหลวทั้งคู่ ต้อง rollback+raise เหมือนเดิม
    ไม่ใช่กลืน error เงียบๆ หรือค้างอยู่ในสถานะกำกวม
    """
    fake = _install_fake_adapter(monkeypatch, _happy_responses(
        studio_status=lambda args: {"status": "error", "error": "Could not retrieve studio status."},
        download_artifact=RuntimeError("network error ระหว่างดาวน์โหลดจริงด้วย"),
    ))

    with pytest.raises(StudioTerminalError):
        await pipeline.run_notebooklm_post_production_pipeline(
            briefing_file, confirm_generation=True,
        )

    content_hash = manifest_mod.compute_content_hash(briefing_file)
    saved = manifest_mod.load_manifest(manifest_mod.manifest_path_for(content_hash))
    assert saved.status == "source_added"
    assert saved.artifact_id is None


# ── Deep Research: research_status บล็อกในตัวเอง ไม่ต้อง poll loop ฝั่งเรา ──────

async def test_research_uses_native_blocking_not_manual_poll_loop(monkeypatch, briefing_file):
    fake = _install_fake_adapter(monkeypatch, _happy_responses(
        research_start={"task_id": "task-1"},
        research_status={"status": "completed"},  # single response พอ — ถ้า pipeline poll loop เอง จะ error (list ว่าง)
        research_import={},
    ))

    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True, with_research=True, research_query="quantum computing",
    )
    assert result.status == "completed"
    assert fake.call_count("research_start") == 1
    assert fake.call_count("research_status") == 1
    assert fake.call_count("research_import") == 1
    # ยืนยันว่า max_wait ถูกส่งไปให้ tool จัดการบล็อกเอง แทนที่เราจะ loop เรียกซ้ำ
    research_status_call = next(c for c in fake.calls if c[0] == "research_status")
    assert "max_wait" in research_status_call[1]


async def test_with_research_false_never_calls_research_tools(monkeypatch, briefing_file):
    fake = _install_fake_adapter(monkeypatch, _happy_responses())
    await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True, with_research=False,
    )
    assert fake.call_count("research_start") == 0
    assert fake.call_count("research_status") == 0
    assert fake.call_count("research_import") == 0


async def test_with_research_requires_query():
    with pytest.raises(ValueError, match="research_query"):
        await pipeline.run_notebooklm_post_production_pipeline(
            Path("irrelevant.md"), confirm_generation=True, with_research=True, research_query=None,
        )


# ── NotebookLM Prompts: notebook_query ต่อ prompt + auto-enable research ────

async def test_notebooklm_prompts_all_types_sent_to_notebook_query(monkeypatch, briefing_file):
    fake = _install_fake_adapter(monkeypatch, _happy_responses(notebook_query={}))
    prompts = [
        _prompt("SOCRATIC", "คำถามที่ 1", prompt_id="P01"),
        _prompt("BLIND_SPOT", "คำถามที่ 2", prompt_id="P02"),
        _prompt("FEYNMAN", "คำถามที่ 3", prompt_id="P03"),
    ]

    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True, notebooklm_prompts=prompts,
    )

    assert result.status == "completed"
    assert fake.call_count("notebook_query") == 3
    queries = [c[1]["query"] for c in fake.calls if c[0] == "notebook_query"]
    assert "คำถามที่ 1" in queries[0]
    assert "คำถามที่ 2" in queries[1]
    assert "คำถามที่ 3" in queries[2]


async def test_notebooklm_prompts_with_research_tag_auto_enables_deep_research(monkeypatch, briefing_file):
    """[RESEARCH] prompt ในไฟล์ต้องเปิด Deep Research เองโดยไม่ต้องส่ง with_research=True มา"""
    fake = _install_fake_adapter(monkeypatch, _happy_responses(
        research_start={"task_id": "task-1"},
        research_status={"status": "completed"},
        research_import={},
        notebook_query={},
    ))
    prompts = [_prompt("RESEARCH", "วิเคราะห์ตัวเลข X", prompt_id="P01")]

    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True, notebooklm_prompts=prompts,
    )

    assert result.status == "completed"
    assert fake.call_count("research_start") == 1
    research_start_call = next(c for c in fake.calls if c[0] == "research_start")
    assert "วิเคราะห์ตัวเลข X" in research_start_call[1]["query"]


async def test_explicit_research_query_takes_precedence_over_prompts(monkeypatch, briefing_file):
    """caller ระบุ research_query เองชัดเจน (เช่น CLI flag) ต้องใช้ค่านั้น ไม่ทับด้วยที่ derive จากไฟล์"""
    fake = _install_fake_adapter(monkeypatch, _happy_responses(
        research_start={"task_id": "task-1"},
        research_status={"status": "completed"},
        research_import={},
        notebook_query={},
    ))
    prompts = [_prompt("RESEARCH", "คำถามจากไฟล์", prompt_id="P01")]

    await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True, with_research=True,
        research_query="คำถามที่ผู้ใช้ระบุเอง", notebooklm_prompts=prompts,
    )

    research_start_call = next(c for c in fake.calls if c[0] == "research_start")
    assert research_start_call[1]["query"] == "คำถามที่ผู้ใช้ระบุเอง"


async def test_no_notebooklm_prompts_never_calls_notebook_query(monkeypatch, briefing_file):
    fake = _install_fake_adapter(monkeypatch, _happy_responses())
    await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True,
    )
    assert fake.call_count("notebook_query") == 0


async def test_resume_from_prompts_queried_skips_notebook_query(monkeypatch, briefing_file):
    content_hash = manifest_mod.compute_content_hash(briefing_file)
    _seed_manifest(
        content_hash, briefing_file,
        notebook_id="nb-1", source_id="src-1", prompts_queried=True, status="prompts_queried",
    )
    fake = _install_fake_adapter(monkeypatch, _happy_responses())

    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True, notebooklm_prompts=[_prompt("SOCRATIC")],
    )
    assert result.status == "completed"
    assert fake.call_count("notebook_query") == 0


async def test_studio_terminal_failure_rolls_back_to_prompts_queried(monkeypatch, briefing_file):
    """ถ้าถาม prompts ไปแล้วแต่ audio generation ล้มเหลว ต้องจำว่าถาม prompts ไปแล้ว ไม่ถามซ้ำตอน retry"""
    content_hash = manifest_mod.compute_content_hash(briefing_file)
    _seed_manifest(
        content_hash, briefing_file,
        notebook_id="nb-1", source_id="src-1", prompts_queried=True, status="prompts_queried",
    )
    fake = _install_fake_adapter(monkeypatch, _happy_responses(
        studio_status=lambda args: {"status": "failed", "artifact_id": "art-1"},
        download_artifact=RuntimeError("recovery download ก็ล้มเหลวเหมือนกัน"),
    ))

    with pytest.raises(StudioTerminalError):
        await pipeline.run_notebooklm_post_production_pipeline(
            briefing_file, confirm_generation=True, notebooklm_prompts=[_prompt("SOCRATIC")],
        )

    saved = manifest_mod.load_manifest(manifest_mod.manifest_path_for(content_hash))
    assert saved.status == "prompts_queried"
    assert saved.artifact_id is None

    fake2 = _install_fake_adapter(monkeypatch, _happy_responses())
    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True, notebooklm_prompts=[_prompt("SOCRATIC")],
    )
    assert result.status == "completed"
    assert fake2.call_count("notebook_query") == 0


# ── Resilience: studio_status call ค้าง / response ไม่รู้จัก (เจอจริงจาก #AG-47/#AG-49) ──

async def test_poll_studio_status_recovers_from_single_call_hang(monkeypatch, briefing_file):
    """call เดียวค้าง (เช่น subprocess/network แฮงค์) ต้องไม่บล็อกไม่มีที่สิ้นสุด — asyncio.wait_for
    ต้องตัดแล้วให้ loop ลองรอบถัดไปแทน ไม่ใช่ค้างจนไม่มีวันถึง timeout_seconds รวมเลย
    """
    _install_fake_adapter(monkeypatch, _happy_responses(
        studio_status=_studio_status_hang_once_then(
            poll_sequence=[{"status": "completed", "artifact_id": "art-1"}],
            detail_response={"status": "completed", "url": "https://example.com/audio.mp3"},
        ),
    ))
    result = await asyncio.wait_for(
        pipeline.run_notebooklm_post_production_pipeline(briefing_file, confirm_generation=True),
        timeout=5,  # กันเทสต์เองค้างถ้า wait_for ใน pipeline ไม่ทำงานจริง
    )
    assert result.status == "completed"


async def test_poll_studio_status_continues_and_logs_on_unrecognized_status(monkeypatch, briefing_file, caplog):
    """response ที่ไม่มี status ตรงกับที่รู้จักเลย (ไม่ completed/failed) ต้องไม่ทำให้ pipeline พัง —
    แค่ log raw response ไว้ debug แล้ว poll ต่อ (notebooklm-mcp เป็น internal API ไม่มีเอกสาร
    โครงสร้าง response จริงอาจต่างจากที่ _extract_artifact_state คาดไว้)
    """
    caplog.set_level(logging.WARNING, logger="tools.content.notebooklm.pipeline")
    _install_fake_adapter(monkeypatch, _happy_responses(
        studio_status=_studio_status_handler(
            poll_sequence=[{"status": "some_unrecognized_value", "artifact_id": "art-1"}, {"status": "completed", "artifact_id": "art-1"}],
            detail_response={"status": "completed", "url": "https://example.com/audio.mp3"},
        ),
    ))
    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True,
    )
    assert result.status == "completed"
    assert "ไม่มี status ที่รู้จัก" in caplog.text
    assert "some_unrecognized_value" in caplog.text


# ── Dynamic extension + collision prevention ────────────────────────────

async def test_download_uses_real_extension_and_hash_prefix(monkeypatch, briefing_file):
    _install_fake_adapter(monkeypatch, _happy_responses(
        studio_status=_studio_status_handler(
            poll_sequence=[{"status": "completed", "artifact_id": "art-1"}],
            detail_response={"status": "completed", "url": "https://cdn.example.com/x/audio-final.mp3?sig=abc"},
        ),
    ))
    content_hash = manifest_mod.compute_content_hash(briefing_file)

    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True,
    )

    assert result.audio_path.suffix == ".mp3"
    assert content_hash[:8] in result.audio_path.name
    assert result.audio_path.stem.startswith("test_briefing_")


async def test_download_falls_back_to_m4a_when_no_url_found(monkeypatch, briefing_file):
    _install_fake_adapter(monkeypatch, _happy_responses(
        studio_status=_studio_status_handler(
            poll_sequence=[{"status": "completed", "artifact_id": "art-1"}],
            detail_response={"status": "completed"},  # ไม่มี url ให้ parse เลย
        ),
    ))
    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True,
    )
    assert result.audio_path.suffix == ".m4a"


# ── Download failure ─────────────────────────────────────────────────────

async def test_download_empty_file_raises_and_cleans_up_temp(monkeypatch, briefing_file, tmp_path):
    _install_fake_adapter(monkeypatch, _happy_responses(
        download_artifact=_fake_download(payload=b""),
    ))
    with pytest.raises(RuntimeError, match="ว่างเปล่า"):
        await pipeline.run_notebooklm_post_production_pipeline(
            briefing_file, confirm_generation=True,
        )
    leftover_temp_files = list((tmp_path / "audio_out").glob(".tmp_*"))
    assert leftover_temp_files == []


# ── Source add / MCP error handling ─────────────────────────────────────

async def test_source_add_error_propagates_and_preserves_notebook_id(monkeypatch, briefing_file):
    _install_fake_adapter(monkeypatch, _happy_responses(
        source_add=RuntimeError("MCP tool 'source_add' คืน error: upload failed"),
    ))
    with pytest.raises(RuntimeError, match="upload failed"):
        await pipeline.run_notebooklm_post_production_pipeline(
            briefing_file, confirm_generation=True,
        )
    content_hash = manifest_mod.compute_content_hash(briefing_file)
    saved = manifest_mod.load_manifest(manifest_mod.manifest_path_for(content_hash))
    assert saved.notebook_id == "nb-1"  # ขั้นก่อนหน้าที่สำเร็จแล้วต้องไม่หาย — resume ต่อได้โดยไม่สร้าง notebook ซ้ำ
    assert saved.source_id is None


# ── Preflight ─────────────────────────────────────────────────────────────

async def test_preflight_auth_not_configured_raises(monkeypatch, briefing_file):
    async def _bad_auth(session):
        raise PreflightError("NotebookLM auth ยังไม่พร้อม (auth_status='stale') — รัน `nlm login` ก่อน")

    fake = FakeAdapter(_happy_responses())
    monkeypatch.setattr(adapter, "call_tool", fake.call_tool)
    monkeypatch.setattr(adapter, "open_session", _fake_open_session)
    monkeypatch.setattr(adapter, "check_auth", _bad_auth)
    monkeypatch.setattr(adapter, "check_binary_available", lambda: None)

    with pytest.raises(PreflightError, match="nlm login"):
        await pipeline.run_notebooklm_post_production_pipeline(
            briefing_file, confirm_generation=True,
        )
    assert fake.call_count("notebook_create") == 0


# ── Adapter decode logic (structuredContent / text-JSON / error) ────────

class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeCallToolResult:
    def __init__(self, *, content=None, structuredContent=None, isError=False):
        self.content = content or []
        self.structuredContent = structuredContent
        self.isError = isError


def test_decode_result_prefers_structured_content():
    result = _FakeCallToolResult(structuredContent={"notebook_id": "nb-1"})
    assert adapter._decode_result("notebook_create", result) == {"notebook_id": "nb-1"}


def test_decode_result_parses_json_text_when_no_structured_content():
    result = _FakeCallToolResult(content=[_FakeTextBlock('{"source_id": "src-1"}')])
    assert adapter._decode_result("source_add", result) == {"source_id": "src-1"}


def test_decode_result_wraps_non_json_text():
    result = _FakeCallToolResult(content=[_FakeTextBlock("plain text, not json")])
    assert adapter._decode_result("studio_status", result) == {"raw_text": "plain text, not json"}


def test_decode_result_raises_on_is_error():
    result = _FakeCallToolResult(content=[_FakeTextBlock("quota exceeded")], isError=True)
    with pytest.raises(RuntimeError, match="quota exceeded"):
        adapter._decode_result("studio_create", result)


# ── Manifest validation ───────────────────────────────────────────────────

def test_load_manifest_returns_never_seen_for_missing_file(tmp_path):
    res = manifest_mod.load_manifest(tmp_path / "does_not_exist.json")
    assert res.load_status == manifest_mod.ManifestStatus.NEVER_SEEN
    assert res.manifest is None
    assert res.is_resolved is False


def test_load_manifest_returns_corrupt_for_corrupt_json(tmp_path):
    bad = tmp_path / "corrupt.json"
    bad.write_text("{not valid json", encoding="utf-8")
    res = manifest_mod.load_manifest(bad)
    assert res.load_status == manifest_mod.ManifestStatus.CORRUPT
    assert res.manifest is None
    assert res.is_corrupt is True


def test_load_manifest_returns_unsupported_version_for_schema_mismatch(tmp_path):
    bad = tmp_path / "schema_mismatch.json"
    bad.write_text('{"unexpected_field_only": true}', encoding="utf-8")  # ขาด required fields
    res = manifest_mod.load_manifest(bad)
    assert res.load_status == manifest_mod.ManifestStatus.UNSUPPORTED_VERSION
    assert res.manifest is None
    assert res.is_corrupt is True


# ── on_step callback: ให้ caller (api/notebooklm_worker.py) เขียน job_logs ได้เอง ──────
# pipeline.py เองไม่รู้จัก state_db เลย (tools/ ห้าม import api/ ผิดชั้นสถาปัตยกรรม)

async def test_on_step_called_at_each_checkpoint_in_order(monkeypatch, briefing_file):
    _install_fake_adapter(monkeypatch, _happy_responses(
        research_start={"task_id": "task-1"},
        research_status={"status": "completed"},
        research_import={},
        notebook_query={},
    ))
    steps: list[tuple[str, str]] = []

    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True,
        notebooklm_prompts=[_prompt("RESEARCH", "วิเคราะห์ X")],
        on_step=lambda node, message: steps.append((node, message)),
    )

    assert result.status == "completed"
    assert [node for node, _ in steps] == [
        "notebook_create", "source_add", "research", "notebook_prompts", "studio_create", "download",
    ]
    assert all(message for _, message in steps)  # ทุกข้อความไม่ว่างเปล่า


async def test_on_step_not_called_for_skipped_steps_on_resume(monkeypatch, briefing_file):
    content_hash = manifest_mod.compute_content_hash(briefing_file)
    _seed_manifest(
        content_hash, briefing_file,
        notebook_id="nb-1", source_id="src-1", artifact_id="art-1", status="audio_generating",
    )
    _install_fake_adapter(monkeypatch, _happy_responses())
    steps: list[str] = []

    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True,
        on_step=lambda node, message: steps.append(node),
    )

    assert result.status == "completed"
    # ทุก step ก่อนหน้าถูก resume ข้าม (มี field ใน manifest ครบแล้ว) — ต้องไม่ log ซ้ำ
    assert steps == ["download"]


async def test_pipeline_works_without_on_step_callback(monkeypatch, briefing_file):
    """caller ไม่ส่ง on_step มาเลย (เช่น CLI) — ต้องไม่ raise, ใช้ no-op default"""
    _install_fake_adapter(monkeypatch, _happy_responses())
    result = await pipeline.run_notebooklm_post_production_pipeline(
        briefing_file, confirm_generation=True,
    )
    assert result.status == "completed"
