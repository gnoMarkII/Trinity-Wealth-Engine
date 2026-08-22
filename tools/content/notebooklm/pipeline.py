"""Pipeline Orchestration — นำ Briefing Book เข้า NotebookLM แล้วสร้าง Audio Overview (Podcast)

Resume: ทุกขั้นตอน guard ด้วย `if not manifest.<field>` ก่อนเรียก MCP — ถ้า field นั้นถูกบันทึกไว้
จาก run ก่อนหน้าแล้ว (manifest โหลดจาก data/notebooklm_runs/<content_hash>.json) จะข้ามไปเลย ทำให้
resume ได้จากจุดที่ค้างจริง ไม่ต้องพึ่ง status string เป็นตัวตัดสินเส้นทาง (status ใช้เป็น audit trail)

Confirmation Gate: `_generate_audio()` เป็นจุดเดียวในทั้งไฟล์ที่เรียก studio_create ได้ — ทุก path ที่
ต้องไปสร้าง audio (ไม่ว่าจะมาจาก linear flow หรือ resume จาก source_added/research_done) ต้องผ่าน
ฟังก์ชันนี้เท่านั้น ซึ่งเช็ค confirm_generation ก่อนเสมอ ป้องกัน bug เดิมใน v4 ที่ resume dispatch
ข้าม confirmation gate ไปได้
"""
import asyncio
import os
from pathlib import Path
from typing import Callable

from langsmith import traceable

from core.logger import get_logger
from schemas.briefing_book_schemas import NotebookLMPromptRecord
from tools.content.notebooklm import adapter
from tools.content.notebooklm.manifest import (
    NotebookLMManifest,
    compute_content_hash,
    load_manifest,
    manifest_path_for,
    new_manifest,
    save_manifest,
)
from tools.content.notebooklm.models import (
    ConfirmationRequiredError,
    NotebookLMRunResult,
    StudioTerminalError,
    StudioTimeoutError,
)
from tools.content.notebooklm.prompts import build_notebook_query, build_research_query

logger = get_logger(__name__)

OUTPUT_DIR = Path("memories/30_Knowledge_Base/NotebookLM_Audio")

_POLL_INITIAL_INTERVAL = 10
_POLL_MAX_INTERVAL = 60
_TERMINAL_SUCCESS = {"completed"}
_TERMINAL_FAILURE = {"failed", "error"}


def _validate_input(briefing_file_path: Path) -> Path:
    p = Path(briefing_file_path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์ briefing: {p}")
    if p.suffix.lower() != ".md":
        raise ValueError(f"briefing_file_path ต้องเป็นไฟล์ .md: {p}")
    if p.stat().st_size == 0:
        raise ValueError(f"ไฟล์ briefing ว่างเปล่า: {p}")
    return p


def _result_from_manifest(manifest: NotebookLMManifest, manifest_path: Path) -> NotebookLMRunResult:
    return NotebookLMRunResult(
        notebook_id=manifest.notebook_id,
        source_id=manifest.source_id,
        artifact_id=manifest.artifact_id,
        audio_path=Path(manifest.audio_path) if manifest.audio_path else None,
        status=manifest.status,
        manifest_path=manifest_path,
        content_hash=manifest.content_hash,
    )


async def _run_research(
    session,
    manifest: NotebookLMManifest,
    research_query: str,
    timeout_seconds: int,
    research_mode: str = "deep",
) -> NotebookLMManifest:
    if not manifest.research_task_id:
        resp = await adapter.call_tool(session, "research_start", {
            "query": research_query,
            "notebook_id": manifest.notebook_id,
            "mode": research_mode,
        })
        if resp.get("status") == "no_research" or not resp.get("task_id"):
            logger.info(f"[NotebookLM Pipeline] ข้ามขั้นตอน Deep research: {resp}")
            manifest.research_completed = True
            manifest.status = "research_done"
            return manifest
        manifest.research_task_id = resp.get("task_id")
        save_manifest(manifest)

    # research_status บล็อกในตัวเอง (มี poll_interval/max_wait ของมันเอง) — ไม่ต้องเขียน poll loop เอง
    status_resp = await adapter.call_tool(session, "research_status", {
        "notebook_id": manifest.notebook_id,
        "task_id": manifest.research_task_id,
        "max_wait": timeout_seconds,
    })
    
    status_str = status_resp.get("status")
    error_str = status_resp.get("error", "")
    
    if status_str == "no_research" or (status_str == "error" and "NOT_FOUND" in str(error_str)):
        logger.info(f"[NotebookLM Pipeline] สถานะ Deep research แจ้งว่าข้ามงาน/ไม่พบงาน (no_research/NOT_FOUND): {status_resp}")
        manifest.research_completed = True
        manifest.status = "research_done"
        return manifest
        
    if status_str != "completed":
        raise StudioTerminalError(f"Deep research ไม่สำเร็จ: {status_resp}")

    await adapter.call_tool(session, "research_import", {
        "notebook_id": manifest.notebook_id,
        "task_id": manifest.research_task_id,
        "cited_only": True,
    })
    manifest.research_completed = True
    manifest.status = "research_done"
    return manifest


async def _run_notebook_prompts(
    session, manifest: NotebookLMManifest, prompts: list[NotebookLMPromptRecord],
) -> NotebookLMManifest:
    """ถามคำถามทั้งหมดจาก section NotebookLM Prompts ของ briefing (ทั้ง 4 ประเภท: BLIND_SPOT,
    SOCRATIC, FEYNMAN, RESEARCH) ผ่าน notebook_query — ไม่สนใจคำตอบ จุดประสงค์คือสร้าง
    context/chat history ให้ notebook ก่อนสร้าง Audio Overview ไม่ใช่เก็บผลลัพธ์กลับมาใช้ต่อ
    """
    for prompt in prompts:
        try:
            await adapter.call_tool(session, "notebook_query", {
                "notebook_id": manifest.notebook_id,
                "query": build_notebook_query(prompt),
            })
        except Exception as e:
            logger.warning(
                f"[NotebookLM Pipeline] notebook_query ล้มเหลว (ข้ามคำถามนี้เนื่องจากจุดประสงค์เพื่อสร้าง context เท่านั้น): {e}"
            )
            continue
            
    manifest.prompts_queried = True
    manifest.status = "prompts_queried"
    return manifest


async def _generate_audio(
    session,
    manifest: NotebookLMManifest,
    *,
    confirm_generation: bool,
    audio_language: str,
) -> NotebookLMManifest:
    """จุดเดียวที่เรียก studio_create ได้ — เช็ค confirm_generation ก่อนเสมอไม่ว่าจะมาจาก resume path ไหน"""
    if not confirm_generation:
        raise ConfirmationRequiredError(
            "Audio generation requires confirm_generation=True — "
            f"resume ค้างอยู่ที่ status={manifest.status}, ยังไม่เคยยืนยัน"
        )
    for attempt in range(60):
        try:
            resp = await adapter.call_tool(session, "studio_create", {
                "notebook_id": manifest.notebook_id,
                "artifact_type": "audio",
                "confirm": True,
                "language": audio_language,
            })
            break
        except Exception as e:
            if attempt < 59 and "Could not retrieve notebook sources" in str(e):
                logger.warning(
                    "[NotebookLM Pipeline] Google API ยังเตรียม Source ไม่เสร็จ รอ 5 วินาทีแล้วลองใหม่ (attempt %d/60)...",
                    attempt + 1,
                )
                import asyncio
                await asyncio.sleep(5)
                continue
            raise
    artifact_id = resp.get("artifact_id")
    if not artifact_id:
        # เจอจริง #AG-49 รอบสอง: studio_create คืน response ที่ไม่มี artifact_id เลย (เช่น
        # error-shaped response แทนที่จะเป็น artifact ใหม่) — ถ้าปล่อยผ่านเงียบๆ manifest จะเข้า
        # สถานะ "audio_generating" ทั้งที่ artifact_id เป็น None แล้ว _poll_studio_status จะพังทันที
        # ด้วย error ที่ทำให้เข้าใจผิดว่า "เช็คสถานะไม่ได้" ทั้งที่จริงคือไม่เคยได้ artifact_id มาตั้งแต่แรก
        # ต้อง raise ก่อน mutate manifest เพื่อไม่ให้ resume ครั้งหน้าค้างอยู่กับ artifact_id ปลอม
        logger.warning(
            "[NotebookLM Pipeline] studio_create ไม่คืน artifact_id ที่ใช้งานได้ (notebook_id=%s): %s",
            manifest.notebook_id, resp,
        )
        raise StudioTerminalError(f"studio_create ไม่คืน artifact_id ที่ใช้งานได้: {resp}")
    manifest.artifact_id = artifact_id
    manifest.status = "audio_generating"
    return manifest


def _extract_artifact_state(resp: dict, artifact_id: str) -> dict:
    """studio_status ไม่มี output schema ทางการ — รองรับทั้งกรณีคืน dict เดี่ยว (ถูก filter แล้ว) และ list"""
    for key in ("artifacts", "items", "results"):
        items = resp.get(key)
        if isinstance(items, list):
            for item in items:
                if item.get("artifact_id") == artifact_id or item.get("id") == artifact_id:
                    return item
    if "status" in resp and ("artifact_id" in resp or "url" in resp or "download_url" in resp):
        return resp
    # If the tool itself returned an error (e.g. rate limit, or artifact expired)
    if resp.get("status") == "error" and "error" in resp:
        return {"status": "failed", "error_reason": resp.get("error")}
    return {}


_STUDIO_STATUS_CALL_TIMEOUT = 60  # วินาที — กัน call เดียวแฮงค์ไม่มีที่สิ้นสุด (ไม่ใช่ timeout รวมของ poll loop)


async def _poll_studio_status(session, notebook_id: str, artifact_id: str, timeout_seconds: int) -> dict:
    """studio_status เป็น single-shot check (ไม่บล็อกในตัวเองแบบ research_status) — ต้อง poll loop เอง

    ห่อแต่ละ call ด้วย asyncio.wait_for แยกจาก timeout รวมของทั้ง loop — ถ้า call เดียวแฮงค์
    (network/subprocess ค้าง) จะไม่บล็อกไม่มีที่สิ้นสุดจนไม่มีทางถึง timeout_seconds ได้เลย
    (เจอจริง: #AG-47/#AG-49 ค้างที่ audio_generating เกิน timeout ไปมาก โดย job status ยังเป็น
    "running" ไม่ error เลย — สงสัยว่า call ค้างหรือ response ไม่ตรงกับที่ _extract_artifact_state
    คาดไว้ เพิ่ม log บรรทัดล่างเพื่อเก็บ raw response ไว้ debug ครั้งถัดไปด้วย)
    """
    elapsed = 0
    interval = _POLL_INITIAL_INTERVAL
    while elapsed < timeout_seconds:
        try:
            resp = await asyncio.wait_for(
                adapter.call_tool(session, "studio_status", {
                    "notebook_id": notebook_id,
                    "artifact_id": artifact_id,
                }),
                timeout=_STUDIO_STATUS_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[NotebookLM Pipeline] studio_status call ค้างเกิน %ss (artifact_id=%s) — ลองรอบถัดไป",
                _STUDIO_STATUS_CALL_TIMEOUT, artifact_id,
            )
            await asyncio.sleep(interval)
            elapsed += _STUDIO_STATUS_CALL_TIMEOUT + interval
            interval = min(interval * 2, _POLL_MAX_INTERVAL)
            continue

        state = _extract_artifact_state(resp, artifact_id)
        status = state.get("status")
        if status in _TERMINAL_SUCCESS:
            return state
        if status in _TERMINAL_FAILURE:
            raise StudioTerminalError(f"Audio generation ล้มเหลว (artifact_id={artifact_id}): {state}")
        if status not in _TERMINAL_SUCCESS and status not in _TERMINAL_FAILURE:
            # status เป็น None หรือค่าที่ไม่รู้จักเลย (ไม่ completed/failed) — เก็บ raw response ไว้
            # debug เพราะ notebooklm-mcp เป็น internal API ไม่มีเอกสาร โครงสร้าง response จริงอาจไม่
            # ตรงกับที่ _extract_artifact_state คาดไว้ (เช่น NotebookLM สร้างเสร็จจริงแล้วแต่เราไม่รู้จัก key)
            logger.warning(
                "[NotebookLM Pipeline] studio_status response ไม่มี status ที่รู้จัก (artifact_id=%s): %s",
                artifact_id, resp,
            )
        await asyncio.sleep(interval)
        elapsed += interval
        interval = min(interval * 2, _POLL_MAX_INTERVAL)
    raise StudioTimeoutError(
        f"studio_status ยังไม่ completed หลังรอ {timeout_seconds}s (artifact_id={artifact_id})"
    )


def _guess_extension(state: dict) -> str:
    """ดึงนามสกุลไฟล์จริงจาก URL ใน studio_status(include_details=True) — ไม่ฮาร์ดโค้ด

    Fallback .m4a เฉพาะกรณีหา URL ไม่เจอเลย — Audio Overview เป็นไฟล์เสียงล้วน .m4a จึงตรงกับ
    เนื้อหาจริงมากกว่า .mp4 (ที่สื่อถึงวิดีโอ)
    """
    for key in ("url", "media_url", "download_url", "audio_url"):
        val = state.get(key)
        if isinstance(val, str) and val:
            suffix = Path(val.split("?")[0]).suffix
            if suffix:
                return suffix
    return ".m4a"


async def _download_audio(
    session,
    manifest: NotebookLMManifest,
    briefing_file_path: Path,
) -> Path:
    try:
        status_resp = await adapter.call_tool(session, "studio_status", {
            "notebook_id": manifest.notebook_id,
            "artifact_id": manifest.artifact_id,
            "include_details": True,
        })
        state = _extract_artifact_state(status_resp, manifest.artifact_id)
    except Exception as e:
        # เจอจริง #AG-49: studio_status พังทั้งที่ Audio สร้างเสร็จสมบูรณ์แล้วจริง (ยืนยันจากเว็บ
        # NotebookLM เอง) — ไม่ให้ความล้มเหลวของ "เช็คนามสกุลไฟล์" มาบล็อกการดาวน์โหลดจริง แค่ใช้
        # นามสกุล default แทน (_guess_extension คืน .m4a เมื่อไม่มี state ให้ parse)
        logger.warning(
            "[NotebookLM Pipeline] เช็ค studio_status(include_details=True) ไม่สำเร็จ (artifact_id=%s): %s "
            "— ใช้นามสกุลไฟล์ default แทนแล้วลองดาวน์โหลดตรงๆ",
            manifest.artifact_id, e,
        )
        state = {}
    ext = _guess_extension(state)

    # ฝัง content_hash prefix ในชื่อไฟล์เสมอ กันชนกับ output เก่าของ briefing คนละเนื้อหาที่ชื่อไฟล์ซ้ำกัน
    dest_name = f"{briefing_file_path.stem}_{manifest.content_hash[:8]}{ext}"
    dest_path = OUTPUT_DIR / dest_name
    tmp_path = OUTPUT_DIR / f".tmp_{dest_name}"

    for attempt in range(60):
        try:
            await adapter.call_tool(session, "download_artifact", {
                "notebook_id": manifest.notebook_id,
                "artifact_type": "audio",
                "artifact_id": manifest.artifact_id,
                "output_path": str(tmp_path),
            })
            break
        except Exception as e:
            if attempt < 59 and "propagating" in str(e).lower():
                logger.warning(
                    "[NotebookLM Pipeline] Audio media URL ยังไม่พร้อม (กำลัง propagate) รอ 10 วินาทีแล้วลองใหม่ (attempt %d/60)...",
                    attempt + 1,
                )
                import asyncio
                await asyncio.sleep(10)
                continue
            raise

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ดาวน์โหลด audio artifact ไม่สำเร็จหรือไฟล์ว่างเปล่า: {tmp_path}")

    os.replace(tmp_path, dest_path)
    return dest_path


async def _finalize_download(
    session,
    manifest: NotebookLMManifest,
    resolved_path: Path,
    on_step: Callable[[str, str], None],
    *,
    note: str = "",
) -> NotebookLMManifest:
    audio_path = await _download_audio(session, manifest, resolved_path)
    try:
        from tools.content.notebooklm.audio_utils import compress_audio_for_discord
        audio_path = compress_audio_for_discord(audio_path)
    except Exception as e:
        logger.warning("[NotebookLM Pipeline] Audio compression step skipped: %s", e)


    manifest.audio_path = str(audio_path)
    manifest.status = "completed"
    save_manifest(manifest)
    on_step("download", f"ดาวน์โหลดและเตรียมไฟล์ Audio สำเร็จ{note}: {audio_path.name}")
    return manifest


@traceable(run_type="chain")
async def run_notebooklm_post_production_pipeline(
    briefing_file_path: Path,
    *,
    confirm_generation: bool,
    with_research: bool = False,
    research_query: str | None = None,
    research_mode: str = "deep",
    notebooklm_prompts: list[NotebookLMPromptRecord] | None = None,
    audio_language: str = "th",
    timeout_seconds: int = 5_400,  # 1 ชม. 30 นาที
    title: str | None = None,
    on_step: Callable[[str, str], None] | None = None,
) -> NotebookLMRunResult:
    """on_step(node, message) ถูกเรียกที่ checkpoint หลักเท่านั้น (ไม่ใช่ทุก poll tick กันสแปม) —
    ตั้งใจให้ไม่รู้จัก state_db/job_logs เลย (tools/ ห้าม import api/ ผิดชั้นสถาปัตยกรรม) caller
    ชั้น api/ (notebooklm_worker.py) เป็นคนตัดสินใจว่าจะเอา node/message นี้ไปเขียนที่ไหน
    """
    on_step = on_step or (lambda node, message: None)
    notebooklm_prompts = notebooklm_prompts or []

    # prompt [RESEARCH] ในไฟล์เปิด Deep Research ให้อัตโนมัติเสมอ — แต่ถ้า caller ระบุ
    # with_research/research_query มาเองชัดเจนแล้ว (เช่น CLI flag) ให้เคารพค่านั้นก่อน ไม่ทับ
    prompts_research_query = build_research_query(notebooklm_prompts)
    effective_with_research = with_research or bool(prompts_research_query)
    effective_research_query = research_query or prompts_research_query

    if effective_with_research and not effective_research_query:
        raise ValueError("research_query จำเป็นเมื่อ with_research=True")

    resolved_path = _validate_input(briefing_file_path)
    content_hash = compute_content_hash(resolved_path)
    manifest_path = manifest_path_for(content_hash)
    manifest = load_manifest(manifest_path) or new_manifest(content_hash=content_hash, briefing_path=resolved_path)

    if manifest.status == "completed" and manifest.audio_path:
        logger.info("[NotebookLM Pipeline] SKIP | reason: already completed | hash %s", content_hash[:8])
        return _result_from_manifest(manifest, manifest_path)

    adapter.check_binary_available()
    adapter.check_output_dir_writable(OUTPUT_DIR)

    async with adapter.open_session() as session:
        await adapter.check_auth(session)

        if not manifest.notebook_id:
            resp = await adapter.call_tool(session, "notebook_create", {
                "title": title or resolved_path.stem,
            })
            manifest.notebook_id = resp.get("notebook_id")
            manifest.status = "notebook_created"
            save_manifest(manifest)
            on_step("notebook_create", "สร้าง Notebook สำเร็จ")

        if not manifest.source_id:
            resp = await adapter.call_tool(session, "source_add", {
                "notebook_id": manifest.notebook_id,
                "source_type": "file",
                "file_path": str(resolved_path),
                "wait": True,
            })
            source_id = resp.get("source_id")
            
            # #AG-49: notebooklm-mcp's wait=True might return prematurely if the Google API
            # briefly returns 500 or 404 during ingestion. We must explicitly poll source_describe
            # until status is 'success' before proceeding to studio_create.
            if source_id:
                for _ in range(12):
                    try:
                        s_resp = await adapter.call_tool(session, "source_describe", {
                            "source_id": source_id
                        })
                        if s_resp.get("status") == "success":
                            break
                    except Exception:
                        pass
                    import asyncio
                    await asyncio.sleep(5)
            
            manifest.source_id = source_id
            manifest.status = "source_added"
            save_manifest(manifest)
            on_step("source_add", "อัปโหลด Briefing Book สำเร็จ")

        if effective_with_research and not manifest.research_completed and not manifest.artifact_id:
            manifest = await _run_research(session, manifest, effective_research_query, timeout_seconds, research_mode=research_mode)
            save_manifest(manifest)
            on_step("research", "Deep Research เสร็จสิ้น นำเข้าแหล่งข้อมูลแล้ว")

        if notebooklm_prompts and not manifest.prompts_queried and not manifest.artifact_id:
            manifest = await _run_notebook_prompts(session, manifest, notebooklm_prompts)
            save_manifest(manifest)
            on_step("notebook_prompts", f"ถามคำถามจาก NotebookLM Prompts ครบ {len(notebooklm_prompts)} ข้อ")

        if not manifest.artifact_id:
            manifest = await _generate_audio(
                session, manifest,
                confirm_generation=confirm_generation,
                audio_language=audio_language,
            )
            save_manifest(manifest)
            on_step("studio_create", "เริ่มสร้าง Audio Overview...")

        if not manifest.audio_path:
            try:
                await _poll_studio_status(session, manifest.notebook_id, manifest.artifact_id, timeout_seconds)
            except StudioTerminalError as terminal_error:
                # เจอจริง #AG-49: studio_status คืน status="error" พร้อมข้อความ "Could not retrieve
                # studio status" (เช็คสถานะไม่ได้) ทั้งที่ Audio สร้างเสร็จสมบูรณ์แล้วจริงในเว็บ
                # NotebookLM — ข้อความนี้บอกว่า "เช็คสถานะพัง" ไม่ใช่ "generation ล้มเหลว" ก่อนจะ
                # rollback+regenerate (เสีย quota ซ้ำ) ลอง download ตรงๆ ด้วย artifact_id เดิมก่อน
                # ถ้าโหลดสำเร็จ = ของจริงเสร็จแล้ว ถือว่า completed ไปเลย ไม่ต้องสร้างใหม่
                try:
                    manifest = await _finalize_download(
                        session, manifest, resolved_path, on_step,
                        note=" (กู้คืนจาก studio_status ที่เช็คสถานะไม่ได้)",
                    )
                except Exception:
                    # โหลดกู้คืนก็ไม่สำเร็จ — ยอมรับว่า generation ล้มเหลวจริง ต้อง regenerate ใหม่
                    # เคลียร์ artifact_id ให้ resume ครั้งหน้าวิ่งเข้า _generate_audio() อีกรอบ (ผ่าน
                    # confirmation gate เหมือนเดิม) — status ย้อนกลับไปที่ checkpoint ที่แท้จริงก่อนหน้า
                    # ตาม field boolean ที่มีจริง (ไม่ใช่ "failed" เฉยๆ) เพื่อไม่ให้ resume ครั้งหน้าลืมว่า
                    # research/ถาม prompts เสร็จไปแล้วหรือยัง แล้วไปทำซ้ำโดยไม่จำเป็น
                    manifest.artifact_id = None
                    if manifest.prompts_queried:
                        manifest.status = "prompts_queried"
                    elif manifest.research_completed:
                        manifest.status = "research_done"
                    else:
                        manifest.status = "source_added"
                    save_manifest(manifest)
                    raise terminal_error
            else:
                manifest = await _finalize_download(session, manifest, resolved_path, on_step)

    logger.info(
        "[NotebookLM Pipeline] DONE | notebook %s | audio %s",
        manifest.notebook_id, manifest.audio_path,
    )
    return _result_from_manifest(manifest, manifest_path)
