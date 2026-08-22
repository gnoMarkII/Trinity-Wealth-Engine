import asyncio
from contextlib import closing
import json
from pathlib import Path
import re
from typing import Any, Optional

from api import state_db
from core.logger import get_logger
from tools.content.notebooklm.pipeline import run_notebooklm_post_production_pipeline
from tools.content.notebooklm.prompts import extract_notebooklm_prompts

logger = get_logger(__name__)


def _extract_briefing_metadata(briefing_path: Path, raw_text: str) -> tuple[str, str, str]:
    """สกัด Title, Summary และ Relative Source Path จากไฟล์ Briefing Book
    - Title: ดึงจากบรรทัด H1 แรก หรือ fallback briefing_path.stem
    - Summary: ดึงเนื้อหาหลัง H1 จนถึงก่อน ## Act I หรือ ## แรก (ตัดไม่เกิน 3000 chars) หรือข้อความว่าง
    - Source Ref: 'NotebookLM_Sources/<filename>.md'
    """
    lines = raw_text.splitlines()
    title = ""
    summary_lines = []
    found_h1 = False

    for line in lines:
        stripped = line.strip()
        if not found_h1:
            if stripped.startswith("# "):
                title = re.sub(r"^#\s*[\U00010000-\U0010ffff\u2600-\u27ff]*\s*", "", stripped).strip()
                found_h1 = True
        else:
            if stripped.startswith("## "):
                # เจอ section ถัดไป (เช่น ## Act I หรือ ## 1.) ให้หยุด
                break
            # ตัด separator line และ template comment
            if stripped.startswith("---") or stripped.startswith("<!--"):
                continue
            if stripped:
                summary_lines.append(stripped)

    if not title:
        title = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", briefing_path.stem).replace("_", " ")

    summary = "\n".join(summary_lines).strip()
    if len(summary) > 3000:
        summary = summary[:2997] + "..."

    source_ref = f"NotebookLM_Sources/{briefing_path.name}"
    return title, summary, source_ref


def notebooklm_run_fn(
    job_id: str,
    thread_id: str,
    instruction: str,
    flow: str = "notebooklm",
    scope: str = "both",
    resume_value: Optional[dict[str, Any]] = None,
) -> None:
    """instruction คือ briefing_file_path (absolute, resolved แล้วตอน dispatch ใน routes_notebooklm.py)

    อ่าน section ## NotebookLM Prompts จากไฟล์ (ถ้ามี) แล้วส่งเข้า pipeline เอง
    ส่ง on_step เข้า pipeline เพื่อเขียน checkpoint หลักลง job_logs
    และดำเนินการส่ง Discord Notification เมื่อการ์ดเปิด toggle ไว้ (Failure-tolerant ไม่ทำให้ job ล้มเหลว)
    """
    def _log_step(node: str, message: str) -> None:
        with closing(state_db.get_connection()) as conn:
            state_db.append_job_log(conn, job_id, node, message, role="reply", label=node)

    briefing_path = Path(instruction)
    raw_text = briefing_path.read_text(encoding="utf-8")
    prompts = extract_notebooklm_prompts(raw_text)

    result = asyncio.run(run_notebooklm_post_production_pipeline(
        briefing_path, confirm_generation=True, notebooklm_prompts=prompts, on_step=_log_step,
    ))

    # Isolated Discord Notification Phase
    if result and result.status == "completed" and result.audio_path:
        try:
            with closing(state_db.get_connection()) as conn:
                job = state_db.get_job(conn, job_id)
                card_id = job["card_id"] if job is not None else None
                card = state_db.get_kanban_card(conn, card_id) if card_id else None

            if card is not None:
                card_dict = dict(card)
                should_notify = bool(card_dict.get("discord_notify", 1))
                if not should_notify:
                    _log_step("discord_skipped", "ข้ามการส่ง Discord (สวิตช์ปิดอยู่)")
                else:
                    delivery_key = f"notebooklm:{result.content_hash}"
                    existing_sent = []
                    try:
                        existing_sent = json.loads(card_dict.get("discord_sent_events") or "[]")
                    except Exception:
                        existing_sent = []


                    if delivery_key in existing_sent:
                        _log_step("discord_skipped", "ข้ามการส่ง Discord (เคยส่งไฟล์เวอร์ชันนี้ไปแล้ว)")
                    else:
                        from core.discord_notifier import send_notebooklm_audio_discord
                        title, summary, source_ref = _extract_briefing_metadata(briefing_path, raw_text)
                        delivery = send_notebooklm_audio_discord(
                            audio_path=result.audio_path,
                            title=title,
                            summary=summary,
                            source_ref=source_ref,
                        )

                        if delivery.status == "sent":
                            with closing(state_db.get_connection()) as conn:
                                state_db.mark_discord_events_sent(conn, card_id, [delivery_key])
                            _log_step("discord", f"🎙️ โพสต์ไฟล์เสียงขึ้น Discord สำเร็จ: {Path(result.audio_path).name}")
                        elif delivery.status == "skipped_oversize":
                            _log_step("discord_skipped_oversize", f"⚠️ {delivery.message}")
                        elif delivery.status == "skipped_disabled":
                            _log_step("discord_skipped", f"ข้ามการส่ง Discord ({delivery.message})")
                        else:
                            _log_step("discord_failed", f"⚠️ ส่ง Discord ไม่สำเร็จ: {delivery.message}")
        except Exception as e:
            logger.warning("Discord notification phase encountered an error (non-fatal): %s", e)
            _log_step("discord_failed", f"ส่ง Discord ไม่สำเร็จ (ไม่กระทบไฟล์เสียง): {e}")

