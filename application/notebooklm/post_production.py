"""Application use case for NotebookLM post-production and notification."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from application.notebooklm.ports import (
    NotebookLMBriefingContentPort,
    NotebookLMNotificationPort,
    NotebookLMWorkerStatePort,
    NotebookLMPipelinePort,
    NotebookLMPromptExtractorPort,
    NotebookLMNotificationOutboxPort,
)


def extract_briefing_metadata(briefing_path: Path, raw_text: str) -> tuple[str, str, str]:
    """Extract stable title/summary/source reference for a Discord delivery."""
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
                break
            if stripped.startswith("---") or stripped.startswith("<!--"):
                continue
            if stripped:
                summary_lines.append(stripped)
    if not title:
        title = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", briefing_path.stem).replace("_", " ")
    summary = "\n".join(summary_lines).strip()
    if len(summary) > 3000:
        summary = summary[:2997] + "..."
    return title, summary, f"NotebookLM_Sources/{briefing_path.name}"


class NotebookLMPostProductionApplicationService:
    """Run post-production and isolate optional Discord delivery side effects."""

    def __init__(
        self,
        state: NotebookLMWorkerStatePort,
        content: NotebookLMBriefingContentPort,
        pipeline_runner: NotebookLMPipelinePort | Callable[..., Any],
        prompt_extractor: NotebookLMPromptExtractorPort | Callable[[str], Any],
        notifier: Optional[NotebookLMNotificationPort] = None,
        outbox: Optional[NotebookLMNotificationOutboxPort] = None,
    ) -> None:
        self._state = state
        self._content = content
        self._pipeline_runner = pipeline_runner
        self._prompt_extractor = prompt_extractor
        self._notifier = notifier
        self._outbox = outbox

    def _log_step(self, job_id: str, node: str, message: str) -> None:
        self._state.append_log(job_id, node, message)

    def execute(self, job_id: str, instruction: str) -> None:
        briefing_path = Path(instruction)
        raw_text = self._content.read(str(briefing_path))
        # ``unittest.mock`` creates arbitrary attributes on mocks.  Inspect
        # the concrete type so a legacy callable/AsyncMock is not mistaken
        # for a Protocol object that happens to expose ``extract``/``run``.
        extractor = (
            self._prompt_extractor.extract
            if getattr(type(self._prompt_extractor), "extract", None) is not None
            else self._prompt_extractor
        )
        prompts = extractor(raw_text)

        runner = (
            self._pipeline_runner.run
            if getattr(type(self._pipeline_runner), "run", None) is not None
            else self._pipeline_runner
        )
        result = asyncio.run(
            runner(
                briefing_path,
                confirm_generation=True,
                notebooklm_prompts=prompts,
                on_step=lambda node, message: self._log_step(job_id, node, message),
            )
        )
        if not result or result.status != "completed" or not result.audio_path or self._notifier is None:
            return

        delivery_key: Optional[str] = None
        try:
            job = self._state.get_job(job_id)
            card_id = job.get("card_id") if job else None
            card = self._state.get_card(card_id) if card_id else None
            if card is None:
                return
            if not bool(card.get("discord_notify", 1)):
                self._log_step(job_id, "discord_skipped", "ข้ามการส่ง Discord (สวิตช์ปิดอยู่)")
                return

            delivery_key = f"notebooklm:{result.content_hash}"
            try:
                sent_events = json.loads(card.get("discord_sent_events") or "[]")
            except Exception:
                sent_events = []
            if delivery_key in sent_events:
                self._log_step(job_id, "discord_skipped", "ข้ามการส่ง Discord (เคยส่งไฟล์เวอร์ชันนี้ไปแล้ว)")
                return

            title, summary, source_ref = extract_briefing_metadata(briefing_path, raw_text)
            if self._outbox is not None:
                existing = self._outbox.get(delivery_key)
                if existing and existing.get("status") == "sent":
                    self._log_step(job_id, "discord_skipped", "ข้ามการส่ง Discord (outbox ส่งสำเร็จแล้ว)")
                    return
                self._outbox.enqueue(
                    event_id=delivery_key,
                    idempotency_key=delivery_key,
                    aggregate_type="notebooklm_briefing",
                    aggregate_id=str(job_id),
                    event_type="discord_audio_ready",
                    payload={
                        "audio_path": str(result.audio_path),
                        "title": title,
                        "summary": summary,
                        "source_ref": source_ref,
                    },
                )
            delivery = self._notifier.send(
                audio_path=result.audio_path,
                title=title,
                summary=summary,
                source_ref=source_ref,
            )
            status = getattr(delivery, "status", None)
            message = getattr(delivery, "message", "")
            if status == "sent":
                if self._outbox is not None:
                    self._outbox.mark_sent(delivery_key)
                self._state.mark_discord_events_sent(card_id, [delivery_key])
                self._log_step(job_id, "discord", f"โพสต์ไฟล์เสียงขึ้น Discord สำเร็จ: {Path(result.audio_path).name}")
            elif status == "skipped_oversize":
                self._log_step(job_id, "discord_skipped_oversize", f"⚠️ {message}")
            elif status == "skipped_disabled":
                self._log_step(job_id, "discord_skipped", f"ข้ามการส่ง Discord ({message})")
            else:
                if self._outbox is not None:
                    self._outbox.mark_failed(delivery_key, message or str(status))
                self._log_step(job_id, "discord_failed", f"⚠️ ส่ง Discord ไม่สำเร็จ: {message}")
        except Exception as exc:
            if self._outbox is not None and delivery_key is not None:
                try:
                    self._outbox.mark_failed(delivery_key, str(exc))
                except Exception:
                    pass
            # Notification is deliberately failure-tolerant; audio generation
            # has already completed and the job must not be marked failed.
            self._log_step(job_id, "discord_failed", f"ส่ง Discord ไม่สำเร็จ (ไม่กระทบไฟล์เสียง): {exc}")


__all__ = ["NotebookLMPostProductionApplicationService", "extract_briefing_metadata"]
