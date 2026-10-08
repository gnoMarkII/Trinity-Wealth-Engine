"""Execution adapter for the NotebookLM post-production use case.

All workflow/state/provider orchestration lives in
``application.notebooklm.post_production``.  This module only wires concrete
adapters and retains the historical function entry point used by JobQueue and
tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from application.notebooklm.post_production import (
    NotebookLMPostProductionApplicationService,
    extract_briefing_metadata,
)
from api.db.legacy_adapter import (
    LegacyNotebookLMWorkerStateAdapter,
    LegacyNotebookLMNotificationOutboxAdapter,
)
from tools.content.notebooklm.adapters.filesystem import FilesystemBriefingContentAdapter
from tools.content.notebooklm.pipeline import run_notebooklm_post_production_pipeline
from tools.content.notebooklm.prompts import extract_notebooklm_prompts


def _extract_briefing_metadata(briefing_path: Path, raw_text: str) -> tuple[str, str, str]:
    """Compatibility alias for the historical worker helper."""
    return extract_briefing_metadata(briefing_path, raw_text)


class _DiscordNotifierAdapter:
    def send(self, *, audio_path: Any, title: str, summary: str, source_ref: str) -> Any:
        from core.discord_notifier import send_notebooklm_audio_discord

        return send_notebooklm_audio_discord(
            audio_path=audio_path,
            title=title,
            summary=summary,
            source_ref=source_ref,
        )


def notebooklm_run_fn(
    job_id: str,
    thread_id: str,
    instruction: str,
    flow: str = "notebooklm",
    scope: str = "both",
    resume_value: Optional[dict[str, Any]] = None,
) -> None:
    """Route NotebookLM jobs based on explicit flow name."""
    del thread_id, scope, resume_value  # reserved by the common queue signature

    if flow == "notebooklm":
        service = NotebookLMPostProductionApplicationService(
            state=LegacyNotebookLMWorkerStateAdapter(),
            content=FilesystemBriefingContentAdapter(),
            pipeline_runner=run_notebooklm_post_production_pipeline,
            prompt_extractor=extract_notebooklm_prompts,
            notifier=_DiscordNotifierAdapter(),
            outbox=LegacyNotebookLMNotificationOutboxAdapter(),
        )
        service.execute(job_id=job_id, instruction=instruction)
    elif flow == "macro_notebooklm":
        from api.dependencies import build_macro_notebooklm_export_service

        macro_export_service = build_macro_notebooklm_export_service()
        state_adapter = LegacyNotebookLMWorkerStateAdapter()
        macro_export_service.execute_sync(
            job_id=job_id,
            instruction=instruction,
            on_step=lambda node, msg: state_adapter.append_log(job_id, node, msg),
        )
    else:
        raise ValueError(f"Unknown flow '{flow}' dispatched to notebooklm worker")


__all__ = [
    "notebooklm_run_fn",
    "run_notebooklm_post_production_pipeline",
    "extract_notebooklm_prompts",
    "_extract_briefing_metadata",
]
