"""Application service for NotebookLM source and generation workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from application.notebooklm.ports import (
    NotebookLMCardRepositoryPort,
    NotebookLMDispatchPort,
    NotebookLMBinaryPort,
    NotebookLMJobRepositoryPort,
    NotebookLMManifestPort,
    NotebookLMSourceCatalogPort,
)
from application.notebooklm.dto import NotebookLMAvailableSourceDTO, NotebookLMStatusDTO
from application.notebooklm.domain import parse_source_filename

NOTEBOOKLM_SOURCES_DIR = Path("memories/30_Knowledge_Base/NotebookLM_Sources").resolve()

# Historical import path retained for callers that used the private helper.
_parse_source_filename = parse_source_filename


class NotebookLMPreflightError(RuntimeError):
    """Outbound binary/storage preflight failed before dispatch."""


class NotebookLMApplicationService:
    """Coordinates NotebookLM use cases through outbound ports."""

    def __init__(
        self,
        repo: NotebookLMJobRepositoryPort,
        sources_dir: Optional[Path] = None,
        card_repo: Optional[NotebookLMCardRepositoryPort] = None,
        dispatcher: Optional[NotebookLMDispatchPort] = None,
        binary: Optional[NotebookLMBinaryPort] = None,
        source_catalog: Optional[NotebookLMSourceCatalogPort] = None,
        manifest_port: Optional[NotebookLMManifestPort] = None,
    ) -> None:
        self._repo = repo
        self._card_repo = card_repo
        self._dispatcher = dispatcher
        self._binary = binary
        self._source_catalog = source_catalog
        self._manifest_port = manifest_port
        # ``sources_dir`` remains in the signature for callers that construct
        # the service directly.  Production composition must provide ports;
        # the explicit error keeps accidental infrastructure access visible.
        self._legacy_sources_dir = sources_dir

    def list_available_sources(self) -> list[NotebookLMAvailableSourceDTO]:
        if self._source_catalog is None:
            if self._legacy_sources_dir is not None:
                raise RuntimeError("NotebookLM source catalog port is not configured")
            return []
        return [
            NotebookLMAvailableSourceDTO(**item)
            for item in self._source_catalog.list_sources()
        ]

    def _resolve_source(self, reference: str) -> dict[str, Any]:
        if self._source_catalog is None:
            raise RuntimeError("NotebookLM source catalog port is not configured")
        return self._source_catalog.resolve_source(reference)

    def validate_briefing_source(self, briefing_path: str) -> Path:
        """Compatibility-facing validation that delegates to the source port."""
        return Path(self._resolve_source(briefing_path)["file_path"])

    def get_status(self, job_id: str) -> Optional[NotebookLMStatusDTO]:
        job = self._repo.get_job(job_id)
        if job is None:
            return None
        manifest = self._manifest_port.get_for_source(job["instruction"]) if self._manifest_port else None
        return NotebookLMStatusDTO(
            job_id=job_id,
            status=job["status"],
            audio_path=manifest.audio_path if manifest else None,
            notebook_id=manifest.notebook_id if manifest else None,
            error=job.get("error_message"),
        )

    def generate(self, card_id: str, briefing_file_path: Optional[str] = None) -> dict[str, Any]:
        if self._card_repo is None or self._dispatcher is None or self._binary is None:
            raise RuntimeError("NotebookLM generation ports are not configured")
        card = self._card_repo.get_card(card_id)
        if card is None:
            raise LookupError("ไม่พบการ์ดนี้")
        if card.get("flow") != "notebooklm":
            raise ValueError("การ์ดนี้ไม่ใช่ NotebookLM Audio Overview")
        reference = card.get("prompt") or briefing_file_path
        if not reference:
            raise ValueError("ต้องเลือกไฟล์ Briefing Book ก่อนสร้าง Audio")
        source = self._resolve_source(reference)
        source_path = source["file_path"]
        if card.get("prompt") != source_path:
            self._card_repo.set_source(card_id, source_path, bool(source.get("is_verified", True)))
        self._binary.check_available()
        job_id = self._dispatcher.dispatch(source_path, card_id, flow="notebooklm")
        self._card_repo.move_card(card_id, "executing", job_id=job_id)
        job = self._repo.get_job(job_id)
        if job is None:
            raise LookupError(f"ไม่พบ job_id: {job_id}")
        return {"job_id": job_id, "status": job["status"]}
