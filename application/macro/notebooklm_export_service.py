"""Macro NotebookLM Research Export Application Service.

Hexagonal Architecture Invariant:
Depends solely on domain protocols defined in application.macro.notebooklm_export_ports.
Does NOT import database connections, frameworks, or concrete adapters directly.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from application.macro.notebooklm_export_ports import (
    MacroCorpusReaderPort,
    MacroCorpusSnapshot,
    MacroExportBundlePort,
    MacroExportDispatchPort,
    MacroExportPipelinePort,
    MacroExportRecord,
    MacroExportRepositoryPort,
)


class MacroNotebookLMExportService:
    """Coordinates Macro research bundle preparation, deduplication, and export execution."""

    def __init__(
        self,
        corpus_reader: MacroCorpusReaderPort,
        bundle_builder: MacroExportBundlePort,
        repo: MacroExportRepositoryPort,
        dispatcher: MacroExportDispatchPort,
        pipeline: Optional[MacroExportPipelinePort] = None,
    ) -> None:
        self._corpus_reader = corpus_reader
        self._bundle_builder = bundle_builder
        self._repo = repo
        self._dispatcher = dispatcher
        self._pipeline = pipeline

    def request_export(self, mode: str = "all_retained") -> MacroExportRecord:
        """Capture current macro corpus, generate frozen bundle, and dispatch export."""
        if mode != "all_retained":
            raise ValueError(f"Unsupported export mode: '{mode}'. Only 'all_retained' is allowed.")

        # 1. Capture snapshot and build bundle with deterministic content hash
        snapshot = self._corpus_reader.capture_snapshot()
        preliminary_id = f"macro_export_{uuid.uuid4().hex[:12]}"
        bundle_dir, content_hash, inventory = self._bundle_builder.build_bundle(preliminary_id, snapshot)

        # 2. Idempotency Check: if identical content hash is already active or ready, reuse it
        existing = self._repo.get_by_content_hash(content_hash)
        if existing is not None and existing.state in (
            "queued",
            "preparing",
            "uploading",
            "verifying",
            "ready",
            "ready_with_warnings",
        ):
            if existing.state == "queued" and not existing.job_id:
                job_id = self._dispatcher.dispatch(
                    instruction=existing.export_id,
                    card_id=None,
                    flow="macro_notebooklm",
                    scope="both",
                )
                updated = self._repo.update_state(
                    export_id=existing.export_id,
                    state="queued",
                    stage="dispatched",
                    job_id=job_id,
                )
                return self._reconcile_with_manifest(updated or existing)
            return self._reconcile_with_manifest(existing)

        # 3. Create new durable export record
        export_id = preliminary_id
        request_key = f"macro_notebooklm:{content_hash}"
        record = MacroExportRecord(
            export_id=export_id,
            request_key=request_key,
            content_hash=content_hash,
            job_id=None,
            state="queued",
            stage="bundle_frozen",
            snapshot_at=snapshot.snapshot_at,
            strategy_report_id=snapshot.strategy_report_id,
            notebook_id=None,
            notebook_url=None,
            manifest_path=str(bundle_dir / "manifest.json"),
            inventory=inventory,
            warnings=snapshot.metadata.get("warnings", []),
            error_code=None,
            error_message=None,
            created_at=time.time(),
            updated_at=time.time(),
        )
        created_record = self._repo.create(record)

        # 4. Dispatch to background queue
        job_id = self._dispatcher.dispatch(
            instruction=export_id,
            card_id=None,
            flow="macro_notebooklm",
            scope="both",
        )

        updated = self._repo.update_state(
            export_id=export_id,
            state="queued",
            stage="dispatched",
            job_id=job_id,
        )
        return updated or created_record

    def get_export(self, export_id: str) -> Optional[MacroExportRecord]:
        """Fetch export record by ID, reconciling with on-disk manifest progress."""
        record = self._repo.get_by_id(export_id)
        if record is None:
            return None
        return self._reconcile_with_manifest(record)

    def get_latest_export(self) -> Optional[MacroExportRecord]:
        """Fetch latest export record, reconciling with on-disk manifest progress."""
        record = self._repo.get_latest()
        if record is None:
            return None
        return self._reconcile_with_manifest(record)

    def retry_export(self, export_id: str) -> MacroExportRecord:
        """Retry a failed or partial export from its existing frozen bundle."""
        record = self._repo.get_by_id(export_id)
        if record is None:
            raise LookupError(f"Macro export '{export_id}' not found")

        # If already in flight, return without duplicate dispatch
        if record.state in ("queued", "uploading", "verifying"):
            return record

        job_id = self._dispatcher.dispatch(
            instruction=export_id,
            card_id=None,
            flow="macro_notebooklm",
            scope="both",
        )

        updated = self._repo.update_state(
            export_id=export_id,
            state="queued",
            stage="retrying",
            job_id=job_id,
            error_code=None,
            error_message=None,
        )
        return updated or record

    async def execute_worker_job(
        self,
        job_id: str,
        instruction: str,
        on_step: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Executed by background worker for flow='macro_notebooklm'."""
        export_id = instruction.strip()
        record = self._repo.get_by_id(export_id)
        if record is None:
            raise LookupError(f"Export record '{export_id}' not found for job '{job_id}'")

        if not self._pipeline:
            raise RuntimeError("MacroExportPipelinePort is not configured on this worker")

        self._repo.update_state(
            export_id=export_id,
            state="uploading",
            stage="uploading_sources",
            job_id=job_id,
        )

        manifest_file = Path(record.manifest_path) if record.manifest_path else None
        bundle_dir = manifest_file.parent if manifest_file else None
        if not bundle_dir or not bundle_dir.is_dir():
            self._repo.update_state(
                export_id=export_id,
                state="failed",
                stage="error",
                error_code="bundle_missing",
                error_message=f"Export bundle directory not found: {bundle_dir}",
            )
            return

        try:
            result = await self._pipeline.execute_export(
                export_id=export_id,
                bundle_dir=bundle_dir,
                on_step=on_step,
            )
            final_status = result.get("status", "failed")
            stage = "completed" if final_status in ("ready", "ready_with_warnings") else "partial"
            self._repo.update_state(
                export_id=export_id,
                state=final_status,
                stage=stage,
                notebook_id=result.get("notebook_id"),
                notebook_url=result.get("notebook_url"),
            )
        except Exception as exc:
            self._repo.update_state(
                export_id=export_id,
                state="failed",
                stage="error",
                error_code="pipeline_error",
                error_message=str(exc),
            )
            raise

    def execute_sync(
        self,
        job_id: str,
        instruction: str,
        on_step: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Synchronous runner for worker thread execution."""
        import asyncio

        asyncio.run(self.execute_worker_job(job_id=job_id, instruction=instruction, on_step=on_step))

    def _reconcile_with_manifest(self, record: MacroExportRecord) -> MacroExportRecord:
        """Syncs the in-memory/DB record with on-disk manifest if available."""
        if not record.manifest_path:
            return record
        manifest_file = Path(record.manifest_path)
        if not manifest_file.is_file():
            return record

        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            m_status = data.get("status")
            m_nb_id = data.get("notebook_id")
            m_nb_url = data.get("notebook_url")

            needs_update = False
            state = record.state
            if m_status and m_status != record.state and record.state != "failed":
                state = m_status
                needs_update = True
            nb_id = record.notebook_id or m_nb_id
            nb_url = record.notebook_url or m_nb_url
            if nb_id != record.notebook_id or nb_url != record.notebook_url:
                needs_update = True

            if needs_update:
                updated = self._repo.update_state(
                    export_id=record.export_id,
                    state=state,
                    stage="updated_from_manifest",
                    notebook_id=nb_id,
                    notebook_url=nb_url,
                )
                if updated:
                    return updated
        except Exception:
            pass
        return record
