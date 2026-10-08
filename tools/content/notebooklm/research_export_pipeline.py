"""NotebookLM Research Export Pipeline — Sources-Only Multi-Part Ingestion.

Dedicated pipeline for uploading frozen Markdown research bundles to NotebookLM.
Invariants:
- Strictly sources-only (no studio_create, no audio generation, no discord, no deep research).
- Durable multi-source checkpoints enabling resumption from any failed source.
- Validates readiness via source_describe before marking ready.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from application.macro.notebooklm_export_ports import MacroExportPipelinePort
from core.logger import get_logger
from tools.content.notebooklm import adapter
from tools.content.notebooklm.research_export_manifest import (
    ManifestCorruptError,
    ResearchExportManifest,
    load_research_export_manifest,
    save_research_export_manifest,
)

logger = get_logger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def run_research_export_pipeline(
    bundle_dir: Path,
    export_id: str,
    *,
    title: Optional[str] = None,
    on_step: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Uploads all sources from bundle_dir to a dedicated NotebookLM research notebook."""
    on_step = on_step or (lambda node, message: None)
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    inventory_path = bundle_dir / "inventory.json"

    if not inventory_path.is_file():
        raise FileNotFoundError(f"Missing inventory.json in export bundle: {inventory_path}")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    content_hash = inventory.get("content_hash", "")
    sources_to_upload: List[Dict[str, Any]] = inventory.get("sources", [])

    # Load or initialize durable manifest
    manifest = load_research_export_manifest(manifest_path)
    if manifest is None:
        manifest = ResearchExportManifest(
            export_id=export_id,
            bundle_hash=content_hash,
            status="uploading",
            updated_at=_utc_now_iso(),
        )
        save_research_export_manifest(manifest_path, manifest)

    # Pre-flight binary verification
    adapter.check_binary_available()

    async with adapter.open_session() as session:
        await adapter.check_auth(session)

        # 1. Create or reuse Notebook
        if not manifest.notebook_id:
            default_title = f"Macro Research — {inventory.get('snapshot_at', _utc_now_iso())[:16]}"
            nb_title = title or default_title
            resp = await adapter.call_tool(session, "notebook_create", {"title": nb_title})
            notebook_id = resp.get("notebook_id")
            if not notebook_id:
                raise RuntimeError(f"notebook_create did not return a valid notebook_id: {resp}")
            manifest.notebook_id = notebook_id
            manifest.notebook_url = f"https://notebooklm.google.com/notebook/{notebook_id}"
            manifest.status = "uploading"
            manifest.updated_at = _utc_now_iso()
            save_research_export_manifest(manifest_path, manifest)
            on_step("notebook_create", f"สร้าง NotebookLM สำเร็จ: {nb_title}")
        else:
            if not manifest.notebook_url:
                manifest.notebook_url = f"https://notebooklm.google.com/notebook/{manifest.notebook_id}"

        # 2. Iterate and upload each source part
        total_sources = len(sources_to_upload)
        uploaded_count = 0
        failed_count = 0

        for idx, src in enumerate(sources_to_upload, start=1):
            file_name = src["file_name"]
            rel_path = src["relative_path"]
            file_path = bundle_dir / rel_path

            if not file_path.is_file():
                err_msg = f"Source file does not exist: {file_path}"
                logger.error("[Macro NotebookLM Pipeline] %s", err_msg)
                manifest.sources[file_name] = {
                    "source_id": None,
                    "file_name": file_name,
                    "sha256": src.get("sha256", ""),
                    "status": "failed",
                    "error": err_msg,
                }
                failed_count += 1
                continue

            # C11: Recompute actual sha256 bytes from disk rather than trusting inventory blindly
            actual_bytes = file_path.read_bytes()
            actual_sha = hashlib.sha256(actual_bytes).hexdigest()

            # Check if this source was already successfully uploaded with matching hash
            existing_rec = manifest.sources.get(file_name)
            if (
                existing_rec
                and existing_rec.get("status") == "success"
                and existing_rec.get("source_id")
                and existing_rec.get("sha256") == actual_sha
            ):
                uploaded_count += 1
                continue

            on_step("source_add", f"กำลังส่งแหล่งข้อมูล {idx}/{total_sources}: {file_name}")

            try:
                add_resp = await adapter.call_tool(
                    session,
                    "source_add",
                    {
                        "notebook_id": manifest.notebook_id,
                        "source_type": "file",
                        "file_path": str(file_path),
                        "wait": True,
                    },
                )
                source_id = add_resp.get("source_id") if isinstance(add_resp, dict) else None
                if not source_id:
                    raise RuntimeError(f"source_add failed to return a valid source_id: {add_resp}")

                # C11: Verify ingestion readiness with source_describe (fail if not confirmed)
                ingestion_ready = False
                last_describe_err = None
                for _ in range(12):
                    try:
                        desc_resp = await adapter.call_tool(session, "source_describe", {"source_id": source_id})
                        if isinstance(desc_resp, dict) and desc_resp.get("status") in ("success", "ready", "ok"):
                            ingestion_ready = True
                            break
                    except Exception as e:
                        last_describe_err = str(e)
                    await asyncio.sleep(2)

                if not ingestion_ready:
                    raise TimeoutError(
                        f"source_describe readiness timed out for {file_name} (source_id: {source_id}): {last_describe_err}"
                    )

                manifest.sources[file_name] = {
                    "source_id": source_id,
                    "file_name": file_name,
                    "sha256": actual_sha,
                    "status": "success",
                    "error": None,
                }
                uploaded_count += 1
            except Exception as exc:
                logger.error("[Macro NotebookLM Pipeline] Failed to upload source %s: %s", file_name, exc)
                manifest.sources[file_name] = {
                    "source_id": None,
                    "file_name": file_name,
                    "sha256": actual_sha,
                    "status": "failed",
                    "error": str(exc),
                }
                failed_count += 1

            manifest.updated_at = _utc_now_iso()
            save_research_export_manifest(manifest_path, manifest)

        # 3. Determine terminal status
        has_warnings = bool(inventory.get("warnings"))
        if failed_count == 0 and uploaded_count == total_sources:
            manifest.status = "ready_with_warnings" if has_warnings else "ready"
            on_step("complete", f"ส่งข้อมูล Macro ทั้งหมด {total_sources} แหล่งข้อมูลเรียบร้อยแล้ว")
        elif uploaded_count > 0:
            manifest.status = "partial"
            on_step("partial", f"ส่งข้อมูลสำเร็จ {uploaded_count}/{total_sources} แหล่งข้อมูล (บางรายการติดขัด)")
        else:
            manifest.status = "failed"
            on_step("failed", "ไม่สามารถส่งแหล่งข้อมูลไปยัง NotebookLM ได้")

        manifest.updated_at = _utc_now_iso()
        save_research_export_manifest(manifest_path, manifest)

        return {
            "notebook_id": manifest.notebook_id,
            "notebook_url": manifest.notebook_url,
            "status": manifest.status,
            "total_sources": total_sources,
            "uploaded_count": uploaded_count,
            "failed_count": failed_count,
            "manifest_path": str(manifest_path),
        }


class NotebookLMResearchExportPipelineAdapter(MacroExportPipelinePort):
    """Adapter implementing MacroExportPipelinePort using the local MCP runner."""

    async def execute_export(
        self,
        export_id: str,
        bundle_dir: Path,
        title: Optional[str] = None,
        on_step: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        return await run_research_export_pipeline(
            bundle_dir=bundle_dir,
            export_id=export_id,
            title=title,
            on_step=on_step,
        )
