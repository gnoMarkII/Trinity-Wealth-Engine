"""Filesystem adapters for NotebookLM source and manifest ports."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from application.notebooklm.domain import parse_source_filename
from tools.content.notebooklm import manifest


class FilesystemSourceCatalogAdapter:
    def __init__(self, sources_dir: Path) -> None:
        self._sources_dir = Path(sources_dir).resolve()

    def list_sources(self) -> list[dict[str, Any]]:
        if not self._sources_dir.is_dir():
            return []
        result = []
        for path in self._sources_dir.rglob("*.md"):
            # Exclude hidden files, temp files, revisions, outbox, quarantine, inbox
            if any(p.startswith(".") or p in ("Inbox", "Revisions", "quarantine", "outbox") for p in path.parts):
                continue
            title, date_part, is_verified = parse_source_filename(path.stem)
            result.append({
                "file_path": str(path.resolve()),
                "title": title,
                "date_part": date_part,
                "is_verified": is_verified,
            })
        result.sort(key=lambda item: (item["date_part"] or "", item["title"]), reverse=True)
        return result

    def resolve_source(self, reference: str) -> dict[str, Any]:
        candidate = Path(reference).resolve()
        if not candidate.is_file() or not candidate.is_relative_to(self._sources_dir):
            raise ValueError("ไฟล์ Briefing Book ที่เลือกไม่ถูกต้อง หรืออยู่นอก NotebookLM_Sources/")
        title, date_part, is_verified = parse_source_filename(candidate.stem)
        return {
            "file_path": str(candidate),
            "title": title,
            "date_part": date_part,
            "is_verified": is_verified,
        }


class FilesystemBriefingContentAdapter:
    """Read-only briefing content adapter used by the worker use case."""

    def read(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")


class FilesystemManifestAdapter:
    def get_for_source(self, source_path: str) -> Any:
        from tools.content.notebooklm.manifest import ManifestLoadResult, ManifestStatus
        try:
            content_hash = manifest.compute_content_hash(Path(source_path))
            return manifest.load_manifest(manifest.manifest_path_for(content_hash))
        except (OSError, ValueError, TypeError) as exc:
            # History store is unreadable — return typed MISSING_HISTORY rather than None.
            # Returning None would make callers treat the source as never-seen and create
            # a new run, silently losing existing history.
            return ManifestLoadResult(
                load_status=ManifestStatus.MISSING_HISTORY,
                error_message=f"History store unreadable: {exc}",
                path=source_path,
            )
