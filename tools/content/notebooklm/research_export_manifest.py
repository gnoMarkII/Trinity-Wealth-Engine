"""Durable Manifest Schema and Checkpointing for NotebookLM Research Exports.

Ensures upload progress across multiple sources is durable, atomic, and resumable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SourceUploadState:
    file_name: str
    sha256: str
    source_id: Optional[str] = None
    status: str = "pending"  # pending, uploading, success, failed
    error_message: Optional[str] = None


@dataclass
class ResearchExportManifest:
    schema_version: str = "2.0-research-export"
    export_id: str = ""
    bundle_hash: str = ""
    notebook_id: Optional[str] = None
    notebook_url: Optional[str] = None
    status: str = "created"  # created, uploading, ready, ready_with_warnings, partial, failed
    sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchExportManifest":
        return cls(
            schema_version=data.get("schema_version", "2.0-research-export"),
            export_id=data.get("export_id", ""),
            bundle_hash=data.get("bundle_hash", ""),
            notebook_id=data.get("notebook_id"),
            notebook_url=data.get("notebook_url"),
            status=data.get("status", "created"),
            sources=data.get("sources", {}),
            updated_at=data.get("updated_at", ""),
        )


def save_research_export_manifest(manifest_path: Path, manifest: ResearchExportManifest) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(".tmp")
    data_str = json.dumps(manifest.to_dict(), indent=2)
    temp_path.write_text(data_str, encoding="utf-8")
    temp_path.replace(manifest_path)


class ManifestCorruptError(Exception):
    """Raised when an existing manifest file cannot be parsed or validated."""
    pass


def load_research_export_manifest(manifest_path: Path) -> Optional[ResearchExportManifest]:
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ManifestCorruptError(f"Manifest JSON must be a dictionary: {manifest_path}")
        return ResearchExportManifest.from_dict(data)
    except ManifestCorruptError:
        raise
    except Exception as exc:
        raise ManifestCorruptError(f"Corrupt manifest file at {manifest_path}: {exc}") from exc
