"""Infrastructure adapter from note-shaped intent to ``KnowledgeWritePort``."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from application.knowledge.note_write_ports import KnowledgeNoteWritePort
from application.knowledge.write_models import KnowledgeWriteCommand
from application.knowledge.write_ports import KnowledgeWritePort
from tools.archivist.artifact_writer import CommittedArtifactSet
from tools.archivist.vault_paths import VaultPaths


def _fingerprint(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class BrokerKnowledgeNoteWriter(KnowledgeNoteWritePort):
    """Translate note-shaped calls into injected durable write commands."""

    def __init__(self, *, vault_paths: VaultPaths, write_port: KnowledgeWritePort) -> None:
        self.vault_paths = vault_paths
        self.write_port = write_port

    def write_note(
        self,
        metadata: dict[str, Any],
        body: str,
        filename: Optional[str] = None,
        target_path: Optional[Union[str, Path]] = None,
        expected_current_hash: Optional[str] = None,
        companion_artifacts: Optional[Mapping[str, Union[str, bytes]]] = None,
    ) -> CommittedArtifactSet:
        meta = dict(metadata)
        meta.setdefault("schema_version", 2)
        meta.setdefault("entity_type", "concept")
        meta.setdefault("title", filename or "Untitled")
        safe_filename = Path(str(filename or target_path or "")).name if (filename or target_path) else None
        intent_hash = _fingerprint({"metadata": meta, "body": body, "companions": companion_artifacts or {}})
        document_key = str(meta.get("document_key") or "") or None
        operation = "publish_capture" if str(meta.get("entity_type") or "").lower() == "capture" else "upsert_note"
        command = KnowledgeWriteCommand(
            operation=operation,
            idempotency_key=f"compat:{document_key or meta.get('title')}:{intent_hash}",
            document_key=document_key,
            entity_type=str(meta.get("entity_type") or "concept"),
            producer=str(meta.get("producer") or "application-note-writer"),
            producer_version=str(meta.get("producer_version") or "r9"),
            actor=str(meta.get("actor") or "app"),
            expected_content_hash=expected_current_hash,
            payload={"metadata": meta, "body": body, "filename": safe_filename, "companion_artifacts": dict(companion_artifacts or {})},
        )
        receipt = self.write_port.submit(command)
        if not receipt.is_success:
            raise RuntimeError(receipt.error_message or f"Vault write failed: {receipt.status}")
        if not receipt.relative_path or not receipt.note_id or not receipt.revision_id:
            raise RuntimeError("Vault write receipt is missing committed artifact identity")
        primary = self.vault_paths.safe_resolve(receipt.relative_path)
        manifest_path = self.vault_paths.revision_path(receipt.note_id, receipt.revision_id, "manifest.json")
        companion_files: list[Path] = []
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            companion_files = [primary.parent / str(name) for name in (manifest.get("required_companions") or [])]
            revision = int(manifest.get("revision", 1))
        except (OSError, ValueError, TypeError):
            revision = 1
        return CommittedArtifactSet(
            note_id=receipt.note_id,
            revision_id=receipt.revision_id,
            revision=revision,
            primary_file=primary,
            companion_files=companion_files,
            content_hash=str(receipt.content_hash or ""),
            is_reused=receipt.status == "duplicate_reused",
            manifest_path=manifest_path,
            artifact_set_hash=receipt.artifact_set_hash or receipt.content_hash,
        )

    def write_capture(self, metadata: dict[str, Any], body: str, *, filename: Optional[str] = None) -> Path:
        meta = dict(metadata)
        meta.setdefault("schema_version", 2)
        meta.setdefault("entity_type", "capture")
        meta.setdefault("title", filename or "Capture")
        command = KnowledgeWriteCommand(
            operation="publish_capture",
            idempotency_key=f"capture:{_fingerprint({'metadata': meta, 'body': body, 'filename': filename})}",
            producer=str(meta.get("producer") or "application-capture-writer"),
            producer_version=str(meta.get("producer_version") or "r9"),
            actor=str(meta.get("actor") or "app"),
            payload={"metadata": meta, "body": body, "filename": filename},
        )
        receipt = self.write_port.submit(command)
        if not receipt.is_success or not receipt.relative_path:
            raise RuntimeError(receipt.error_message or f"capture write failed: {receipt.status}")
        return self.vault_paths.safe_resolve(receipt.relative_path)


def build_knowledge_note_writer(*, vault_paths: VaultPaths, write_port: KnowledgeWritePort) -> BrokerKnowledgeNoteWriter:
    """Build the note adapter only from an already-composed write port."""
    return BrokerKnowledgeNoteWriter(vault_paths=vault_paths, write_port=write_port)
