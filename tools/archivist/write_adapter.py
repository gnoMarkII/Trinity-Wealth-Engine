"""Concrete executor behind the application-level Vault write boundary.

The application layer knows only about commands and receipts.  This adapter is
the one place where a command is translated into the existing atomic
``ArtifactWriter`` contract.  It intentionally does not expose a filesystem
path API to callers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

from filelock import FileLock

from application.knowledge.write_models import KnowledgeWriteCommand
from tools.archivist.artifact_writer import (
    ArtifactWriter,
    CommittedArtifactSet,
    StaleWriteConflictError,
)
from tools.archivist.artifact_store import ArtifactError, DurableArtifactStore
from tools.archivist.core import _atomic_write_text
from tools.archivist.maintenance_guard import assert_write_allowed
from tools.archivist.metadata import dump_note, normalize_legacy_metadata
from tools.archivist.metadata import parse_note
from tools.archivist.schema_registry import SchemaRegistry, load_default_registry
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.vault_policy import sanitize_filename


class WriteAdapterError(RuntimeError):
    """A command cannot be translated into a safe canonical write."""


@dataclass(frozen=True)
class AdapterCommit:
    """Durable result of one canonical write operation."""

    note_id: Optional[str] = None
    revision_id: Optional[str] = None
    relative_path: Optional[str] = None
    content_hash: Optional[str] = None
    artifact_set_hash: Optional[str] = None
    warnings: tuple[str, ...] = ()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ArtifactWriterKnowledgeAdapter:
    """Translate broker commands into the single canonical writer."""

    def __init__(
        self,
        *,
        vault_paths: Optional[VaultPaths] = None,
        registry: Optional[SchemaRegistry] = None,
        writer: Optional[ArtifactWriter] = None,
        fencing_guard: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        self.vault_paths = vault_paths or VaultPaths()
        self.registry = registry or load_default_registry()
        self.writer = writer or ArtifactWriter(vault_paths=self.vault_paths)
        self.fencing_guard = fencing_guard

    def commit(self, command: KnowledgeWriteCommand, *, fencing_token: int = 0) -> AdapterCommit:
        payload = dict(command.payload)
        operation = command.operation
        raw_operation_metadata = payload.get("metadata") or payload.get("frontmatter")
        if operation == "publish_capture" and isinstance(raw_operation_metadata, Mapping) and str(raw_operation_metadata.get("entity_type") or "").strip().lower() == "capture" and not payload.get("published_entity_type"):
            return self._commit_capture(command, payload, fencing_token=fencing_token)
        if operation in {"upsert_note", "publish_capture", "append_journal_entry", "register_attachment"}:
            return self._commit_note(command, payload, fencing_token=fencing_token)
        if operation == "retire_note":
            return self._retire(command, payload, fencing_token=fencing_token)
        if operation == "restore_note":
            return self._restore(command, payload, fencing_token=fencing_token)
        if operation == "regenerate_projection":
            raise WriteAdapterError(
                "regenerate_projection requires a projection worker; submit a worker-specific command"
            )
        raise WriteAdapterError(f"Unsupported write operation: {operation}")

    def _commit_note(
        self,
        command: KnowledgeWriteCommand,
        payload: dict[str, Any],
        *,
        fencing_token: int,
    ) -> AdapterCommit:
        raw_metadata = payload.get("metadata")
        if raw_metadata is None:
            raw_metadata = payload.get("frontmatter")
        if not isinstance(raw_metadata, Mapping):
            raise WriteAdapterError("note write payload requires metadata mapping")
        metadata, _ = normalize_legacy_metadata(dict(raw_metadata), producer=command.producer)

        self._assert_expected_revision(command, metadata)

        # A capture is an input state, not a publishable knowledge profile.  A
        # publisher must explicitly supply the canonical entity type it wants
        # to materialize; the old capture envelope remains rejected.
        if command.operation == "publish_capture":
            if str(metadata.get("entity_type") or "").strip().lower() == "capture":
                promoted = payload.get("published_entity_type")
                if not promoted:
                    raise WriteAdapterError(
                        "publish_capture requires payload.published_entity_type when metadata is capture"
                    )
                metadata["entity_type"] = str(promoted).strip().lower()
            metadata.pop("capture_status", None)
            metadata.pop("capture_source", None)
            metadata.pop("captured_at", None)
            metadata["search_scope"] = "included"

        profile_id = payload.get("profile_id")
        profile = self.registry.profile_for(metadata.get("entity_type"), profile_id=profile_id)
        if profile is None:
            raise WriteAdapterError(
                f"metadata has no registered profile for entity_type={metadata.get('entity_type')!r}"
            )
        metadata.setdefault("search_scope", profile.search_scope_default)
        metadata.setdefault("retention_class", self.registry.retention_class_for(profile.profile_id))
        metadata.setdefault(
            "content_status",
            "generated" if profile.profile_id in {"navigation", "derived_artifact"} else "published",
        )
        metadata.setdefault("document_role", "knowledge" if profile.profile_id == "published" else profile.profile_id)
        valid, issues = self.registry.validate_metadata(
            metadata,
            profile_id=profile_id,
            allow_identity_allocation=True,
        )
        if not valid:
            details = "; ".join(f"{item['field']}: {item['reason']}" for item in issues)
            raise WriteAdapterError(f"metadata violates Vault registry: {details}")

        body = payload.get("body", "")
        if not isinstance(body, str):
            raise WriteAdapterError("note write payload.body must be a string")
        companion_artifacts = payload.get("companion_artifacts")
        if companion_artifacts is not None and not isinstance(companion_artifacts, Mapping):
            raise WriteAdapterError("companion_artifacts must be a mapping")
        filename = payload.get("filename")
        target_path = payload.get("target_path")
        if target_path is not None and command.actor not in {"migration", "reconciliation"}:
            raise WriteAdapterError("client-supplied target_path is not allowed for application writes")
        expected_hash = command.expected_content_hash or payload.get("expected_current_hash")
        committed = self.writer.write_note(
            metadata,
            body,
            filename=str(filename) if filename is not None else None,
            target_path=str(target_path) if target_path is not None else None,
            expected_current_hash=str(expected_hash) if expected_hash else None,
            companion_artifacts=dict(companion_artifacts) if companion_artifacts is not None else None,
            commit_context={
                "command_id": command.command_id,
                "actor": command.actor,
                "producer": command.producer,
                "registry_digest": self.registry.digest(),
                "fencing_token": fencing_token,
                "correlation_id": command.correlation_id,
            },
            commit_guard=(
                (lambda: self.fencing_guard(command.command_id, fencing_token))
                if self.fencing_guard is not None and fencing_token
                else None
            ),
            force_new_revision=(command.actor == "reconciliation" and command.expected_revision_id is not None),
        )
        return self._from_committed(committed)

    def _assert_expected_revision(
        self,
        command: KnowledgeWriteCommand,
        metadata: Mapping[str, Any],
    ) -> None:
        """Apply revision-level OCC before ArtifactWriter allocates/stages."""
        expected = str(command.expected_revision_id or "").strip()
        if not expected:
            return
        note_id = str(metadata.get("note_id") or "").strip()
        if not note_id:
            raise WriteAdapterError(
                "expected_revision_id requires metadata.note_id so the current head can be checked"
            )
        head_path = self.vault_paths.root / ".system" / "artifacts" / "heads" / f"{note_id}.json"
        try:
            head = json.loads(head_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise StaleWriteConflictError(
                f"expected revision {expected!r} has no durable head for note {note_id!r}"
            ) from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WriteAdapterError(f"cannot read durable head for {note_id!r}: {exc}") from exc
        actual = str(head.get("revision_id") or "") if isinstance(head, Mapping) else ""
        if actual != expected:
            raise StaleWriteConflictError(
                f"stale revision for {note_id}: expected {expected}, actual {actual or '<missing>'}"
            )
        # A revision head alone cannot detect a human edit made directly in
        # Obsidian after the head was created.  Application updates therefore
        # also compare the current projection with the immutable baseline;
        # reconciliation/migration actors are explicitly allowed to import or
        # repair that difference through their own guarded paths.
        if command.actor in {"reconciliation", "migration", "vault-maintenance"}:
            return
        try:
            baseline = DurableArtifactStore(vault_paths=self.vault_paths).get_revision_artifact(note_id, expected)
            projection_path = str(baseline.manifest.get("projection_path") or "").strip()
            if not projection_path:
                return
            projection = self.vault_paths.safe_resolve(projection_path)
            if not projection.is_file():
                raise StaleWriteConflictError(f"current projection is missing for note {note_id!r}")
            current_metadata, current_body, issues = parse_note(projection.read_text(encoding="utf-8"))
            if issues:
                raise StaleWriteConflictError(f"current projection metadata is malformed for note {note_id!r}")
            if current_metadata != baseline.metadata or current_body != baseline.body:
                raise StaleWriteConflictError(
                    f"current projection changed outside the broker for note {note_id!r}; reconcile the human edit first"
                )
        except StaleWriteConflictError:
            raise
        except (ArtifactError, OSError, UnicodeDecodeError, ValueError) as exc:
            raise WriteAdapterError(f"cannot verify current projection for note {note_id!r}: {exc}") from exc

    def _commit_capture(
        self,
        command: KnowledgeWriteCommand,
        payload: dict[str, Any],
        *,
        fencing_token: int,
    ) -> AdapterCommit:
        raw_metadata = payload.get("metadata") or payload.get("frontmatter")
        if not isinstance(raw_metadata, Mapping):
            raise WriteAdapterError("capture payload requires metadata mapping")
        metadata, _ = normalize_legacy_metadata(dict(raw_metadata), producer=command.producer)
        valid, issues = self.registry.validate_metadata(metadata, profile_id="capture")
        if not valid:
            details = "; ".join(f"{item['field']}: {item['reason']}" for item in issues)
            raise WriteAdapterError(f"capture metadata violates Vault registry: {details}")
        body = payload.get("body", "")
        if not isinstance(body, str):
            raise WriteAdapterError("capture payload.body must be a string")
        filename = sanitize_filename(str(payload.get("filename") or metadata.get("title") or "capture"))
        if not filename.lower().endswith(".md"):
            filename += ".md"
        target = self.vault_paths.safe_resolve(Path("00_Inbox") / filename)
        if self.fencing_guard is not None and fencing_token:
            self.fencing_guard(command.command_id, fencing_token)
        assert_write_allowed(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = dump_note(metadata, body)
        _atomic_write_text(target, text)
        digest = _sha256(text.encode("utf-8"))
        return AdapterCommit(
            relative_path=target.relative_to(self.vault_paths.root).as_posix(),
            content_hash=digest,
            artifact_set_hash=digest,
        )

    def _from_committed(self, committed: CommittedArtifactSet) -> AdapterCommit:
        return AdapterCommit(
            note_id=committed.note_id,
            revision_id=committed.revision_id,
            relative_path=committed.primary_file.resolve().relative_to(self.vault_paths.root).as_posix(),
            content_hash=committed.content_hash,
            artifact_set_hash=committed.artifact_set_hash or committed.content_hash,
            warnings=("revision_reused",) if committed.is_reused else (),
        )

    def _retire(
        self,
        command: KnowledgeWriteCommand,
        payload: dict[str, Any],
        *,
        fencing_token: int,
    ) -> AdapterCommit:
        note_id = str(payload.get("note_id") or command.document_key or "").strip() or None
        document_key = str(payload.get("document_key") or command.document_key or "").strip() or None
        if not note_id and not document_key:
            raise WriteAdapterError("retire_note requires note_id or document_key")
        return self._append_tombstone(
            note_id=note_id,
            document_key=document_key,
            status="retired",
            reason=str(payload.get("reason") or "application_retire"),
            command_id=command.command_id,
            fencing_token=fencing_token,
        )

    def _restore(
        self,
        command: KnowledgeWriteCommand,
        payload: dict[str, Any],
        *,
        fencing_token: int,
    ) -> AdapterCommit:
        note_id = str(payload.get("note_id") or command.document_key or "").strip() or None
        document_key = str(payload.get("document_key") or command.document_key or "").strip() or None
        if not note_id and not document_key:
            raise WriteAdapterError("restore_note requires note_id or document_key")
        return self._append_tombstone(
            note_id=note_id,
            document_key=document_key,
            status="restored",
            reason=str(payload.get("reason") or "application_restore"),
            command_id=command.command_id,
            fencing_token=fencing_token,
        )

    def _append_tombstone(
        self,
        *,
        note_id: Optional[str],
        document_key: Optional[str],
        status: str,
        reason: str,
        command_id: str = "",
        fencing_token: int = 0,
    ) -> AdapterCommit:
        path = self.vault_paths.root / ".system" / "retired_notes.jsonl"
        if self.fencing_guard is not None and fencing_token:
            self.fencing_guard(command_id or "", fencing_token)
        assert_write_allowed(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event_version": 2,
            "status": status,
            "note_id": note_id,
            "document_key": document_key,
            "reason": reason,
            "changed_at": _utc_now(),
        }
        lock = FileLock(str(path.with_suffix(path.suffix + ".lock")), timeout=15)
        with lock:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            new_text = existing.rstrip("\n") + ("\n" if existing.strip() else "") + json.dumps(event, ensure_ascii=False) + "\n"
            _atomic_write_text(path, new_text)
        return AdapterCommit(
            relative_path=path.resolve().relative_to(self.vault_paths.root).as_posix(),
            content_hash=_sha256(new_text.encode("utf-8")),
            artifact_set_hash=_sha256(new_text.encode("utf-8")),
        )
