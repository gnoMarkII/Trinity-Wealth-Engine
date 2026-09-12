"""Two-phase atomic artifact writer and revision freezer for Obsidian Vault V2.

Guarantees:
- Cross-process reservation of note_id via DurableIdentityStore.
- Optimistic concurrency control (stale competing changes raise StaleWriteConflictError).
- Content-addressed revisions frozen in 40_Archive/Revisions/NOTE_ID/REVISION_ID/.
- Same payload reuses revision without adding redundant historical records.
- Crash recovery via journaled stages.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

from filelock import FileLock

from application.knowledge.identity import build_document_key
from tools.archivist.core import _atomic_write_text
from tools.archivist.identity_store import DurableIdentityStore
from tools.archivist.maintenance_guard import assert_write_allowed
from tools.archivist.metadata import dump_note, normalize_legacy_metadata, parse_note, validate_note
from tools.archivist.vault_policy import is_retired_note
from tools.archivist.vault_paths import VaultPaths

log = logging.getLogger(__name__)


class StaleWriteConflictError(Exception):
    """Raised when an update targets an outdated revision or content hash."""
    pass


class ArtifactStateCorruptError(Exception):
    """Raised when durable head/manifest state cannot be trusted for a write."""
    pass


@dataclass
class CommittedArtifactSet:
    note_id: str
    revision_id: str
    revision: int
    primary_file: Path
    companion_files: list[Path]
    content_hash: str
    is_reused: bool = False
    manifest_path: Optional[Path] = None
    artifact_set_hash: Optional[str] = None


def _compute_content_hash(text: str) -> str:
    """Computes a canonical SHA-256 hash of text content."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _compute_exact_hash(data: bytes) -> str:
    """SHA-256 of exact persisted bytes (including trailing newlines)."""
    return hashlib.sha256(data).hexdigest()


def _manifest_digest(payload: dict[str, Any]) -> str:
    return _compute_exact_hash(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _validate_companion_name(name: str) -> str:
    """Allow only one filename component for a companion artifact."""
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError(f"Companion filename must stay inside the note directory: {name!r}")
    if name in {".", "..", "manifest.json"}:
        raise ValueError(f"Reserved companion filename: {name!r}")
    return name


def _compute_artifact_set_hash(
    body: str,
    companion_artifacts: Optional[dict[str, Union[str, bytes]]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Computes SHA-256 hash over the multi-artifact set (primary body + companions + semantic metadata)."""
    hasher = hashlib.sha256()
    hasher.update(body.strip().encode("utf-8"))

    if metadata:
        # Include semantic metadata (excluding dynamic run/revision counters)
        excluded_meta = {"revision", "updated_at", "mtime", "file_size", "content_sha256"}
        clean_meta = {k: v for k, v in metadata.items() if k not in excluded_meta}
        meta_str = json.dumps(clean_meta, sort_keys=True, ensure_ascii=False, default=str)
        hasher.update(f"metadata:{meta_str}".encode("utf-8"))

    if companion_artifacts:
        for comp_name in sorted(companion_artifacts.keys()):
            hasher.update(f"companion:{comp_name}:".encode("utf-8"))
            comp_data = companion_artifacts[comp_name]
            if isinstance(comp_data, bytes):
                hasher.update(comp_data)
            else:
                hasher.update(str(comp_data).strip().encode("utf-8"))
    return hasher.hexdigest()


class ArtifactWriter:
    """Atomic writer managing note identities, revision freezing, and companion artifacts."""

    def __init__(
        self,
        vault_paths: Optional[VaultPaths] = None,
        identity_store: Optional[DurableIdentityStore] = None,
    ) -> None:
        self._vault_paths = vault_paths or VaultPaths()
        self._identity_store = identity_store or DurableIdentityStore(root=self._vault_paths.root)

    @property
    def vault_paths(self) -> VaultPaths:
        return self._vault_paths

    @property
    def identity_store(self) -> DurableIdentityStore:
        return self._identity_store

    @property
    def _artifact_root(self) -> Path:
        return self._vault_paths.root / ".system" / "artifacts"

    def _read_head(self, note_id: str) -> dict[str, Any] | None:
        head = self._artifact_root / "heads" / f"{note_id}.json"
        if not head.exists():
            return None
        try:
            value = json.loads(head.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("head must be a JSON object")
            if value.get("note_id") not in (None, note_id):
                raise ValueError("head note_id does not match requested note")
            if not value.get("revision_id"):
                raise ValueError("head is missing revision_id")
            return value
        except (OSError, ValueError) as exc:
            # A corrupt head must never cause a new identity or a silent current
            # fallback. Leave the evidence for an operator/recovery workflow.
            raise ArtifactStateCorruptError(f"Corrupt durable head for {note_id}: {exc}") from exc

    def _current_manifest(self, note_id: str) -> dict[str, Any] | None:
        head = self._read_head(note_id)
        if not head or not head.get("revision_id"):
            return None
        path = self._vault_paths.revision_path(note_id, str(head["revision_id"]), "manifest.json")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("manifest must be a JSON object")
            if value.get("note_id") != note_id or value.get("revision_id") != head.get("revision_id"):
                raise ValueError("manifest identity does not match head")
            if value.get("committed") is not True:
                raise ValueError("current manifest is not committed")
            if head.get("manifest_digest") and _manifest_digest(value) != head["manifest_digest"]:
                raise ValueError("head manifest digest mismatch")
            return value
        except (OSError, ValueError) as exc:
            raise ArtifactStateCorruptError(f"Corrupt current manifest for {note_id}: {exc}") from exc

    def _register_revision(self, manifest: dict[str, Any]) -> Path:
        """Register an immutable revision before exposing its projection."""
        registry_dir = self._artifact_root
        registry_dir.mkdir(parents=True, exist_ok=True)
        registry = registry_dir / "registry.json"
        lock = FileLock(str(registry_dir / "registry.lock"), timeout=15)
        with lock:
            if registry.exists():
                try:
                    data = json.loads(registry.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        raise ValueError("registry must be an object")
                except (OSError, ValueError) as exc:
                    raise OSError(f"Artifact registry is corrupt: {exc}") from exc
            else:
                data = {"version": 1, "revisions": {}, "updated_at": None}
            data.setdefault("version", 1)
            data.setdefault("revisions", {})
            key = f"{manifest['note_id']}:{manifest['revision_id']}"
            digest = _manifest_digest(manifest)
            old = data["revisions"].get(key)
            if old and old.get("manifest_digest") != digest:
                raise OSError(f"Immutable revision registration conflict for {key}")
            data["revisions"][key] = {
                "note_id": manifest["note_id"],
                "revision_id": manifest["revision_id"],
                "revision": manifest["revision"],
                "manifest_digest": digest,
                "manifest_path": str(
                    self._vault_paths.revision_path(
                        manifest["note_id"], manifest["revision_id"], "manifest.json"
                    ).relative_to(self._vault_paths.root).as_posix()
                ),
                "artifact_set_hash": manifest.get("artifact_set_hash"),
            }
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            _atomic_write_text(registry, json.dumps(data, indent=2, ensure_ascii=False))
        return registry

    def _activate_head(self, manifest: dict[str, Any]) -> Path:
        heads = self._artifact_root / "heads"
        heads.mkdir(parents=True, exist_ok=True)
        head = heads / f"{manifest['note_id']}.json"
        payload = {
            "version": 1,
            "note_id": manifest["note_id"],
            "revision_id": manifest["revision_id"],
            "revision": manifest["revision"],
            "manifest_digest": _manifest_digest(manifest),
            "artifact_set_hash": manifest.get("artifact_set_hash"),
            "activated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_text(head, json.dumps(payload, indent=2, ensure_ascii=False))
        return head

    def _identity_for_write(self, meta: dict[str, Any], target_path: Path, doc_key: str):
        """Import an existing opaque ID before reserving a new one."""
        supplied_id = meta.get("note_id")
        if supplied_id:
            return self._identity_store.import_note_identity(
                note_id=str(supplied_id),
                document_key=doc_key,
                entity_id=meta.get("entity_id"),
            )
        if target_path.exists():
            try:
                existing_meta, _, issues = parse_note(target_path.read_text(encoding="utf-8"))
                existing_id = existing_meta.get("note_id")
                existing_key = existing_meta.get("document_key") or doc_key
                if existing_id:
                    # The current file is the strongest local evidence for its
                    # identity. Import it against its stored key and use that key
                    # for future writes, preserving old IDs during metadata repair.
                    if existing_key != doc_key and not meta.get("document_key"):
                        doc_key = existing_key
                        meta["document_key"] = existing_key
                    return self._identity_store.import_note_identity(
                        note_id=str(existing_id),
                        document_key=doc_key,
                        entity_id=meta.get("entity_id"),
                    )
            except UnicodeDecodeError:
                raise ValueError(f"Existing note is not valid UTF-8: {target_path}")
        return self._identity_store.reserve_note_identity(
            document_key=doc_key, entity_id=meta.get("entity_id")
        )

    def write_note(
        self,
        metadata: dict[str, Any],
        body: str,
        filename: Optional[str] = None,
        target_path: Optional[Union[str, Path]] = None,
        expected_current_hash: Optional[str] = None,
        companion_artifacts: Optional[dict[str, Union[str, bytes]]] = None,
        commit_context: Optional[dict[str, Any]] = None,
        commit_guard: Optional[Callable[[], None]] = None,
        force_new_revision: bool = False,
    ) -> CommittedArtifactSet:
        """Writes or updates a note with identity reservation and revision freezing.

        Args:
            metadata: Frontmatter metadata dictionary.
            body: Markdown body text.
            filename: Optional explicit filename.
            target_path: Optional explicit path inside the vault for generic/capture notes.
            expected_current_hash: Expected SHA-256 hash of existing file (for OCC).
            companion_artifacts: Dict of {companion_filename: content} to commit alongside note.
            commit_context: Broker provenance such as command_id, actor, registry
                digest, and fencing token.  It is additive and optional for R7
                compatibility callers.
            commit_guard: Optional fencing callback checked before staging and
                before every irreversible commit phase.  Broker callers use it
                to prevent an expired lease from becoming a filesystem commit.
            force_new_revision: Treat the current projection as an external
                human edit even when the incoming bytes already match it.  The
                durable head remains the OCC baseline and must receive a new
                immutable revision.
        """
        # Check before identity allocation, directory creation, or lock-file
        # creation. A remediation lease must fence every managed writer.
        assert_write_allowed(self._vault_paths.root)
        if commit_guard is not None:
            commit_guard()
        meta, normalization_issues = normalize_legacy_metadata(dict(metadata), producer="artifact_writer")
        if normalization_issues:
            log.debug("Metadata normalization warnings: %s", normalization_issues)
        if companion_artifacts is not None:
            companion_artifacts = {
                _validate_companion_name(name): value
                for name, value in companion_artifacts.items()
            }
        entity_type = meta.get("entity_type", "concept")
        ticker = meta.get("ticker") or (meta.get("tickers")[0] if meta.get("tickers") else None)
        title = meta.get("title") or filename or "Untitled"
        # Every managed write carries the V2 common fields.  Legacy callers may
        # omit schema_version, but they must not be allowed to create another
        # unversioned note through this writer.
        meta.setdefault("schema_version", 2)
        meta.setdefault("entity_type", entity_type)
        meta.setdefault("title", title)

        # 1. Determine document_key and target path.  The path itself is not an
        # identity, but it lets an import preserve an existing legacy note_id.
        doc_key = meta.get("document_key")
        if not doc_key:
            source_id = (
                ticker
                or meta.get("video_id")
                or meta.get("source_url")
                or meta.get("url")
                or meta.get("event_id")
                or meta.get("period")
                or title
            )
            as_of = meta.get("date") or meta.get("as_of")
            scope = meta.get("scope")
            doc_key = build_document_key(
                kind=entity_type,
                source_identity=source_id,
                role="primary",
                as_of=as_of,
                scope=scope,
            )
            meta["document_key"] = doc_key
        # Validate before reserving/importing an identity, then validate again
        # after the durable note_id is attached.  This prevents malformed
        # domain payloads from leaving an identity allocation behind.
        candidate_meta = dict(meta)
        candidate_meta.setdefault("note_id", "pending")
        candidate_model, candidate_issues = validate_note(candidate_meta, mode="strict")
        if candidate_model is None:
            raise ValueError(f"Invalid V2 metadata: {candidate_issues}")
        if is_retired_note(
            self._vault_paths.root,
            note_id=meta.get("note_id"),
            document_key=str(doc_key),
        ):
            raise ValueError(
                "Refusing to materialize a retired note identity/document_key: "
                f"{meta.get('note_id') or doc_key}"
            )
        # Resolve the canonical target before identity import; note_id is not part
        # of the path contract.
        target_path = (
            self._vault_paths.safe_resolve(target_path)
            if target_path is not None
            else self._vault_paths.note_path(meta, filename=filename)
        )
        identity = self._identity_for_write(meta, target_path, str(doc_key))
        meta["note_id"] = identity.note_id
        if identity.entity_id and not meta.get("entity_id"):
            meta["entity_id"] = identity.entity_id
        validated_meta, validation_issues = validate_note(meta, mode="strict")
        if validated_meta is None:
            raise ValueError(f"Invalid V2 metadata: {validation_issues}")

        # 2. Compare current projection and calculate the next revision.
        target_dir = target_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        lock_file = target_dir / f".{target_path.name}.lock"
        lock = FileLock(str(lock_file), timeout=15)

        with lock:
            if commit_guard is not None:
                commit_guard()
            current_revision = 1
            is_reused = False
            existing_content = ""
            effective_companions: Optional[dict[str, Union[str, bytes]]] = companion_artifacts
            removed_companions: list[Path] = []

            # Check existing file under lock
            if target_path.exists():
                try:
                    existing_content = target_path.read_text(encoding="utf-8")
                    existing_meta, existing_body, _ = parse_note(existing_content)
                    current_revision = int(existing_meta.get("revision", 1))
                    current_manifest = self._current_manifest(identity.note_id)
                    if current_manifest:
                        current_revision = max(
                            current_revision, int(current_manifest.get("revision", current_revision))
                        )

                    # If the caller omits companions, preserve the current set.
                    # If it supplies a set, that set is the complete next set;
                    # deleting a companion is therefore an explicit revision.
                    existing_companion_names = list(
                        (current_manifest or {}).get("required_companions", [])
                    )
                    if not existing_companion_names and current_manifest:
                        existing_companion_names = list(
                            (current_manifest or {}).get("companions", {}).keys()
                        )
                    existing_companions: dict[str, Union[str, bytes]] = {}
                    for comp_name in existing_companion_names:
                        comp_p = target_dir / comp_name
                        if comp_p.exists():
                            existing_companions[comp_name] = comp_p.read_bytes()
                    effective_companions = (
                        companion_artifacts
                        if companion_artifacts is not None
                        else existing_companions
                    )
                    if companion_artifacts is not None:
                        removed_companions = [
                            target_dir / name
                            for name in existing_companion_names
                            if name not in companion_artifacts
                        ]

                    existing_set_hash = _compute_artifact_set_hash(
                        existing_body,
                        existing_companions if existing_companions else None,
                        metadata=existing_meta,
                    )
                    existing_body_hash = _compute_content_hash(existing_body)

                    # Check for stale competing write
                    if expected_current_hash is not None and expected_current_hash not in (existing_set_hash, existing_body_hash):
                        raise StaleWriteConflictError(
                            f"Stale write conflict for {target_path}: expected hash "
                            f"{expected_current_hash[:10]}, actual {existing_set_hash[:10]}"
                        )

                    incoming_set_hash = _compute_artifact_set_hash(
                        body, effective_companions, metadata=meta
                    )
                    if incoming_set_hash == existing_set_hash and not force_new_revision:
                        is_reused = True
                        # No-op reuse: preserve the exact revision ID registered
                        # by the current head.  A hash-derived fallback is only a
                        # legacy compatibility path and is never written as a new
                        # revision ID.
                        revision_id = (
                            (current_manifest or {}).get("revision_id")
                            or (self._read_head(identity.note_id) or {}).get("revision_id")
                            or existing_meta.get("_revision_id")
                        )
                        if not revision_id:
                            raise OSError(
                                "Cannot reuse current payload without a registered current revision"
                            )
                        meta["revision"] = current_revision
                    else:
                        current_revision += 1
                        meta["revision"] = current_revision
                        # Use opaque UUID so A→B→A yields three distinct revision_ids,
                        # not a collision between revision-1 and revision-3.
                        revision_id = f"rev_{uuid.uuid4().hex[:16]}"
                except (StaleWriteConflictError, ArtifactStateCorruptError):
                    raise
                except Exception as e:
                    log.warning("Could not parse existing note for revision check: %s", e)
                    current_revision += 1
                    meta["revision"] = current_revision
                    incoming_set_hash = _compute_artifact_set_hash(
                        body,
                        companion_artifacts,
                        metadata=meta,
                    )
                    revision_id = f"rev_{uuid.uuid4().hex[:16]}"
            else:
                meta["revision"] = 1
                incoming_set_hash = _compute_artifact_set_hash(body, companion_artifacts, metadata=meta)
                revision_id = f"rev_{uuid.uuid4().hex[:16]}"

            # The rest of the commit uses the exact set that participated in the
            # comparison, so the frozen snapshot and current projection cannot
            # silently disagree about companions.
            companion_artifacts = effective_companions
            target_companions = [
                target_dir / name for name in (companion_artifacts or {})
            ]
            new_content = dump_note(meta, body)
            content_hash = incoming_set_hash

            safe_commit_context = None
            if commit_context:
                safe_commit_context = {
                    key: commit_context[key]
                    for key in (
                        "command_id",
                        "actor",
                        "producer",
                        "registry_digest",
                        "fencing_token",
                        "correlation_id",
                    )
                    if commit_context.get(key) is not None
                }

            if is_reused:
                # A successful idempotent retry must not rewrite either the
                # immutable snapshot or the current projection.  The returned
                # reference is the one read from the durable head above.
                manifest_path = self._vault_paths.revision_path(
                    identity.note_id, revision_id, "manifest.json"
                )
                return CommittedArtifactSet(
                    note_id=identity.note_id,
                    revision_id=str(revision_id),
                    revision=current_revision,
                    primary_file=target_path,
                    companion_files=target_companions,
                    content_hash=content_hash,
                    is_reused=True,
                    manifest_path=manifest_path,
                    artifact_set_hash=content_hash,
                )

            # Stage files into .system/pending_writes/{write_id}/
            if commit_guard is not None:
                commit_guard()
            write_id = uuid.uuid4().hex[:8]
            stage_dir = self._vault_paths.root / ".system" / "pending_writes" / write_id
            stage_dir.mkdir(parents=True, exist_ok=True)

            try:
                staged_primary = stage_dir / target_path.name
                _atomic_write_text(staged_primary, new_content)

                staged_companions: list[Path] = []
                companion_hashes: dict[str, str] = {}

                if companion_artifacts:
                    for comp_name, comp_data in companion_artifacts.items():
                        staged_comp = stage_dir / comp_name
                        if isinstance(comp_data, bytes):
                            staged_comp.write_bytes(comp_data)
                            companion_hashes[comp_name] = _compute_exact_hash(comp_data)
                        else:
                            str_data = str(comp_data)
                            _atomic_write_text(staged_comp, str_data)
                            # Hash the bytes that were actually persisted. On
                            # Windows TextIO may normalize line endings.
                            companion_hashes[comp_name] = _compute_exact_hash(staged_comp.read_bytes())
                        staged_companions.append(staged_comp)

                # Hash the staged bytes, not the pre-write Python string, so the
                # manifest describes the exact filesystem representation.
                primary_sha256 = _compute_exact_hash(staged_primary.read_bytes())
                artifact_manifest = {
                    "manifest_version": 1,
                    "committed": True,
                    "note_id": identity.note_id,
                    "revision_id": revision_id,
                    "revision": current_revision,
                    "artifact_set_hash": content_hash,
                    "primary_file": target_path.name,
                    "projection_path": target_path.relative_to(self._vault_paths.root).as_posix(),
                    "primary_sha256": primary_sha256,
                    "companions": companion_hashes,
                    "required_companions": sorted(companion_hashes),
                    "committed_at": datetime.now(timezone.utc).isoformat(),
                }
                if safe_commit_context:
                    artifact_manifest["commit_context"] = safe_commit_context

                # Write journal marker BEFORE any mutations.  It names exact
                # bytes and the predecessor so recovery can distinguish a user
                # edit from an interrupted projection update.
                journal_data = {
                    "journal_version": 2,
                    "write_id": write_id,
                    "target": str(target_path),
                    "note_id": identity.note_id,
                    "revision_id": revision_id,
                    "revision": current_revision,
                    "content_hash": content_hash,
                    "primary_sha256": primary_sha256,
                    "companion_hashes": companion_hashes,
                    "companions": [str(p) for p in target_companions],
                    "removed_companions": [str(p) for p in removed_companions],
                    "previous_primary_sha256": _compute_exact_hash(existing_content.encode("utf-8")) if existing_content else None,
                    "phase": "staged",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if safe_commit_context:
                    journal_data["commit_context"] = safe_commit_context
                journal_file = stage_dir / "journal.json"
                _atomic_write_text(journal_file, json.dumps(journal_data, indent=2))

                # PHASE 1: Freeze revision snapshot and manifest into 40_Archive/Revisions/NOTE_ID/REVISION_ID/
                if commit_guard is not None:
                    commit_guard()
                rev_file = self._vault_paths.revision_path(
                    note_id=identity.note_id,
                    revision_id=revision_id,
                    filename=target_path.name,
                )
                rev_dir = rev_file.parent
                rev_dir.mkdir(parents=True, exist_ok=True)

                # Only write frozen files if not already existing (immutable past revisions)
                if not rev_file.exists():
                    _atomic_write_text(rev_file, new_content)
                for t_comp, s_comp in zip(target_companions, staged_companions):
                    comp_rev_file = rev_dir / t_comp.name
                    if not comp_rev_file.exists():
                        if t_comp.suffix.lower() == ".json" or not isinstance(companion_artifacts[s_comp.name], bytes):
                            _atomic_write_text(comp_rev_file, s_comp.read_text(encoding="utf-8"))
                        else:
                            comp_rev_file.write_bytes(s_comp.read_bytes())

                # Write the complete immutable manifest.  The manifest itself
                # is not included in its own digest, avoiding a hash cycle.
                rev_manifest = rev_dir / "manifest.json"
                if not rev_manifest.exists():
                    _atomic_write_text(
                        rev_manifest,
                        json.dumps(artifact_manifest, indent=2, ensure_ascii=False),
                    )
                else:
                    existing_manifest = json.loads(rev_manifest.read_text(encoding="utf-8"))
                    if _manifest_digest(existing_manifest) != _manifest_digest(artifact_manifest):
                        raise OSError(f"Immutable revision already exists with different bytes: {rev_dir}")

                # Register and activate the durable head before projection. A
                # reader can therefore always choose a complete frozen set even
                # if the current Markdown refresh is interrupted.
                if commit_guard is not None:
                    commit_guard()
                self._register_revision(artifact_manifest)
                self._activate_head(artifact_manifest)
                journal_data["revision_dir"] = str(rev_dir)
                journal_data["manifest_digest"] = _manifest_digest(artifact_manifest)
                journal_data["phase"] = "head_activated"
                _atomic_write_text(journal_file, json.dumps(journal_data, indent=2, ensure_ascii=False))

                # PHASE 2: Atomic pointer switch into target locations
                if commit_guard is not None:
                    commit_guard()
                _atomic_write_text(target_path, new_content)
                for s_comp, t_comp in zip(staged_companions, target_companions):
                    if s_comp.suffix.lower() == ".json" or not isinstance(companion_artifacts[s_comp.name], bytes):
                        _atomic_write_text(t_comp, s_comp.read_text(encoding="utf-8"))
                    else:
                        t_comp.write_bytes(s_comp.read_bytes())
                for removed in removed_companions:
                    if removed.exists():
                        removed.unlink()

                journal_data["phase"] = "complete"
                _atomic_write_text(journal_file, json.dumps(journal_data, indent=2, ensure_ascii=False))

                # Success! Remove staging directory now.  A reader never needs
                # this staging directory after the committed head exists.
                shutil.rmtree(stage_dir, ignore_errors=True)

                return CommittedArtifactSet(
                    note_id=identity.note_id,
                    revision_id=revision_id,
                    revision=current_revision,
                    primary_file=target_path,
                    companion_files=target_companions,
                    content_hash=content_hash,
                    is_reused=is_reused,
                    manifest_path=rev_manifest,
                    artifact_set_hash=content_hash,
                )
            except Exception:
                # Do NOT clean up staging or journal on failure so recover_pending_writes can handle it
                raise
            finally:
                # FileLock owns lock lifecycle.  Do not unlink the lock path:
                # another process may have opened the same lock between release
                # and this finally block.
                pass


def recover_pending_writes(root: Union[str, Path, None] = None) -> list[str]:
    """Recover pending projections without discarding failed evidence.

    New journals are replayed only after the staged bytes and immutable manifest
    match their recorded hashes.  Older journals from the first V2 migration are
    supported as a compatibility path; if they fail, their directory remains for
    an operator to inspect.
    """
    vp = VaultPaths(root=root)
    pending_dir = vp.root / ".system" / "pending_writes"
    recovered: list[str] = []

    if not pending_dir.exists():
        return recovered

    for entry in sorted(pending_dir.iterdir()):
        if not entry.is_dir():
            continue
        journal = entry / "journal.json"
        if not journal.is_file():
            continue
        try:
            data = json.loads(journal.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("journal must be an object")
            write_id = str(data.get("write_id") or entry.name)
            target_str = data.get("target")
            if not target_str:
                # A legacy crash can leave only a write marker. It cannot be
                # replayed safely. There are no staged bytes or target metadata
                # to preserve in this marker-only case, so remove the empty
                # marker after recording an explicit recovery-required result.
                recovered.append(f"recovery_required_{write_id}")
                shutil.rmtree(entry, ignore_errors=True)
                continue
            target = Path(target_str).resolve()
            if not target.is_relative_to(vp.root):
                raise ValueError(f"journal target escapes vault root: {target}")
            staged_primary = entry / target.name
            if not staged_primary.is_file():
                raise FileNotFoundError(f"staged primary missing: {staged_primary}")

            journal_version = int(data.get("journal_version", 1) or 1)
            if journal_version >= 2:
                # A v2 journal is replayable only after the immutable revision
                # was registered and activated.  In particular, a journal that
                # stopped after staging must remain evidence; projecting those
                # bytes would create a current note with no durable history.
                revision_dir = data.get("revision_dir")
                expected_manifest_digest = data.get("manifest_digest")
                if not revision_dir or not expected_manifest_digest:
                    raise ValueError(
                        "v2 journal is missing immutable revision reference; retaining evidence"
                    )
                rev_dir = Path(revision_dir).resolve()
                if not rev_dir.is_relative_to(vp.root):
                    raise ValueError("journal revision_dir escapes vault root")
                manifest_path = rev_dir / "manifest.json"
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"immutable manifest missing: {manifest_path}")
                try:
                    immutable_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise OSError(f"immutable manifest unreadable: {manifest_path}") from exc
                if not isinstance(immutable_manifest, dict):
                    raise OSError(f"immutable manifest is not an object: {manifest_path}")
                if (
                    immutable_manifest.get("committed") is not True
                    or immutable_manifest.get("note_id") != data.get("note_id")
                    or immutable_manifest.get("revision_id") != data.get("revision_id")
                ):
                    raise OSError("immutable manifest identity/commit marker mismatch")
                if _manifest_digest(immutable_manifest) != expected_manifest_digest:
                    raise OSError(f"immutable manifest digest mismatch: {manifest_path}")
                manifest_companions = immutable_manifest.get("companions") or {}
                journal_companions = data.get("companion_hashes") or {}
                if manifest_companions != journal_companions:
                    raise OSError("journal companion set differs from immutable manifest")
                if immutable_manifest.get("primary_file") != target.name:
                    raise OSError("journal target differs from immutable primary file")
                registry_path = vp.root / ".system" / "artifacts" / "registry.json"
                if not registry_path.is_file():
                    raise FileNotFoundError(f"artifact registry missing: {registry_path}")
                try:
                    registry_data = json.loads(registry_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise OSError(f"artifact registry unreadable: {registry_path}") from exc
                registry_key = f"{data.get('note_id')}:{data.get('revision_id')}"
                registry_entry = (registry_data.get("revisions") or {}).get(registry_key)
                if not isinstance(registry_entry, dict) or registry_entry.get("manifest_digest") != expected_manifest_digest:
                    raise OSError("immutable revision is not registered with the journal digest")
                head_path = vp.root / ".system" / "artifacts" / "heads" / f"{data.get('note_id')}.json"
                if not head_path.is_file():
                    raise FileNotFoundError(f"artifact head missing: {head_path}")
                try:
                    head_data = json.loads(head_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise OSError(f"artifact head unreadable: {head_path}") from exc
                if (
                    not isinstance(head_data, dict)
                    or head_data.get("revision_id") != data.get("revision_id")
                    or head_data.get("manifest_digest") != expected_manifest_digest
                ):
                    raise OSError("artifact head does not point at journal revision")

            # Verify bytes before touching the current projection when hashes are
            # present. Legacy journals do not have these fields and use the
            # compatibility branch below.
            primary_bytes = staged_primary.read_bytes()
            expected_primary = data.get("primary_sha256")
            if expected_primary and _compute_exact_hash(primary_bytes) != expected_primary:
                raise OSError("staged primary digest mismatch")

            previous = data.get("previous_primary_sha256")
            if previous and target.exists():
                current_hash = _compute_exact_hash(target.read_bytes())
                if current_hash not in (previous, expected_primary):
                    raise StaleWriteConflictError(
                        f"Recovery conflict: current projection was edited at {target}"
                    )

            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(target, primary_bytes.decode("utf-8"))

            for comp_str in data.get("companions", []):
                comp_path = Path(comp_str).resolve()
                if not comp_path.is_relative_to(vp.root):
                    raise ValueError(f"journal companion escapes vault root: {comp_path}")
                staged_comp = entry / comp_path.name
                if not staged_comp.is_file():
                    raise FileNotFoundError(f"staged companion missing: {staged_comp}")
                comp_bytes = staged_comp.read_bytes()
                expected = (data.get("companion_hashes") or {}).get(comp_path.name)
                if expected and _compute_exact_hash(comp_bytes) != expected:
                    raise OSError(f"staged companion digest mismatch: {comp_path.name}")
                comp_path.parent.mkdir(parents=True, exist_ok=True)
                if comp_path.suffix.lower() == ".json":
                    _atomic_write_text(comp_path, comp_bytes.decode("utf-8"))
                else:
                    comp_path.write_bytes(comp_bytes)

            for removed_str in data.get("removed_companions", []):
                removed_path = Path(removed_str).resolve()
                if not removed_path.is_relative_to(vp.root):
                    raise ValueError(f"journal removed companion escapes vault root: {removed_path}")
                removed_path.unlink(missing_ok=True)

            data["phase"] = "complete"
            _atomic_write_text(journal, json.dumps(data, indent=2, ensure_ascii=False))
            recovered.append(f"recovered_{write_id}")
            shutil.rmtree(entry, ignore_errors=True)
        except Exception as exc:
            # Keep journal and staged files for a later retry/operator review.
            log.warning("Failed to recover pending write %s: %s", entry, exc)

    return recovered
