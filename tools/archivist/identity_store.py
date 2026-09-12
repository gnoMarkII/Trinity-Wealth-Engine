"""Durable Identity Store for Obsidian Vault V2.

Persists note identity allocations cross-process using file locking.
Guarantees that concurrent workers or retries for the same document_key
always receive the exact same note_id.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Optional, Union

import threading
from filelock import FileLock

from application.knowledge.identity import IdentityReservationPort, NoteIdentity
from tools.archivist.core import _atomic_write_text
from tools.archivist.maintenance_guard import assert_write_allowed

log = logging.getLogger(__name__)


class IdentityConflictError(Exception):
    """Raised when import_note_identity detects a conflicting existing mapping."""


class DuplicateNoteIdError(Exception):
    """Raised when a note_id is already mapped to a different document_key."""


class StoreCorruptError(Exception):
    """Raised when the identity store file cannot be parsed after all retry attempts."""


_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


def _get_process_lock(path_str: str) -> threading.RLock:
    import os
    norm_path = os.path.normcase(os.path.abspath(path_str))
    with _PROCESS_LOCKS_GUARD:
        if norm_path not in _PROCESS_LOCKS:
            _PROCESS_LOCKS[norm_path] = threading.RLock()
        return _PROCESS_LOCKS[norm_path]


class DurableIdentityStore(IdentityReservationPort):
    """File-backed identity store with cross-process locking."""

    def __init__(self, root: Union[str, Path, None] = None) -> None:
        if root is not None:
            self._root = Path(root).resolve()
        else:
            from tools.archivist.vault_paths import VaultPaths
            self._root = VaultPaths().root

        self._store_dir = self._root / ".system"
        self._store_file = self._store_dir / "identity_allocations.json"
        self._lock_file = self._store_dir / "identity_allocations.lock"

    def _ensure_store(self) -> None:
        self._store_dir.mkdir(parents=True, exist_ok=True)
        if not self._store_file.exists():
            _atomic_write_text(self._store_file, "{}")

    def _read_allocations(self) -> dict[str, dict[str, Any]]:
        """Read allocations from disk. Raises StoreCorruptError if unrecoverable.

        NEVER returns {} silently when the file exists but is corrupt — that would
        allow re-allocation of previously assigned IDs. Instead, the corrupt file is
        quarantined and StoreCorruptError is raised so callers can respond explicitly.
        """
        import time
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                if not self._store_file.exists():
                    return {}
                with self._store_file.open("r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        return {}
                    data = json.loads(content)
                    if isinstance(data, dict):
                        return data
                    # File contains valid JSON but not a dict — treat as corrupt
                    raise ValueError(f"Identity store is not a JSON object (got {type(data).__name__})")
            except json.JSONDecodeError as e:
                last_exc = e
                time.sleep(0.02)
            except Exception as e:
                last_exc = e
                time.sleep(0.02)

        # All attempts failed — quarantine the corrupt file before raising
        quarantine_path = self._store_file.with_suffix(".corrupt")
        try:
            corrupt_bytes = self._store_file.read_bytes()
            assert_write_allowed(quarantine_path)
            quarantine_path.write_bytes(corrupt_bytes)
            log.error(
                "Identity store corrupt at %s after 5 attempts; quarantined to %s. Error: %s",
                self._store_file, quarantine_path, last_exc,
            )
        except Exception as qe:
            log.error("Failed to quarantine corrupt store: %s", qe)

        raise StoreCorruptError(
            f"Identity store at {self._store_file} is corrupt and could not be parsed "
            f"after 5 attempts. Quarantined to {quarantine_path}. Last error: {last_exc}"
        )


    def _save_allocations(self, data: dict[str, dict[str, Any]]) -> None:
        _atomic_write_text(
            self._store_file,
            json.dumps(data, indent=2, ensure_ascii=False),
        )

    def reserve_note_identity(
        self,
        document_key: str,
        entity_id: Optional[str] = None,
    ) -> NoteIdentity:
        """Atomically reserves or retrieves the existing NoteIdentity for document_key."""
        thread_lock = _get_process_lock(str(self._lock_file))
        file_lock = FileLock(str(self._lock_file), timeout=15)

        with thread_lock:
            with file_lock:
                self._ensure_store()
                allocations = self._read_allocations()
                if document_key in allocations:
                    rec = allocations[document_key]
                    return NoteIdentity(
                        note_id=rec["note_id"],
                        document_key=document_key,
                        entity_id=rec.get("entity_id") or entity_id,
                        created_at=rec.get("created_at"),
                    )

                # Allocate new stable note_id
                now_str = datetime.now(timezone.utc).isoformat()
                new_id = f"note_{uuid.uuid4().hex[:12]}"
                allocations[document_key] = {
                    "note_id": new_id,
                    "document_key": document_key,
                    "entity_id": entity_id,
                    "created_at": now_str,
                }
                self._save_allocations(allocations)

                return NoteIdentity(
                    note_id=new_id,
                    document_key=document_key,
                    entity_id=entity_id,
                    created_at=now_str,
                )

    def import_note_identity(
        self,
        note_id: str,
        document_key: str,
        entity_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> NoteIdentity:
        """Imports and pins an existing note_id for document_key.

        Raises IdentityConflictError if the document_key already maps to a DIFFERENT note_id.
        The store is never mutated when a conflict is detected.
        """
        thread_lock = _get_process_lock(str(self._lock_file))
        file_lock = FileLock(str(self._lock_file), timeout=15)

        with thread_lock:
            with file_lock:
                self._ensure_store()
                allocations = self._read_allocations()
                if document_key in allocations:
                    rec = allocations[document_key]
                    if rec["note_id"] != note_id:
                        raise IdentityConflictError(
                            f"Import conflict for '{document_key}': "
                            f"existing note_id='{rec['note_id']}' conflicts with "
                            f"imported note_id='{note_id}'. "
                            f"The store was NOT mutated."
                        )
                    # Same note_id → idempotent, return existing record
                    return NoteIdentity(
                        note_id=rec["note_id"],
                        document_key=document_key,
                        entity_id=rec.get("entity_id"),
                        created_at=rec["created_at"],
                    )

                # Check reverse index: same note_id already mapped to a different doc_key?
                for existing_key, existing_rec in allocations.items():
                    if existing_rec["note_id"] == note_id and existing_key != document_key:
                        raise DuplicateNoteIdError(
                            f"note_id='{note_id}' is already mapped to document_key='{existing_key}'. "
                            f"Cannot also map it to '{document_key}'."
                        )

                now_str = created_at or datetime.now(timezone.utc).isoformat()
                allocations[document_key] = {
                    "note_id": note_id,
                    "document_key": document_key,
                    "entity_id": entity_id,
                    "created_at": now_str,
                }
                self._save_allocations(allocations)
                return NoteIdentity(
                    note_id=note_id,
                    document_key=document_key,
                    entity_id=entity_id,
                    created_at=now_str,
                )

    def bulk_import_note_identities(
        self,
        records: Iterable[NoteIdentity | dict[str, Any]],
    ) -> dict[str, Any]:
        """Atomically import a batch of proven identities with one store write.

        The complete batch and the existing store are validated before mutation.
        This makes a large legacy-vault import both efficient and all-or-nothing:
        one conflicting document key or reused note ID aborts the whole batch.
        """
        incoming: list[NoteIdentity] = []
        for raw in records:
            if isinstance(raw, NoteIdentity):
                identity = raw
            elif isinstance(raw, dict):
                identity = NoteIdentity(
                    note_id=str(raw.get("note_id") or "").strip(),
                    document_key=str(raw.get("document_key") or "").strip(),
                    entity_id=raw.get("entity_id"),
                    created_at=raw.get("created_at"),
                )
            else:
                raise TypeError(
                    "bulk identity records must be NoteIdentity instances or dictionaries"
                )
            if not identity.note_id or not identity.document_key:
                raise ValueError("note_id and document_key are required for bulk import")
            incoming.append(identity)

        thread_lock = _get_process_lock(str(self._lock_file))
        file_lock = FileLock(str(self._lock_file), timeout=15)
        with thread_lock:
            with file_lock:
                self._ensure_store()
                allocations = self._read_allocations()
                candidate = dict(allocations)
                key_to_id = {
                    str(key): str(record.get("note_id") or "")
                    for key, record in candidate.items()
                }
                id_to_key: dict[str, str] = {}
                for key, note_id in key_to_id.items():
                    if not note_id:
                        raise StoreCorruptError(
                            f"Identity store record '{key}' has no note_id"
                        )
                    prior_key = id_to_key.get(note_id)
                    if prior_key is not None and prior_key != key:
                        raise StoreCorruptError(
                            f"Identity store maps note_id='{note_id}' to both "
                            f"'{prior_key}' and '{key}'"
                        )
                    id_to_key[note_id] = key

                imported = 0
                reused = 0
                now_str = datetime.now(timezone.utc).isoformat()
                results: list[NoteIdentity] = []
                for identity in incoming:
                    existing_id = key_to_id.get(identity.document_key)
                    if existing_id is not None:
                        if existing_id != identity.note_id:
                            raise IdentityConflictError(
                                f"Import conflict for '{identity.document_key}': "
                                f"existing note_id='{existing_id}' conflicts with "
                                f"imported note_id='{identity.note_id}'. "
                                "The store was NOT mutated."
                            )
                        rec = candidate[identity.document_key]
                        reused += 1
                        results.append(
                            NoteIdentity(
                                note_id=str(rec["note_id"]),
                                document_key=identity.document_key,
                                entity_id=rec.get("entity_id"),
                                created_at=rec.get("created_at"),
                            )
                        )
                        continue

                    existing_key = id_to_key.get(identity.note_id)
                    if existing_key is not None:
                        raise DuplicateNoteIdError(
                            f"note_id='{identity.note_id}' is already mapped to "
                            f"document_key='{existing_key}'. Cannot also map it to "
                            f"'{identity.document_key}'."
                        )

                    created_at = identity.created_at or now_str
                    rec = {
                        "note_id": identity.note_id,
                        "document_key": identity.document_key,
                        "entity_id": identity.entity_id,
                        "created_at": created_at,
                    }
                    candidate[identity.document_key] = rec
                    key_to_id[identity.document_key] = identity.note_id
                    id_to_key[identity.note_id] = identity.document_key
                    imported += 1
                    results.append(
                        NoteIdentity(
                            note_id=identity.note_id,
                            document_key=identity.document_key,
                            entity_id=identity.entity_id,
                            created_at=created_at,
                        )
                    )

                if imported:
                    self._save_allocations(candidate)
                return {
                    "imported": imported,
                    "reused": reused,
                    "total": len(results),
                    "identities": results,
                }


    def get_note_identity(self, document_key: str) -> Optional[NoteIdentity]:
        """Retrieves existing NoteIdentity by document_key if already allocated."""
        thread_lock = _get_process_lock(str(self._lock_file))
        file_lock = FileLock(str(self._lock_file), timeout=10)

        with thread_lock:
            with file_lock:
                self._ensure_store()
                allocations = self._read_allocations()
                if document_key in allocations:
                    rec = allocations[document_key]
                    return NoteIdentity(
                        note_id=rec["note_id"],
                        document_key=document_key,
                        entity_id=rec.get("entity_id"),
                        created_at=rec.get("created_at"),
                    )
                return None

    def get_note_identity_by_note_id(self, note_id: str) -> Optional[NoteIdentity]:
        """Find an existing durable identity by opaque note_id for drift repair."""
        target_id = str(note_id or "").strip()
        if not target_id:
            return None
        thread_lock = _get_process_lock(str(self._lock_file))
        file_lock = FileLock(str(self._lock_file), timeout=10)

        with thread_lock:
            with file_lock:
                self._ensure_store()
                allocations = self._read_allocations()
                for document_key, record in allocations.items():
                    if str(record.get("note_id") or "") == target_id:
                        return NoteIdentity(
                            note_id=target_id,
                            document_key=str(document_key),
                            entity_id=record.get("entity_id"),
                            created_at=record.get("created_at"),
                        )
        return None
