"""Tests for DurableIdentityStore cross-process identity allocation and locking."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.archivist.identity_store import DurableIdentityStore
from tools.archivist.identity_store import DuplicateNoteIdError, IdentityConflictError


def test_reserve_identity_deterministic_and_reusable(tmp_path: Path) -> None:
    """Reserving identity for the same document_key returns the exact same note_id."""
    store = DurableIdentityStore(root=tmp_path)
    doc_key = "v1:stock_hub:FTNT:hub"

    id1 = store.reserve_note_identity(doc_key, entity_id="entity_ftnt_01")
    assert id1.note_id.startswith("note_")
    assert id1.document_key == doc_key
    assert id1.entity_id == "entity_ftnt_01"

    # Second call for the same key must return identical note_id
    id2 = store.reserve_note_identity(doc_key)
    assert id2.note_id == id1.note_id
    assert id2.entity_id == "entity_ftnt_01"

    # get_note_identity must match
    id_get = store.get_note_identity(doc_key)
    assert id_get is not None
    assert id_get.note_id == id1.note_id


def test_concurrent_identity_reservation(tmp_path: Path) -> None:
    """Multiple concurrent workers requesting the same document_key must receive identical note_id."""
    store = DurableIdentityStore(root=tmp_path)
    doc_key = "v1:equity_analysis:FTNT:primary"

    def _worker(worker_id: int):
        s = DurableIdentityStore(root=tmp_path)
        return s.reserve_note_identity(doc_key)

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(_worker, range(10)))

    # All workers must have gotten the exact same note_id
    allocated_ids = {r.note_id for r in results}
    assert len(allocated_ids) == 1


def test_bulk_import_is_atomic_and_idempotent(tmp_path: Path) -> None:
    store = DurableIdentityStore(root=tmp_path)
    records = [
        {"note_id": "note_legacy_a", "document_key": "legacy:a"},
        {"note_id": "note_legacy_b", "document_key": "legacy:b"},
    ]

    first = store.bulk_import_note_identities(records)
    assert first["imported"] == 2
    assert first["reused"] == 0

    second = store.bulk_import_note_identities(records)
    assert second["imported"] == 0
    assert second["reused"] == 2
    assert store.get_note_identity("legacy:a").note_id == "note_legacy_a"


def test_bulk_import_conflict_does_not_partially_mutate_store(tmp_path: Path) -> None:
    store = DurableIdentityStore(root=tmp_path)
    store.import_note_identity("note_existing", "legacy:existing")

    try:
        store.bulk_import_note_identities(
            [
                {"note_id": "note_new", "document_key": "legacy:new"},
                {"note_id": "note_other", "document_key": "legacy:existing"},
            ]
        )
    except IdentityConflictError:
        pass
    else:
        raise AssertionError("expected bulk document-key conflict")
    assert store.get_note_identity("legacy:new") is None

    try:
        store.bulk_import_note_identities(
            [{"note_id": "note_existing", "document_key": "legacy:duplicate-id"}]
        )
    except DuplicateNoteIdError:
        pass
    else:
        raise AssertionError("expected bulk note-id conflict")
    assert store.get_note_identity("legacy:duplicate-id") is None
