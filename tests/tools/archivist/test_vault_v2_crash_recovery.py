"""Acceptance & crash recovery tests for Vault V2 ArtifactWriter and ArtifactStore (Task F03)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.archivist.artifact_store import (
    ArtifactNotFoundError,
    DurableArtifactStore,
    ManifestDigestMismatchError,
)
from tools.archivist.artifact_writer import (
    ArtifactWriter,
    StaleWriteConflictError,
    recover_pending_writes,
)
from tools.archivist.vault_paths import VaultPaths


def test_artifact_writer_commits_frozen_manifest_and_store_reads(tmp_path: Path) -> None:
    vp = VaultPaths(root=tmp_path)
    writer = ArtifactWriter(vault_paths=vp)
    store = DurableArtifactStore(vault_paths=vp)

    meta = {
        "title": "Alpha Research Note",
        "entity_type": "stock",
        "ticker": "AAPL",
        "date": "2026-09-08",
    }
    body = "# AAPL Analysis\nStrong earnings expected."
    companions = {
        "metrics.json": json.dumps({"pe_ratio": 28.5, "revenue": 100_000_000}),
    }

    committed = writer.write_note(
        metadata=meta,
        body=body,
        companion_artifacts=companions,
    )

    assert committed.note_id.startswith("note_")
    assert committed.revision == 1
    assert committed.primary_file.exists()

    # Verify frozen revision archive
    rev_dir = tmp_path / "40_Archive" / "Revisions" / committed.note_id / committed.revision_id
    assert rev_dir.exists()
    assert (rev_dir / "manifest.json").exists()
    assert (rev_dir / "metrics.json").exists()

    # Verify reading through DurableArtifactStore
    stored = store.get_revision_artifact(
        note_id=committed.note_id,
        revision_id=committed.revision_id,
        verify_digests=True,
    )
    assert stored.note_id == committed.note_id
    assert stored.revision == 1
    assert "metrics.json" in stored.companions
    assert json.loads(stored.companions["metrics.json"].decode("utf-8"))["pe_ratio"] == 28.5


def test_artifact_store_detects_corrupted_companion(tmp_path: Path) -> None:
    vp = VaultPaths(root=tmp_path)
    writer = ArtifactWriter(vault_paths=vp)
    store = DurableArtifactStore(vault_paths=vp)

    meta = {"title": "Beta Note", "entity_type": "concept"}
    body = "Body content"
    companions = {"data.json": '{"status": "ok"}'}

    committed = writer.write_note(metadata=meta, body=body, companion_artifacts=companions)

    # Tamper with the companion in frozen revision
    rev_dir = tmp_path / "40_Archive" / "Revisions" / committed.note_id / committed.revision_id
    comp_file = rev_dir / "data.json"
    comp_file.write_text('{"status": "tampered"}', encoding="utf-8")

    with pytest.raises(ManifestDigestMismatchError, match="Companion data.json SHA-256 mismatch"):
        store.get_revision_artifact(committed.note_id, committed.revision_id, verify_digests=True)


def test_crash_recovery_completes_interrupted_pointer_switch(tmp_path: Path) -> None:
    vp = VaultPaths(root=tmp_path)
    writer = ArtifactWriter(vault_paths=vp)

    meta = {"title": "Crash Note", "entity_type": "concept"}
    body = "Initial uncommitted body"

    # Simulate crash right after staging and journal write
    stage_dir = tmp_path / ".system" / "pending_writes" / "crash123"
    stage_dir.mkdir(parents=True, exist_ok=True)

    target_note = tmp_path / "10_Concepts" / "Crash_Note.md"
    target_note.parent.mkdir(parents=True, exist_ok=True)
    staged_primary = stage_dir / "Crash_Note.md"
    staged_primary.write_text("---\ntitle: Crash Note\nnote_id: note_crash_test\n---\nRecovered body", encoding="utf-8")

    journal_data = {
        "write_id": "crash123",
        "target": str(target_note),
        "note_id": "note_crash_test",
        "revision_id": "rev_crash123",
        "revision": 1,
        "companions": [],
        "timestamp": "2026-09-08T00:00:00Z",
    }
    (stage_dir / "journal.json").write_text(json.dumps(journal_data), encoding="utf-8")

    # Before recovery: target does not have recovered body
    assert not target_note.exists()

    # Run crash recovery
    recovered = recover_pending_writes(root=tmp_path)
    assert "recovered_crash123" in recovered

    # After recovery: target note has recovered content and staging is cleaned up
    assert target_note.exists()
    assert "Recovered body" in target_note.read_text(encoding="utf-8")
    assert not stage_dir.exists()


def test_v2_staged_journal_without_immutable_reference_is_retained(tmp_path: Path) -> None:
    """Staged v2 bytes must not become current content before durable commit."""
    stage_dir = tmp_path / ".system" / "pending_writes" / "staged_only"
    stage_dir.mkdir(parents=True, exist_ok=True)
    target_note = tmp_path / "10_Concepts" / "Should_Not_Project.md"
    (stage_dir / target_note.name).write_text(
        "---\nnote_id: note_staged_only\n---\nunsafe staged bytes",
        encoding="utf-8",
    )
    (stage_dir / "journal.json").write_text(
        json.dumps(
            {
                "journal_version": 2,
                "write_id": "staged_only",
                "target": str(target_note),
                "note_id": "note_staged_only",
                "revision_id": "rev_staged_only",
                "companions": [],
                "companion_hashes": {},
            }
        ),
        encoding="utf-8",
    )

    assert recover_pending_writes(root=tmp_path) == []
    assert not target_note.exists()
    assert (stage_dir / "journal.json").exists()


def test_stale_write_conflict_rejected(tmp_path: Path) -> None:
    vp = VaultPaths(root=tmp_path)
    writer = ArtifactWriter(vault_paths=vp)

    meta = {"title": "Concurrency Note", "entity_type": "concept"}
    writer.write_note(metadata=meta, body="Version 1")

    with pytest.raises(StaleWriteConflictError, match="Stale write conflict"):
        writer.write_note(
            metadata=meta,
            body="Version 2 competing",
            expected_current_hash="bad_stale_hash_12345",
        )
