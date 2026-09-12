"""Tests for ArtifactWriter, optimistic concurrency, and revision snapshot freezing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.archivist.artifact_writer import (
    ArtifactWriter,
    StaleWriteConflictError,
    recover_pending_writes,
)
from tools.archivist.identity_store import DurableIdentityStore
from tools.archivist.vault_paths import VaultPaths


def test_write_note_new_and_revision_freeze(tmp_path: Path) -> None:
    """Writing a new note creates primary file and freezes revision in 40_Archive/Revisions/."""
    vp = VaultPaths(root=tmp_path)
    store = DurableIdentityStore(root=tmp_path)
    writer = ArtifactWriter(vault_paths=vp, identity_store=store)

    meta = {
        "entity_type": "equity_analysis",
        "ticker": "FTNT",
        "title": "2026-09-05 FTNT Equity Analysis",
        "date": "2026-09-05",
    }
    body = "# FTNT Analysis\nBullish sentiment with high score."

    res = writer.write_note(metadata=meta, body=body, filename="2026-09-05 FTNT Equity Analysis")

    assert res.revision == 1
    assert res.is_reused is False
    assert res.primary_file.exists()
    assert "FTNT/Analysis" in res.primary_file.as_posix()

    # Check frozen revision snapshot
    rev_path = vp.revision_path(note_id=res.note_id, revision_id=res.revision_id, filename="2026-09-05 FTNT Equity Analysis.md")
    assert rev_path.exists()
    assert body in rev_path.read_text(encoding="utf-8")


def test_write_note_idempotent_reuse(tmp_path: Path) -> None:
    """Writing identical payload reuses existing revision without incrementing counter."""
    vp = VaultPaths(root=tmp_path)
    writer = ArtifactWriter(vault_paths=vp)

    meta = {"entity_type": "stock_hub", "ticker": "FTNT", "title": "FTNT"}
    body = "# FTNT Hub\nInitial content."

    res1 = writer.write_note(meta, body)
    assert res1.revision == 1
    assert res1.is_reused is False

    # Second write with exact same body and metadata
    res2 = writer.write_note(meta, body)
    assert res2.revision == 1
    assert res2.is_reused is True
    assert res2.note_id == res1.note_id
    assert res2.revision_id == res1.revision_id


def test_write_note_stale_write_conflict(tmp_path: Path) -> None:
    """Mismatch in expected_current_hash must raise StaleWriteConflictError."""
    vp = VaultPaths(root=tmp_path)
    writer = ArtifactWriter(vault_paths=vp)

    meta = {"entity_type": "stock_hub", "ticker": "FTNT", "title": "FTNT"}
    body1 = "# FTNT Hub\nInitial version."
    res1 = writer.write_note(meta, body1)

    # Competing change modified the file
    body2 = "# FTNT Hub\nModified by competitor."
    res2 = writer.write_note(meta, body2)
    assert res2.revision == 2

    # Now an agent tries to write expecting hash from version 1
    with pytest.raises(StaleWriteConflictError, match="Stale write conflict"):
        writer.write_note(meta, "# FTNT Hub\nStale update.", expected_current_hash=res1.content_hash)


def test_write_note_with_companion_artifacts(tmp_path: Path) -> None:
    """Writes note along with companion JSON sidecar and freezes both in revision."""
    vp = VaultPaths(root=tmp_path)
    writer = ArtifactWriter(vault_paths=vp)

    meta = {
        "entity_type": "equity_analysis",
        "ticker": "FTNT",
        "title": "2026-09-05 FTNT Equity Analysis",
    }
    body = "# Analysis MD"
    sidecar_json = json.dumps({"ticker": "FTNT", "score": 85})

    res = writer.write_note(
        metadata=meta,
        body=body,
        filename="2026-09-05 FTNT Equity Analysis",
        companion_artifacts={"2026-09-05 FTNT Equity Analysis.json": sidecar_json},
    )

    assert len(res.companion_files) == 1
    sidecar_path = res.companion_files[0]
    assert sidecar_path.exists()
    assert "85" in sidecar_path.read_text(encoding="utf-8")

    # Check companion in frozen revision
    frozen_sidecar = vp.revision_path(
        note_id=res.note_id,
        revision_id=res.revision_id,
        filename="2026-09-05 FTNT Equity Analysis.json",
    )
    assert frozen_sidecar.exists()
    assert "85" in frozen_sidecar.read_text(encoding="utf-8")


def test_recover_pending_writes(tmp_path: Path) -> None:
    """recover_pending_writes cleans up any incomplete stages."""
    pending_dir = tmp_path / ".system" / "pending_writes" / "crash1"
    pending_dir.mkdir(parents=True)
    journal = pending_dir / "journal.json"
    journal.write_text(json.dumps({"write_id": "crash1"}), encoding="utf-8")

    recovered = recover_pending_writes(root=tmp_path)
    assert len(recovered) == 1
    assert "crash1" in recovered[0]
    assert not pending_dir.exists()
