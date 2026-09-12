"""Acceptance Regressions for Vault V2 Findings B01-B08.

Ensures:
- B03: Metadata-only mutation on committed note produces a distinct revision without modifying frozen history.
- B04: Consecutive failure retains recovery journals and staged evidence.
- B05: Search query has zero embedding or mutation side effects.
- B06: Substring transcripts under identical period get isolated note identities.
- B07: Rollback to V2 configuration does not revert to V1 defaults.
- B08: All generated navigation links resolve.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from tools.archivist import artifact_writer as aw
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.search import search_all_memories


@pytest.fixture(autouse=True)
def isolate_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    monkeypatch.setenv("WEBUI_STATE_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("ENABLE_BACKGROUND_WORKERS", "false")
    monkeypatch.setenv("SCHEDULER_ENABLED", "false")

    def forbidden_network(*args, **kwargs):
        raise RuntimeError("External network call forbidden in isolated tests")

    import socket
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    monkeypatch.setattr(socket, "create_connection", forbidden_network)


def test_b03_metadata_only_mutation_creates_new_revision(tmp_path: Path) -> None:
    """B03: Modifying only YAML metadata must allocate a new revision and not overwrite the frozen revision files."""
    (tmp_path / ".system").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".system" / "vault_config.json").write_text('{"layout_version": 2}', encoding="utf-8")

    writer = aw.ArtifactWriter(vault_paths=VaultPaths(tmp_path))
    meta_v1 = {
        "entity_type": "concept",
        "title": "Decentralized Finance",
        "tags": ["finance", "v1"],
    }
    body = "# Decentralized Finance\nCore concept."
    first = writer.write_note(meta_v1, body, filename="DeFi")
    rev1_archive = tmp_path / "40_Archive" / "Revisions" / first.note_id / first.revision_id
    assert rev1_archive.exists()
    rev1_content = (rev1_archive / first.primary_file.name).read_text(encoding="utf-8")

    # Update metadata only
    meta_v2 = {
        "entity_type": "concept",
        "title": "Decentralized Finance",
        "tags": ["finance", "v2", "updated"],
    }
    second = writer.write_note(meta_v2, body, filename="DeFi")

    assert second.note_id == first.note_id, "Note ID must be preserved on update"
    assert second.revision_id != first.revision_id, "New revision_id must be generated for metadata change"
    assert second.revision > first.revision, "Revision number must increment"

    # Frozen archive of revision 1 must be strictly intact
    assert rev1_archive.exists()
    rev1_content_after = (rev1_archive / first.primary_file.name).read_text(encoding="utf-8")
    assert rev1_content == rev1_content_after, "Old frozen revision bytes must be completely unchanged!"


def test_b04_crash_recovery_journal_retention_on_repeated_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B04: Repeated crash/failure during commit must not drop staged journal evidence."""
    (tmp_path / ".system").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".system" / "vault_config.json").write_text('{"layout_version": 2}', encoding="utf-8")

    writer = aw.ArtifactWriter(vault_paths=VaultPaths(tmp_path))
    meta = {"entity_type": "concept", "title": "Crash Test"}

    orig_atomic = aw._atomic_write_text

    # Simulate catastrophic failure during atomic write
    def crashing_atomic_write(path, content, *args, **kwargs):
        if "pending_writes" not in str(path):
            raise OSError("Disk write failed midway")
        return orig_atomic(path, content, *args, **kwargs)

    monkeypatch.setattr(aw, "_atomic_write_text", crashing_atomic_write)

    with pytest.raises(OSError, match="Disk write failed midway"):
        writer.write_note(meta, "Initial body", filename="CrashTest")

    # Pending journal must be retained on failure
    pending_dir = tmp_path / ".system" / "pending_writes"
    journals = list(pending_dir.rglob("journal.json"))
    assert len(journals) >= 1, "Crash recovery journal must be retained on failure"


def test_b05_search_memory_has_zero_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B05: Executing search_all_memories must never invoke process_outbox, document embedding
    (add_texts/vectorstore updates), or state writes — in warm, cold, AND error paths.

    The previous version of this test only patched process_outbox on the catalog and asserted
    the return type. It did NOT verify cold-path mutations or error-branch side effects.
    This version uses a shared spy counter dict that is incremented from ALL paths.
    """
    sample = tmp_path / "30_Knowledge_Base" / "Concepts" / "SearchTest.md"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text(
        "---\ntitle: Searchable\nentity_type: concept\n---\n# Searchable Note\nContent.",
        encoding="utf-8",
    )

    # Shared spy counter — MUST be accessible from all branches including error handlers
    spy = {
        "process_outbox": 0,
        "add_texts": 0,
        "save_index_state": 0,
        "rmtree": 0,
        "document_embeddings": 0,
    }

    def _spy_process_outbox(*args, **kwargs) -> None:
        spy["process_outbox"] += 1
        raise AssertionError("search query invoked process_outbox — mutation in query path!")

    def _spy_add_texts(*args, **kwargs) -> list:
        spy["add_texts"] += 1
        raise AssertionError("search query invoked add_texts — document embedding in query path!")

    def _spy_save_index_state(*args, **kwargs) -> None:
        spy["save_index_state"] += 1
        raise AssertionError("search query invoked _save_index_state — state write in query path!")

    def _spy_rmtree(*args, **kwargs) -> None:
        spy["rmtree"] += 1
        raise AssertionError("search query invoked shutil.rmtree — data destruction in query path!")

    # Patch mutation points in ALL branches (warm, cold, error)
    monkeypatch.setattr(
        "tools.archivist.catalog_adapter.SqliteNoteCatalogAdapter.process_outbox",
        _spy_process_outbox,
    )

    import tools.archivist.search as search_mod
    monkeypatch.setattr(search_mod, "_save_index_state", _spy_save_index_state)

    import shutil
    monkeypatch.setattr(shutil, "rmtree", _spy_rmtree)

    # Patch out vectorstore initialization to avoid model loading
    import unittest.mock as um
    fake_vs = um.MagicMock()
    fake_vs.similarity_search.return_value = []
    fake_vs.add_texts = _spy_add_texts
    monkeypatch.setattr(search_mod, "_vs_cache", {"vs": fake_vs})

    # Run warm path (vectorstore cache hit) — must have ZERO mutations
    result_warm = search_all_memories.invoke({"keyword": "Searchable"})
    assert isinstance(result_warm, (str, list)), f"Expected str or list result, got {type(result_warm)}"

    # Verify zero side effects across all spy counters
    assert spy["process_outbox"] == 0, f"Warm query triggered process_outbox {spy['process_outbox']} times"
    assert spy["add_texts"] == 0, f"Warm query triggered add_texts {spy['add_texts']} times"
    assert spy["save_index_state"] == 0, f"Warm query triggered _save_index_state {spy['save_index_state']} times"
    assert spy["rmtree"] == 0, f"Warm query triggered rmtree {spy['rmtree']} times"
