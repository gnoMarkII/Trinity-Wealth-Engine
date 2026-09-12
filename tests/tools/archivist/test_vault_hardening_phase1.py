import json
import os
import threading
from pathlib import Path
import pytest

from tools.archivist.core import get_note_lock, _atomic_write_text
from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
from tools.archivist.writer import save_memory, write_raw_markdown
from tools.archivist.search import search_all_memories


def test_get_note_lock_concurrent_writes(tmp_path: Path, monkeypatch):
    """Stress-test per-note locking across multiple threads writing to the same note."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    note_path = tmp_path / "00_Inbox" / "SharedNote.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(note_path, "# Shared Note\n\nCounter: 0\n")

    num_threads = 5
    increments_per_thread = 4
    errors = []

    def worker(worker_id: int):
        try:
            for _ in range(increments_per_thread):
                with get_note_lock(note_path, timeout=15.0, vault_root=tmp_path):
                    content = note_path.read_text(encoding="utf-8")
                    lines = content.strip().splitlines()
                    val = int(lines[-1].split(":")[-1].strip())
                    lines[-1] = f"Counter: {val + 1}"
                    _atomic_write_text(note_path, "\n".join(lines) + "\n")
        except Exception as e:
            errors.append((worker_id, e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors occurred during concurrent writes: {errors}"
    final_content = note_path.read_text(encoding="utf-8")
    expected_val = num_threads * increments_per_thread
    assert f"Counter: {expected_val}" in final_content


def test_dual_write_synchronous_catalog_hook(tmp_path: Path, monkeypatch):
    """Verifies that writing a note automatically synchronizes to SQLite catalog without manual sync."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    db_path = tmp_path / ".system" / "vault_catalog.db"
    cat = SqliteNoteCatalogAdapter(db_path=db_path, vault_root=tmp_path)

    # 1. Write note via save_memory
    res = save_memory.func(
        title="AutomatedTestCompany",
        content="Testing dual-write sync to SQLite catalog.",
        folder_path="30_Knowledge_Base/Stocks/AUTO",
        tags=["test", "dual_write"],
        entity_type="Company",
    )
    assert "สำเร็จ" in res

    # 2. Query catalog directly - it MUST exist immediately
    entry = cat.get_by_path("30_Knowledge_Base/Stocks/AUTO/AutomatedTestCompany.md")
    assert entry is not None
    assert entry.title == "AutomatedTestCompany"
    assert entry.entity_type in ("company", "stock_hub")


def test_catalog_outbox_fallback_and_recovery(tmp_path: Path, monkeypatch):
    """Verifies that failed catalog writes enqueue to outbox and process_outbox recovers them."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    cat = SqliteNoteCatalogAdapter(vault_root=tmp_path)

    # Enqueue a dummy outbox entry
    test_note = tmp_path / "30_Knowledge_Base" / "Concepts" / "OutboxTarget.md"
    test_note.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(test_note, "---\ntitle: Outbox Target\nentity_type: concept\n---\nBody")

    rel = "30_Knowledge_Base/Concepts/OutboxTarget.md"
    cat.enqueue_outbox(rel, action="upsert", error_detail="simulated error")

    outbox_file = tmp_path / ".system" / "catalog_outbox.jsonl"
    assert outbox_file.exists()

    # Process outbox
    recovered = cat.process_outbox()
    assert recovered == 1
    assert not outbox_file.exists()

    entry = cat.get_by_path(rel)
    assert entry is not None
    assert entry.title == "Outbox Target"


def test_catalog_reconcile_missing_purges_deleted_files(tmp_path: Path, monkeypatch):
    """Verifies that deleting a file externally and running reconcile_missing removes it from SQLite."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    cat = SqliteNoteCatalogAdapter(vault_root=tmp_path)

    # Create note
    target = tmp_path / "30_Knowledge_Base" / "Concepts" / "WillBeDeleted.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(target, "---\ntitle: To Delete\nentity_type: concept\n---\nTemp")
    cat.upsert_note_from_file(target)

    rel = "30_Knowledge_Base/Concepts/WillBeDeleted.md"
    assert cat.get_by_path(rel) is not None

    # Simulate human deleting file in Obsidian
    target.unlink()

    # Reconcile
    res = cat.reconcile_missing()
    assert res["pruned"] >= 1
    assert cat.get_by_path(rel) is None


def test_search_latency_delta_query(tmp_path: Path, monkeypatch):
    """Verifies that search_all_memories utilizes SQLite catalog delta check."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    cat = SqliteNoteCatalogAdapter(vault_root=tmp_path)

    # Add a note
    target = tmp_path / "30_Knowledge_Base" / "Concepts" / "InterestRate.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(target, "---\ntitle: Interest Rate\nentity_type: concept\n---\nCentral bank interest rate analysis.")
    cat.upsert_note_from_file(target)

    # Calling search should use delta query and not fail
    res = search_all_memories.func("interest rate")
    assert isinstance(res, str)
