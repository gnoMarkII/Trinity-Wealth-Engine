from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from tools.archivist.hybrid_retriever import hybrid_search


class _Catalog:
    def __init__(self, entry: SimpleNamespace) -> None:
        self.entry = entry

    def iter_notes(self, *, page_size: int = 500):
        yield self.entry


def test_vector_match_returns_the_matched_chunk_instead_of_note_prefix(tmp_path: Path) -> None:
    vault = tmp_path / "memories"
    note = vault / "30_Knowledge_Base" / "Concepts" / "Long.md"
    note.parent.mkdir(parents=True)
    body = "prefix " + ("x" * 9000) + "\nneedle appears after the old read limit\n"
    raw = f"---\ntitle: Long\nentity_type: concept\nsearch_scope: included\n---\n{body}"
    note.write_text(raw, encoding="utf-8")
    entry = SimpleNamespace(
        relative_path="30_Knowledge_Base/Concepts/Long.md",
        note_id="note-long",
        title="Long",
        ticker="",
        content_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )
    catalog = _Catalog(entry)
    catalog_path = tmp_path / "catalog.db"
    catalog_path.write_bytes(b"catalog")
    matched = Document(
        page_content="needle appears after the old read limit",
        metadata={
            "relative_path": entry.relative_path,
            "note_id": entry.note_id,
            "chunk_index": 4,
        },
    )

    result = hybrid_search(
        vault,
        catalog,
        catalog_path,
        "needle",
        vector_results=[matched],
        k=1,
    )[0]

    assert "needle appears after the old read limit" in result.page_content
    assert "prefix" not in result.page_content
    assert result.metadata["content_kind"] == "matched_chunk"
    assert result.metadata["content_truncated"] is False
    assert result.metadata["matched_chunk_index"] == 4


def test_hybrid_retriever_reuses_sha256_record_cache_across_catalog_updates(tmp_path: Path, monkeypatch) -> None:
    vault = tmp_path / "memories"
    note1 = vault / "30_Knowledge_Base" / "Concepts" / "Note1.md"
    note2 = vault / "30_Knowledge_Base" / "Concepts" / "Note2.md"
    note3 = vault / "30_Knowledge_Base" / "Concepts" / "Note3.md"
    note1.parent.mkdir(parents=True, exist_ok=True)

    raw1 = "---\ntitle: Alpha\nentity_type: concept\n---\nFirst note content"
    raw2 = "---\ntitle: Beta\nentity_type: concept\n---\nSecond note content"
    raw3 = "---\ntitle: Gamma\nentity_type: concept\n---\nThird note content"
    note1.write_text(raw1, encoding="utf-8")
    note2.write_text(raw2, encoding="utf-8")
    note3.write_text(raw3, encoding="utf-8")

    entry1 = SimpleNamespace(
        relative_path="30_Knowledge_Base/Concepts/Note1.md",
        note_id="n1",
        title="Alpha",
        ticker="",
        content_sha256=hashlib.sha256(raw1.encode("utf-8")).hexdigest(),
    )
    entry2 = SimpleNamespace(
        relative_path="30_Knowledge_Base/Concepts/Note2.md",
        note_id="n2",
        title="Beta",
        ticker="",
        content_sha256=hashlib.sha256(raw2.encode("utf-8")).hexdigest(),
    )
    entry3 = SimpleNamespace(
        relative_path="30_Knowledge_Base/Concepts/Note3.md",
        note_id="n3",
        title="Gamma",
        ticker="",
        content_sha256=hashlib.sha256(raw3.encode("utf-8")).hexdigest(),
    )

    class _MultiCatalog:
        def __init__(self, entries):
            self.entries = entries
        def iter_notes(self, *, page_size: int = 500):
            yield from self.entries

    catalog_path = tmp_path / "catalog_multi.db"
    catalog_path.write_bytes(b"v1")

    # Pass 1: notes 1 and 2
    cat_v1 = _MultiCatalog([entry1, entry2])
    res1 = hybrid_search(vault, cat_v1, catalog_path, "Alpha", k=2)
    assert any("First note" in d.page_content for d in res1)

    # Pass 2: catalog changed (mtime/size altered), note 3 added.
    catalog_path.write_bytes(b"v2_longer_content")

    read_calls: list[str] = []
    original_read_text = Path.read_text

    def spy_read_text(self, *args, **kwargs):
        read_calls.append(str(self))
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy_read_text)

    cat_v2 = _MultiCatalog([entry1, entry2, entry3])
    res2 = hybrid_search(vault, cat_v2, catalog_path, "Gamma", k=3)
    assert any("Third note" in d.page_content for d in res2)

    # Note 1 and Note 2 must NOT be read from disk again because their SHA-256 matched the cache!
    read_paths = [p.replace("\\", "/") for p in read_calls]
    assert not any("Note1.md" in p for p in read_paths)
    assert not any("Note2.md" in p for p in read_paths)
    assert any("Note3.md" in p for p in read_paths)


class _MultiCatalog:
    def __init__(self, entries):
        self.entries = entries

    def iter_notes(self, *, page_size: int = 500):
        yield from self.entries


def test_r17_deleted_and_moved_note_evicts_warm_cache_even_with_stale_catalog(tmp_path: Path) -> None:
    import tools.archivist.hybrid_retriever as hr

    hr._CACHE.clear()
    hr._RECORD_CACHE.clear()

    vault = tmp_path / "memories"
    note = vault / "30_Knowledge_Base" / "Concepts" / "Ephemeral.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    raw = "---\ntitle: Ephemeral\nentity_type: concept\n---\nUnique phrase dragonfruit"
    note.write_text(raw, encoding="utf-8")

    rel = "30_Knowledge_Base/Concepts/Ephemeral.md"
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    entry = SimpleNamespace(
        relative_path=rel,
        note_id="n-ephemeral",
        title="Ephemeral",
        ticker="",
        content_sha256=sha,
    )
    catalog_path = tmp_path / "catalog_r17_del.db"
    catalog_path.write_bytes(b"v1")

    warm = hr.HybridRetriever(vault, _MultiCatalog([entry]), catalog_path)
    assert len(warm.records) == 1
    assert any(k[1] == rel for k in hr._RECORD_CACHE)

    # Delete source file, then rebuild against a new catalog generation that still contains the stale entry
    note.unlink()
    catalog_path.write_bytes(b"v2_stale_entry_still_present")
    rebuilt = hr.HybridRetriever(vault, _MultiCatalog([entry]), catalog_path)
    assert rebuilt.records == []
    assert not any(k[1] == rel for k in hr._RECORD_CACHE)

    # Vector chunk alone must not resurrect a deleted note
    stale_chunk = Document(
        page_content="dragonfruit chunk",
        metadata={"relative_path": rel, "note_id": "n-ephemeral"},
    )
    assert hr.hybrid_search(vault, _MultiCatalog([entry]), catalog_path, "dragonfruit", vector_results=[stale_chunk], k=3) == []

    # Same behavior when a note is moved away from its old catalog path (directory or moved target)
    moved_target = vault / "30_Knowledge_Base" / "Concepts" / "Moved.md"
    moved_target.write_text(raw, encoding="utf-8")
    rebuilt_moved = hr.HybridRetriever(vault, _MultiCatalog([entry]), catalog_path)
    assert rebuilt_moved.records == []


def test_r17_unreadable_note_excluded_and_evicts_warm_cache(tmp_path: Path, monkeypatch) -> None:
    import os
    import tools.archivist.hybrid_retriever as hr

    hr._CACHE.clear()
    hr._RECORD_CACHE.clear()

    vault = tmp_path / "memories"
    note = vault / "30_Knowledge_Base" / "Concepts" / "Locked.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    raw = "---\ntitle: Locked\nentity_type: concept\n---\nConfidential body"
    note.write_text(raw, encoding="utf-8")

    rel = "30_Knowledge_Base/Concepts/Locked.md"
    entry = SimpleNamespace(
        relative_path=rel,
        note_id="n-locked",
        title="Locked",
        ticker="",
        content_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )
    catalog_path = tmp_path / "catalog_r17_lock.db"
    catalog_path.write_bytes(b"v1")

    assert len(hr.HybridRetriever(vault, _MultiCatalog([entry]), catalog_path).records) == 1

    # Simulate OS permission denial via os.access
    real_access = os.access
    monkeypatch.setattr(os, "access", lambda p, mode: False if "Locked.md" in str(p) else real_access(p, mode))
    catalog_path.write_bytes(b"v2")
    rebuilt = hr.HybridRetriever(vault, _MultiCatalog([entry]), catalog_path)
    assert rebuilt.records == []
    assert not any(k[1] == rel for k in hr._RECORD_CACHE)

    # Restore os.access, warm cache again, then fail Path.open with PermissionError
    monkeypatch.undo()
    catalog_path.write_bytes(b"v3")
    assert len(hr.HybridRetriever(vault, _MultiCatalog([entry]), catalog_path).records) == 1

    real_open = Path.open

    def denying_open(self, *args, **kwargs):
        if self.name == "Locked.md":
            raise PermissionError("access denied")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denying_open)
    catalog_path.write_bytes(b"v4")
    rebuilt_open_err = hr.HybridRetriever(vault, _MultiCatalog([entry]), catalog_path)
    assert rebuilt_open_err.records == []
    assert not any(k[1] == rel for k in hr._RECORD_CACHE)


def test_r17_stale_catalog_row_and_metadata_only_updates_and_retirement(tmp_path: Path, monkeypatch) -> None:
    import json
    import time
    import tools.archivist.hybrid_retriever as hr

    hr._CACHE.clear()
    hr._RECORD_CACHE.clear()

    vault = tmp_path / "memories"
    note = vault / "30_Knowledge_Base" / "Concepts" / "Mutable.md"
    stock = vault / "30_Knowledge_Base" / "Stocks" / "PTT" / "PTT.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    stock.parent.mkdir(parents=True, exist_ok=True)

    raw_v1 = "---\ntitle: Alpha\nentity_type: concept\n---\nOld secret obsidian phrase"
    stock_raw = "---\ntitle: PTT Thesis\nentity_type: stock\nticker: PTT\n---\nEnergy giant thesis"
    note.write_text(raw_v1, encoding="utf-8")
    stock.write_text(stock_raw, encoding="utf-8")

    rel = "30_Knowledge_Base/Concepts/Mutable.md"
    rel_stock = "30_Knowledge_Base/Stocks/PTT/PTT.md"
    sha_v1 = hashlib.sha256(raw_v1.encode("utf-8")).hexdigest()
    sha_stock = hashlib.sha256(stock_raw.encode("utf-8")).hexdigest()

    entry_v1 = SimpleNamespace(
        relative_path=rel,
        note_id="n-mut",
        title="Alpha",
        ticker="",
        content_sha256=sha_v1,
        current_revision=1,
        current_revision_id="rev-1",
    )
    entry_stock = SimpleNamespace(
        relative_path=rel_stock,
        note_id="n-ptt",
        title="PTT Thesis",
        ticker="PTT",
        content_sha256=sha_stock,
    )
    catalog_path = tmp_path / "catalog_r17_meta.db"
    catalog_path.write_bytes(b"v1")

    # Warm cache
    res_init = hr.hybrid_search(vault, _MultiCatalog([entry_v1, entry_stock]), catalog_path, "obsidian", k=2)
    assert len(res_init) == 1

    # 1) Metadata-only catalog update: unchanged body on disk MUST NOT re-read file, yet refreshes note_id/title/ticker/rev
    entry_meta_v2 = SimpleNamespace(
        relative_path=rel,
        note_id="n-mut-renamed",
        title="Renamed Title",
        ticker="NVDA",
        content_sha256=sha_v1,
        current_revision=2,
        current_revision_id="rev-2",
    )
    read_calls: list[str] = []
    original_read_text = Path.read_text

    def spy_read_text(self, *args, **kwargs):
        read_calls.append(str(self).replace("\\", "/"))
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy_read_text)
    catalog_path.write_bytes(b"v2_meta_only")
    res_meta = hr.hybrid_search(vault, _MultiCatalog([entry_meta_v2, entry_stock]), catalog_path, "NVDA", k=2)
    assert len(res_meta) == 1
    assert res_meta[0].metadata["note_id"] == "n-mut-renamed"
    assert res_meta[0].metadata["title"] == "Renamed Title"
    assert res_meta[0].metadata["ticker"] == "NVDA"
    assert res_meta[0].metadata["current_revision"] == 2
    assert not any("Mutable.md" in p for p in read_calls)
    monkeypatch.undo()

    # 2) Retirement tombstones (by note_id & stock path) + restore event even with warm cache
    tombstone_path = vault / ".system" / "retired_notes.jsonl"
    tombstone_path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {"status": "retired", "note_id": "n-mut-renamed"},
        {"status": "retired", "relative_path": rel_stock},
    ]
    tombstone_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    assert hr.hybrid_search(vault, _MultiCatalog([entry_meta_v2, entry_stock]), catalog_path, "obsidian", k=5) == []
    assert hr.hybrid_search(vault, _MultiCatalog([entry_meta_v2, entry_stock]), catalog_path, "PTT", k=5) == []

    # Restore concept note only; PTT remains retired
    events.append({"status": "restored", "note_id": "n-mut-renamed"})
    tombstone_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    res_restored = hr.hybrid_search(vault, _MultiCatalog([entry_meta_v2, entry_stock]), catalog_path, "obsidian", k=5)
    assert len(res_restored) == 1
    assert hr.hybrid_search(vault, _MultiCatalog([entry_meta_v2, entry_stock]), catalog_path, "PTT", k=5) == []

    # 3) Stale catalog row: inactive record_state / storage_scope must exclude & evict cache
    entry_inactive = SimpleNamespace(
        relative_path=rel,
        note_id="n-mut-renamed",
        title="Renamed Title",
        ticker="NVDA",
        content_sha256=sha_v1,
        record_state="archived",
        storage_scope="active",
    )
    catalog_path.write_bytes(b"v3_inactive")
    assert hr.HybridRetriever(vault, _MultiCatalog([entry_inactive]), catalog_path).records == []
    assert not any(k[1] == rel for k in hr._RECORD_CACHE)

    # 4) Stale catalog row: body altered on disk while catalog still carries old sha_v1 must NOT resurrect old cached phrase
    # Re-warm with v1 first
    catalog_path.write_bytes(b"v4_rewarm")
    assert len(hr.HybridRetriever(vault, _MultiCatalog([entry_meta_v2]), catalog_path).records) == 1
    time.sleep(0.01)
    raw_v2 = "---\ntitle: Alpha\nentity_type: concept\n---\nFresh ruby phrase on disk"
    note.write_text(raw_v2, encoding="utf-8")
    catalog_path.write_bytes(b"v5_stale_sha")
    assert hr.hybrid_search(vault, _MultiCatalog([entry_meta_v2]), catalog_path, "obsidian", k=5) == []
    res_fresh = hr.hybrid_search(vault, _MultiCatalog([entry_meta_v2]), catalog_path, "ruby", k=5)
    assert len(res_fresh) == 1
    assert (str(vault.resolve()), rel, sha_v1) not in hr._RECORD_CACHE

