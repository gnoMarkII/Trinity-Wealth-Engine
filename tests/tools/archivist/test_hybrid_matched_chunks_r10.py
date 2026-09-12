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

