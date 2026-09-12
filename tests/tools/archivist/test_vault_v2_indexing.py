"""Unit tests for Incremental Vector Indexing Worker (T09b)."""
from pathlib import Path
import pytest

from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
from tools.archivist.indexing_worker import FakeEmbeddings, IndexingWorker


def test_incremental_vector_indexing_with_fake_embeddings(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    kb_dir = vault / "30_Knowledge_Base" / "Concepts"
    kb_dir.mkdir(parents=True)

    note1 = kb_dir / "Inflation.md"
    note1.write_text("""---
schema_version: 2
note_id: concept_inflation
document_key: "v1:concept:inflation:primary"
entity_type: concept
title: Inflation Overview
---
# Inflation Dynamics
Inflation is the rate of increase in prices over a given period of time.
""", encoding="utf-8")

    db_file = vault / ".system" / "vault_catalog.db"
    cat = SqliteNoteCatalogAdapter(db_path=db_file, vault_root=vault)

    chroma_dir = vault / ".chroma_test"
    worker = IndexingWorker(
        catalog=cat,
        vault_root=vault,
        chroma_dir=chroma_dir,
        embeddings=FakeEmbeddings(size=64),
    )

    # 1. Initial indexing
    res1 = worker.sync_index()
    assert res1["added"] == 1
    assert res1["updated"] == 0
    assert res1["deleted"] == 0
    assert res1["total_indexed"] == 1

    # 2. Second indexing with no changes
    res2 = worker.sync_index()
    assert res2["added"] == 0
    assert res2["updated"] == 0
    assert res2["deleted"] == 0

    # 3. Add second note
    note2 = kb_dir / "Interest_Rates.md"
    note2.write_text("""---
schema_version: 2
note_id: concept_interest_rates
document_key: "v1:concept:interest_rates:primary"
entity_type: concept
title: Interest Rates Overview
---
# Interest Rates and Monetary Policy
Interest rates are determined by central banks to control economic expansion.
""", encoding="utf-8")

    res3 = worker.sync_index()
    assert res3["added"] == 1
    assert res3["total_indexed"] == 2

    # 4. Modify first note
    note1.write_text("""---
schema_version: 2
note_id: concept_inflation
document_key: "v1:concept:inflation:primary"
entity_type: concept
title: Inflation Overview
---
# Inflation Dynamics Updated
Stagflation occurs when inflation remains high during economic stagnation.
""", encoding="utf-8")
    import time, os
    t = time.time() + 10
    os.utime(note1, (t, t))

    res4 = worker.sync_index()
    assert res4["updated"] == 1
    assert res4["added"] == 0

    # 5. Delete second note
    note2.unlink()
    res5 = worker.sync_index()
    assert res5["deleted"] == 1
    assert res5["total_indexed"] == 1
