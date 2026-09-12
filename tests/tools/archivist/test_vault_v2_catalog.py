"""Unit tests for SQLite Note Catalog and Vault Link Resolver (T09a)."""
import json
from pathlib import Path
import pytest

from application.knowledge.ports import NoteCatalogEntry
from application.knowledge.query_service import KnowledgeQueryService
from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
from tools.archivist.link_resolver import VaultLinkResolver


def test_sqlite_catalog_basic_crud(tmp_path):
    db_file = tmp_path / "test_catalog.db"
    cat = SqliteNoteCatalogAdapter(db_path=db_file, vault_root=tmp_path)

    entry = NoteCatalogEntry(
        note_id="note_ftnt_hub",
        document_key="v1:stock_hub:FTNT:hub",
        relative_path="30_Knowledge_Base/Stocks/FTNT/FTNT.md",
        entity_type="stock_hub",
        title="Fortinet Hub",
        ticker="FTNT",
        date="2026-09-01",
        mtime=12345.67,
        file_size=1024,
        content_sha256="abc1234",
    )

    # Insert
    cat.upsert_note(entry)
    assert cat.count_notes() == 1

    # Get by ID
    by_id = cat.get_by_id("note_ftnt_hub")
    assert by_id is not None
    assert by_id.title == "Fortinet Hub"
    assert by_id.ticker == "FTNT"

    # Get by path
    by_path = cat.get_by_path("30_Knowledge_Base/Stocks/FTNT/FTNT.md")
    assert by_path is not None
    assert by_path.note_id == "note_ftnt_hub"

    # Get by document key
    by_key = cat.get_by_document_key("v1:stock_hub:FTNT:hub")
    assert by_key is not None
    assert by_key.note_id == "note_ftnt_hub"

    # Find notes
    found = cat.find_notes(ticker="FTNT")
    assert len(found) == 1

    # Update
    entry.title = "Fortinet Inc Hub Updated"
    cat.upsert_note(entry)
    assert cat.get_by_id("note_ftnt_hub").title == "Fortinet Inc Hub Updated"

    # Delete
    cat.delete_note("note_ftnt_hub")
    assert cat.count_notes() == 0
    assert cat.get_by_id("note_ftnt_hub") is None


def test_catalog_incremental_sync_from_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    stocks_dir = vault / "30_Knowledge_Base" / "Stocks" / "NVDA"
    stocks_dir.mkdir(parents=True)

    note_file = stocks_dir / "NVDA.md"
    note_content = """---
schema_version: 2
note_id: nvda_hub_01
document_key: "v1:stock_hub:NVDA:hub"
entity_type: stock_hub
title: NVIDIA Corporation
ticker: NVDA
date: "2026-09-01"
---
# NVIDIA Hub
"""
    note_file.write_text(note_content, encoding="utf-8")

    db_file = vault / ".system" / "vault_catalog.db"
    cat = SqliteNoteCatalogAdapter(db_path=db_file, vault_root=vault)

    # Initial sync
    res1 = cat.sync_from_vault(vault)
    assert res1["scanned"] == 1
    assert res1["added"] == 1
    assert res1["updated"] == 0
    assert res1["deleted"] == 0

    # Second sync without changes -> all unchanged
    res2 = cat.sync_from_vault(vault)
    assert res2["unchanged"] == 1
    assert res2["added"] == 0

    # Modify file
    note_file.write_text(note_content + "\n## Updated Section\n", encoding="utf-8")
    # Touch mtime
    import time
    t = time.time() + 10
    import os
    os.utime(note_file, (t, t))

    res3 = cat.sync_from_vault(vault)
    assert res3["updated"] == 1
    assert res3["added"] == 0

    # Delete file from disk
    note_file.unlink()
    res4 = cat.sync_from_vault(vault)
    assert res4["deleted"] == 1
    assert cat.count_notes() == 0


def test_link_resolver_and_query_service(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    hub_dir = vault / "30_Knowledge_Base" / "Stocks" / "AAPL"
    hub_dir.mkdir(parents=True)
    hub_file = hub_dir / "AAPL.md"
    hub_file.write_text("""---
schema_version: 2
note_id: aapl_hub_01
document_key: "v1:stock_hub:AAPL:hub"
entity_type: stock_hub
title: Apple Inc
ticker: AAPL
---
# Apple Hub
""", encoding="utf-8")

    db_file = vault / ".system" / "vault_catalog.db"
    cat = SqliteNoteCatalogAdapter(db_path=db_file, vault_root=vault)
    cat.sync_from_vault(vault)

    resolver = VaultLinkResolver(catalog=cat, vault_root=vault)
    svc = KnowledgeQueryService(catalog=cat, link_resolver=resolver)

    # Resolve ticker link [[AAPL]]
    resolved = svc.resolve_wikilink("[[AAPL]]")
    assert resolved == "30_Knowledge_Base/Stocks/AAPL/AAPL.md"

    # Resolve with alias [[AAPL|Apple Inc]]
    resolved_alias = svc.resolve_wikilink("[[AAPL|Apple Inc]]")
    assert resolved_alias == "30_Knowledge_Base/Stocks/AAPL/AAPL.md"

    # Resolve with section [[AAPL#Financials]]
    resolved_section = svc.resolve_wikilink("[[AAPL#Financials]]")
    assert resolved_section == "30_Knowledge_Base/Stocks/AAPL/AAPL.md"

    # Query service find by ticker
    entries = svc.find_notes_for_ticker("AAPL")
    assert len(entries) == 1
    assert entries[0].note_id == "aapl_hub_01"
