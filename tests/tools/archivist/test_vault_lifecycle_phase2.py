import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest

from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
from tools.archivist.core import _atomic_write_text
from tools.archivist.vault_backup import create_vault_snapshot, rotate_snapshots, restore_vault_snapshot
from tools.archivist.vault_link_healer import heal_broken_links
from tools.archivist.vault_metadata_backfill import backfill_vault_metadata
from tools.archivist.vault_archival_policy import apply_archival_policy


def test_vault_backup_and_recovery(tmp_path: Path):
    """Verifies that vault snapshot creation, rotation, and restoration work with integrity checks."""
    vault = tmp_path / "vault"
    vault.mkdir()
    sample = vault / "30_Knowledge_Base" / "Concepts" / "Test.md"
    sample.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(sample, "# Test Note")

    backup_dir = tmp_path / "backups"
    zip_path, checksum = create_vault_snapshot(vault_root=vault, backup_dir=backup_dir)

    assert zip_path.exists()
    assert len(checksum) == 64

    # Rotate
    deleted = rotate_snapshots(backup_dir=backup_dir, max_keep=1)
    assert len(deleted) == 0  # 1 snapshot, max 1, none deleted

    # Restore to new location
    restore_target = tmp_path / "restored"
    extracted_count = restore_vault_snapshot(zip_path, restore_target)
    assert extracted_count >= 1
    assert (restore_target / "30_Knowledge_Base" / "Concepts" / "Test.md").exists()


def test_vault_link_healer_stub_generation(tmp_path: Path):
    """Verifies that heal_broken_links detects unlinked wikilinks and generates concept stubs."""
    vault = tmp_path / "vault"
    vault.mkdir()
    hub = vault / "30_Knowledge_Base" / "Stocks" / "AAPL" / "AAPL.md"
    hub.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(hub, "---\ntitle: AAPL\nentity_type: company\n---\n# AAPL\nMentions [[SupplyChainBottleneck]].")

    # Run link healer
    res = heal_broken_links(vault_root=vault, dry_run=False, allow_stub_creation=True)
    assert res["stubs_created"] >= 1

    stub = vault / "30_Knowledge_Base" / "Concepts" / "SupplyChainBottleneck.md"
    assert stub.exists()
    content = stub.read_text(encoding="utf-8")
    assert "SupplyChainBottleneck" in content
    assert "concept/stub" in content


def test_vault_metadata_backfill(tmp_path: Path):
    """Verifies that backfill_vault_metadata non-destructively injects YAML frontmatter."""
    vault = tmp_path / "vault"
    vault.mkdir()
    raw_note = vault / "30_Knowledge_Base" / "Concepts" / "QuantumComputing.md"
    raw_note.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(raw_note, "# Quantum Computing\n\nDiscussion about quantum qubits in 2026-05-12.")

    # Run backfill
    res = backfill_vault_metadata(vault_root=vault, dry_run=False)
    assert res["backfilled"] == 1

    updated = raw_note.read_text(encoding="utf-8")
    assert updated.startswith("---")
    assert "entity_type: concept" in updated
    assert "title: Quantum Computing" in updated
    assert "legacy-backfilled" in updated
    assert "Discussion about quantum qubits" in updated


def test_vault_archival_policy(tmp_path: Path):
    """Verifies that apply_archival_policy archives news older than cutoff date."""
    vault = tmp_path / "vault"
    vault.mkdir()
    old_news = vault / "30_Knowledge_Base" / "News" / "2025-01-15 Fed Rate Cut.md"
    old_news.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(old_news, "# Fed Rate Cut\nNews content from last year.")

    # Run archival policy with 90 days cutoff
    res = apply_archival_policy(vault_root=vault, max_age_days=90, dry_run=False)
    assert res["archived"] == 1

    # File should have moved to 40_Archive/News/2025/
    archived_dest = vault / "40_Archive" / "News" / "2025" / "2025-01-15 Fed Rate Cut.md"
    assert archived_dest.exists()
    assert not old_news.exists()
