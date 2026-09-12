from __future__ import annotations

import json
from pathlib import Path

from scripts.scan_vault_writers_r9 import (
    APPROVED_ARTIFACT_ROOTS,
    APPROVED_BROKER_ROOTS,
    scan,
)
from tools.archivist.runtime_layout import runtime_layout


ROOT = Path(__file__).resolve().parents[2]


def test_r9_writer_inventory_is_deny_by_default() -> None:
    report = scan(ROOT)
    counts = report["counts"]
    assert counts.get("parse_error", 0) == 0
    assert counts.get("unresolved", 0) == 0
    assert counts.get("review", 0) == 0
    assert counts.get("expired", 0) == 0
    assert counts.get("broad_allowlist", 0) == 0
    assert counts.get("vault_write_capability", 0) > 0


def test_r9_production_compatibility_and_construction_boundaries() -> None:
    report = scan(ROOT)
    rows = report["rows"]
    assert not [row for row in rows if row.get("operation") == "compatibility_import"]
    broker_outside = [
        row
        for row in rows
        if row.get("operation") == "KnowledgeWriteBroker.construct"
        and not str(row.get("source_file", "")).startswith("scripts/")
        and row.get("source_file") not in APPROVED_BROKER_ROOTS
    ]
    artifact_outside = [
        row
        for row in rows
        if row.get("operation") == "ArtifactWriter.construct"
        and not str(row.get("source_file", "")).startswith("scripts/")
        and row.get("source_file") not in APPROVED_ARTIFACT_ROOTS
    ]
    assert broker_outside == []
    assert artifact_outside == []


def test_r9_allowlist_has_effective_expiry_and_classification_only_prefixes() -> None:
    contract = json.loads((ROOT / "memories/.system/storage_contract.json").read_text(encoding="utf-8"))
    default_expiry = str(contract["allowlist_policy"]["default_expires_on"])
    rules = contract["direct_writer_allowlist"]
    assert rules
    for rule in rules:
        assert all(str(rule.get(field) or "").strip() for field in ("source", "profile", "owner", "reason", "target_pattern", "test_id"))
        assert str(rule.get("expires_on") or default_expiry).strip() == default_expiry
    assert all(bool(rule.get("classification_only")) for rule in contract["source_prefix_rules"])


def test_r9_production_vault_has_no_runtime_database_files() -> None:
    vault = ROOT / "memories"
    runtime_files = [
        path.relative_to(vault).as_posix()
        for path in vault.rglob("*")
        if path.is_file()
        and (path.suffix.lower() in {".sqlite", ".sqlite3", ".db"} or path.name.endswith(("-wal", "-shm", "-journal")))
    ]
    assert runtime_files == []


def test_r9_production_runtime_layout_is_one_vault_scoped_root() -> None:
    vault = ROOT / "memories"
    layout = runtime_layout(vault, ROOT / "data" / "vault_runtime")
    assert layout.vault_id == "memories"
    assert layout.root == (ROOT / "data" / "vault_runtime" / "memories").resolve()
    assert not layout.root.is_relative_to(vault)
    assert layout.broker_db.parent.parent == layout.root
    assert layout.portfolio_db.parent.parent == layout.root
    assert layout.vector_root.parent == layout.root
