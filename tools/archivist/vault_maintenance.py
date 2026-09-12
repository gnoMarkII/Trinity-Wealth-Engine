"""Vault Maintenance and Database Compaction Engine for Obsidian Vault V2.

Provides enterprise database administration operations:
- VACUUM, ANALYZE, and PRAGMA optimize on SQLite catalog
- Outbox compaction and dead-letter cleanup
- Detection of orphaned sidecars (JSON files without markdown companions or vice-versa)
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Optional, Union

from core.logger import get_logger
from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
from tools.archivist.catalog_runtime import catalog_outbox_path, load_catalog_pointer, resolve_catalog_path
from tools.archivist.maintenance_guard import assert_write_allowed

logger = get_logger(__name__)


def run_catalog_maintenance(
    vault_root: Optional[Union[str, Path]] = None,
    vacuum: bool = True,
) -> dict[str, Any]:
    """Runs VACUUM, ANALYZE, and PRAGMA optimize on the SQLite catalog database."""
    v_root = Path(vault_root).resolve() if vault_root else Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
    assert_write_allowed(v_root)
    try:
        cat_db = resolve_catalog_path(v_root, require_exists=True)
    except FileNotFoundError:
        return {"status": "skipped", "reason": "catalog_db_not_found"}

    if load_catalog_pointer(v_root) is not None:
        outbox = catalog_outbox_path(v_root)
        pending = len(outbox.read_text(encoding="utf-8").splitlines()) if outbox.is_file() else 0
        return {
            "status": "deferred",
            "reason": "active catalog generation is immutable; rebuild and publish a new generation",
            "catalog_path": str(cat_db),
            "pending_outbox_operations": pending,
        }

    initial_size = cat_db.stat().st_size
    conn = sqlite3.connect(str(cat_db), timeout=60.0)
    try:
        conn.execute("PRAGMA optimize;")
        conn.execute("ANALYZE;")
        if vacuum:
            conn.execute("VACUUM;")
        final_size = cat_db.stat().st_size
    finally:
        conn.close()

    # Reconcile missing entries and process outbox
    adapter = SqliteNoteCatalogAdapter(db_path=cat_db, vault_root=v_root)
    pruned_stats = adapter.reconcile_missing()
    resolved_outbox = adapter.process_outbox()

    return {
        "status": "success",
        "initial_size_bytes": initial_size,
        "final_size_bytes": final_size,
        "bytes_reclaimed": max(0, initial_size - final_size),
        "pruned_missing_notes": pruned_stats.get("pruned", 0),
        "resolved_outbox_operations": resolved_outbox,
    }


def detect_orphaned_sidecars(
    vault_root: Optional[Union[str, Path]] = None,
) -> list[dict[str, str]]:
    """Detects JSON sidecars in knowledge base or system sidecars without companion markdown hubs."""
    v_root = Path(vault_root).resolve() if vault_root else Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
    orphans: list[dict[str, str]] = []

    stocks_kb = v_root / "30_Knowledge_Base" / "Stocks"
    if not stocks_kb.exists():
        return orphans

    # 1. Inspect knowledge base stocks
    for ticker_dir in stocks_kb.iterdir():
        if not ticker_dir.is_dir() or ticker_dir.name.startswith("."):
            continue
        ticker = ticker_dir.name.upper()
        hub_md = ticker_dir / f"{ticker}.md"
        hub_exists = hub_md.exists()

        # Check analysis json sidecars
        analysis_dir = ticker_dir / "Analysis"
        json_targets = []
        if analysis_dir.exists():
            json_targets.extend(analysis_dir.glob("*.json"))
        json_targets.extend(ticker_dir.glob("*.json"))

        for json_path in json_targets:
            if not hub_exists:
                orphans.append({
                    "ticker": ticker,
                    "sidecar_path": json_path.relative_to(v_root).as_posix(),
                    "reason": "missing_hub_markdown",
                })

    return orphans
