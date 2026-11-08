"""Back up, rehearse, and explicitly apply the approved round-5 storage migration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.config import get_state_db_path
from api.db.connection import migrate_round5_storage


def _connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _inventory(path: Path) -> dict[str, Any]:
    conn = _connect_readonly(path)
    try:
        quick = str(conn.execute("PRAGMA quick_check").fetchone()[0])
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        tables = [
            str(row[0]) for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        counts: dict[str, int] = {}
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
        pm_columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(pm_review_signoffs)")
        } if "pm_review_signoffs" in tables else set()
        packet_expr = "portfolio_id" if "portfolio_id" in pm_columns else "'default'"
        duplicate_rows = []
        if "pm_review_signoffs" in tables:
            duplicate_rows = [
                dict(row) for row in conn.execute(
                    f"""
                    SELECT {packet_expr} AS portfolio_id, packet_sha256, COUNT(*) AS duplicate_count
                    FROM pm_review_signoffs
                    GROUP BY {packet_expr}, packet_sha256
                    HAVING COUNT(*) > 1
                    """
                ).fetchall()
            ]
        indexes = [
            str(row[0]) for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='pm_review_signoffs' ORDER BY name"
            )
        ] if "pm_review_signoffs" in tables else []
        return {
            "quick_check": quick,
            "user_version": version,
            "tables": tables,
            "row_counts": counts,
            "pm_signoff_columns": sorted(pm_columns),
            "pm_signoff_indexes": indexes,
            "duplicate_signoff_keys": duplicate_rows,
        }
    finally:
        conn.close()


def _copy_sqlite_snapshot(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.unlink(missing_ok=True)
    src_conn = sqlite3.connect(str(source), timeout=30)
    dst_conn = sqlite3.connect(str(temp), timeout=30)
    try:
        src_conn.backup(dst_conn)
        dst_conn.commit()
    finally:
        dst_conn.close()
        src_conn.close()
    os.replace(temp, target)


def _copy_exact(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".round5-rehearsal-", suffix=".sqlite", dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, target)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rehearse(backup: Path, rehearsal: Path, before: dict[str, Any]) -> dict[str, Any]:
    _copy_exact(backup, rehearsal)
    conn = sqlite3.connect(str(rehearsal), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        migrate_round5_storage(conn)
    finally:
        conn.close()
    after = _inventory(rehearsal)
    for table, count in before["row_counts"].items():
        if after["row_counts"].get(table) != count:
            raise RuntimeError(f"rehearsal changed existing row count in {table}")
    if after["quick_check"] != "ok" or after["user_version"] != 6:
        raise RuntimeError("rehearsal failed SQLite integrity or schema-version checks")
    required = {
        "pm_review_packets", "routine_execution_events", "catalyst_revisions",
        "source_captures", "catalyst_assessments", "estimate_records", "state_import_receipts",
    }
    if not required.issubset(after["tables"]):
        raise RuntimeError("rehearsal did not create every approved durable table")
    return after


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path(get_state_db_path()))
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--apply", action="store_true", help="apply after backup, preflight, and rehearsal")
    args = parser.parse_args()
    database = args.database.resolve()
    if not database.is_file():
        raise FileNotFoundError(f"database does not exist: {database}")
    before = _inventory(database)
    if before["quick_check"] != "ok":
        raise RuntimeError(f"database quick_check failed: {before['quick_check']}")
    if before["duplicate_signoff_keys"]:
        raise RuntimeError(
            "duplicate PM sign-off keys require manual review; migration was not applied"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = (args.backup_dir or (database.parent / "backups" / "round5-storage")).resolve()
    backup = backup_dir / f"{database.stem}.pre-{stamp}.sqlite"
    _copy_sqlite_snapshot(database, backup)
    backup_hash = _sha256(backup)
    rehearsal = backup_dir / f".{database.stem}.rehearsal-{stamp}.sqlite"
    rehearsed = _rehearse(backup, rehearsal, before)
    rehearsal.unlink(missing_ok=True)

    result: dict[str, Any] = {
        "database": str(database),
        "backup": str(backup),
        "backup_sha256": backup_hash,
        "before": before,
        "rehearsal": rehearsed,
        "applied": False,
    }
    if args.apply:
        fresh = _inventory(database)
        if fresh != before:
            raise RuntimeError("database changed after preflight; take a new backup and rehearse again")
        conn = sqlite3.connect(str(database), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            migrate_round5_storage(conn)
        finally:
            conn.close()
        after = _inventory(database)
        for table, count in before["row_counts"].items():
            if after["row_counts"].get(table) != count:
                raise RuntimeError(f"migration changed existing row count in {table}; backup retained at {backup}")
        if after["quick_check"] != "ok" or after["user_version"] != 6:
            raise RuntimeError(f"post-migration verification failed; backup retained at {backup}")
        result["after"] = after
        result["applied"] = True
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
