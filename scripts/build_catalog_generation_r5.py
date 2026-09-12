"""Build and publish an external SQLite catalog generation for R5."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.catalog_runtime import (  # noqa: E402
    catalog_generation_path,
    catalog_runtime_root,
    legacy_catalog_path,
    write_catalog_pointer,
)
from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter  # noqa: E402
from tools.archivist.policy_snapshot import eligible_set_fingerprint, policy_snapshot  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _catalog_counts(path: Path) -> dict[str, Any]:
    uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True, timeout=60.0) as conn:
        conn.row_factory = sqlite3.Row
        integrity = str(conn.execute("PRAGMA integrity_check;").fetchone()[0])
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        counts: dict[str, int] = {}
        for table in tables:
            if table.startswith("sqlite_"):
                continue
            counts[table] = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        schema_rows = conn.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
        ).fetchall()
    schema_digest = hashlib.sha256(
        json.dumps([dict(row) for row in schema_rows], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {"integrity_check": integrity, "tables": tables, "counts": counts, "schema_sha256": schema_digest}


def _checkpoint_staging(path: Path) -> None:
    """Checkpoint the mutable edge projection before publishing the file."""
    conn = sqlite3.connect(str(path), timeout=120.0)
    try:
        conn.execute("PRAGMA busy_timeout=120000")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.commit()
    finally:
        conn.close()
    for suffix in ("-wal", "-shm"):
        sidecar = path.with_name(path.name + suffix)
        if sidecar.exists():
            if sidecar.stat().st_size:
                raise RuntimeError(f"staging catalog sidecar is not empty: {sidecar}")
            sidecar.unlink()


def build(vault: Path, run_dir: Path) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    source = legacy_catalog_path(vault)
    if not source.is_file():
        raise FileNotFoundError(f"legacy catalog not found: {source}")
    if run_dir.is_relative_to(vault):
        raise ValueError("R5 catalog evidence must be outside the vault")
    run_dir.mkdir(parents=True, exist_ok=True)

    source_hash = _sha256(source)
    source_counts = _catalog_counts(source)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generation_id = f"catalog_{timestamp}_{source_hash[:12]}"
    runtime = catalog_runtime_root(vault, create=True)
    target = catalog_generation_path(vault, generation_id, create=True)
    staging = target.with_name(f".{target.name}.{generation_id}.staging")
    staging.unlink(missing_ok=True)

    source_uri = f"file:{source.as_posix()}?mode=ro"
    source_conn = sqlite3.connect(source_uri, uri=True, timeout=120.0)
    target_conn = sqlite3.connect(str(staging), timeout=120.0)
    try:
        source_conn.execute("PRAGMA busy_timeout=120000")
        source_conn.backup(target_conn, pages=1000, sleep=0.05)
        target_conn.commit()
    finally:
        target_conn.close()
        source_conn.close()
    target_adapter = SqliteNoteCatalogAdapter(db_path=staging, vault_root=vault)
    target_adapter.rebuild_link_edges(vault)
    _checkpoint_staging(staging)
    os.replace(staging, target)

    target_counts = _catalog_counts(target)
    if target_counts["integrity_check"] != "ok":
        raise RuntimeError(f"external catalog integrity check failed: {target_counts}")
    source_compare = {key: value for key, value in source_counts["counts"].items() if key != "note_links"}
    target_compare = {key: value for key, value in target_counts["counts"].items() if key != "note_links"}
    if source_compare != target_compare:
        raise RuntimeError(
            f"catalog row counts changed during backup: source={source_compare} target={target_compare}"
        )

    target_reader = SqliteNoteCatalogAdapter(db_path=target, vault_root=vault, read_only=True)
    eligible_digest, eligible_count = eligible_set_fingerprint(target_reader.iter_notes(page_size=500), vector=False)
    policy = policy_snapshot()

    manifest = {
        "manifest_version": 1,
        **policy,
        "eligible_set_fingerprint": eligible_digest,
        "eligible_note_count": eligible_count,
        "generation_id": generation_id,
        "vault_root": str(vault),
        "runtime_root": str(runtime),
        "database_relative_path": target.relative_to(runtime).as_posix(),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_database": str(source),
        "source_database_sha256": source_hash,
        "database_sha256": _sha256(target),
        "source": source_counts,
        "target": target_counts,
        "external_only": True,
        "read_only_query_contract": "mode=ro+immutable=1",
    }
    manifest_path = target.with_name("manifest.json")
    _write_json(manifest_path, manifest)
    pointer = write_catalog_pointer(
        vault,
        generation_id=generation_id,
        database_path=target,
        manifest_path=manifest_path,
    )
    result = {
        "status": "PASS",
        "generation_id": generation_id,
        "runtime_root": str(runtime),
        "database": str(target),
        "manifest": str(manifest_path),
        "pointer": str(pointer),
        "source_database": str(source),
        "source_sha256": source_hash,
        "database_sha256": manifest["database_sha256"],
        "counts": target_counts["counts"],
        "integrity_check": target_counts["integrity_check"],
    }
    _write_json(run_dir / "catalog-generation.json", {**manifest, "result": result})
    print(json.dumps(result, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.vault, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
