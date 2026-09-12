"""Rebuild a fresh immutable catalog generation from the active generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter  # noqa: E402
from tools.archivist.catalog_runtime import (  # noqa: E402
    catalog_outbox_path,
    catalog_generation_path,
    catalog_runtime_root,
    resolve_catalog_path,
    write_catalog_pointer,
)
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


def _counts(path: Path) -> dict[str, Any]:
    uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True, timeout=60.0) as conn:
        conn.row_factory = sqlite3.Row
        integrity = str(conn.execute("PRAGMA integrity_check;").fetchone()[0])
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        counts = {
            table: int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in tables
            if not str(table).startswith("sqlite_")
        }
    return {"integrity_check": integrity, "tables": tables, "counts": counts}


def _checkpoint_without_sidecars(path: Path) -> None:
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


def _reconcile_outbox(vault: Path, target: Path, generation_id: str) -> dict[str, Any]:
    """Advance external catalog events after the new immutable generation is published."""
    outbox = catalog_outbox_path(vault)
    if not outbox.is_file():
        return {"path": str(outbox), "record_count": 0, "consumed": 0, "failed": 0}
    try:
        records = [
            json.loads(line)
            for line in outbox.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {"path": str(outbox), "record_count": 0, "consumed": 0, "failed": 1, "error": str(exc)}

    reader = SqliteNoteCatalogAdapter(db_path=target, vault_root=vault, read_only=True)
    consumed = 0
    failed = 0
    for item in records:
        if not isinstance(item, dict) or item.get("state") in {"consumed", "failed"}:
            continue
        item["state"] = "processing"
        item["attempts"] = int(item.get("attempts") or 0) + 1
        item["processing_at"] = datetime.now(timezone.utc).isoformat()
        rel = str(item.get("relative_path") or "").replace("\\", "/")
        action = str(item.get("action") or "upsert")
        entry = reader.get_by_path(rel) if rel else None
        path = vault / rel if rel else None
        if action == "upsert" and entry is not None and path is not None and path.is_file():
            item["state"] = "consumed"
            item["consumed_at"] = datetime.now(timezone.utc).isoformat()
            item["consumed_generation_id"] = generation_id
            consumed += 1
        elif action == "delete" and entry is None:
            item["state"] = "consumed"
            item["consumed_at"] = datetime.now(timezone.utc).isoformat()
            item["consumed_generation_id"] = generation_id
            consumed += 1
        else:
            item["state"] = "failed"
            item["failed_at"] = datetime.now(timezone.utc).isoformat()
            item["error"] = "new generation did not contain the requested projection"
            item["failed_generation_id"] = generation_id
            failed += 1

    temp = outbox.with_suffix(outbox.suffix + ".r5.tmp")
    temp.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
    os.replace(temp, outbox)
    return {"path": str(outbox), "record_count": len(records), "consumed": consumed, "failed": failed}


def rebuild(vault: Path, run_dir: Path) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    source = resolve_catalog_path(vault, require_exists=True)
    if source.is_relative_to(vault):
        raise RuntimeError(f"source catalog is still inside vault: {source}")
    runtime = catalog_runtime_root(vault, create=True)
    source_hash = _sha256(source)
    source_counts = _counts(source)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generation_id = f"catalog_{timestamp}_{source_hash[:12]}_r5"
    target = catalog_generation_path(vault, generation_id, create=True)
    staging = target.with_name(f".{target.name}.{generation_id}.staging")
    staging.unlink(missing_ok=True)

    source_uri = f"file:{source.as_posix()}?mode=ro&immutable=1"
    source_conn = sqlite3.connect(source_uri, uri=True, timeout=120.0)
    target_conn = sqlite3.connect(str(staging), timeout=120.0)
    try:
        source_conn.backup(target_conn, pages=1000, sleep=0.05)
        target_conn.commit()
    finally:
        target_conn.close()
        source_conn.close()

    adapter = SqliteNoteCatalogAdapter(db_path=staging, vault_root=vault)
    sync_result = adapter.sync_from_vault(vault_root=vault, force=True)
    _checkpoint_without_sidecars(staging)
    os.replace(staging, target)
    target_counts = _counts(target)
    if target_counts["integrity_check"] != "ok":
        raise RuntimeError(f"new catalog integrity check failed: {target_counts}")
    # The source generation may intentionally lag the live Vault after a
    # metadata repair.  ``sync_from_vault`` is the reconciliation boundary,
    # so the note row count is expected to move to the live active set.  The
    # lifecycle/tombstone and sidecar tables must remain stable unless a
    # dedicated migration explicitly changes them.
    stable_tables = ("note_tombstones", "sidecar_catalog")
    count_drift = {
        table: {
            "source": source_counts["counts"].get(table, 0),
            "target": target_counts["counts"].get(table, 0),
        }
        for table in stable_tables
        if source_counts["counts"].get(table, 0) != target_counts["counts"].get(table, 0)
    }
    if count_drift:
        raise RuntimeError(
            f"catalog lifecycle counts changed unexpectedly: {count_drift}"
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
        "source_generation_database": str(source),
        "source_database_sha256": source_hash,
        "database_sha256": _sha256(target),
        "source_counts": source_counts,
        "target_counts": target_counts,
        "sync_result": sync_result,
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
    outbox_lifecycle = _reconcile_outbox(vault, target, generation_id)
    result = {
        "status": "PASS",
        "generation_id": generation_id,
        "database": str(target),
        "manifest": str(manifest_path),
        "pointer": str(pointer),
        "source": str(source),
        "source_sha256": source_hash,
        "database_sha256": manifest["database_sha256"],
        "counts": target_counts["counts"],
        "sync_result": sync_result,
        "outbox_lifecycle": outbox_lifecycle,
    }
    _write_json(run_dir / "catalog-rebuild.json", {**manifest, "result": result})
    print(json.dumps(result, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.vault, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
