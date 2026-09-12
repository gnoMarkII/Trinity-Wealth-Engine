"""Create the guarded R5 baseline, lease, and snapshot/restore proof.

The command deliberately keeps all evidence outside the Obsidian vault.  It
does not change note content.  A successful run leaves the R5 maintenance
lease active so that the following mutation steps can share the same owner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.maintenance_guard import (  # noqa: E402
    acquire_maintenance_lease,
    release_maintenance_lease,
)
from tools.archivist.vault_backup import create_vault_snapshot, restore_vault_snapshot  # noqa: E402


OWNER = "codex-vault-r5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        content_hash = _sha256(path)
        row = {
            "relative_path": rel,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": content_hash,
        }
        rows.append(row)
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return rows, digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _snapshot_hash_match(snapshot: Path, vault: Path, restore_dir: Path) -> dict[str, Any]:
    restored_count = restore_vault_snapshot(snapshot, restore_dir, verify_checksum=True)
    mismatches: list[str] = []
    with zipfile.ZipFile(snapshot, "r") as archive:
        members = sorted(item.filename for item in archive.infolist() if not item.is_dir())
    for rel in members:
        live = vault / Path(rel)
        restored = restore_dir / Path(rel)
        if not live.is_file() or not restored.is_file() or _sha256(live) != _sha256(restored):
            mismatches.append(rel)
            if len(mismatches) >= 50:
                break
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "snapshot": str(snapshot),
        "snapshot_sha256": _sha256(snapshot),
        "archive_file_count": len(members),
        "restored_file_count": restored_count,
        "hashes_match": not mismatches,
        "mismatches": mismatches,
        "restore_dir": str(restore_dir),
    }


def run(vault: Path, run_dir: Path, *, owner: str = OWNER, ttl_seconds: int = 4 * 60 * 60) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    if not vault.is_dir():
        raise FileNotFoundError(f"vault does not exist: {vault}")
    if run_dir.is_relative_to(vault):
        raise ValueError(f"R5 evidence must be outside the vault: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty R5 run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    baseline_rows, baseline_fingerprint = _inventory(vault)
    _write_jsonl(run_dir / "baseline-inventory.jsonl", baseline_rows)
    _write_json(
        run_dir / "baseline.json",
        {
            "phase": "F00",
            "started_at": started_at,
            "vault_root": str(vault),
            "file_count": len(baseline_rows),
            "tree_fingerprint": baseline_fingerprint,
            "owner": owner,
        },
    )

    lease = None
    try:
        lease = acquire_maintenance_lease(
            vault,
            owner=owner,
            purpose="R5 Obsidian AI-readiness hardening",
            ttl_seconds=ttl_seconds,
            baseline_tree_fingerprint=baseline_fingerprint,
        )
        _write_json(run_dir / "lease.json", lease.__dict__)

        snapshot_dir = run_dir / "snapshots"
        snapshot, snapshot_hash = create_vault_snapshot(vault_root=vault, backup_dir=snapshot_dir)
        restore_dir = run_dir / "snapshot-restore"
        restore_proof = _snapshot_hash_match(snapshot, vault, restore_dir)
        restore_proof.update({"snapshot_checksum": snapshot_hash, "baseline_fingerprint": baseline_fingerprint})
        _write_json(run_dir / "snapshot-restore-proof.json", restore_proof)
        if restore_proof["status"] != "PASS":
            raise RuntimeError(f"R5 snapshot restore proof failed: {restore_proof['mismatches'][:3]}")

        after_rows, after_fingerprint = _inventory(vault)
        _write_jsonl(run_dir / "post-snapshot-inventory.jsonl", after_rows)
        result = {
            "status": "PASS",
            "phase": "F00",
            "run_dir": str(run_dir),
            "vault_root": str(vault),
            "owner": owner,
            "lease_id": lease.lease_id,
            "lease_status": lease.status,
            "baseline_file_count": len(baseline_rows),
            "post_snapshot_file_count": len(after_rows),
            "baseline_tree_fingerprint": baseline_fingerprint,
            "post_snapshot_tree_fingerprint": after_fingerprint,
            "snapshot_restore_status": restore_proof["status"],
            "snapshot": str(snapshot),
            "snapshot_sha256": snapshot_hash,
            "lease_left_active": True,
        }
        _write_json(run_dir / "run.json", result)
        print(json.dumps(result, ensure_ascii=False))
        return result
    except Exception:
        if lease is not None:
            try:
                release_maintenance_lease(vault, owner=owner, reason="F00 preflight failed")
            except Exception:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--ttl-seconds", type=int, default=4 * 60 * 60)
    args = parser.parse_args()
    run(args.vault, args.run_dir, owner=args.owner, ttl_seconds=args.ttl_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
