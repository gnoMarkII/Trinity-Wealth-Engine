"""Move legacy backup/parking/journal artefacts to a recoverable external queue."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.catalog_runtime import catalog_runtime_root  # noqa: E402
from tools.archivist.maintenance_guard import assert_write_allowed  # noqa: E402


OWNER = "codex-vault-r5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    if root.is_file():
        return {root.name: _sha256(root)}
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _move_one(source: Path, destination: Path, *, category: str, reason: str) -> dict[str, Any]:
    before = _tree_hashes(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"quarantine destination already exists: {destination}")
    shutil.move(str(source), str(destination))
    after = _tree_hashes(destination)
    return {
        "category": category,
        "source": str(source),
        "destination": str(destination),
        "file_count": len(before),
        "sha256_by_relative_path": before,
        "hashes_preserved": before == after,
        "disposition": "quarantined_external_recoverable",
        "reason": reason,
    }


def run(vault: Path, run_dir: Path, *, owner: str = OWNER) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    if not vault.is_dir():
        raise FileNotFoundError(vault)
    if run_dir.is_relative_to(vault):
        raise ValueError("cleanup evidence must be outside the vault")
    assert_write_allowed(vault, owner=owner)
    runtime = catalog_runtime_root(vault, create=True)
    run_id = datetime.now(timezone.utc).strftime("cleanup_%Y%m%dT%H%M%SZ")
    quarantine = runtime / "quarantine" / run_id
    records: list[dict[str, Any]] = []

    legacy_backup = next(
        (path for path in vault.iterdir() if path.is_dir() and path.name.startswith(".pre_migration_backup_")),
        None,
    )
    if legacy_backup is not None:
        records.append(
            _move_one(
                legacy_backup,
                quarantine / "legacy_backups" / legacy_backup.name,
                category="legacy_migration_backup",
                reason="migration backup is no longer an active source; retained outside the vault for recovery",
            )
        )

    outbox = vault / "NotebookLM_Sources" / "outbox"
    if outbox.is_dir():
        for source in sorted(outbox.glob("parking_unknown_*.json")):
            records.append(
                _move_one(
                    source,
                    quarantine / "pending_outbox" / source.name,
                    category="unresolved_notebooklm_outbox",
                    reason="sync_status is present but no durable job/output owner was found; pending review, not deleted",
                )
            )
        try:
            if not any(outbox.iterdir()):
                outbox.rmdir()
        except OSError:
            pass

    system = vault / ".system"
    for source in sorted(system.glob("migration_journal*.jsonl")):
        records.append(
            _move_one(
                source,
                quarantine / "migration_journals" / source.name,
                category="completed_migration_journal_archive",
                reason="append-only completed migration evidence is archived outside the Obsidian user-content tree",
            )
        )

    result = {
        "status": "PASS",
        "phase": "F04",
        "run_id": run_id,
        "vault_root": str(vault),
        "runtime_root": str(runtime),
        "quarantine_root": str(quarantine),
        "record_count": len(records),
        "hash_failures": [record for record in records if not record["hashes_preserved"]],
        "records": records,
    }
    if result["hash_failures"]:
        result["status"] = "FAIL"
        raise RuntimeError(f"quarantine hash verification failed: {result['hash_failures']}")
    _write_json(run_dir / "cleanup-dispositions.json", result)
    _write_json(quarantine / "cleanup-dispositions.json", result)
    # Windows PowerShell may expose a legacy code page; evidence is already
    # UTF-8 on disk, so keep stdout ASCII-safe for a successful exit status.
    print(json.dumps(result, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner", default=OWNER)
    args = parser.parse_args()
    run(args.vault, args.run_dir, owner=args.owner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
