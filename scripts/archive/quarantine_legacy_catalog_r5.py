"""Quarantine the superseded in-vault SQLite catalog after external publish."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.catalog_runtime import catalog_runtime_root, resolve_catalog_path  # noqa: E402
from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter  # noqa: E402
from tools.archivist.maintenance_guard import assert_write_allowed  # noqa: E402


OWNER = "codex-vault-r5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def run(vault: Path, run_dir: Path, *, owner: str = OWNER) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    assert_write_allowed(vault, owner=owner)
    active = resolve_catalog_path(vault, require_exists=True)
    if active.is_relative_to(vault):
        raise RuntimeError(f"active catalog is still inside vault: {active}")
    legacy_base = vault / ".system" / "vault_catalog.db"
    sources = [
        legacy_base,
        legacy_base.with_name(legacy_base.name + "-wal"),
        legacy_base.with_name(legacy_base.name + "-shm"),
    ]
    timestamp = datetime.now(timezone.utc).strftime("catalog_legacy_%Y%m%dT%H%M%SZ")
    destination_root = catalog_runtime_root(vault, create=True) / "quarantine" / timestamp
    records: list[dict[str, Any]] = []
    for source in sources:
        if not source.is_file():
            continue
        before = _sha256(source)
        destination = destination_root / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        after = _sha256(destination)
        records.append(
            {
                "source": str(source),
                "destination": str(destination),
                "sha256": before,
                "hash_preserved": before == after,
                "disposition": "quarantined_external_recoverable",
            }
        )

    catalog = SqliteNoteCatalogAdapter(vault_root=vault, read_only=True)
    result = {
        "status": "PASS" if all(item["hash_preserved"] for item in records) else "FAIL",
        "phase": "F02",
        "vault_root": str(vault),
        "active_catalog": str(active),
        "active_catalog_outside_vault": not active.is_relative_to(vault),
        "legacy_sources": records,
        "legacy_remaining": [str(path) for path in sources if path.exists()],
        "active_note_count": catalog.count_notes(),
        "quarantine_root": str(destination_root),
    }
    if result["legacy_remaining"] or result["status"] != "PASS":
        result["status"] = "FAIL"
        raise RuntimeError(json.dumps(result, ensure_ascii=True))
    _write_json(run_dir / "legacy-catalog-quarantine.json", result)
    _write_json(destination_root / "legacy-catalog-quarantine.json", result)
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
