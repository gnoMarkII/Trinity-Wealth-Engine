"""Atomically switch the catalog pointer to a previously validated generation."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.catalog_runtime import (  # noqa: E402
    catalog_generation_path,
    catalog_runtime_root,
    load_catalog_pointer,
    write_catalog_pointer,
)
from tools.archivist.maintenance_guard import assert_write_allowed  # noqa: E402


def run(vault: Path, generation_id: str, output: Path) -> dict:
    vault = vault.resolve()
    output = output.resolve()
    assert_write_allowed(vault)
    current = load_catalog_pointer(vault) or {}
    target = catalog_generation_path(vault, generation_id)
    runtime = catalog_runtime_root(vault)
    if not target.is_file() or not target.resolve().is_relative_to(runtime):
        raise FileNotFoundError(f"catalog generation not found inside runtime: {target}")
    uri = f"file:{target.as_posix()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as conn:
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        count = int(conn.execute("SELECT COUNT(*) FROM note_catalog").fetchone()[0])
    if integrity != "ok":
        raise RuntimeError(f"target catalog integrity check failed: {integrity}")
    manifest = target.with_name("manifest.json")
    pointer = write_catalog_pointer(
        vault,
        generation_id=generation_id,
        database_path=target,
        manifest_path=manifest if manifest.is_file() else None,
    )
    result = {
        "status": "PASS",
        "changed": current.get("generation_id") != generation_id,
        "previous_generation_id": current.get("generation_id"),
        "active_generation_id": generation_id,
        "database": str(target),
        "note_count": count,
        "integrity": integrity,
        "pointer": str(pointer),
        "switched_at": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if run(args.vault, args.generation_id, args.output)["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
