"""Move legacy in-vault vector manifests to external R5 quarantine."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.maintenance_guard import assert_write_allowed
from tools.archivist.vector_generation import load_active_manifest, vector_runtime_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(vault: Path, output: Path) -> dict:
    vault = vault.resolve()
    output = output.resolve()
    assert_write_allowed(vault)
    active = load_active_manifest(vault)
    if not active:
        raise RuntimeError("no validated active vector generation")
    legacy_root = vault / ".system" / "vector_generations"
    files = sorted(legacy_root.glob("*.json")) if legacy_root.is_dir() else []
    runtime = vector_runtime_path(vault)
    quarantine = runtime / "quarantine" / f"legacy_vector_manifests_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    records = []
    for source in files:
        before = _sha256(source)
        destination = quarantine / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        after = _sha256(destination)
        records.append({
            "source": str(source),
            "destination": str(destination),
            "sha256": before,
            "destination_sha256": after,
            "status": "PASS" if before == after else "FAIL",
        })
    remaining = [str(path) for path in legacy_root.glob("*")] if legacy_root.is_dir() else []
    result = {
        "status": "PASS" if not remaining and all(item["status"] == "PASS" for item in records) else "FAIL",
        "vault": str(vault),
        "runtime": str(runtime),
        "active_generation_id": active["generation_id"],
        "legacy_root": str(legacy_root),
        "quarantine": str(quarantine),
        "records": records,
        "remaining": remaining,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if run(args.vault, args.output)["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
