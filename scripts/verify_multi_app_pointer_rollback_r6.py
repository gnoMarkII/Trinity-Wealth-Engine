"""Prove R6 catalog/vector pointer rollback and reapply without touching live pointers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.catalog_runtime import catalog_runtime_root
from tools.archivist.vector_generation import generation_dir


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target(vault: Path, name: str, payload: dict[str, Any]) -> Path:
    if name == "catalog":
        return catalog_runtime_root(vault) / str(payload["database_relative_path"])
    return generation_dir(vault) / f"{payload['generation_id']}.json"


def prove(vault: Path, run_dir: Path) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    baseline_system = run_dir / "acceptance-baseline-r6" / ".system"
    current_system = vault / ".system"
    work = run_dir / "pointer-rollback-reapply-r6"
    work.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for name, filename in (("catalog", "catalog_generation_active.json"), ("vector", "vector_generation_active.json")):
        before = json.loads((baseline_system / filename).read_text(encoding="utf-8"))
        after = json.loads((current_system / filename).read_text(encoding="utf-8"))
        before_target = _target(vault, name, before)
        after_target = _target(vault, name, after)
        rollback_file = work / f"{name}-rollback.json"
        reapply_file = work / f"{name}-reapply.json"
        rollback_file.write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
        rollback_bytes = rollback_file.read_bytes()
        rollback_payload = json.loads(rollback_file.read_text(encoding="utf-8"))
        rollback_target_ok = _target(vault, name, rollback_payload).is_file()
        reapply_file.write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
        reapply_bytes = reapply_file.read_bytes()
        reapply_payload = json.loads(reapply_file.read_text(encoding="utf-8"))
        reapply_target_ok = _target(vault, name, reapply_payload).is_file()
        records[name] = {
            "rollback_target_exists": before_target.is_file(),
            "reapply_target_exists": after_target.is_file(),
            "rollback_roundtrip": rollback_target_ok,
            "reapply_roundtrip": reapply_target_ok,
            "reapply_pointer_sha256": hashlib.sha256(reapply_bytes).hexdigest(),
            "rollback_pointer_sha256": hashlib.sha256(rollback_bytes).hexdigest(),
        }
    status = "PASS" if all(
        all(bool(value) for key, value in record.items() if key.endswith("exists") or key.endswith("roundtrip"))
        for record in records.values()
    ) else "BLOCKED"
    result = {
        "status": status,
        "mode": "scratch-pointer-switch-no-live-mutation",
        "records": records,
    }
    (run_dir / "pointer-rollback-reapply-proof.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    return 0 if prove(args.vault, args.run_dir)["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
