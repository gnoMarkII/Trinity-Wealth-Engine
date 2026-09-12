"""Prove R6 snapshot rollback and deterministic reapply on isolated clones."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.vault_backup import restore_vault_snapshot  # noqa: E402


IGNORED = {".system/maintenance.json"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in IGNORED
    }


def _fingerprint(tree: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for rel, sha in sorted(tree.items()):
        digest.update(f"{rel}\0{sha}\n".encode("utf-8"))
    return digest.hexdigest()


def _copy_lease(source_vault: Path, target_vault: Path) -> None:
    lease = source_vault / ".system" / "maintenance.json"
    target = target_vault / ".system" / "maintenance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lease, target)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prove(root: Path, source_vault: Path, run_dir: Path, owner: str) -> dict[str, Any]:
    root = root.resolve()
    source_vault = source_vault.resolve()
    run_dir = run_dir.resolve()
    snapshot = next(iter(sorted((run_dir / "snapshots").glob("vault_snapshot_*.zip"))), None)
    if snapshot is None:
        raise FileNotFoundError("R6 snapshot zip not found")

    baseline_rows = [
        json.loads(line)
        for line in (run_dir / "baseline-inventory.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_tree = {
        str(row["relative_path"]): str(row["sha256"])
        for row in baseline_rows
        if str(row["relative_path"]) not in IGNORED
    }

    rollback_root = run_dir / "rollback-proof-baseline-r6-v4"
    reapply_root = run_dir / "rollback-proof-reapply-r6-v4"
    restore_vault_snapshot(snapshot, rollback_root, verify_checksum=True)
    restore_vault_snapshot(snapshot, reapply_root, verify_checksum=True)
    _copy_lease(source_vault, rollback_root)
    _copy_lease(source_vault, reapply_root)

    rollback_tree = _tree(rollback_root)
    rollback_exact = rollback_tree == expected_tree

    from scripts.apply_multi_app_migration_r6 import apply as apply_migration
    from scripts.apply_multi_app_config_r6 import apply as apply_config
    apply_migration(reapply_root, run_dir, owner=owner)
    apply_config(reapply_root, run_dir, owner=owner)

    os.environ["VAULT_MAINTENANCE_OWNER"] = owner
    from tools.archivist.navigation_builder import build_navigation_indices
    from tools.archivist import indexer
    build_navigation_indices(reapply_root)
    indexer._build_cache_from_disk(reapply_root)
    indexer._write_index_from_cache(reapply_root)

    expected_final = _tree(run_dir / "rehearsal-clean-r6")
    reapply_tree = _tree(reapply_root)
    missing = sorted(set(expected_final) - set(reapply_tree))
    added = sorted(set(reapply_tree) - set(expected_final))
    changed = sorted(rel for rel in set(expected_final) & set(reapply_tree) if expected_final[rel] != reapply_tree[rel])
    result = {
        "status": "PASS" if rollback_exact and not missing and not added and not changed else "BLOCKED",
        "snapshot": str(snapshot),
        "rollback_exact": rollback_exact,
        "baseline_fingerprint": _fingerprint(expected_tree),
        "rollback_fingerprint": _fingerprint(rollback_tree),
        "final_reference": str(run_dir / "rehearsal-clean-r6"),
        "reapply_fingerprint": _fingerprint(reapply_tree),
        "final_reference_fingerprint": _fingerprint(expected_final),
        "reapply_missing_count": len(missing),
        "reapply_added_count": len(added),
        "reapply_changed_count": len(changed),
        "reapply_missing": missing[:50],
        "reapply_added": added[:50],
        "reapply_changed": changed[:50],
    }
    _write_json(run_dir / "rollback-reapply-proof.json", result)
    print(json.dumps(result, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--owner", default="codex-vault-r6")
    args = parser.parse_args()
    result = prove(Path("."), args.vault, args.run_dir, args.owner)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
