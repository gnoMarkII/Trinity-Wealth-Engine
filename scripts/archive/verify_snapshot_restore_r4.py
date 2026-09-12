"""Verify a R4 vault snapshot by restoring every archive member and hashing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.vault_backup import restore_vault_snapshot


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(snapshot: Path, vault: Path, restore_dir: Path, output: Path) -> dict:
    snapshot = snapshot.resolve()
    vault = vault.resolve()
    restore_dir = restore_dir.resolve()
    output = output.resolve()
    if restore_dir.exists():
        raise RuntimeError(f"restore directory already exists: {restore_dir}")
    restored_count = restore_vault_snapshot(snapshot, restore_dir, verify_checksum=True)
    with zipfile.ZipFile(snapshot, "r") as archive:
        members = sorted(item.filename for item in archive.infolist() if not item.is_dir())

    mismatches: list[str] = []
    for rel in members:
        live_path = vault / Path(rel)
        restored_path = restore_dir / Path(rel)
        if not live_path.is_file() or not restored_path.is_file() or sha256(live_path) != sha256(restored_path):
            mismatches.append(rel)
            if len(mismatches) >= 20:
                break

    result = {
        "status": "PASS" if not mismatches else "FAIL",
        "snapshot": str(snapshot),
        "snapshot_sha256": sha256(snapshot),
        "files_extracted": len(members),
        "restored_files": restored_count,
        "hashes_match": not mismatches,
        "mismatches": mismatches,
        "restore_dir": str(restore_dir),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--restore-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if verify(args.snapshot, args.vault, args.restore_dir, args.output)["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
