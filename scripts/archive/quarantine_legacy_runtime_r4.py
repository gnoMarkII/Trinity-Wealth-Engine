"""Quarantine exactly the legacy vector-runtime files approved by R4 evidence.

The operation is intentionally fail-closed: every manifest file must exist,
match its recorded size and SHA-256, and be inside the vault before anything
is moved.  The quarantine is outside the vault so the active vault cannot
continue to discover the obsolete runtime.
"""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_target(vault: Path, target: dict) -> list[dict]:
    expected = target.get("files") or []
    expected_paths = {str(item["relative_path"]).replace("\\", "/") for item in expected}
    target_path = vault / str(target["target"])
    if not _inside(target_path, vault):
        raise RuntimeError(f"manifest target escapes vault: {target_path}")
    if not target_path.exists():
        raise RuntimeError(f"manifest target is already absent: {target_path}")

    actual_paths: set[str] = set()
    if target_path.is_dir():
        for path in target_path.rglob("*"):
            if path.is_file():
                actual_paths.add(path.relative_to(vault).as_posix())
    else:
        actual_paths.add(target_path.relative_to(vault).as_posix())
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise RuntimeError(
            f"manifest inventory mismatch for {target['target']}: missing={missing}, unexpected={unexpected}"
        )

    for item in expected:
        path = vault / str(item["relative_path"])
        if not _inside(path, vault) or not path.is_file():
            raise RuntimeError(f"manifest file is missing or escapes vault: {path}")
        size = path.stat().st_size
        digest = _sha256(path)
        if size != int(item["size"]) or digest != str(item["sha256"]):
            raise RuntimeError(
                f"manifest hash mismatch for {path}: size={size}/{item['size']} sha256={digest}/{item['sha256']}"
            )
    return expected


def quarantine(vault: Path, manifest_path: Path, quarantine_root: Path) -> dict:
    vault = vault.resolve()
    manifest_path = manifest_path.resolve()
    quarantine_root = quarantine_root.resolve()
    if _inside(quarantine_root, vault):
        raise RuntimeError(f"quarantine must be outside vault: {quarantine_root}")
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest not found: {manifest_path}")

    # Moving runtime files is a vault mutation, so the same lease contract as
    # other maintenance writers applies.
    assert_write_allowed(vault)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"approved-for-delete", "quarantined"}:
        raise RuntimeError(f"unexpected manifest status: {manifest.get('status')!r}")
    if manifest.get("status") == "quarantined":
        return {
            "status": "already-quarantined",
            "manifest": str(manifest_path),
            "quarantine_root": str(quarantine_root),
        }

    targets = manifest.get("targets") or []
    for target in targets:
        _validate_target(vault, target)

    if quarantine_root.exists():
        existing = list(quarantine_root.rglob("*"))
        if existing:
            raise RuntimeError(f"quarantine destination is not empty: {quarantine_root}")
    quarantine_root.mkdir(parents=True, exist_ok=True)

    moved: list[dict] = []
    for target in targets:
        relative = Path(str(target["target"]))
        source = vault / relative
        destination = quarantine_root / relative
        if destination.exists():
            raise RuntimeError(f"quarantine destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        if source.exists() or not destination.exists():
            raise RuntimeError(f"move verification failed for {source}")
        moved.append({"target": relative.as_posix(), "destination": str(destination)})

    timestamp = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "quarantined"
    manifest["quarantined_at"] = timestamp
    manifest["quarantine_root"] = str(quarantine_root)
    for target in manifest["targets"]:
        target["disposition"] = "quarantined-obsolete-runtime"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = {
        "status": "PASS",
        "manifest": str(manifest_path),
        "quarantine_root": str(quarantine_root),
        "quarantined_at": timestamp,
        "moved": moved,
    }
    result_path = manifest_path.parent / "legacy-runtime-quarantine-result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--quarantine-root", type=Path, required=True)
    args = parser.parse_args()
    result = quarantine(args.vault, args.manifest, args.quarantine_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
