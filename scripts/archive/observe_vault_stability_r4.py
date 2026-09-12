"""Run the R4 post-cleanup vault stability observation window.

The observer is read-only with respect to the vault and the external vector
runtime.  It records a baseline content fingerprint, watches file inventory
and metadata at a bounded interval, hashes any changed files, and emits one
machine-readable evidence file outside the vault when the window completes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_root(vault: Path) -> Path:
    configured = os.getenv("OBSIDIAN_VECTOR_RUNTIME_PATH")
    runtime = Path(configured).expanduser().resolve() if configured else vault.parent / "data" / "vector_runtime" / "vault_v2"
    if runtime.is_relative_to(vault):
        raise RuntimeError(f"vector runtime must be outside vault: {runtime}")
    return runtime


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    if not root.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        result[rel] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": None,
        }
    return result


def _hydrate_hashes(root: Path, inventory: dict[str, dict[str, Any]], paths: set[str] | None = None) -> None:
    selected = paths if paths is not None else set(inventory)
    for rel in sorted(selected):
        record = inventory.get(rel)
        path = root / Path(rel)
        if record is not None and path.is_file():
            record["sha256"] = _sha256(path)


def _content_fingerprint(inventory: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(inventory):
        item = inventory[rel]
        digest.update(f"{rel}\0{item['size']}\0{item['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _stat_fingerprint(inventory: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(inventory):
        item = inventory[rel]
        digest.update(f"{rel}\0{item['size']}\0{item['mtime_ns']}\n".encode("utf-8"))
    return digest.hexdigest()


def _changes(
    root: Path,
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    changed = {
        rel
        for rel in set(before) | set(after)
        if rel not in before
        or rel not in after
        or before[rel]["size"] != after[rel]["size"]
        or before[rel]["mtime_ns"] != after[rel]["mtime_ns"]
    }
    _hydrate_hashes(root, after, changed)
    result: list[dict[str, Any]] = []
    for rel in sorted(changed):
        result.append({
            "relative_path": rel,
            "before": before.get(rel),
            "after": after.get(rel),
        })
    return result


def observe(vault: Path, output: Path, *, duration_seconds: int, interval_seconds: int) -> dict[str, Any]:
    vault = vault.resolve()
    output = output.resolve()
    runtime = _runtime_root(vault)
    if duration_seconds <= 0 or interval_seconds <= 0 or interval_seconds > 60:
        raise ValueError("duration_seconds must be positive and interval_seconds must be between 1 and 60")

    started_at = datetime.now(timezone.utc)
    vault_baseline = _inventory(vault)
    runtime_baseline = _inventory(runtime)
    _hydrate_hashes(vault, vault_baseline)
    _hydrate_hashes(runtime, runtime_baseline)
    baseline_vault_fingerprint = _content_fingerprint(vault_baseline)
    baseline_runtime_fingerprint = _content_fingerprint(runtime_baseline)
    baseline_vault_stat = _stat_fingerprint(vault_baseline)
    baseline_runtime_stat = _stat_fingerprint(runtime_baseline)

    events: list[dict[str, Any]] = []
    samples = 0
    deadline = time.monotonic() + duration_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval_seconds, remaining))
        now = datetime.now(timezone.utc)
        current_vault = _inventory(vault)
        current_runtime = _inventory(runtime)
        vault_changes = _changes(vault, vault_baseline, current_vault)
        runtime_changes = _changes(runtime, runtime_baseline, current_runtime)
        if vault_changes or runtime_changes:
            events.append({
                "observed_at": now.isoformat(),
                "vault_changes": vault_changes,
                "runtime_changes": runtime_changes,
            })
        vault_baseline = current_vault
        runtime_baseline = current_runtime
        samples += 1

    ended_at = datetime.now(timezone.utc)
    _hydrate_hashes(vault, vault_baseline)
    _hydrate_hashes(runtime, runtime_baseline)
    result = {
        "status": "PASS" if not events else "FAIL",
        "reason": None if not events else "unexplained vault or vector-runtime mutation observed",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": (ended_at - started_at).total_seconds(),
        "requested_duration_seconds": duration_seconds,
        "interval_seconds": interval_seconds,
        "samples": samples,
        "vault": str(vault),
        "runtime": str(runtime),
        "baseline_vault_fingerprint": baseline_vault_fingerprint,
        "final_vault_fingerprint": _content_fingerprint(vault_baseline),
        "baseline_runtime_fingerprint": baseline_runtime_fingerprint,
        "final_runtime_fingerprint": _content_fingerprint(runtime_baseline),
        "baseline_vault_stat_fingerprint": baseline_vault_stat,
        "baseline_runtime_stat_fingerprint": baseline_runtime_stat,
        "events": events,
        "event_count": len(events),
        "read_only": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--interval-seconds", type=int, default=30)
    args = parser.parse_args()
    result = observe(
        args.vault,
        args.output,
        duration_seconds=args.duration_seconds,
        interval_seconds=args.interval_seconds,
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
