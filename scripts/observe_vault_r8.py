"""Observe writer, broker, and policy invariants for a bounded interval."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.scan_vault_writers_r8 import scan  # noqa: E402
from tools.archivist.vault_paths import VaultPaths  # noqa: E402
from tools.archivist.write_broker import KnowledgeWriteBroker  # noqa: E402


def _sample(root: Path, broker: KnowledgeWriteBroker) -> dict[str, Any]:
    inventory = scan(root)
    health = broker.health()
    violations = {
        "unresolved_writers": inventory["counts"].get("unresolved", 0),
        "review_writers": inventory["counts"].get("review", 0),
        "expired_allowlist": inventory["counts"].get("expired", 0),
        "parse_errors": inventory["counts"].get("parse_error", 0),
        "pending_commands": health.get("queue_depth", 0),
    }
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "health": health,
        "writer_counts": inventory["counts"],
        "violations": violations,
        "status": "PASS" if not any(violations.values()) else "FAIL",
    }


def run(vault: Path, runtime_base: Path | None, duration_seconds: int, interval_seconds: int) -> dict[str, Any]:
    started = time.monotonic()
    broker = KnowledgeWriteBroker(vault_paths=VaultPaths(vault), runtime_base=runtime_base)
    samples: list[dict[str, Any]] = []
    while True:
        samples.append(_sample(vault.parent, broker))
        elapsed = time.monotonic() - started
        if elapsed >= max(0, duration_seconds):
            break
        time.sleep(max(1, min(interval_seconds, duration_seconds - elapsed)))
    elapsed = time.monotonic() - started
    return {
        "schema": "vault-r8-observation-v1",
        "status": "PASS" if all(item["status"] == "PASS" for item in samples) else "FAIL",
        "requested_duration_seconds": duration_seconds,
        "duration_seconds": elapsed,
        "sample_count": len(samples),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--runtime-base", "--runtime", dest="runtime_base", type=Path)
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("scratch/vault-r8/observation-r8.json"))
    args = parser.parse_args()
    report = run(args.vault.resolve(), args.runtime_base, max(0, args.duration_seconds), max(1, args.interval_seconds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
