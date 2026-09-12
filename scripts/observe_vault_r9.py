"""Observe the R9 runtime boundary and SLOs for a bounded interval."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.scan_vault_writers_r9 import scan  # noqa: E402
from tools.archivist.recovery_bundle import capture_runtime_state  # noqa: E402
from tools.archivist.runtime_layout import runtime_layout  # noqa: E402
from tools.archivist.vault_paths import VaultPaths  # noqa: E402
from tools.archivist.write_broker import KnowledgeWriteBroker  # noqa: E402


def _sample(vault: Path, layout, broker: KnowledgeWriteBroker, inventory: dict[str, Any]) -> dict[str, Any]:
    health = broker.health()
    state = capture_runtime_state(vault, runtime_root=layout.root)
    catalog = state.get("catalog") or {}
    vector = state.get("vector") or {}
    violations = {
        "queue_oldest_age": int(float(health.get("oldest_pending_age_seconds") or 0) >= 60),
        "dead_letters": int(int(health.get("dead_letters", 0)) > 0),
        "runtime_ambiguity": 0,
        "registry_policy_mismatch": int(not (
            state.get("registry_digest")
            and state.get("policy_digest")
            and catalog.get("registry_digest") == state.get("registry_digest")
            and catalog.get("policy_digest") == state.get("policy_digest")
            and vector.get("registry_digest") == state.get("registry_digest")
            and vector.get("policy_digest") == state.get("policy_digest")
        )),
        "writer_inventory_debt": int(any(inventory["counts"].get(key, 0) for key in ("unresolved", "review", "expired", "parse_error", "broad_allowlist"))),
    }
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "health": health,
        "runtime_state": state,
        "writer_counts": inventory["counts"],
        "violations": violations,
        "status": "PASS" if not any(violations.values()) else "FAIL",
    }


def run(vault: Path, runtime_base: Path | None, duration_seconds: int, interval_seconds: int) -> dict[str, Any]:
    vault = vault.resolve()
    layout = runtime_layout(vault, runtime_base, create=False)
    broker = KnowledgeWriteBroker(vault_paths=VaultPaths(vault), runtime_base=runtime_base, broker_id="r9-observation")
    inventory = scan(vault.parent)
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    while True:
        samples.append(_sample(vault, layout, broker, inventory))
        elapsed = time.monotonic() - started
        if elapsed >= max(0, duration_seconds):
            break
        time.sleep(max(1, min(interval_seconds, duration_seconds - elapsed)))
    elapsed = time.monotonic() - started
    return {
        "schema": "vault-r9-observation-v1",
        "status": "PASS" if samples and all(item["status"] == "PASS" for item in samples) else "FAIL",
        "requested_duration_seconds": duration_seconds,
        "duration_seconds": elapsed,
        "sample_count": len(samples),
        "interval_seconds": interval_seconds,
        "writer_inventory_mode": "single_source_scan_at_observation_start",
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--runtime-base", type=Path, default=None)
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.vault, args.runtime_base, max(0, args.duration_seconds), max(1, args.interval_seconds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "duration_seconds": report["duration_seconds"], "sample_count": report["sample_count"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
