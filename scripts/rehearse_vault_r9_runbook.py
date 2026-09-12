"""Execute the non-destructive production runbook rehearsal and record evidence."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_vault_r9_preflight import run as run_preflight  # noqa: E402
from tools.archivist.recovery_bundle import capture_runtime_state  # noqa: E402
from tools.archivist.runtime_layout import runtime_layout  # noqa: E402
from tools.archivist.vault_paths import VaultPaths  # noqa: E402
from tools.archivist.write_broker import KnowledgeWriteBroker  # noqa: E402


REQUIRED_RUNBOOK_MARKERS = (
    "run_vault_r9_cycle.py",
    "backup_vault_platform_r9.py",
    "restore_vault_platform_r9.py",
    "rebuild_vault_derived_r9.py",
    "retry <command-id>",
    "Rollback",
    "Release and freeze gate",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _latest_bundle(root: Path) -> Path | None:
    paths = sorted((root / "scratch" / "vault-r9" / "recovery").glob("*/bundle-manifest.json"))
    if not paths:
        return None
    return max(paths, key=lambda path: (str((_read_json(path) or {}).get("created_at") or ""), path.name)).parent


def _latest_pass_cycle(root: Path) -> Path | None:
    paths = sorted((root / "scratch" / "vault-r9" / "operations").glob("cycle-*.json"))
    for path in reversed(paths):
        payload = _read_json(path)
        if payload and payload.get("status") == "PASS":
            return path
    return None


def run(vault: Path, runtime_base: Path | None, *, operator: str, output: Path) -> dict[str, Any]:
    root = vault.parent.resolve()
    vault = vault.resolve()
    layout = runtime_layout(vault, runtime_base, create=False)
    runbook_path = root / "docs" / "runbooks" / "vault-platform-production.md"
    runbook_text = runbook_path.read_text(encoding="utf-8") if runbook_path.is_file() else ""
    runbook_checks = {marker: marker in runbook_text for marker in REQUIRED_RUNBOOK_MARKERS}

    preflight = run_preflight(vault, runtime_base)
    broker = KnowledgeWriteBroker(vault_paths=VaultPaths(vault), runtime_base=runtime_base)
    broker_health = broker.health()
    runtime_state = capture_runtime_state(vault, runtime_root=layout.root)
    bundle_root = _latest_bundle(root)
    bundle_manifest = _read_json(bundle_root / "bundle-manifest.json") if bundle_root else None
    restore_reports = sorted((root / "scratch" / "vault-r9" / "recovery").glob("restore-*/restore-report.json"))
    restore_report = _read_json(restore_reports[-1]) if restore_reports else None
    cycle_path = _latest_pass_cycle(root)
    cycle = _read_json(cycle_path) if cycle_path else None

    checks = {
        "runbook_present": runbook_path.is_file(),
        "runbook_commands_complete": all(runbook_checks.values()),
        "preflight_pass": preflight.get("status") == "PASS",
        "broker_queue_empty": int(broker_health.get("queue_depth", 0)) == 0,
        "dead_letters_zero": int(broker_health.get("dead_letters", 0)) == 0,
        "runtime_state_captured": bool(runtime_state.get("runtime_layout")),
        "verified_bundle_present": bool(
            bundle_manifest
            and bundle_manifest.get("bundle_version") == 1
            and bundle_manifest.get("vault_snapshot", {}).get("sha256")
            and bundle_manifest.get("runtime_state")
            and bundle_manifest.get("contract", {}).get("registry_digest")
            and bundle_manifest.get("contract", {}).get("policy_digest")
        ),
        "verified_restore_present": bool(restore_report and restore_report.get("status") == "PASS"),
        "passing_cycle_present": bool(cycle and cycle.get("status") == "PASS"),
    }
    report = {
        "schema": "vault-r9-runbook-rehearsal-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "operator": operator,
        "operator_mode": "codex-runbook-driven",
        "mutation_mode": "staging_only",
        "performed_at": datetime.now(timezone.utc).isoformat(),
        "vault": str(vault),
        "runtime_root": str(layout.root),
        "runbook": str(runbook_path),
        "runbook_checks": runbook_checks,
        "checks": checks,
        "preflight": {"status": preflight.get("status"), "path": str(root / "scratch/vault-r9/preflight/r9-preflight-after-derived.json")},
        "broker_health": broker_health,
        "bundle": str(bundle_root) if bundle_root else None,
        "restore_report": str(restore_reports[-1]) if restore_reports else None,
        "cycle": str(cycle_path) if cycle_path else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--runtime-base", type=Path, default=None)
    parser.add_argument("--operator", default="vault-platform-owner")
    parser.add_argument("--output", type=Path, default=Path("scratch/vault-r9/operations/runbook-rehearsal.json"))
    args = parser.parse_args()
    report = run(args.vault, args.runtime_base, operator=args.operator, output=args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve()), "checks": report["checks"]}, ensure_ascii=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
