"""Assemble R8 preflight, rehearsal, acceptance, and observation evidence."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, default=Path("scratch/vault-r8/f00-preflight-r8.json"))
    parser.add_argument("--rehearsal", type=Path, default=Path("scratch/vault-r8/f03-broker-rehearsal.json"))
    parser.add_argument("--acceptance", type=Path, default=Path("scratch/vault-r8/acceptance/acceptance-report.json"))
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/vault-r8/evidence"))
    args = parser.parse_args()
    parts = {
        "preflight": _read(args.preflight),
        "broker_rehearsal": _read(args.rehearsal),
        "acceptance": _read(args.acceptance),
        "observation": _read(args.observation),
    }
    acceptance = parts["acceptance"] or {}
    observation = parts["observation"] or {}
    report = {
        "schema": "vault-r8-evidence-bundle-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if acceptance.get("status") == "PASS" and observation.get("status") == "PASS" and float(observation.get("duration_seconds", 0)) >= 3600 else "BLOCKED",
        "components": parts,
        "remaining_items": [] if acceptance.get("status") == "PASS" else ["repeat or resolve acceptance gates marked NOT_RUN/FAIL"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evidence-bundle.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    lines = ["# Vault R8 Evidence Bundle", "", f"Status: **{report['status']}**", "", "| Evidence | Status |", "|---|---|"]
    lines.extend(f"| {key} | {(value or {}).get('status', 'NOT_RUN')} |" for key, value in parts.items())
    (args.output_dir / "evidence-bundle.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
