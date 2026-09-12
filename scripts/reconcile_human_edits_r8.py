"""Run the R8 human-edit scanner, optionally through the write boundary."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.human_edit_reconciler import HumanEditReconciler
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.write_broker import KnowledgeWriteBroker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--runtime-base", "--runtime", dest="runtime_base", type=Path)
    parser.add_argument("--output", type=Path, default=Path("scratch/vault-r8/f06-reconciliation.json"))
    parser.add_argument("--write-enabled", action="store_true")
    args = parser.parse_args()
    paths = VaultPaths(args.vault.resolve())
    broker = KnowledgeWriteBroker(vault_paths=paths, runtime_base=args.runtime_base) if args.write_enabled else None
    report = HumanEditReconciler(vault_paths=paths, broker=broker, runtime_base=args.runtime_base).reconcile_once(write_enabled=args.write_enabled)
    payload = report.to_dict()
    payload["mode"] = "write_enabled" if args.write_enabled else "shadow"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if payload["counts"].get("conflict", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
