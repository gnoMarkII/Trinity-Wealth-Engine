"""Build a deterministic, reviewable Concepts cleanup plan without mutation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.concepts_cleanup import (  # noqa: E402
    build_cleanup_plan,
    scan_concepts,
    write_jsonl,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _load_inventory(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("concepts"), list):
        raise ValueError(f"invalid R10 inventory: {path}")
    return payload


def build(vault: Path, output_dir: Path, inventory: Path | None = None) -> dict:
    vault = vault.resolve()
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(vault):
        raise ValueError("R10 cleanup plan must be outside the Vault")
    snapshot = _load_inventory(inventory.resolve()) if inventory else scan_concepts(vault)
    plan = build_cleanup_plan(snapshot)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "concepts-inventory.json", snapshot)
    write_jsonl(output_dir / "concepts-inventory.jsonl", snapshot.get("concepts") or [])
    _write_json(output_dir / "cleanup-plan.json", plan)
    write_jsonl(output_dir / "cleanup-plan.jsonl", plan.get("concepts") or [])
    summary = {
        "snapshot_fingerprint": plan.get("snapshot_fingerprint"),
        "policy_digest": plan.get("policy_digest"),
        "concept_file_count": plan.get("concept_file_count", 0),
        "disposition_counts": plan.get("disposition_counts", {}),
        "apply_item_count": plan.get("apply_item_count", 0),
    }
    _write_json(output_dir / "cleanup-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    build(args.vault, args.output_dir, args.inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
