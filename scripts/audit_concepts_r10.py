"""Create a read-only, hashable Concepts and link-graph inventory for R10."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.concepts_cleanup import scan_concepts, write_jsonl  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def audit(vault: Path, output_dir: Path) -> dict:
    vault = vault.resolve()
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(vault):
        raise ValueError("R10 inventory must be outside the Vault")
    snapshot = scan_concepts(vault)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "concepts-inventory.json", snapshot)
    write_jsonl(output_dir / "concepts-inventory.jsonl", snapshot.get("concepts") or [])
    summary = {
        key: snapshot[key]
        for key in (
            "snapshot_fingerprint",
            "total_markdown_files",
            "concept_file_count",
            "eligible_note_count",
            "edge_count",
            "unresolved_edge_count",
        )
    }
    _write_json(output_dir / "inventory-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.vault, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
