"""Atomically switch the vector pointer to a validated external generation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.maintenance_guard import assert_write_allowed  # noqa: E402
from tools.archivist.vector_generation import (  # noqa: E402
    activate_manifest,
    generation_dir,
    load_manifest,
    load_active_manifest,
    vector_runtime_path,
)


def run(vault: Path, generation_id: str, output: Path) -> dict:
    vault = vault.resolve()
    output = output.resolve()
    assert_write_allowed(vault)
    manifest_path = generation_dir(vault) / f"{generation_id}.json"
    manifest = load_manifest(manifest_path)
    if manifest.get("build_status") != "validated":
        raise RuntimeError(f"target vector generation is not validated: {generation_id}")
    previous = load_active_manifest(vault)
    pointer = activate_manifest(vault, manifest)
    result = {
        "status": "PASS",
        "changed": not previous or previous.get("generation_id") != generation_id,
        "previous_generation_id": previous.get("generation_id") if previous else None,
        "active_generation_id": generation_id,
        "manifest": str(manifest_path),
        "runtime": str(vector_runtime_path(vault)),
        "pointer": str(pointer),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return 0 if run(args.vault, args.generation_id, args.output)["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
