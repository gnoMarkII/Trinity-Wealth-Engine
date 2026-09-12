"""Read-only R8 preflight for the Vault registry and write boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.metadata import parse_note  # noqa: E402
from tools.archivist.schema_registry import load_default_registry  # noqa: E402
from tools.archivist.vault_policy import is_searchable_note  # noqa: E402
from scripts.scan_vault_writers_r8 import scan  # noqa: E402


def _tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = 0
    markdown = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{rel}\0{path.stat().st_size}\0{value}\n".encode("utf-8"))
        files += 1
        markdown += int(path.suffix.lower() == ".md")
    return {"sha256": digest.hexdigest(), "file_count": files, "markdown_count": markdown}


def run(vault: Path) -> dict[str, Any]:
    registry = load_default_registry()
    searchable = 0
    invalid: list[dict[str, Any]] = []
    for path in sorted(vault.rglob("*.md")):
        if not is_searchable_note(path, vault_root=vault):
            continue
        searchable += 1
        try:
            metadata, _body, issues = parse_note(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            invalid.append({"path": path.relative_to(vault).as_posix(), "kind": "read", "issues": [str(exc)]})
            continue
        if issues:
            invalid.append({"path": path.relative_to(vault).as_posix(), "kind": "yaml", "issues": issues})
            continue
        valid, registry_issues = registry.validate_metadata(
            metadata,
            require_schema_version=False,
            allow_identity_allocation=True,
        )
        if not valid:
            invalid.append({"path": path.relative_to(vault).as_posix(), "kind": "registry", "issues": registry_issues})

    inventory = scan(vault.parent)
    checks = {
        "registry_loaded": bool(registry.profiles),
        "active_notes_registry_valid": not invalid,
        "writer_inventory_unresolved": inventory["counts"].get("unresolved", 0) == 0,
        "writer_inventory_review": inventory["counts"].get("review", 0) == 0,
        "writer_inventory_expired": inventory["counts"].get("expired", 0) == 0,
        "writer_inventory_parse_errors": inventory["counts"].get("parse_error", 0) == 0,
    }
    return {
        "schema": "vault-r8-preflight-v1",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "vault": str(vault),
        "checks": checks,
        "registry": {
            "registry_digest": registry.digest(),
            "policy_digest": registry.policy_digest(),
            "profile_count": len(registry.profiles),
            "alias_count": len(registry.entity_profiles),
        },
        "active_notes": {"searchable": searchable, "invalid_count": len(invalid), "invalid": invalid[:100]},
        "tree": _tree_fingerprint(vault),
        "writer_inventory": {"counts": inventory["counts"], "row_count": len(inventory["rows"])},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--output", type=Path, default=Path("scratch/vault-r8/f00-preflight-r8.json"))
    args = parser.parse_args()
    report = run(args.vault.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
