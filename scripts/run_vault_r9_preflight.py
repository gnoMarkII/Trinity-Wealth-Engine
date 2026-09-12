"""Read-only production preflight for the R9 Vault platform boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.scan_vault_writers_r9 import scan  # noqa: E402
from tools.archivist.metadata import parse_note  # noqa: E402
from tools.archivist.recovery_bundle import capture_runtime_state  # noqa: E402
from tools.archivist.runtime_layout import runtime_layout  # noqa: E402
from tools.archivist.schema_registry import load_default_registry  # noqa: E402
from tools.archivist.vault_policy import is_searchable_note  # noqa: E402


def _tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    markdown_count = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{relative}\0{path.stat().st_size}\0{file_hash}\n".encode("utf-8"))
        file_count += 1
        markdown_count += int(path.suffix.lower() == ".md")
    return {"sha256": digest.hexdigest(), "file_count": file_count, "markdown_count": markdown_count}


def _runtime_files_inside_vault(vault: Path) -> list[str]:
    suffixes = (".sqlite", ".sqlite3", ".db", "-wal", "-shm", "-journal")
    return sorted(
        path.relative_to(vault).as_posix()
        for path in vault.rglob("*")
        if path.is_file() and (path.suffix.lower() in {".sqlite", ".sqlite3", ".db"} or path.name.endswith(suffixes[3:]))
    )


def run(vault: Path, runtime_base: Path | None) -> dict[str, Any]:
    vault = vault.resolve()
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

    layout = runtime_layout(vault, runtime_base, create=False)
    runtime_state = capture_runtime_state(vault, runtime_root=layout.root)
    inventory = scan(vault.parent)
    in_vault_runtime = _runtime_files_inside_vault(vault)
    checks = {
        "registry_loaded": bool(registry.profiles),
        "active_notes_registry_valid": not invalid,
        "writer_inventory_unresolved": inventory["counts"].get("unresolved", 0) == 0,
        "writer_inventory_review": inventory["counts"].get("review", 0) == 0,
        "writer_inventory_expired": inventory["counts"].get("expired", 0) == 0,
        "writer_inventory_parse_errors": inventory["counts"].get("parse_error", 0) == 0,
        "writer_inventory_broad_allowlist": inventory["counts"].get("broad_allowlist", 0) == 0,
        "runtime_outside_vault": not layout.root.is_relative_to(vault),
        "no_runtime_sqlite_inside_vault": not in_vault_runtime,
        "catalog_integrity": runtime_state["catalog"].get("integrity_check") in {None, "ok"},
        "registry_digest_parity": runtime_state.get("registry_digest") == registry.digest(),
        "policy_digest_parity": runtime_state.get("policy_digest") == registry.policy_digest(),
        "active_vector_policy_parity": runtime_state["vector"].get("policy_digest") == registry.policy_digest(),
    }
    return {
        "schema": "vault-r9-preflight-v1",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "vault": str(vault),
        "checks": checks,
        "runtime_layout": layout.as_dict(),
        "runtime_state": runtime_state,
        "active_notes": {"searchable": searchable, "invalid_count": len(invalid), "invalid": invalid[:100]},
        "tree": _tree_fingerprint(vault),
        "writer_inventory": {"counts": inventory["counts"], "row_count": len(inventory["rows"])},
        "runtime_files_inside_vault": in_vault_runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--runtime-base", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("scratch/vault-r9/preflight/r9-preflight.json"))
    args = parser.parse_args()
    report = run(args.vault, args.runtime_base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
