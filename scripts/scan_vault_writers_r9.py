"""Deny-by-default inventory for Vault-capable writers.

The R8 inventory used source-directory prefixes as an implicit approval.  R9
keeps those prefixes only as classification metadata and treats a direct
writer/broker construction as approved only at an explicit composition or
maintenance boundary.  Ordinary filesystem writes remain visible as
``generic_filesystem_write`` and are not silently treated as Vault writes.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.scan_vault_writers_r8 import INTERNAL_SOURCES, scan as scan_r8  # noqa: E402


APPROVED_BROKER_ROOTS = {
    "tools/archivist/composition.py",
    "tools/archivist/broker_cli.py",
    "api/workers/vault_write_worker.py",
    "scripts/observe_vault_r8.py",
    "scripts/reconcile_human_edits_r8.py",
    "scripts/rebuild_portfolio_projections_r8.py",
    "scripts/rehearse_vault_broker_r8.py",
    "scripts/run_vault_r8_acceptance.py",
    "scripts/run_vault_r9_acceptance.py",
}

APPROVED_ARTIFACT_ROOTS = {
    "tools/archivist/write_adapter.py",
    "tools/archivist/vault_link_healer.py",
    "tools/archivist/vault_metadata_backfill.py",
}

WRITE_LIKE = {
    "write_text",
    "write_bytes",
    "path.open_write",
    "_atomic_write_text",
    "_atomic_write_to",
    "write_raw_markdown",
    "save_memory",
    "os.replace",
    "shutil.move",
    "shutil.rmtree",
    "unlink",
    "mkdir",
    "ArtifactWriter.construct",
    "KnowledgeWriteBroker.construct",
}


def _contract(root: Path) -> dict[str, Any]:
    path = root / "memories" / ".system" / "storage_contract.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _direct_rules(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in contract.get("direct_writer_allowlist") or []:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source") or "").replace("\\", "/").strip("/")
        if source:
            result[source] = dict(raw)
    return result


def _effective_expiry(rule: dict[str, Any], contract: dict[str, Any]) -> str | None:
    return str(rule.get("expires_on") or (contract.get("allowlist_policy") or {}).get("default_expires_on") or "") or None


def _expiry_disposition(expires_on: str | None) -> str | None:
    if not expires_on:
        return None
    try:
        return "expired" if date.fromisoformat(expires_on) <= date.today() else None
    except ValueError:
        return "expired"


def _base_row(row: dict[str, Any], *, category: str, profile: str, owner: str | None, reason: str | None, rule: dict[str, Any] | None, disposition: str) -> dict[str, Any]:
    value = dict(row)
    value.update(
        {
            "category": category,
            "profile": profile,
            "owner": owner,
            "reason": reason,
            "target_pattern": (rule or {}).get("target_pattern"),
            "test_id": (rule or {}).get("test_id"),
            "expires_on": _effective_expiry(rule or {}, value.get("_contract") or {}),
            "disposition": disposition,
        }
    )
    value.pop("_contract", None)
    return value


def _classify_legacy_row(row: dict[str, Any], *, direct: dict[str, dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    source = str(row.get("source_file") or "").replace("\\", "/")
    operation = str(row.get("operation") or "")
    rule = direct.get(source)
    if rule is not None:
        profile = str(rule.get("profile") or "allowlisted")
        target_pattern = str(rule.get("target_pattern") or "")
        category = "vault_write_capability" if any(token in target_pattern for token in ("30_Knowledge_Base", "20_Portfolio_Management", "90_Attachments", ".system")) else "generic_filesystem_write"
        disposition = "allowlisted"
        required = ("source", "profile", "owner", "reason", "target_pattern", "test_id")
        if any(not str(rule.get(field) or "").strip() for field in required):
            disposition = "review"
        expiry = _effective_expiry(rule, contract)
        if _expiry_disposition(expiry):
            disposition = "expired"
        value = _base_row(row, category=category, profile=profile, owner=str(rule.get("owner") or "") or None, reason=str(rule.get("reason") or "") or None, rule=rule, disposition=disposition)
        value["expires_on"] = expiry
        value["allowlist_scope"] = "file"
        value["observed_symbol"] = row.get("function")
        value["observed_sink"] = operation
        return value

    if source in INTERNAL_SOURCES:
        profile, owner = INTERNAL_SOURCES[source]
        return _base_row(row, category="vault_write_capability" if operation in WRITE_LIKE else "generic_filesystem_write", profile=profile, owner=owner, reason="approved infrastructure boundary", rule=None, disposition="allowlisted")

    if source.startswith("scripts/"):
        return _base_row(row, category="maintenance", profile="maintenance", owner="vault-maintenance", reason="explicit script entrypoint; runtime policy requires lease/apply guard", rule=None, disposition="maintenance")
    if source.startswith("tests/"):
        return _base_row(row, category="test", profile="test", owner="test-suite", reason="temporary test/rehearsal filesystem", rule=None, disposition="test")
    if source.startswith("generated/"):
        return _base_row(row, category="generated", profile="generated", owner="build", reason="generated artifact", rule=None, disposition="generated")
    return _base_row(row, category="generic_filesystem_write", profile="generic_filesystem_write", owner=None, reason="not proven to target the Vault", rule=None, disposition="classified")


def _iter_python(root: Path):
    for path in sorted(root.rglob("*.py")):
        rel_parts = path.resolve().relative_to(root.resolve()).parts
        if not rel_parts or rel_parts[0] not in {"tools", "application", "api", "agents", "scripts"}:
            continue
        if any(part in {".venv", "__pycache__", ".git"} for part in rel_parts):
            continue
        yield path, "/".join(rel_parts)


def _structural_rows(root: Path, contract: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for path, source in _iter_python(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=source)
        except (OSError, SyntaxError) as exc:
            parse_errors.append({"source_file": source, "error": str(exc)})
            continue
        stack: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Import(self, node: ast.Import) -> Any:
                for alias in node.names:
                    if "write_port_compat" in alias.name:
                        rows.append({"source_file": source, "line": node.lineno, "function": stack[-1] if stack else "<module>", "operation": "compatibility_import", "target_expression": alias.name, "category": "vault_write_capability", "profile": "compatibility", "owner": "vault-contract", "reason": "R9 forbids production compatibility imports", "disposition": "unresolved"})
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
                module = str(node.module or "")
                if "write_port_compat" in module:
                    rows.append({"source_file": source, "line": node.lineno, "function": stack[-1] if stack else "<module>", "operation": "compatibility_import", "target_expression": module, "category": "vault_write_capability", "profile": "compatibility", "owner": "vault-contract", "reason": "R9 forbids production compatibility imports", "disposition": "unresolved"})
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> Any:
                name = ""
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name in {"KnowledgeWriteBroker", "ArtifactWriter"}:
                    operation = f"{name}.construct"
                    approved = source in (APPROVED_BROKER_ROOTS if name == "KnowledgeWriteBroker" else APPROVED_ARTIFACT_ROOTS)
                    if source.startswith("scripts/"):
                        category, profile, owner, reason, disposition = "maintenance", "maintenance", "vault-maintenance", "explicit script entrypoint; must be leased/apply guarded", "maintenance"
                    elif approved:
                        category, profile, owner, reason, disposition = "vault_write_capability", "composition_root", "vault-contract", "approved composition/infrastructure boundary", "allowlisted"
                    else:
                        category, profile, owner, reason, disposition = "vault_write_capability", "composition_root", None, "direct infrastructure construction outside approved roots", "unresolved"
                    rows.append({"source_file": source, "line": node.lineno, "function": stack[-1] if stack else "<module>", "operation": operation, "target_expression": ast.unparse(node), "category": category, "profile": profile, "owner": owner, "reason": reason, "disposition": disposition, "allowlist_scope": "file" if approved else None})
                self.generic_visit(node)

        Visitor().visit(tree)
    return rows, parse_errors


def scan(root: Path) -> dict[str, Any]:
    root = root.resolve()
    contract = _contract(root)
    direct = _direct_rules(contract)
    legacy = scan_r8(root)
    rows = [_classify_legacy_row(dict(row), direct=direct, contract=contract) for row in legacy.get("rows") or []]
    structural, parse_errors = _structural_rows(root, contract)
    rows.extend(structural)
    counts: dict[str, int] = {}
    for row in rows:
        disposition = str(row.get("disposition") or "classified")
        counts[disposition] = counts.get(disposition, 0) + 1
        category = str(row.get("category") or "generic_filesystem_write")
        counts[category] = counts.get(category, 0) + 1
    broad_rules = [
        dict(item) for item in (contract.get("source_prefix_rules") or [])
        if isinstance(item, dict) and not bool(item.get("classification_only"))
    ]
    counts["parse_error"] = len(parse_errors) + int(legacy.get("counts", {}).get("parse_error", 0))
    counts["broad_allowlist"] = len(broad_rules)
    for key in ("unresolved", "review", "expired", "allowlisted", "maintenance", "test", "generated", "vault_write_capability", "generic_filesystem_write"):
        counts.setdefault(key, 0)
    return {
        "schema": "vault-r9-writer-inventory-v1",
        "root": str(root),
        "manifest": "memories/.system/storage_contract.json",
        "counts": counts,
        "broad_allowlist": broad_rules,
        "rows": rows,
        "parse_errors": parse_errors,
        "classification_prefixes": [dict(item) for item in (contract.get("source_prefix_rules") or []) if isinstance(item, dict)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on", default="unresolved,review,expired,broad_allowlist")
    args = parser.parse_args()
    report = scan(args.root)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    blocked = {item.strip() for item in str(args.fail_on).split(",") if item.strip()}
    if any(report["counts"].get(item, 0) for item in blocked):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
