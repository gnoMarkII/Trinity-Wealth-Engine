"""Path/profile-based inventory of production Vault write intent.

R7 used a filename allowlist.  R8 resolves the source file against the
machine-readable storage-contract entries, records the operation and target
expression, and fails closed when a production write has no owner/profile.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WRITE_ATTRS = {"write_text", "write_bytes", "unlink", "mkdir"}
PRODUCTION_ROOTS = {"tools", "application", "api", "agents", "scripts"}
INTERNAL_SOURCES = {
    "tools/_atomic_io.py": ("infrastructure", "atomic_io"),
    "tools/archivist/artifact_writer.py": ("infrastructure", "artifact_engine"),
    "tools/archivist/write_adapter.py": ("infrastructure", "broker_executor"),
    "tools/archivist/write_broker.py": ("external_runtime", "broker_runtime"),
    "tools/archivist/write_port_compat.py": ("compatibility", "broker_boundary"),
    "tools/archivist/managed_blocks.py": ("infrastructure", "managed_blocks"),
    "tools/archivist/writer.py": ("compatibility", "broker_boundary"),
}


def _contract_path(root: Path) -> Path:
    return root / "memories" / ".system" / "storage_contract.json"


def _load_manifest(root: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(_contract_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    values = payload.get("direct_writer_allowlist") if isinstance(payload, dict) else []
    return [dict(item) for item in values if isinstance(item, dict)]


def _load_prefix_rules(root: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(_contract_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    values = payload.get("source_prefix_rules") if isinstance(payload, dict) else []
    return [dict(item) for item in values if isinstance(item, dict)]


def _load_allowlist_policy(root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(_contract_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    value = payload.get("allowlist_policy") if isinstance(payload, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _norm(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _rule_for(
    source: str,
    rules: list[dict[str, Any]],
    prefix_rules: list[dict[str, Any]],
) -> tuple[str, str, str, str | None]:
    if source in INTERNAL_SOURCES:
        profile, owner = INTERNAL_SOURCES[source]
        return profile, owner, "internal infrastructure boundary", None
    for rule in rules:
        candidate = str(rule.get("source") or "").replace("\\", "/").strip("/")
        if candidate and (source == candidate or source.endswith("/" + candidate)):
            return (
                str(rule.get("profile") or "allowlisted"),
                str(rule.get("owner") or ""),
                str(rule.get("reason") or ""),
                str(rule.get("expires_on") or "") or None,
            )
    for rule in prefix_rules:
        prefix = str(rule.get("prefix") or "").replace("\\", "/")
        if prefix and source.startswith(prefix):
            return (
                str(rule.get("profile") or "allowlisted"),
                str(rule.get("owner") or ""),
                str(rule.get("reason") or ""),
                str(rule.get("expires_on") or "") or None,
            )
    if source.startswith("scripts/"):
        return "privileged_or_rehearsal", "vault-maintenance", "offline rehearsal or leased maintenance entrypoint", None
    return "", "", "", None


class Visitor(ast.NodeVisitor):
    def __init__(self, source: str, rules: list[dict[str, Any]], prefix_rules: list[dict[str, Any]], allowlist_policy: dict[str, Any]) -> None:
        self.source = source
        self.rules = rules
        self.prefix_rules = prefix_rules
        self.allowlist_policy = allowlist_policy
        self.function_stack: list[str] = []
        self.rows: list[dict[str, Any]] = []

    def _add(self, node: ast.AST, operation: str, target: str, *, severity: str = "write") -> None:
        profile, owner, reason, expires_on = _rule_for(self.source, self.rules, self.prefix_rules)
        disposition = "allowlisted" if profile else "unresolved"
        if self.source.startswith("scripts/") and profile:
            disposition = "maintenance_or_rehearsal"
        effective_expires_on = expires_on or str(self.allowlist_policy.get("default_expires_on") or "") or None
        if profile and effective_expires_on:
            try:
                if date.fromisoformat(effective_expires_on) <= date.today():
                    disposition = "expired"
            except ValueError:
                disposition = "expired"
        self.rows.append(
            {
                "source_file": self.source,
                "line": getattr(node, "lineno", None),
                "function": self.function_stack[-1] if self.function_stack else "<module>",
                "operation": operation,
                "target_expression": target,
                "profile": profile or "unresolved",
                "owner": owner or None,
                "reason": reason or None,
                "expires_on": effective_expires_on,
                "disposition": disposition,
                "severity": severity,
            }
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> Any:
        operation = ""
        target = ""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in WRITE_ATTRS:
                operation = node.func.attr
                target = ast.unparse(node.func.value)
            elif node.func.attr == "write_note" and isinstance(node.func.value, ast.Call):
                operation = "writer.write_note"
                target = ast.unparse(node.func.value)
            elif node.func.attr == "open" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and any(flag in str(node.args[1].value) for flag in ("w", "a", "x", "+")):
                operation = "path.open_write"
                target = ast.unparse(node.func.value)
            elif node.func.attr in {"move", "rmtree"} or (
                node.func.attr == "replace"
                and ast.unparse(node.func.value).strip().lower() in {"os", "shutil"}
            ):
                operation = f"{ast.unparse(node.func.value)}.{node.func.attr}"
                target = ast.unparse(node.args[0]) if node.args else ""
        elif isinstance(node.func, ast.Name):
            if node.func.id in {"open", "_atomic_write_text", "_atomic_write_to", "write_raw_markdown", "save_memory"}:
                if node.func.id in {"open"} and len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and not any(flag in str(node.args[1].value) for flag in ("w", "a", "x", "+")):
                    operation = ""
                else:
                    operation = node.func.id
                    target = ast.unparse(node.args[0]) if node.args else ""
            elif node.func.id == "ArtifactWriter":
                operation = "ArtifactWriter.construct"
                target = ast.unparse(node)
        if operation:
            self._add(node, operation, target)
        self.generic_visit(node)


def scan(root: Path) -> dict[str, Any]:
    rules = _load_manifest(root)
    prefix_rules = _load_prefix_rules(root)
    allowlist_policy = _load_allowlist_policy(root)
    rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        rel_parts = path.resolve().relative_to(root.resolve()).parts
        if not rel_parts or rel_parts[0] not in PRODUCTION_ROOTS or any(part in {".venv", "__pycache__", ".git"} for part in rel_parts):
            continue
        source = "/".join(rel_parts)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=source)
        except (OSError, SyntaxError) as exc:
            parse_errors.append({"source_file": source, "error": str(exc)})
            continue
        visitor = Visitor(source, rules, prefix_rules, allowlist_policy)
        visitor.visit(tree)
        rows.extend(visitor.rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    counts["parse_error"] = len(parse_errors)
    counts.setdefault("expired", 0)
    return {
        "schema": "vault-r8-writer-inventory-v1",
        "root": str(root),
        "manifest": "memories/.system/storage_contract.json",
        "counts": counts,
        "rows": rows,
        "parse_errors": parse_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on", default="", help="comma-separated dispositions, e.g. unresolved,review")
    args = parser.parse_args()
    report = scan(args.root.resolve())
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    blocked = {item.strip() for item in str(args.fail_on).split(",") if item.strip()}
    if blocked and any(row.get("disposition") in blocked for row in report["rows"]):
        return 2
    if report["parse_errors"] and "parse_error" in blocked:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
