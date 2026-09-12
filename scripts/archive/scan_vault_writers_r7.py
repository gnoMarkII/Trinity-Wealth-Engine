"""AST inventory of filesystem writers that may reach the Obsidian Vault."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


WRITE_METHODS = {"write_text", "write_bytes", "unlink", "mkdir"}
WRITE_CALLS = {"_atomic_write_to", "_atomic_write_text", "_stage_text", "os.replace", "frontmatter.dumps", "fm.dumps", "yaml.dump", "yaml.safe_dump"}
SHARED_FILES = {"artifact_writer.py", "_atomic_io.py", "core.py"}
ALLOWLISTS = {
    "_atomic_io.py": "shared_writer_internal",
    "navigation_builder.py": "navigation_profile",
    "repository_adapter.py": "portfolio_state_profile",
    "watchlist_adapter.py": "portfolio_state_profile",
    "goals_adapter.py": "portfolio_state_profile",
    "journal_vault_adapter.py": "portfolio_state_profile",
    "performance_adapter.py": "portfolio_derived_profile",
    "goals.py": "portfolio_state_legacy_wrapper",
    "journal.py": "portfolio_state_legacy_wrapper",
    "performance.py": "portfolio_derived_legacy_wrapper",
    "vault_migration.py": "migration_profile",
    "vault_metadata_backfill.py": "migration_profile",
    "vault_link_healer.py": "migration_profile",
    "identity_store.py": "control_metadata_profile",
    "metadata.py": "shared_serializer_internal",
    "search.py": "derived_runtime_profile",
    "vault_acceptance.py": "evidence_report_profile",
    "vault_backup.py": "external_backup_profile",
    "writer.py": "legacy_v1_fail_closed_profile",
    "adapter.py": "external_runtime_probe_profile",
    "core.py": "formatter_only_profile",
    "youtube.py": "formatter_only_profile",
    "finnomena_adapter.py": "external_cache_profile",
    "entity_registry.py": "control_metadata_profile",
    "maintenance_guard.py": "control_metadata_profile",
    "vault_archival_policy.py": "migration_profile",
    "vault_audit.py": "evidence_report_profile",
    "vault_benchmark.py": "scratch_evidence_profile",
    "briefing_artifacts.py": "shared_writer_boundary",
    "obsidian_adapter.py": "shared_writer_boundary",
    "audio_utils.py": "external_runtime_profile",
    "manifest.py": "external_runtime_profile",
    "pipeline.py": "external_runtime_profile",
    "baselines.py": "derived_runtime_profile",
    "dashboard.py": "derived_runtime_profile",
    "evaluation.py": "derived_runtime_profile",
    "news_funnel.py": "migration_profile",
    "report_formatter.py": "derived_runtime_profile",
    "sqlite_mirror_decorator.py": "external_runtime_profile",
    "dime_sync_service.py": "external_runtime_profile",
    "scbam_sync_service.py": "external_runtime_profile",
    "wealthx_sync_service.py": "external_runtime_profile",
    "quant_history.py": "shared_writer_boundary",
    "news_funnel_store.py": "external_runtime_profile",
    "pit_fundamentals_ledger.py": "derived_runtime_profile",
    "paths.py": "portfolio_state_profile",
}


class Visitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.function_stack: list[str] = []
        self.rows: list[dict[str, Any]] = []

    def _record(self, node: ast.AST, operation: str, target: str = "") -> None:
        function = self.function_stack[-1] if self.function_stack else "<module>"
        filename = self.path.name
        if filename in SHARED_FILES and "archivist" in self.path.parts:
            disposition = "shared_writer_internal"
        elif filename in ALLOWLISTS:
            disposition = ALLOWLISTS[filename]
        elif any(token in self.path.as_posix().lower() for token in ("cache", "outbox", "sidecar", "evidence", "vector", "catalog", "index")):
            disposition = "derived_or_control_candidate"
        elif operation in {"write_text", "write_bytes", "path.open_write", "open_write", "frontmatter.dumps", "fm.dumps", "yaml.dump", "yaml.safe_dump"}:
            disposition = "unresolved"
        else:
            disposition = "review"
        self.rows.append({
            "source_file": self.path.as_posix(),
            "line": getattr(node, "lineno", None),
            "function": function,
            "operation": operation,
            "target_expression": target,
            "disposition": disposition,
        })

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> Any:
        operation = ""
        target = ""
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in WRITE_METHODS:
                operation = attr
                target = ast.unparse(node.func.value)
            elif attr == "replace" and isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                operation = "os.replace"
                target = ast.unparse(node.args[1]) if len(node.args) > 1 else ""
            elif attr == "rmtree" and isinstance(node.func.value, ast.Name) and node.func.value.id == "shutil":
                operation = "shutil.rmtree"
                target = ast.unparse(node.args[0]) if node.args else ""
            elif attr == "open":
                mode = ""
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = str(node.args[1].value)
                for keyword in node.keywords:
                    if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                        mode = str(keyword.value.value)
                if any(flag in mode for flag in ("w", "a", "x", "+")):
                    operation = "path.open_write"
                    target = ast.unparse(node.func.value)
            elif attr in {"dumps"} and isinstance(node.func.value, ast.Name) and node.func.value.id in {"frontmatter", "fm"}:
                operation = f"{node.func.value.id}.dumps"
            elif attr in {"dump", "safe_dump"} and isinstance(node.func.value, ast.Name) and node.func.value.id == "yaml":
                operation = f"yaml.{attr}"
        elif isinstance(node.func, ast.Name) and node.func.id in {"_atomic_write_to", "_atomic_write_text", "_stage_text"}:
            operation = node.func.id
            target = ast.unparse(node.args[0]) if node.args else ""
        elif isinstance(node.func, ast.Name) and node.func.id == "open":
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for keyword in node.keywords:
                if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                    mode = str(keyword.value.value)
            if any(flag in mode for flag in ("w", "a", "x", "+")):
                operation = "open_write"
                target = ast.unparse(node.args[0]) if node.args else ""
        if operation:
            self._record(node, operation, target)
        self.generic_visit(node)


def scan(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in {".venv", "__pycache__", ".git"} for part in path.parts):
            continue
        try:
            top_level = path.relative_to(root).parts[0]
        except (ValueError, IndexError):
            continue
        if top_level not in {"tools", "application", "api"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            rows.append({"source_file": path.as_posix(), "line": None, "function": "<parse>", "operation": "parse_error", "target_expression": str(exc), "disposition": "blocked"})
            continue
        visitor = Visitor(path)
        visitor.visit(tree)
        rows.extend(visitor.rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    return {"schema": "vault-r7-writer-inventory-v1", "root": str(root), "counts": counts, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = scan(args.root.resolve())
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
