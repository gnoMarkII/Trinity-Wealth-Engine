"""Prepare a reviewable, read-only F13 live change set.

This command inventories the current live vault and writes all proposed
actions under ``scratch``.  It deliberately does not call an apply, repair,
writer, catalog sync, or vector indexing operation against ``memories``.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
from tools.archivist.catalog_runtime import resolve_catalog_path
from tools.archivist.vault_audit import scan_vault, write_audit_report
from tools.archivist.metadata import parse_note
from tools.archivist.vault_migration import create_migration_plan


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_fingerprint(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".chroma_index" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        file_hash = _sha256_file(path)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


def _git_head(workspace: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _infer_entity_type(parts: tuple[str, ...], filename: str) -> str | None:
    """Infer only from an unambiguous canonical folder, never from ticker text."""
    if "Earnings" in parts:
        return "earnings_call"
    if "Quant" in parts:
        return "quant_snapshot"
    if "Analysis" in parts:
        return "equity_analysis"
    if "Stocks" in parts:
        return "stock_hub" if Path(filename).stem.upper() in {parts[-2].upper(), parts[-1].split(".")[0].upper()} else "equity_analysis"
    if "YouTube_Summaries" in parts:
        return "youtube_summary"
    if "News" in parts:
        return "company_news"
    if "Books" in parts:
        return "book_note"
    if "Daily_Snapshots" in parts:
        return "macro_snapshot"
    if "Macroeconomics" in parts or "Strategies" in parts:
        return "macro_strategy"
    if "NotebookLM_Sources" in parts:
        return "briefing_book"
    if "Concepts" in parts:
        return "concept"
    return None


def _metadata_backfill_plan(live: Path, audit: Any, out: Path) -> dict[str, Any]:
    """Record proof-backed metadata proposals without allocating identities."""
    jsonl = out / "metadata-backfill-plan.jsonl"
    counts: dict[str, int] = {
        "records": 0,
        "schema_version": 0,
        "title": 0,
        "entity_type": 0,
        "identity_required": 0,
        "future_schema_read_only": 0,
        "malformed_blocked": 0,
    }
    h1_re = re.compile(r"^#\s+(.+)$", re.MULTILINE)
    with jsonl.open("w", encoding="utf-8") as stream:
        for record in audit.inventory:
            if record.extension != ".md" or record.is_excluded:
                continue
            path = live / record.relative_path
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            meta, body, issues = parse_note(raw)
            if issues:
                counts["malformed_blocked"] += 1
                continue
            proposed: dict[str, Any] = {}
            evidence: dict[str, str] = {}
            schema = meta.get("schema_version")
            if schema is not None and isinstance(schema, int) and schema > 2:
                counts["future_schema_read_only"] += 1
                continue
            if "schema_version" not in meta:
                proposed["schema_version"] = 2
                evidence["schema_version"] = "V2 contract"
                counts["schema_version"] += 1
            if not meta.get("title"):
                title_match = h1_re.search(body)
                title = title_match.group(1).strip() if title_match else path.stem.replace("_", " ")
                if title:
                    proposed["title"] = title
                    evidence["title"] = "H1" if title_match else "filename_stem"
                    counts["title"] += 1
            if not meta.get("entity_type") and not meta.get("type"):
                inferred = _infer_entity_type(Path(record.relative_path).parts, path.name)
                if inferred:
                    proposed["entity_type"] = inferred
                    evidence["entity_type"] = "canonical_folder"
                    counts["entity_type"] += 1
            if not meta.get("note_id"):
                counts["identity_required"] += 1
            if proposed or not meta.get("note_id"):
                counts["records"] += 1
                payload = {
                    "relative_path": record.relative_path,
                    "pre_hash": record.sha256,
                    "proposed_fields": proposed,
                    "evidence": evidence,
                    "identity_status": "requires_explicit_import" if not meta.get("note_id") else "present",
                    "disposition": "proposed" if proposed else "blocked_identity_missing",
                }
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    summary = {"path": str(jsonl), **counts}
    _write_json(out / "metadata-backfill-summary.json", summary)
    return summary


def _openapi_drift_report(workspace: Path, out: Path) -> dict[str, Any]:
    """Compare the generated API schema with the checked-in golden manifest.

    The report classifies route ownership/intent for review; it does not alter
    the golden fixture or silently turn drift into a pass.
    """
    fixture = workspace / "tests" / "fixtures" / "manifest_openapi_schema.json"
    report: dict[str, Any] = {
        "fixture": str(fixture),
        "fixture_exists": fixture.is_file(),
        "status": "NOT_RUN",
        "missing_paths": [],
        "extra_paths": [],
        "method_drift": [],
        "extra_route_ownership": [],
        "error": None,
    }
    if not fixture.is_file():
        report["error"] = "golden manifest missing"
        _write_json(out / "openapi-drift.json", report)
        return report
    try:
        from api.main import app

        expected = json.loads(fixture.read_text(encoding="utf-8"))
        expected_paths = expected.get("paths", {})
        actual_paths = app.openapi().get("paths", {})
        missing = sorted(set(expected_paths) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected_paths))
        method_drift: list[dict[str, Any]] = []
        for path in sorted(set(expected_paths) & set(actual_paths)):
            exp_methods = set(expected_paths[path])
            act_methods = set(actual_paths[path])
            if exp_methods != act_methods:
                method_drift.append({
                    "path": path,
                    "expected_methods": sorted(exp_methods),
                    "actual_methods": sorted(act_methods),
                })
        ownership = []
        for path in extra:
            if "/portfolio/scb/" in path:
                owner = "SCB portfolio connector"
            elif "/portfolio/wealthx/" in path:
                owner = "WealthX portfolio connector"
            elif "/portfolio/dime/" in path:
                owner = "Dime portfolio connector"
            else:
                owner = "unclassified"
            ownership.append({
                "path": path,
                "owner": owner,
                "methods": sorted(actual_paths[path]),
                "operation_ids": sorted(
                    str(actual_paths[path][method].get("operationId", ""))
                    for method in actual_paths[path]
                ),
            })
        report.update({
            "status": "PASS" if not missing and not extra and not method_drift else "DRIFT",
            "missing_paths": missing,
            "extra_paths": extra,
            "method_drift": method_drift,
            "extra_route_ownership": ownership,
        })
    except Exception as exc:
        report["status"] = "NOT_RUN"
        report["error"] = str(exc)
    _write_json(out / "openapi-drift.json", report)
    md_lines = [
        "# OpenAPI R3 Drift Review",
        "",
        f"- Status: **{report['status']}**",
        f"- Golden fixture: `{fixture}`",
        f"- Missing paths: {len(report['missing_paths'])}",
        f"- Extra paths: {len(report['extra_paths'])}",
        f"- Method drift: {len(report['method_drift'])}",
        "",
        "## Extra route ownership candidates",
        "",
        "| Path | Owner candidate | Methods | Operation IDs |",
        "|---|---|---|---|",
    ]
    for item in report["extra_route_ownership"]:
        md_lines.append(
            f"| `{item['path']}` | {item['owner']} | {', '.join(item['methods'])} | "
            f"{', '.join(item['operation_ids'])} |"
        )
    (out / "openapi-drift.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    live = Path(os.getenv("OBSIDIAN_VAULT_PATH", str(workspace / "memories"))).resolve()
    if not live.is_dir():
        raise FileNotFoundError(f"live vault does not exist: {live}")

    run_id = datetime.now(timezone.utc).strftime("f13_preflight_%Y%m%dT%H%M%SZ")
    out = workspace / "scratch" / "vault-v2" / "remediation-r3" / run_id
    baseline_dir = out / "baseline"
    out.mkdir(parents=True, exist_ok=True)

    before_fp, before_count = _tree_fingerprint(live)
    audit = scan_vault(live)
    inventory_json, audit_md = write_audit_report(audit, baseline_dir)
    metadata_plan = _metadata_backfill_plan(live, audit, out)
    openapi_drift = _openapi_drift_report(workspace, out)

    plan_path = out / "proposed-migration-plan.json"
    plan = create_migration_plan(live, output_file=plan_path)

    try:
        catalog_path = resolve_catalog_path(live, require_exists=True)
    except FileNotFoundError:
        catalog_path = live / ".system" / "vault_catalog.db"
    active_markdown_count = sum(
        1 for record in audit.inventory if record.extension == ".md" and not record.is_excluded
    )
    catalog_stats: dict[str, Any] = {
        "path": str(catalog_path),
        "exists": catalog_path.is_file(),
        "rows": None,
        "active_markdown_paths": active_markdown_count,
        "catalog_missing_active_paths": None,
        "catalog_extra_paths": None,
        "error": None,
    }
    if catalog_path.is_file():
        try:
            catalog = SqliteNoteCatalogAdapter(
                db_path=catalog_path,
                vault_root=live,
                read_only=True,
            )
            entries = list(catalog.iter_notes(page_size=500))
            catalog_paths = {entry.relative_path.replace("\\", "/") for entry in entries}
            active_paths = {
                record.relative_path
                for record in audit.inventory
                if record.extension == ".md" and not record.is_excluded
            }
            catalog_stats.update(
                {
                    "rows": len(entries),
                    "catalog_missing_active_paths": len(active_paths - catalog_paths),
                    "catalog_extra_paths": len(catalog_paths - active_paths),
                }
            )
        except Exception as exc:
            catalog_stats["error"] = str(exc)

    artifacts_root = live / ".system" / "artifacts"
    identity_file = live / ".system" / "identity_allocations.json"
    registry_file = artifacts_root / "registry.json"
    active_generation = live / ".system" / "vector_generation_active.json"
    durable_stats = {
        "identity_store_exists": identity_file.is_file(),
        "identity_store_sha256": _sha256_file(identity_file) if identity_file.is_file() else None,
        "identity_allocations": None,
        "artifact_registry_exists": registry_file.is_file(),
        "artifact_revisions": None,
        "artifact_heads": len(list((artifacts_root / "heads").glob("*.json")))
        if (artifacts_root / "heads").is_dir()
        else 0,
        "active_vector_generation_exists": active_generation.is_file(),
        "active_vector_generation_sha256": _sha256_file(active_generation)
        if active_generation.is_file()
        else None,
    }
    if identity_file.is_file():
        try:
            identity_data = json.loads(identity_file.read_text(encoding="utf-8"))
            durable_stats["identity_allocations"] = len(identity_data) if isinstance(identity_data, dict) else None
        except Exception:
            durable_stats["identity_allocations"] = "corrupt"
    if registry_file.is_file():
        try:
            registry_data = json.loads(registry_file.read_text(encoding="utf-8"))
            revisions = registry_data.get("revisions", {}) if isinstance(registry_data, dict) else {}
            durable_stats["artifact_revisions"] = len(revisions) if isinstance(revisions, dict) else None
        except Exception:
            durable_stats["artifact_revisions"] = "corrupt"

    after_fp, after_count = _tree_fingerprint(live)
    live_read_only = before_fp == after_fp and before_count == after_count

    proposed = {
        "run_id": run_id,
        "scope": "F13 preflight only; live apply not executed",
        "vault_root": str(live),
        "baseline": {
            "tree_fingerprint": before_fp,
            "file_count": before_count,
            "audit_total_files": audit.total_files,
            "audit_active_files": audit.active_files_count,
            "audit_excluded_files": audit.excluded_files_count,
            "audit_stats": audit.stats,
            "metadata_backfill": metadata_plan,
        },
        "migration_plan": {
            "path": str(plan_path),
            "plan_id": plan.plan_id,
            "total_files": plan.total_files,
            "summary": plan.summary,
            "contract_version": plan.contract_version,
            "config_before": plan.config_before,
            "config_before_hash": plan.config_before_hash,
        },
        "catalog": catalog_stats,
        "durable_state": durable_stats,
        "openapi_drift": openapi_drift,
        "read_only_guard": {
            "tree_fingerprint_after": after_fp,
            "file_count_after": after_count,
            "unchanged": live_read_only,
        },
        "exact_commands": {
            "plan": f".venv\\Scripts\\python.exe -m tools.archivist.vault_v2_cli plan --vault \"{live}\" --output \"{plan_path}\"",
            "snapshot": f".venv\\Scripts\\python.exe -m tools.archivist.vault_v2_cli snapshot --vault \"{live}\" --output <approved-backup-dir>",
            "apply": f".venv\\Scripts\\python.exe -m tools.archivist.vault_v2_cli apply --plan \"{plan_path}\" --vault \"{live}\" --allow-live",
            "verify": f".venv\\Scripts\\python.exe -m tools.archivist.vault_v2_cli verify --plan \"{plan_path}\" --vault \"{live}\"",
            "rollback": ".venv\\Scripts\\python.exe -m tools.archivist.vault_v2_cli rollback --journal <journal-from-apply> --vault " + f'"{live}"',
        },
        "preconditions": [
            "F12 rehearsal report is PASS with immutable evidence hashes",
            "maintenance window pauses Obsidian and all managed writers",
            "approved backup includes vault plus durable state and has restore proof",
            "fresh pre-apply hash/config/registry/owner-reference check equals this baseline",
            "explicit operator approval for the listed apply command",
        ],
        "not_executed": [
            "live migration/apply",
            "live metadata backfill or identity allocation",
            "live catalog mutation",
            "live vector model build or activation",
            "live Obsidian/UI acceptance",
        ],
    }

    _write_json(out / "run.json", {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_head": _git_head(workspace),
        "vault_root": str(live),
        "scope": "read-only F13 preflight",
        "tree_fingerprint": before_fp,
        "file_count": before_count,
        "f12_report": str(workspace / "scratch" / "vault-v2" / "remediation-r3" / "latest-run.json"),
    })
    _write_json(out / "proposed-live-changes.json", proposed)

    traceability = {
        "run_id": run_id,
        "f12_acceptance_report": str(workspace / "scratch" / "vault-v2" / "remediation-r3" / "latest-run.json"),
        "findings": {
            "R3-01": {"tasks": ["F02", "F05", "F07", "F09"], "cases": ["C01", "C02", "C03", "C20", "C21"], "evidence": [str(out / "baseline" / "inventory.json")]},
            "R3-02": {"tasks": ["F03"], "cases": ["C10", "C11", "C12"], "evidence": [str(workspace / "tests" / "tools" / "archivist" / "test_vault_v2_crash_recovery.py")]},
            "R3-03": {"tasks": ["F03"], "cases": ["C04", "C05", "C06", "C07", "C08", "C09"], "evidence": [str(workspace / "tests" / "tools" / "archivist" / "test_vault_v2_regression_probes.py")]},
            "R3-04": {"tasks": ["F01", "F06"], "cases": ["C15", "C16"], "evidence": [str(workspace / "tests" / "unit" / "application" / "test_notebooklm_service.py")]},
            "R3-05": {"tasks": ["F07", "F08"], "cases": ["C17", "C18", "C19"], "evidence": [str(workspace / "tests" / "tools" / "archivist" / "test_vault_v2_acceptance_regressions.py")]},
            "R3-06": {"tasks": ["F04"], "cases": ["C13", "C14"], "evidence": [str(workspace / "tests" / "tools" / "content" / "test_notebooklm_vault_migration.py")]},
            "R3-07": {"tasks": ["F09", "F11", "F12"], "cases": ["C22", "C23"], "evidence": [str(workspace / "scratch" / "vault-v2" / "remediation-r3")]},
            "R3-08": {"tasks": ["F10"], "cases": ["C24"], "evidence": [str(workspace / "tests" / "tools" / "archivist" / "test_vault_v2_navigation.py")]},
        },
    }
    _write_json(out / "traceability.json", traceability)

    unresolved = [
        "Live apply is intentionally not executed by this preflight; it requires the listed maintenance and approval preconditions.",
        f"Metadata audit reports {audit.stats.get('missing_metadata', 0)} missing-recommended-property findings; no facts are invented here.",
        f"Broken-link findings: {audit.stats.get('broken_links', 0)}; ambiguous-link findings: {audit.stats.get('ambiguous_links', 0)}.",
        "Historical NotebookLM/source records without proof-backed mapping remain blocked or read-only until F06 owner references are supplied.",
        "Real provider/model and 1k/10k/50k acceptance runs are not claimed by this preflight.",
        "OpenAPI drift ownership and interactive Obsidian/UI inspection require their respective owner evidence.",
    ]
    (out / "unresolved-items.md").write_text(
        "# F13 Preflight Unresolved Items\n\n" + "\n".join(f"- {item}" for item in unresolved) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Proposed Live Changes — Vault V2 R3",
        "",
        f"- Run: `{run_id}`",
        f"- Live root: `{live}`",
        f"- Baseline tree fingerprint: `{before_fp}` ({before_count:,} files)",
        f"- Read-only guard: **{'PASS' if live_read_only else 'FAIL'}**",
        f"- Migration plan: `{plan.plan_id}` ({plan.total_files:,} files; {plan.summary})",
        f"- Metadata backfill candidates: `{metadata_plan['records']:,}` records (identity imports remain explicit)",
        "",
        "## Exact proposed operations",
        "",
        "1. Create and verify an approved snapshot containing vault and durable state.",
        "2. Recheck the baseline fingerprint/config/registry and pause affected writers.",
        "3. Apply the generated plan with the explicit `--allow-live` flag.",
        "4. Verify every target hash, then build/validate the production vector generation before activation.",
        "5. On any mismatch, retain journal/evidence and use the recorded rollback command; do not force-overwrite user edits.",
        "6. Run post-apply catalog, ingestion/restart, search-isolation, navigation, and owner acceptance checks.",
        "",
        "## Commands",
        "",
    ]
    for name, command in proposed["exact_commands"].items():
        lines.extend([f"### {name}", "", "```powershell", command, "```", ""])
    lines.extend([
        "## Preconditions",
        "",
        *[f"- {item}" for item in proposed["preconditions"]],
        "",
        "## Scope explicitly not executed",
        "",
        *[f"- {item}" for item in proposed["not_executed"]],
        "",
        "Machine-readable details are in `proposed-live-changes.json`, `metadata-backfill-plan.jsonl`, `openapi-drift.json`, `run.json`, `traceability.json`, and `baseline/`.",
        "",
    ])
    (out / "proposed-live-changes.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json(workspace / "scratch" / "vault-v2" / "remediation-r3" / "latest-preflight.json", {
        "run_id": run_id,
        "report_dir": str(out),
        "status": "READY_FOR_REVIEW" if live_read_only else "FAIL",
        "plan": str(plan_path),
    })

    print(json.dumps({
        "run_id": run_id,
        "status": "READY_FOR_REVIEW" if live_read_only else "FAIL",
        "output": str(out),
        "plan": str(plan_path),
        "total_files": plan.total_files,
        "summary": plan.summary,
        "audit": audit.stats,
        "catalog": catalog_stats,
    }, ensure_ascii=False))
    return 0 if live_read_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
