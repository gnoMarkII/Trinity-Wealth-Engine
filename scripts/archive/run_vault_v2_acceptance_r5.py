"""Run the R5 acceptance extension A21-A30 and publish one A01-A30 report."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document  # noqa: E402

from tools.archivist.ai_answer_contract import collect_evidence  # noqa: E402
from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter  # noqa: E402
from tools.archivist.catalog_runtime import (  # noqa: E402
    catalog_outbox_path,
    catalog_runtime_root,
    load_catalog_pointer,
    resolve_catalog_path,
)
from tools.archivist.metadata import parse_note  # noqa: E402
from tools.archivist.navigation_builder import build_navigation_indices  # noqa: E402
from tools.archivist.vault_acceptance import (  # noqa: E402
    aggregate_acceptance_records,
    hash_file,
    record_assertion,
    write_acceptance_report,
)
from tools.archivist.vault_audit import scan_vault  # noqa: E402
from tools.archivist.vault_backup import create_vault_snapshot, restore_vault_snapshot  # noqa: E402
from tools.archivist.vault_maintenance import detect_orphaned_sidecars  # noqa: E402
from tools.archivist.vault_policy import is_searchable_note  # noqa: E402
from tools.archivist.vector_generation import (  # noqa: E402
    corpus_fingerprint,
    load_active_manifest,
    vector_runtime_path,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        digest.update(f"{rel}\0{path.stat().st_size}\0{file_hash}\n".encode("utf-8"))
        count += 1
    return {"sha256": digest.hexdigest(), "file_count": count}


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return default


def _sqlite_rows(path: Path, table: str) -> list[dict[str, Any]]:
    uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]


def _nav_paths(vault: Path) -> list[Path]:
    paths = sorted((vault / "00_Index").glob("*.md"))
    source_index = vault / "30_Knowledge_Base" / "NotebookLM_Sources" / "index.md"
    if source_index.is_file():
        paths.append(source_index)
    return paths


def _navigation_contract(vault: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for path in _nav_paths(vault):
        rel = path.relative_to(vault).as_posix()
        text = path.read_text(encoding="utf-8")
        metadata, body, issues = parse_note(text)
        starts_with_yaml = text.startswith("---")
        identity_keys = {
            key: len(re.findall(rf"(?m)^\s*{key}\s*:", text))
            for key in ("note_id", "document_key")
        }
        body_duplicate_keys = {
            key: len(re.findall(rf"(?m)^\s*{key}\s*:", body))
            for key in ("schema_version", "note_id", "document_key")
        }
        item = {
            "relative_path": rel,
            "starts_with_yaml": starts_with_yaml,
            "parse_issues": issues,
            "metadata_keys": sorted(metadata),
            "identity_key_counts": identity_keys,
            "body_duplicate_key_counts": body_duplicate_keys,
        }
        details.append(item)
        if (
            not starts_with_yaml
            or issues
            or identity_keys["note_id"] != 1
            or identity_keys["document_key"] != 1
            or any(body_duplicate_keys.values())
        ):
            errors.append(item)
    return {"paths": [item["relative_path"] for item in details], "details": details, "errors": errors}


def _navigation_idempotency(vault: Path, run_dir: Path) -> dict[str, Any]:
    clone = run_dir / "navigation-idempotency-clone"
    if clone.exists():
        clone = run_dir / f"navigation-idempotency-clone-{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
    shutil.copytree(vault, clone, ignore=shutil.ignore_patterns(".trash", ".sync_history"))
    lease = clone / ".system" / "maintenance.json"
    lease.unlink(missing_ok=True)
    build_navigation_indices(vault_root=clone)
    first = {
        path.relative_to(clone).as_posix(): _sha256(path)
        for path in clone.rglob("*.md")
    }
    build_navigation_indices(vault_root=clone)
    second = {
        path.relative_to(clone).as_posix(): _sha256(path)
        for path in clone.rglob("*.md")
    }
    changed = sorted(
        rel for rel in set(first) | set(second)
        if first.get(rel) != second.get(rel)
    )
    return {
        "clone": str(clone),
        "first_markdown_count": len(first),
        "second_markdown_count": len(second),
        "changed_files_second_run": changed,
        "status": "PASS" if not changed else "FAIL",
    }


def _catalog_generation_contract(vault: Path) -> dict[str, Any]:
    pointer = load_catalog_pointer(vault) or {}
    database = resolve_catalog_path(vault, require_exists=True)
    runtime = catalog_runtime_root(vault)
    manifest_rel = str(pointer.get("manifest_relative_path") or "")
    manifest = (runtime / manifest_rel).resolve() if manifest_rel else None
    database_rel = str(pointer.get("database_relative_path") or "")
    absolute_pointer_fields = [
        key for key, value in pointer.items()
        if isinstance(value, str) and (Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value))
    ]
    sidecars = [database.with_name(database.name + suffix) for suffix in ("-wal", "-shm")]
    integrity: list[dict[str, Any]] = []
    candidate_paths = sorted((runtime / "catalog").glob("*/vault_catalog.db"))
    for candidate in candidate_paths:
        uri = f"file:{candidate.as_posix()}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as conn:
            check = conn.execute("PRAGMA integrity_check").fetchone()[0]
            count = conn.execute("SELECT COUNT(*) FROM note_catalog").fetchone()[0]
        integrity.append({"path": str(candidate), "integrity": check, "note_count": count})
    prior = [item for item in integrity if Path(item["path"]).resolve() != database.resolve()]
    valid = (
        bool(pointer)
        and bool(database_rel)
        and database.is_file()
        and database.is_relative_to(runtime)
        and not database.is_relative_to(vault)
        and not absolute_pointer_fields
        and manifest is not None
        and manifest.is_file()
        and not any(path.exists() for path in sidecars)
        and bool(prior)
        and all(item["integrity"] == "ok" for item in integrity)
    )
    return {
        "status": "PASS" if valid else "FAIL",
        "pointer": pointer,
        "database": str(database),
        "manifest": str(manifest) if manifest else None,
        "runtime": str(runtime),
        "absolute_pointer_fields": absolute_pointer_fields,
        "active_sidecars": [str(path) for path in sidecars if path.exists()],
        "generation_count": len(integrity),
        "rollback_candidate_count": len(prior),
        "integrity": integrity,
    }


def _sidecar_contract(vault: Path) -> dict[str, Any]:
    database = resolve_catalog_path(vault, require_exists=True)
    rows = _sqlite_rows(database, "sidecar_catalog")
    errors: list[dict[str, Any]] = []
    for row in rows:
        path = vault / str(row.get("relative_path") or "")
        if not path.is_file():
            errors.append({"relative_path": row.get("relative_path"), "reason": "missing"})
            continue
        actual_hash = _sha256(path)
        actual_size = path.stat().st_size
        if actual_hash != str(row.get("sha256")) or actual_size != int(row.get("file_size") or -1):
            errors.append({
                "relative_path": row.get("relative_path"),
                "reason": "hash_or_size_mismatch",
                "catalog_sha256": row.get("sha256"),
                "actual_sha256": actual_hash,
                "catalog_size": row.get("file_size"),
                "actual_size": actual_size,
            })
        if not row.get("ticker") or not row.get("evaluation_date") or not row.get("storage_tier"):
            errors.append({"relative_path": row.get("relative_path"), "reason": "missing_owner_or_role"})
    errors.extend({"relative_path": item.get("sidecar_path"), "reason": item.get("reason")} for item in detect_orphaned_sidecars(vault_root=vault))
    return {"row_count": len(rows), "expected_row_count": 26, "errors": errors, "status": "PASS" if len(rows) == 26 and not errors else "FAIL"}


def _obsidian_contract(vault: Path) -> dict[str, Any]:
    path = vault / ".obsidian" / "app.json"
    settings = _read_json(path, {}) or {}
    expected = {
        "promptDelete": True,
        "alwaysUpdateLinks": True,
        "showUnsupportedFiles": False,
        "newFileLocation": "folder",
        "newLinkFormat": "absolute",
        "useMarkdownLinks": False,
        "attachmentFolderPath": "90_Attachments",
    }
    mismatches = {
        key: {"expected": value, "actual": settings.get(key)}
        for key, value in expected.items()
        if settings.get(key) != value
    }
    audit = scan_vault(vault)
    errors = int(audit.stats.get("broken_links", 0)) + int(audit.stats.get("ambiguous_links", 0))
    attachment = vault / "90_Attachments"
    return {
        "status": "PASS" if not mismatches and errors == 0 and attachment.is_dir() else "FAIL",
        "settings": settings,
        "mismatches": mismatches,
        "broken_links": audit.stats.get("broken_links", 0),
        "ambiguous_links": audit.stats.get("ambiguous_links", 0),
        "attachment_folder": str(attachment),
        "attachment_folder_exists": attachment.is_dir(),
    }


def _cleanup_contract(vault: Path, run_dir: Path) -> dict[str, Any]:
    cleanup = _read_json(run_dir / "cleanup-dispositions.json", {}) or {}
    legacy = [
        str(path.relative_to(vault)).replace("\\", "/")
        for path in vault.rglob("*")
        if path.is_file()
        and (path.name.startswith(".pre_migration_backup_") or path.name.startswith("parking_unknown_"))
    ]
    journals = [
        str(path.relative_to(vault)).replace("\\", "/")
        for path in vault.rglob("*.jsonl")
        if "migration_journal" in path.name
    ]
    legacy_catalog = [
        str(path.relative_to(vault)).replace("\\", "/")
        for path in (vault / ".system").glob("vault_catalog.db*")
        if path.exists()
    ]
    outbox = catalog_outbox_path(vault)
    outbox_errors: list[str] = []
    outbox_records = []
    if outbox.is_file():
        for line in outbox.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            outbox_records.append(item)
            if item.get("state") not in {"pending", "processing", "consumed", "failed"} or not item.get("idempotency_key"):
                outbox_errors.append(str(item))
    cleanup_ok = cleanup.get("status") == "PASS" and not cleanup.get("hash_failures")
    errors = legacy + journals + legacy_catalog + outbox_errors
    return {
        "status": "PASS" if cleanup_ok and not errors else "FAIL",
        "cleanup_evidence_status": cleanup.get("status"),
        "cleanup_record_count": cleanup.get("record_count", 0),
        "in_vault_undisposed": errors,
        "outbox_path": str(outbox),
        "outbox_records": outbox_records,
    }


def _trust_contract(vault: Path) -> dict[str, Any]:
    database = resolve_catalog_path(vault, require_exists=True)
    rows = _sqlite_rows(database, "note_catalog")
    tiers: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    production_eligible_count = 0
    for row in rows:
        path = vault / str(row.get("relative_path") or "")
        metadata, _, issues = parse_note(path.read_text(encoding="utf-8")) if path.is_file() else ({}, "", ["missing"])
        tier = str(metadata.get("trust_tier") or "")
        tiers[tier] += 1
        eligible = metadata.get("production_eligible") is True or str(metadata.get("production_eligible")).lower() in {"true", "1", "yes"}
        if eligible:
            production_eligible_count += 1
        if tier not in {"T1", "T2", "T3", "TX"} or issues:
            errors.append({"path": str(row.get("relative_path")), "reason": "invalid_contract", "tier": tier, "issues": issues})
        if tier == "TX" and not str(metadata.get("source_unavailable_reason") or "").strip():
            errors.append({"path": str(row.get("relative_path")), "reason": "TX_without_source_unavailable_reason"})
        if eligible and not (
            tier in {"T1", "T2"}
            and str(metadata.get("source_verification_status")) == "verified"
            and str(metadata.get("content_verification_status")) == "verified"
        ):
            errors.append({"path": str(row.get("relative_path")), "reason": "unsupported_production_eligibility"})
    return {
        "status": "PASS" if len(rows) == 4229 and not errors and production_eligible_count == 0 else "FAIL",
        "row_count": len(rows),
        "trust_tiers": dict(tiers),
        "production_eligible_count": production_eligible_count,
        "errors": errors[:100],
        "error_count": len(errors),
    }


def _benchmark_contract(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "retrieval-benchmark-r5-semantic-v2.json"
    payload = _read_json(path, {}) or {}
    thresholds = payload.get("thresholds") or {}
    metrics = {
        "precision_at_5": float(payload.get("precision_at_5", 0)),
        "mrr_at_5": float(payload.get("mrr_at_5", 0)),
        "exact_ticker_top1": float(payload.get("exact_ticker_top1", 0)),
        "forbidden_retired_results": int(payload.get("forbidden_retired_results", -1)),
        "citation_join_missing": int(payload.get("citation_join_missing", -1)),
        "language_gap": float(payload.get("language_gap", 999)),
        "warm_p95_ms": float(payload.get("warm_p95_ms", 999999)),
    }
    met = (
        payload.get("status") == "PASS"
        and metrics["precision_at_5"] >= float(thresholds.get("precision_at_5", 0.8))
        and metrics["mrr_at_5"] >= float(thresholds.get("mrr_at_5", 0.8))
        and metrics["exact_ticker_top1"] >= float(thresholds.get("exact_ticker_top1", 1.0))
        and metrics["forbidden_retired_results"] == 0
        and metrics["citation_join_missing"] == 0
        and metrics["language_gap"] <= float(thresholds.get("language_gap", 0.1))
        and metrics["warm_p95_ms"] <= float(thresholds.get("warm_p95_ms", 1000.0))
    )
    return {"status": "PASS" if met else "FAIL", "payload": payload, "metrics": metrics, "labels": payload.get("dataset_label_status")}


def _leakage_contract(run_dir: Path, vault: Path) -> dict[str, Any]:
    benchmark = _read_json(run_dir / "retrieval-benchmark-r5-semantic-v2.json", {}) or {}
    catalog = SqliteNoteCatalogAdapter(vault_root=vault, read_only=True)
    excluded_in_active = [
        entry.relative_path
        for entry in catalog.iter_notes(page_size=500)
        if not is_searchable_note(vault / entry.relative_path, vault_root=vault)
    ]
    errors = []
    if int(benchmark.get("forbidden_retired_results", -1)) != 0:
        errors.append("forbidden_retired_results")
    if excluded_in_active:
        errors.append("excluded_notes_in_active_vector_corpus")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "forbidden_retired_results": benchmark.get("forbidden_retired_results"),
        "excluded_in_active": excluded_in_active[:100],
    }


def _citation_contract(vault: Path, run_dir: Path) -> dict[str, Any]:
    answer = _read_json(run_dir / "answer-contract-r5.json", {}) or {}
    catalog = SqliteNoteCatalogAdapter(vault_root=vault, read_only=True)
    research = answer.get("research_mode") or {}
    production = answer.get("production_mode") or {}
    cited = research.get("citations") or []
    labels_ok = all(
        item.get("relative_path")
        and item.get("content_sha256")
        and item.get("trust_tier")
        and "source_verification_status" in item
        and "content_verification_status" in item
        for item in cited
    )
    join_ok = bool(cited) and not (research.get("missing_citations") or [])
    contract_ok = answer.get("status") == "PASS" and research.get("status") == "PASS" and production.get("status") == "BLOCKED"
    # Re-join the sample citation through the active catalog so this gate is
    # independent of the smoke-test's serialized output.
    current_ok = all(catalog.get_by_path(str(item.get("relative_path"))) is not None for item in cited)
    return {
        "status": "PASS" if contract_ok and join_ok and labels_ok and current_ok else "FAIL",
        "answer_contract_status": answer.get("status"),
        "research_status": research.get("status"),
        "production_status": production.get("status"),
        "citation_count": len(cited),
        "join_ok": join_ok and current_ok,
        "labels_ok": labels_ok,
    }


def _final_snapshot_contract(vault: Path, run_dir: Path, rehearsal_report: Path | None) -> dict[str, Any]:
    backup_dir = run_dir / "final-snapshots"
    snapshot, snapshot_hash = create_vault_snapshot(vault_root=vault, backup_dir=backup_dir)
    restore_dir = run_dir / "final-snapshot-restore"
    if restore_dir.exists():
        restore_dir = run_dir / f"final-snapshot-restore-{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
    restored = restore_vault_snapshot(snapshot, restore_dir, verify_checksum=True)
    live_hashes = {
        path.relative_to(vault).as_posix(): _sha256(path)
        for path in vault.rglob("*")
        if path.is_file()
        and ".system" not in path.relative_to(vault).parts
        or path.is_file() and ".system" in path.relative_to(vault).parts and "locks" not in path.relative_to(vault).parts
    }
    restored_hashes = {
        path.relative_to(restore_dir).as_posix(): _sha256(path)
        for path in restore_dir.rglob("*")
        if path.is_file()
    }
    # Snapshot excludes only lock/transient members; compare the restored set
    # directly, which also proves every archive member against its live source.
    mismatches = sorted(
        rel for rel in set(restored_hashes) if live_hashes.get(rel) != restored_hashes[rel]
    )
    rehearsal = _read_json(rehearsal_report, {}) if rehearsal_report else {}
    rehearsal_records = (rehearsal or {}).get("records") or []
    rollback_ok = any(item.get("case_id") == "F12.ROLLBACK" and item.get("status") == "PASS" for item in rehearsal_records)
    reapply_ok = any(item.get("case_id") == "F12.REAPPLY" and item.get("status") == "PASS" for item in rehearsal_records)
    rehearsal_ok = bool(rehearsal and rehearsal.get("overall_status") == "PASS" and rollback_ok and reapply_ok)
    result = {
        "status": "PASS" if not mismatches and rehearsal_ok else "FAIL",
        "snapshot": str(snapshot),
        "snapshot_sha256": snapshot_hash,
        "restore_dir": str(restore_dir),
        "restored_files": restored,
        "hash_mismatches": mismatches[:50],
        "rehearsal_report": str(rehearsal_report) if rehearsal_report else None,
        "rehearsal_overall_status": (rehearsal or {}).get("overall_status"),
        "rollback_ok": rollback_ok,
        "reapply_ok": reapply_ok,
    }
    _write_json(run_dir / "final-snapshot-restore-proof.json", result)
    return result


def run(vault: Path, run_dir: Path, *, r4_report: Path, rehearsal_report: Path | None) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    run_id = "r5_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = run_dir / "acceptance" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    code_fingerprint = _sha256(Path(__file__))
    input_fingerprint = _tree_digest(vault)["sha256"]
    records: list[dict[str, Any]] = []

    prior = _read_json(r4_report, {}) or {}
    for item in prior.get("records", []):
        copied = dict(item)
        copied["run_id"] = run_id
        records.append(copied)

    def add(case_id: str, phase: str, actual: Any, expected: Any, payload: Any, *, status: str | None = None, reason: str | None = None) -> None:
        evidence = _write_json(evidence_dir / f"{case_id}.json", {
            "run_id": run_id,
            "case_id": case_id,
            "code_fingerprint": code_fingerprint,
            "input_fingerprint": input_fingerprint,
            "actual": actual,
            "expected": expected,
            "payload": payload,
        })
        records.append(record_assertion(
            case_id=case_id,
            gate_ids=(case_id,),
            run_id=run_id,
            command="python scripts/run_vault_v2_acceptance_r5.py",
            phase=phase,
            actual=actual,
            expected=expected,
            evidence_paths=(evidence,),
            status=status,
            reason=reason,
            code_fingerprint=code_fingerprint,
            input_fingerprint=input_fingerprint,
        ).to_dict())

    nav = _navigation_contract(vault)
    add("A21", "F03", len(nav["errors"]), 0, nav)

    try:
        nav_idempotency = _navigation_idempotency(vault, run_dir)
        add("A22", "F03", len(nav_idempotency["changed_files_second_run"]), 0, nav_idempotency)
    except Exception as exc:
        add("A22", "F03", str(exc), 0, {"error": str(exc)}, status="FAIL", reason=str(exc))

    catalog = _catalog_generation_contract(vault)
    add("A23", "F02", catalog["status"], "PASS", catalog)

    obsidian = _obsidian_contract(vault)
    add("A24", "F05", obsidian["status"], "PASS", obsidian)

    cleanup = _cleanup_contract(vault, run_dir)
    add("A25", "F04", cleanup["status"], "PASS", cleanup)

    trust = _trust_contract(vault)
    add("A26", "F06", trust["status"], "PASS", trust)

    benchmark = _benchmark_contract(run_dir)
    add("A27", "F07", benchmark["status"], "PASS", benchmark)

    leakage = _leakage_contract(run_dir, vault)
    add("A28", "F07", leakage["status"], "PASS", leakage)

    citation = _citation_contract(vault, run_dir)
    add("A29", "F08", citation["status"], "PASS", citation)

    try:
        final_snapshot = _final_snapshot_contract(vault, run_dir, rehearsal_report)
        add("A30", "F11/F13", final_snapshot["status"], "PASS", final_snapshot)
    except Exception as exc:
        add("A30", "F11/F13", str(exc), "PASS", {"error": str(exc)}, status="FAIL", reason=str(exc))

    report = aggregate_acceptance_records(
        records,
        expected_run_id=run_id,
        mandatory_case_ids=[f"A{i:02d}" for i in range(1, 31)],
    )
    report.update({
        "run_id": run_id,
        "vault_root": str(vault),
        "generated_by": "scripts/run_vault_v2_acceptance_r5.py",
        "code_fingerprint": code_fingerprint,
        "input_fingerprint": input_fingerprint,
        "retrieval_readiness": benchmark.get("labels"),
    })
    write_acceptance_report(report, run_dir / "acceptance", run_id=run_id)
    remaining = [
        {"case_id": item.get("case_id"), "status": item.get("status"), "reason": item.get("reason")}
        for item in report.get("records", [])
        if item.get("status") != "PASS"
    ]
    _write_json(run_dir / "remaining-items.json", remaining)
    _write_json(run_dir / "r5-final-run.json", {
        "run_id": run_id,
        "overall_status": report.get("overall_status"),
        "counts": report.get("counts"),
        "remaining_count": len(remaining),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"run_id": run_id, "overall_status": report.get("overall_status"), "counts": report.get("counts"), "remaining": remaining}, ensure_ascii=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--r4-report", type=Path, required=True)
    parser.add_argument("--rehearsal-report", type=Path, default=None)
    args = parser.parse_args()
    report = run(args.vault, args.run_dir, r4_report=args.r4_report, rehearsal_report=args.rehearsal_report)
    return 0 if report.get("overall_status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
