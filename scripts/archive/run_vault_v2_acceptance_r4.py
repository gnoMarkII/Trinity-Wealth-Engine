"""Run assertion-derived A01-A20 evidence for the R4 vault remediation.

The runner is intentionally read-only with respect to the vault.  It writes
only evidence and reports below the supplied remediation run directory.  A
separate observation process may write ``observation-60m.json`` there before
the final invocation so A20 is never inferred from an incomplete wait.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
from tools.archivist.catalog_runtime import resolve_catalog_path
from tools.archivist.metadata import normalize_legacy_metadata, parse_note
from tools.archivist.vault_acceptance import (
    AcceptanceRecord,
    aggregate_acceptance_records,
    record_assertion,
    write_acceptance_report,
)
from tools.archivist.vault_audit import scan_vault
from tools.archivist.vault_maintenance import detect_orphaned_sidecars
from tools.archivist.vault_policy import is_searchable_note
from tools.archivist.search import get_query_vector_runtime
from tools.archivist.vector_generation import (
    corpus_fingerprint,
    load_active_manifest,
    vector_runtime_path,
)


RUN_ID = "r4_20260909T165700Z"
CANONICAL_TYPES = {
    "company_news",
    "youtube_summary",
    "equity_analysis",
    "stock_hub",
    "quant_snapshot",
    "earnings_call",
    "macro_strategy",
    "macro_snapshot",
    "briefing_book",
    "book_note",
    "concept",
    "index",
    "operational_log",
    "portfolio_snapshot",
    "holding",
    "goal",
    "watchlist_item",
    "dashboard",
}
SOURCE_BEARING_TYPES = {
    "company_news",
    "youtube_summary",
    "equity_analysis",
    "stock_hub",
    "quant_snapshot",
    "earnings_call",
    "macro_strategy",
    "macro_snapshot",
    "briefing_book",
    "book_note",
}
DATE_KEYS = ("date", "published_date", "analysis_date", "as_of", "authored_date", "date_read")
PROVENANCE_KEYS = (
    "source_url",
    "source_key",
    "source",
    "source_id",
    "source_file",
    "source_path",
    "url",
    "provider",
    "publisher",
)
RETIRED_TICKERS = {
    "AAPL",
    "ADVANC",
    "AMZN",
    "AOT",
    "CPALL",
    "GOOGL",
    "KBANK",
    "MSFT",
    "NVDA",
    "PTT",
    "SPY",
    "TSLA",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if "query_cache" in path.relative_to(root).parts:
            continue
        rel = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        size = path.stat().st_size
        encoded = f"{rel}\0{size}\0{file_hash}\n".encode("utf-8")
        digest.update(encoded)
        count += 1
        total_bytes += size
    return {"sha256": digest.hexdigest(), "file_count": count, "total_bytes": total_bytes}


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def _rows(vault: Path) -> list[dict[str, Any]]:
    try:
        db = resolve_catalog_path(vault, require_exists=True)
    except FileNotFoundError:
        return []
    uri = f"file:{db.as_posix()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM note_catalog ORDER BY relative_path")]


def _active_notes(audit: Any, vault: Path) -> list[tuple[str, dict[str, Any], str]]:
    notes: list[tuple[str, dict[str, Any], str]] = []
    for record in audit.inventory:
        if record.extension.lower() != ".md" or record.is_excluded:
            continue
        path = vault / record.relative_path
        try:
            metadata, body, issues = parse_note(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            notes.append((record.relative_path, {"_read_error": str(exc)}, ""))
            continue
        if issues:
            notes.append((record.relative_path, {"_parse_issues": issues}, body))
        else:
            normalized, _ = normalize_legacy_metadata(metadata)
            notes.append((record.relative_path, normalized, body))
    return notes


def _retirement_paths(run_dir: Path) -> set[str]:
    plan = run_dir / "retirement-plan.jsonl"
    paths: set[str] = set()
    if plan.is_file():
        for line in plan.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if value.get("relative_path"):
                    paths.add(str(value["relative_path"]).replace("\\", "/"))
    return paths


def _retired_tombstone_paths(vault: Path) -> set[str]:
    path = vault / ".system" / "retired_notes.jsonl"
    result: set[str] = set()
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if value.get("relative_path"):
                result.add(str(value["relative_path"]).replace("\\", "/"))
    return result


def _write_delete_manifest(run_dir: Path, vault: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Combine already-recorded retire/quarantine evidence into one manifest."""
    records: list[dict[str, Any]] = []
    retirement = run_dir / "retirement-plan.jsonl"
    if retirement.is_file():
        for line in retirement.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if value.get("disposition") or value.get("reason"):
                    records.append({"kind": "retirement", **value})

    artifact_summary = run_dir / "f02-retirement-summary.json"
    if artifact_summary.is_file():
        value = json.loads(artifact_summary.read_text(encoding="utf-8"))
        for artifact in value.get("artifact_records", []):
            records.append({"kind": "artifact", **artifact})

    stub = run_dir / "ftnt-stub-retirement.json"
    if stub.is_file():
        records.append({"kind": "stub", **json.loads(stub.read_text(encoding="utf-8"))})

    runtime = run_dir / "legacy-runtime-delete-manifest.json"
    if runtime.is_file():
        value = json.loads(runtime.read_text(encoding="utf-8"))
        for target in value.get("targets", []):
            records.append({"kind": "legacy-runtime", **target})

    path = run_dir / "delete-manifest.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path, records


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(vault: Path, run_dir: Path, observation_path: Path | None = None) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    evidence_dir = run_dir / "acceptance" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    audit = scan_vault(vault)
    notes = _active_notes(audit, vault)
    current_paths = {path for path, _, _ in notes}
    retirement_paths = _retirement_paths(run_dir)
    tombstone_paths = _retired_tombstone_paths(vault)

    def evidence(case_id: str, payload: Any) -> Path:
        return _write_json(evidence_dir / f"{case_id}.json", payload)

    records: list[AcceptanceRecord] = []

    def add(
        case_id: str,
        phase: str,
        actual: Any,
        expected: Any,
        payload: Any,
        *,
        status: str | None = None,
        reason: str | None = None,
    ) -> None:
        path = evidence(case_id, payload)
        records.append(
            record_assertion(
                case_id=case_id,
                gate_ids=(case_id,),
                run_id=RUN_ID,
                command="python scripts/run_vault_v2_acceptance_r4.py",
                phase=phase,
                actual=actual,
                expected=expected,
                evidence_paths=(path,),
                status=status,
                reason=reason,
            )
        )

    parse_errors = int(audit.stats.get("parse_errors", 0)) + sum(
        1 for _, meta, _ in notes if meta.get("_parse_issues") or meta.get("_read_error")
    )
    add("A01", "F15", parse_errors, 0, {"parse_errors": parse_errors, "audit": audit.stats})

    required = {"schema_version", "note_id", "entity_type", "title"}
    missing_metadata = [
        {"path": path, "missing": sorted(required - set(meta))}
        for path, meta, _ in notes
        if not (meta.get("_parse_issues") or meta.get("_read_error")) and required - set(meta)
    ]
    add("A02", "F15", len(missing_metadata), 0, {"missing": missing_metadata[:100], "count": len(missing_metadata)})

    ids = [(str(meta.get("note_id") or ""), path) for path, meta, _ in notes]
    docs = [(str(meta.get("document_key") or ""), path) for path, meta, _ in notes]
    id_dupes = [key for key, count in Counter(key for key, _ in ids if key).items() if count > 1]
    doc_dupes = [key for key, count in Counter(key for key, _ in docs if key).items() if count > 1]
    identity_errors = {
        "missing_note_id": [path for key, path in ids if not key],
        "duplicate_note_id": id_dupes,
        "missing_document_key": [path for key, path in docs if not key],
        "duplicate_document_key": doc_dupes,
    }
    identity_count = sum(len(value) for value in identity_errors.values())
    add("A03", "F15", identity_count, 0, identity_errors)

    unknown_types = sorted(
        {str(meta.get("entity_type") or "") for _, meta, _ in notes if str(meta.get("entity_type") or "") not in CANONICAL_TYPES}
    )
    add("A04", "F05", len(unknown_types), 0, {"unknown_entity_types": unknown_types, "counts": Counter(str(meta.get("entity_type") or "") for _, meta, _ in notes)})

    invalid_dates: list[dict[str, str]] = []
    for path, meta, _ in notes:
        values = [str(meta.get(key) or "")[:10] for key in DATE_KEYS if str(meta.get(key) or "").strip()]
        date_status = str(meta.get("date_status") or "").lower()
        if not values and date_status != "unknown":
            invalid_dates.append({"path": path, "reason": "missing date and date_status is not unknown"})
        for value in values:
            try:
                datetime.fromisoformat(value)
            except ValueError:
                invalid_dates.append({"path": path, "reason": f"invalid date {value!r}"})
    add("A05", "F05", len(invalid_dates), 0, {"invalid_or_missing": invalid_dates[:100], "count": len(invalid_dates)})

    provenance_missing = [
        {"path": path, "entity_type": meta.get("entity_type")}
        for path, meta, _ in notes
        if str(meta.get("entity_type") or "") in SOURCE_BEARING_TYPES
        and not any(str(meta.get(key) or "").strip() for key in PROVENANCE_KEYS)
        and not str(meta.get("source_unavailable_reason") or "").strip()
    ]
    add("A06", "F07", len(provenance_missing), 0, {"missing": provenance_missing[:100], "count": len(provenance_missing)})

    valid_verification = {"verified", "unverified", "not_reviewed", "unavailable", "pending"}
    verification_errors: list[dict[str, Any]] = []
    for path, meta, _ in notes:
        for field in ("source_verification_status", "content_verification_status"):
            value = str(meta.get(field) or "").strip().lower()
            if value not in valid_verification:
                verification_errors.append({"path": path, "field": field, "value": value})
        if str(meta.get("source_verification_status") or "").lower() == "verified" and not (
            any(str(meta.get(key) or "").strip() for key in PROVENANCE_KEYS)
            or str(meta.get("source_unavailable_reason") or "").strip()
        ):
            verification_errors.append({"path": path, "field": "source_verification_status", "value": "verified_without_source"})
    add("A07", "F07", len(verification_errors), 0, {"errors": verification_errors[:100], "count": len(verification_errors)})

    link_errors = int(audit.stats.get("broken_links", 0)) + int(audit.stats.get("ambiguous_links", 0))
    add("A08", "F04", link_errors, 0, {"broken_links": audit.stats.get("broken_links", 0), "ambiguous_links": audit.stats.get("ambiguous_links", 0)})

    duplicate_errors = int(audit.stats.get("duplicate_filenames", 0))
    add("A09", "F06", duplicate_errors, 0, {"duplicate_filenames": duplicate_errors})

    rows = _rows(vault)
    active_catalog_paths = {
        str(row.get("relative_path") or "").replace("\\", "/")
        for row in rows
        if str(row.get("record_state") or "active") == "active"
        and str(row.get("storage_scope") or "active") == "active"
    }
    searchable_paths = {
        path
        for path, _, _ in notes
        if is_searchable_note(vault / path, vault_root=vault)
    }
    catalog_diff = {
        "missing_active_rows": sorted(searchable_paths - active_catalog_paths),
        "extra_active_rows": sorted(active_catalog_paths - searchable_paths),
        "retirement_plan_count": len(retirement_paths),
        # The retirement plan covers the stock retirement batch.  Other
        # controlled retirements (for example a generated concept stub) are
        # validated by their own evidence and must not make F03 fail merely
        # because they share the tombstone log.
        "retired_tombstone_count": len(tombstone_paths & retirement_paths),
        "extra_controlled_tombstones": sorted(tombstone_paths - retirement_paths),
    }
    catalog_error_count = len(catalog_diff["missing_active_rows"]) + len(catalog_diff["extra_active_rows"])
    if retirement_paths != (tombstone_paths & retirement_paths):
        catalog_error_count += len(retirement_paths - tombstone_paths)
    add("A10", "F03", catalog_error_count, 0, catalog_diff)

    registry_path = vault / ".system" / "artifacts" / "registry.json"
    heads_dir = vault / ".system" / "artifacts" / "heads"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {"revisions": {}}
    head_files = sorted(heads_dir.glob("*.json")) if heads_dir.is_dir() else []
    artifact_errors: list[str] = []
    for head_path in head_files:
        try:
            head = json.loads(head_path.read_text(encoding="utf-8"))
            note_id = str(head.get("note_id") or head_path.stem)
            rev = str(head.get("revision_id") or "")
            manifest_path = vault / "40_Archive" / "Revisions" / note_id / rev / "manifest.json"
            primary = vault / "40_Archive" / "Revisions" / note_id / rev / str(json.loads(manifest_path.read_text(encoding="utf-8")).get("primary_file"))
            if not manifest_path.is_file() or not primary.is_file():
                artifact_errors.append(str(head_path.relative_to(vault)))
        except (OSError, ValueError, TypeError):
            artifact_errors.append(str(head_path.relative_to(vault)))
    artifact_errors.extend(str(key) for key in registry.get("revisions", {}) if not key)
    f02 = run_dir / "f02-retirement-summary.json"
    retired_artifacts = 0
    if f02.is_file():
        retired_artifacts = len(json.loads(f02.read_text(encoding="utf-8")).get("artifact_records", []))
    add("A11", "F03", len(artifact_errors), 0, {"errors": artifact_errors, "active_head_count": len(head_files), "registry_revision_count": len(registry.get("revisions", {})), "retired_artifact_records": retired_artifacts})

    sidecar_rows = [row for row in rows if row.get("relative_path") and row.get("sidecar_id")]
    sidecar_missing = [row["relative_path"] for row in sidecar_rows if not (vault / str(row["relative_path"])).is_file()]
    orphans = detect_orphaned_sidecars(vault_root=vault)
    add("A12", "F08", len(sidecar_missing) + len(orphans), 0, {"cataloged": len(sidecar_rows), "missing_files": sidecar_missing, "orphans": orphans})

    excluded_scope_errors = [path for path in active_catalog_paths if not is_searchable_note(vault / path, vault_root=vault)]
    add("A13", "F08", len(excluded_scope_errors), 0, {"excluded_paths_in_active_catalog": excluded_scope_errors[:100], "searchable_path_count": len(searchable_paths)})

    vector_errors: list[str] = []
    manifest: dict[str, Any] | None = None
    vector_note_count = 0
    vector_chunk_count = 0
    try:
        manifest = load_active_manifest(vault)
        cat = SqliteNoteCatalogAdapter(vault_root=vault, read_only=True)
        corpus_hash, vector_note_count, _ = corpus_fingerprint(cat.iter_notes(page_size=500))
        if corpus_hash != manifest.get("corpus_fingerprint"):
            vector_errors.append("corpus_fingerprint_mismatch")
        if vector_note_count != int(manifest.get("eligible_note_count", -1)):
            vector_errors.append("eligible_note_count_mismatch")
        runtime = get_query_vector_runtime(vault, vector_runtime_path(vault), manifest)
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(runtime))
            collection = client.get_collection(name=str(manifest["collection_name"]))
            vector_chunk_count = int(collection.count())
            metadata = collection.get(include=["metadatas"], limit=max(vector_chunk_count, 1)).get("metadatas") or []
            for item in metadata:
                if not item or not item.get("content_sha256"):
                    vector_errors.append("chunk_missing_content_sha256")
                    break
                rel = str(item.get("relative_path") or item.get("source") or "").replace("\\", "/")
                if rel and any(rel.startswith(f"30_Knowledge_Base/Stocks/{ticker}/") for ticker in RETIRED_TICKERS):
                    vector_errors.append(f"retired_vector_reference:{rel}")
                    break
        except Exception as exc:
            vector_errors.append(f"vector_store_unavailable:{exc}")
    except Exception as exc:
        vector_errors.append(str(exc))
    add("A14", "F10", len(vector_errors), 0, {"errors": vector_errors[:20], "manifest": manifest, "note_count": vector_note_count, "chunk_count": vector_chunk_count})

    os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
    os.environ["VAULT_EMBEDDINGS_BACKEND"] = "offline"
    query_results: dict[str, str] = {}
    query_errors: list[str] = []
    try:
        from tools.archivist import search as search_module

        search_module.VAULT_PATH = vault
        search_module.CHROMA_PATH = vector_runtime_path(vault)
        search_module._vs_cache.clear()
        for query in ("อัตราดอกเบี้ย", "interest rate", "FTNT"):
            result = search_module.search_all_memories.func(query)
            query_results[query] = str(result)
            if "เกิดข้อผิดพลาด" in str(result) or "ไม่สามารถค้นหาได้" in str(result):
                query_errors.append(query)
        for query in ("AMZN", "NVDA"):
            result = search_module.search_all_memories.func(query)
            query_results[query] = str(result)
            for ticker in ("AMZN", "NVDA"):
                if f"30_Knowledge_Base/Stocks/{ticker}/" in str(result):
                    query_errors.append(f"retired_stock_projection:{query}:{ticker}")
    except Exception as exc:
        query_errors.append(str(exc))
    add("A15", "F10", len(query_errors), 0, {"errors": query_errors, "queries": query_results})

    before_vault = _tree_fingerprint(vault)
    before_runtime = _tree_fingerprint(vector_runtime_path(vault))
    try:
        if "search_module" in locals():
            search_module.search_all_memories.func("read only smoke query")
    except Exception:
        pass
    after_vault = _tree_fingerprint(vault)
    after_runtime = _tree_fingerprint(vector_runtime_path(vault))
    readonly_errors = []
    if before_vault != after_vault:
        readonly_errors.append("vault_changed")
    if before_runtime != after_runtime:
        readonly_errors.append("vector_runtime_changed")
    add("A16", "F10", len(readonly_errors), 0, {"errors": readonly_errors, "before_vault": before_vault, "after_vault": after_vault, "before_runtime": before_runtime, "after_runtime": after_runtime})

    nav_paths = [
        "00_Index/Home.md",
        "00_Index/Stocks_Hub.md",
        "00_Index/Macro_Hub.md",
        "00_Index/News_Hub.md",
        "00_Index/Audio_Hub.md",
    ]
    nav_missing = [path for path in nav_paths if not (vault / path).is_file()]
    nav_link_missing: list[str] = []
    for rel in nav_paths:
        path = vault / rel
        if not path.is_file():
            continue
        _, body, _ = parse_note(path.read_text(encoding="utf-8"))
        for token in body.split("[[")[1:]:
            target = token.split("]]")[0].split("|", 1)[0].split("#", 1)[0].strip()
            if not target:
                continue
            candidate = vault / (target if target.lower().endswith(".md") else f"{target}.md")
            if not candidate.is_file():
                nav_link_missing.append(f"{rel}->{target}")
    add("A17", "F11", len(nav_missing) + len(nav_link_missing) + link_errors, 0, {"missing_navigation": nav_missing, "missing_navigation_links": nav_link_missing[:100], "audit_link_errors": link_errors})

    runtime_root = vector_runtime_path(vault)
    legacy_runtime = [vault / ".chroma_index", vault / ".chroma_mtime", vault / ".system" / "vector_index_state.json"]
    runtime_errors = [str(path.relative_to(vault)) for path in legacy_runtime if path.exists()]
    if runtime_root.is_relative_to(vault):
        runtime_errors.append("active_runtime_inside_vault")
    add("A18", "F09", len(runtime_errors), 0, {"errors": runtime_errors, "runtime_root": str(runtime_root)})

    delete_manifest, delete_records = _write_delete_manifest(run_dir, vault)
    delete_errors = [
        index
        for index, record in enumerate(delete_records)
        if not (record.get("disposition") or record.get("reason") or record.get("quarantine_path") or record.get("target"))
    ]
    # An empty manifest is valid when this run has no destructive operations;
    # the gate is that every recorded operation has an explicit disposition.
    add("A19", "F14", len(delete_errors), 0, {"manifest": str(delete_manifest), "record_count": len(delete_records), "invalid_records": delete_errors})

    if observation_path and observation_path.is_file():
        observation = json.loads(observation_path.read_text(encoding="utf-8"))
        observation_status = str(observation.get("status") or "NOT_RUN").upper()
        add("A20", "F14", observation_status, "PASS", observation, status=observation_status, reason=observation.get("reason"))
    else:
        add("A20", "F14", "NOT_RUN", "PASS", {"observation": None}, status="NOT_RUN", reason="60-minute observation evidence is not present")

    report = aggregate_acceptance_records(
        records,
        expected_run_id=RUN_ID,
        mandatory_case_ids=[f"A{i:02d}" for i in range(1, 21)],
    )
    report["run_id"] = RUN_ID
    report["vault_root"] = str(vault)
    report["git_head"] = _git_head()
    report["vault_fingerprint"] = _tree_fingerprint(vault)
    report["generated_by"] = "scripts/run_vault_v2_acceptance_r4.py"
    write_acceptance_report(report, run_dir / "acceptance", run_id=RUN_ID)
    remaining = [
        {
            "case_id": item.get("case_id"),
            "status": item.get("status"),
            "reason": item.get("reason"),
        }
        for item in report.get("records", [])
        if item.get("status") != "PASS"
    ]
    _write_json(run_dir / "remaining-items.json", remaining)
    _write_json(
        run_dir / "run.json",
        {
            "run_id": RUN_ID,
            "phase": "F15_acceptance",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "vault_root": str(vault),
            "git_head": _git_head(),
            "overall_status": report.get("overall_status"),
            "remaining_count": len(remaining),
        },
    )
    print(json.dumps({"run_id": RUN_ID, "overall_status": report.get("overall_status"), "remaining": remaining}, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", default="memories")
    parser.add_argument("--run-dir", default=f"scratch/vault-v2/remediation-r4/{RUN_ID}")
    parser.add_argument("--observation", default="")
    args = parser.parse_args()
    run(Path(args.vault), Path(args.run_dir), Path(args.observation).resolve() if args.observation else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
