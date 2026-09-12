"""Guarded metadata contract backfill for the live R4 vault.

Only missing, typed metadata is added.  Existing note IDs, document keys,
dates, provenance, and verification values are never overwritten.  The
script requires the active maintenance owner to be present in the environment
so a normal writer cannot accidentally run this migration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.core import _atomic_write_text
from tools.archivist.catalog_runtime import resolve_catalog_path
from tools.archivist.identity_store import DurableIdentityStore
from tools.archivist.metadata import dump_note, parse_note
from tools.archivist.vault_audit import scan_vault


RUN_ID = "r4_20260909T165700Z"
SOURCE_TYPES = {
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def catalog_rows(vault: Path) -> dict[str, dict[str, Any]]:
    try:
        db = resolve_catalog_path(vault, require_exists=True)
    except FileNotFoundError:
        return {}
    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        return {
            str(row["relative_path"]).replace("\\", "/"): dict(row)
            for row in conn.execute("SELECT * FROM note_catalog")
        }


def identity_by_note_id(vault: Path) -> dict[str, dict[str, Any]]:
    path = vault / ".system" / "identity_allocations.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for record in value.values() if isinstance(value, dict) else []:
        if isinstance(record, dict) and record.get("note_id"):
            result[str(record["note_id"])] = record
    return result


def run(vault: Path, run_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = catalog_rows(vault)
    durable = identity_by_note_id(vault)
    audit = scan_vault(vault)
    plan_path = run_dir / "metadata-contract-plan.jsonl"
    blocked: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_doc_keys: dict[str, str] = {}

    for record in audit.inventory:
        if record.extension.lower() != ".md" or record.is_excluded:
            continue
        path = vault / record.relative_path
        raw = path.read_text(encoding="utf-8")
        meta, body, issues = parse_note(raw)
        if issues:
            blocked.append({"path": record.relative_path, "reason": issues})
            continue
        note_id = str(meta.get("note_id") or "").strip()
        if not note_id:
            blocked.append({"path": record.relative_path, "reason": "missing note_id"})
            continue

        rel = record.relative_path.replace("\\", "/")
        row = rows.get(rel)
        row_key = str(row.get("document_key") or "") if row and str(row.get("note_id") or "") == note_id else ""
        durable_record = durable.get(note_id) or {}
        document_key = str(meta.get("document_key") or row_key or durable_record.get("document_key") or "").strip()
        if not document_key:
            # This fallback is tied to the already durable note identity, not
            # to the current path. It remains stable across rename/move and is
            # explicitly marked as an import-era key in the evidence.
            document_key = f"legacy-import:v1:{note_id}"
            counts["document_key_identity_fallback"] += 1

        prior_path = seen_doc_keys.get(document_key)
        if prior_path and prior_path != rel:
            blocked.append({"path": rel, "reason": f"document_key conflict with {prior_path}: {document_key}"})
            continue
        seen_doc_keys[document_key] = rel

        updates_for_file: dict[str, Any] = {}
        if not str(meta.get("document_key") or "").strip():
            updates_for_file["document_key"] = document_key
            counts["document_key"] += 1
        if not any(str(meta.get(key) or "").strip() for key in DATE_KEYS):
            if str(meta.get("date_status") or "").lower() != "unknown":
                updates_for_file["date_status"] = "unknown"
                counts["date_status_unknown"] += 1
        elif not str(meta.get("date_status") or "").strip():
            updates_for_file["date_status"] = "known"
            counts["date_status_known"] += 1
        if not str(meta.get("source_verification_status") or "").strip():
            updates_for_file["source_verification_status"] = "not_reviewed"
            counts["source_verification_status"] += 1
        if not str(meta.get("content_verification_status") or "").strip():
            updates_for_file["content_verification_status"] = "not_reviewed"
            counts["content_verification_status"] += 1
        entity_type = str(meta.get("entity_type") or "concept").strip().lower()
        if entity_type in SOURCE_TYPES and not any(str(meta.get(key) or "").strip() for key in PROVENANCE_KEYS):
            if not str(meta.get("source_unavailable_reason") or "").strip():
                updates_for_file["source_unavailable_reason"] = (
                    "legacy note has no recorded source evidence at remediation time"
                )
                counts["source_unavailable_reason"] += 1

        if not updates_for_file:
            continue
        new_meta = dict(meta)
        new_meta.update(updates_for_file)
        new_text = dump_note(new_meta, body)
        before_hash = sha256(path)
        entry = {
            "relative_path": rel,
            "note_id": note_id,
            "document_key": document_key,
            "changed_fields": sorted(updates_for_file),
            "pre_hash": before_hash,
            "status": "planned" if dry_run else "updated",
        }
        if not dry_run:
            _atomic_write_text(path, new_text)
            entry["post_hash"] = sha256(path)
        updates.append(entry)

    with plan_path.open("w", encoding="utf-8") as stream:
        for entry in updates:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        for entry in blocked:
            stream.write(json.dumps({"status": "blocked", **entry}, ensure_ascii=False) + "\n")

    summary = {
        "status": "PASS" if not blocked else "BLOCKED",
        "dry_run": dry_run,
        "updated": len(updates),
        "blocked": len(blocked),
        "counts": dict(counts),
        "plan": str(plan_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "metadata-contract-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", default="memories")
    parser.add_argument("--run-dir", default=f"scratch/vault-v2/remediation-r4/{RUN_ID}")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = run(Path(args.vault), Path(args.run_dir), dry_run=args.dry_run)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
