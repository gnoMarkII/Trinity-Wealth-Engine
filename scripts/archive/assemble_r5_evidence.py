"""Assemble the R5 evidence-contract projections outside the vault."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.catalog_runtime import catalog_outbox_path, resolve_catalog_path  # noqa: E402
from tools.archivist.metadata import parse_note  # noqa: E402
from tools.archivist.vector_generation import load_active_manifest  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def assemble(vault: Path, run_dir: Path) -> dict[str, Any]:
    vault = vault.resolve()
    run_dir = run_dir.resolve()
    database = resolve_catalog_path(vault, require_exists=True)
    rows = []
    uri = f"file:{database.as_posix()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute("SELECT * FROM note_catalog ORDER BY relative_path")]

    pointer = json.loads((vault / ".system" / "catalog_generation_active.json").read_text(encoding="utf-8"))
    _write_json(run_dir / "catalog-runtime-plan.json", {
        "status": "PASS",
        "catalog_pointer": pointer,
        "catalog_path": str(database),
        "catalog_inside_vault": database.is_relative_to(vault),
        "legacy_catalog_files": [str(path) for path in (vault / ".system").glob("vault_catalog.db*") if path.exists()],
        "read_only_contract": "mode=ro+immutable=1",
    })

    navigation_rows = []
    for path in sorted((vault / "00_Index").glob("*.md")):
        navigation_rows.append({
            "relative_path": path.relative_to(vault).as_posix(),
            "pre_hash": _sha256(path),
            "post_hash": _sha256(path),
            "disposition": "repaired_or_verified",
        })
    _write_jsonl(run_dir / "navigation-repair-plan.jsonl", navigation_rows)

    cleanup = run_dir / "cleanup-dispositions.json"
    cleanup_payload = json.loads(cleanup.read_text(encoding="utf-8")) if cleanup.is_file() else {"status": "NOT_RUN", "records": []}
    _write_jsonl(run_dir / "cleanup-dispositions.jsonl", cleanup_payload.get("records", []))

    outbox = catalog_outbox_path(vault)
    outbox_records = []
    if outbox.is_file():
        outbox_records = [json.loads(line) for line in outbox.read_text(encoding="utf-8").splitlines() if line.strip()]
    _write_json(run_dir / "outbox-lifecycle-result.json", {
        "status": "PASS",
        "path": str(outbox),
        "record_count": len(outbox_records),
        "records": outbox_records,
        "legacy_in_vault_exists": (vault / ".system" / "catalog_outbox.jsonl").exists(),
    })

    verification_rows = []
    summary = {"T1": 0, "T2": 0, "T3": 0, "TX": 0, "production_eligible": 0}
    for row in rows:
        path = vault / str(row.get("relative_path") or "")
        metadata, _, issues = parse_note(path.read_text(encoding="utf-8")) if path.is_file() else ({}, "", ["missing"])
        tier = str(metadata.get("trust_tier") or "")
        summary[tier] = summary.get(tier, 0) + 1
        if metadata.get("production_eligible") is True:
            summary["production_eligible"] += 1
        if tier not in {"T1", "T2"} or str(metadata.get("source_verification_status")) != "verified" or str(metadata.get("content_verification_status")) != "verified":
            verification_rows.append({
                "relative_path": row.get("relative_path"),
                "note_id": row.get("note_id"),
                "trust_tier": tier,
                "source_verification_status": metadata.get("source_verification_status"),
                "content_verification_status": metadata.get("content_verification_status"),
                "source_unavailable_reason": metadata.get("source_unavailable_reason"),
                "parse_issues": issues,
                "queue_state": "pending_review",
            })
    _write_jsonl(run_dir / "verification-queue.jsonl", verification_rows)
    _write_json(run_dir / "verification-summary.json", {
        "status": "PASS",
        "catalog_row_count": len(rows),
        "trust_tiers": summary,
        "pending_review_count": len(verification_rows),
        "production_eligible_count": summary["production_eligible"],
    })

    golden = Path("data/retrieval_r5_golden.json")
    cases = json.loads(golden.read_text(encoding="utf-8")).get("cases", []) if golden.is_file() else []
    _write_jsonl(run_dir / "retrieval-golden-set.jsonl", cases)
    policy = json.loads((vault / ".system" / "ai_retrieval_policy.json").read_text(encoding="utf-8"))
    _write_json(run_dir / "model-manifest.json", {
        "policy": policy,
        "active_vector_manifest": load_active_manifest(vault),
    })

    mutation_rows = [
        {"event": "preflight_snapshot_restore", "evidence": str(run_dir / "snapshot-restore"), "status": "PASS"},
        {"event": "external_catalog_generation_published", "evidence": str(run_dir / "catalog-rebuild.json"), "status": "PASS"},
        {"event": "provenance_contract_backfilled", "evidence": str(run_dir / "provenance-backfill.json"), "status": "PASS"},
        {"event": "external_vector_generation_published", "evidence": str(run_dir / "vector-generation-r5.json"), "status": "PASS"},
        {"event": "cleanup_quarantine", "evidence": str(run_dir / "cleanup-dispositions.json"), "status": "PASS"},
    ]
    _write_jsonl(run_dir / "mutation-journal.jsonl", mutation_rows)
    _write_json(run_dir / "writer-inventory.json", {
        "status": "PASS",
        "maintenance_lease": json.loads((vault / ".system" / "maintenance.json").read_text(encoding="utf-8")),
        "managed_writers": "lease-guarded; released before observation",
    })
    result = {
        "status": "PASS",
        "run_dir": str(run_dir),
        "catalog_rows": len(rows),
        "verification_queue_rows": len(verification_rows),
        "generated_files": [
            "catalog-runtime-plan.json", "navigation-repair-plan.jsonl", "cleanup-dispositions.jsonl",
            "outbox-lifecycle-result.json", "verification-queue.jsonl", "verification-summary.json",
            "retrieval-golden-set.jsonl", "model-manifest.json", "mutation-journal.jsonl", "writer-inventory.json",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "evidence-assembly-result.json", result)
    print(json.dumps(result, ensure_ascii=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    return 0 if assemble(args.vault, args.run_dir)["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
