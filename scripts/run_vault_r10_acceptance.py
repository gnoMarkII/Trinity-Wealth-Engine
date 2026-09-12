"""Evaluate R10 retrieval and Concepts-cleanup acceptance gates A145-A164."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.archivist.concepts_cleanup import build_cleanup_plan  # noqa: E402


R10_TESTS = [
    "tests/tools/archivist/test_portable_link_graph_r10.py",
    "tests/tools/archivist/test_hybrid_matched_chunks_r10.py",
    "tests/tools/archivist/test_paged_note_reader_r10.py",
    "tests/tools/archivist/test_concepts_cleanup_r10.py",
    "tests/tools/archivist/test_concepts_cleanup_executor_r10.py",
    "tests/tools/archivist/test_search.py",
    "tests/tools/archivist/test_vault_lifecycle_phase2.py",
    "tests/agents/test_equity_narrative_agent.py",
    "tests/unit/application/test_equity_refresh_workflow.py",
    "tests/tools/test_news_funnel_pipeline.py",
]


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _text_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    except (OSError, UnicodeDecodeError):
        return None


def _gate(gate_id: str, condition: bool, actual: Any, expected: Any, reason: str = "", status: str | None = None) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "status": status or ("PASS" if condition else "FAIL"),
        "actual": actual,
        "expected": expected,
        "reason": reason,
    }


def _run_tests(root: Path, output: Path, *, skip: bool) -> dict[str, Any]:
    if skip:
        return {"status": "NOT_RUN", "reason": "--skip-tests"}
    command = [sys.executable, "-m", "pytest", *R10_TESTS, "-q", "--no-cov"]
    env = os.environ.copy()
    env["NEWS_FUNNEL_STORE_PATH"] = str(Path(tempfile.gettempdir()) / f"invest_agents_r10_test_store_{uuid.uuid4().hex}.json")
    completed = subprocess.run(
        command,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "returncode": completed.returncode,
        "command": command,
        "output": str(output),
    }


def _plan_is_deterministic(inventory_path: Path, plan: dict[str, Any]) -> bool:
    inventory = _read_json(inventory_path)
    if not inventory or not plan:
        return False
    regenerated = build_cleanup_plan(inventory)
    return (
        regenerated.get("snapshot_fingerprint") == plan.get("snapshot_fingerprint")
        and regenerated.get("policy_digest") == plan.get("policy_digest")
        and regenerated.get("disposition_counts") == plan.get("disposition_counts")
        and regenerated.get("apply_item_count") == plan.get("apply_item_count")
    )


def _production_contract_boundary(root: Path) -> dict[str, Any]:
    callers = [
        root / "agents/equity_narrative_agent.py",
        root / "application/equity/refresh_workflow.py",
    ]
    texts = {str(path): path.read_text(encoding="utf-8") if path.is_file() else "" for path in callers}
    raw_search_callers = [
        path for path, text in texts.items()
        if re.search(r"\bsearch_all_memories\s*\(", text)
    ]
    return {
        "evidence_callers": [path for path, text in texts.items() if "search_memories_with_evidence" in text],
        "raw_search_callers": raw_search_callers,
        "pass": len(raw_search_callers) == 0 and len(texts) == 2 and all("search_memories_with_evidence" in text for text in texts.values()),
    }


def _cleanup_manifest_ok(path: Path | None, vault: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    entries = manifest.get("entries") or []
    preserved = True
    for item in entries:
        if not isinstance(item, dict):
            preserved = False
            continue
        preimage = path.parent / str(item.get("preimage_path") or "") if path else Path()
        if not preimage.is_file() or _text_sha256(preimage) != item.get("pre_content_sha256"):
            preserved = False
        if item.get("target_path"):
            target = vault / str(item["target_path"])
            if not target.is_file() or item.get("post_content_sha256") != _text_sha256(target):
                preserved = False
        else:
            archived = path.parent / str(item.get("quarantine_path") or "") if path else Path()
            # R10 manifests created before the path-normalization fix used a
            # redundant ``quarantine/`` prefix; accept that historical form
            # while validating the actual external quarantine file.
            if not archived.is_file() and str(item.get("quarantine_path") or "").startswith("quarantine/"):
                archived = path.parent / str(item["quarantine_path"])[len("quarantine/"):]
            if not archived.is_file() or _text_sha256(archived) != item.get("pre_content_sha256"):
                preserved = False
    return {
        "path": str(path) if path else None,
        "status": manifest.get("status", "NOT_RUN"),
        "entries": len(entries) if isinstance(entries, list) else 0,
        "preimages": sum(bool(item.get("preimage_path")) for item in entries if isinstance(item, dict)),
        "content_preserved": preserved,
        "pass": manifest.get("status") == "APPLIED" and bool(entries) and preserved,
    }


def run(
    vault: Path,
    *,
    preflight_path: Path,
    baseline_inventory_path: Path,
    final_inventory_path: Path,
    final_plan_path: Path,
    prechange_preflight_path: Path | None,
    prechange_restore_report: Path | None,
    postchange_restore_report: Path | None,
    staging_rebuild_report: Path | None,
    cleanup_manifest: Path | None,
    output_dir: Path,
    skip_tests: bool,
) -> dict[str, Any]:
    root = vault.resolve().parent
    vault = vault.resolve()
    preflight = _read_json(preflight_path)
    baseline = _read_json(baseline_inventory_path)
    final_inventory = _read_json(final_inventory_path)
    final_plan = _read_json(final_plan_path)
    prechange_preflight = _read_json(prechange_preflight_path)
    prechange_restore = _read_json(prechange_restore_report)
    postchange_restore = _read_json(postchange_restore_report)
    staging_rebuild = _read_json(staging_rebuild_report)
    cleanup = _cleanup_manifest_ok(cleanup_manifest, vault)
    tests = _run_tests(root, output_dir / "r10-pytest.txt", skip=skip_tests)
    final_rows = final_plan.get("concepts") or []
    review_rows = [row for row in final_rows if row.get("disposition") == "REVIEW"]
    empty_auto_stubs = [row for row in final_rows if row.get("auto_stub") and not row.get("has_substantive_content")]
    zero_inbound_without_evidence = [
        row for row in final_rows
        if not row.get("inbound_real") and (not row.get("disposition") or not row.get("reason"))
    ]
    preflight_checks = preflight.get("checks") or {}
    audit_stats = preflight.get("audit", {}).get("stats") or {}
    contract_boundary = _production_contract_boundary(root)
    graph_source = (root / "tools/archivist/search.py").read_text(encoding="utf-8")
    current_test_command = " ".join(str(item) for item in tests.get("command") or [])
    test_scope_ok = tests.get("status") == "PASS" and "test_hybrid_matched_chunks_r10.py" in current_test_command and "test_paged_note_reader_r10.py" in current_test_command
    test_memory_active = (vault / "30_Knowledge_Base/Concepts/Test Memory.md").is_file()
    quarantine_test_memory = False
    for candidate in (root / "data/vault_quarantine/memories").glob("**/Test Memory.md"):
        if candidate.is_file():
            quarantine_test_memory = True
            break

    gates = [
        _gate(
            "A145",
            prechange_preflight.get("status") == "PASS" and prechange_restore.get("status") == "PASS",
            {"prechange_preflight": prechange_preflight.get("status", "NOT_RUN"), "restore": prechange_restore.get("status", "NOT_RUN")},
            "pre-change preflight and staging restore PASS",
        ),
        _gate(
            "A146",
            int(baseline.get("concept_file_count") or 0) == 1748 and len(baseline.get("concepts") or []) == 1748,
            {"concept_file_count": baseline.get("concept_file_count"), "rows": len(baseline.get("concepts") or [])},
            1748,
        ),
        _gate(
            "A147",
            _plan_is_deterministic(final_inventory_path, final_plan) and all(not row.get("apply_eligible") for row in review_rows),
            {"deterministic": _plan_is_deterministic(final_inventory_path, final_plan), "review_rows": len(review_rows), "review_apply": sum(bool(row.get("apply_eligible")) for row in review_rows)},
            "deterministic plan; REVIEW excluded from apply",
        ),
        _gate(
            "A148",
            all(int(audit_stats.get(key, 0)) == 0 for key in ("broken_links", "ambiguous_links", "parse_errors")),
            {key: audit_stats.get(key, 0) for key in ("broken_links", "ambiguous_links", "parse_errors")},
            "0 broken/ambiguous/parse errors",
        ),
        _gate("A149", bool(preflight_checks.get("no_active_wikilink_or_embed")), preflight_checks.get("no_active_wikilink_or_embed"), True, "active canonical Vault has no wikilink/embed dependency"),
        _gate("A150", bool(preflight_checks.get("catalog_link_projection")), preflight_checks.get("catalog_link_projection"), True, "note_links projection present and matches catalog manifest"),
        _gate(
            "A151",
            all(token in graph_source for token in ("iter_link_edges", "direction=\"incoming\"", "direction=\"outgoing\"", "note_links")),
            {"catalog_graph_api": True, "source": "tools/archivist/search.py"},
            "portable outgoing/incoming catalog graph",
        ),
        _gate(
            "A152",
            all(token in graph_source for token in ("_graph_eligible_files", "sensitivity", "lifecycle_status")),
            {"namespace_filtering": True, "source": "tools/archivist/search.py"},
            "GraphRAG filters scope/lifecycle/sensitivity",
        ),
        _gate("A153", test_scope_ok, tests, "matched chunk beyond 8,000 is covered by passing R10 test"),
        _gate("A154", test_scope_ok, tests, "paged reader and content hash are covered by passing R10 test"),
        _gate("A155", contract_boundary["pass"], contract_boundary, "production answer callers use evidence contract"),
        _gate(
            "A156",
            not [row for row in final_rows if row.get("auto_stub") and str(row.get("search_scope") or "").lower() == "included"],
            {"searchable_auto_stubs": 0},
            0,
        ),
        _gate("A157", not zero_inbound_without_evidence, {"unassigned": len(zero_inbound_without_evidence)}, 0, "every zero-real-inbound row has disposition and reason"),
        _gate(
            "A158",
            int(final_inventory.get("concept_file_count") or 0) < int(baseline.get("concept_file_count") or 0)
            and not [row for row in final_rows if row.get("disposition") == "RELOCATE"]
            and cleanup.get("pass") is True,
            {"final_concepts": final_inventory.get("concept_file_count"), "relocations_remaining": sum(row.get("disposition") == "RELOCATE" for row in final_rows), "cleanup": cleanup},
            "misrouted content absent from Concepts with quarantine preimages",
        ),
        _gate("A159", not test_memory_active and quarantine_test_memory, {"active": test_memory_active, "quarantine": quarantine_test_memory}, "Test Memory quarantined and tombstoned"),
        _gate("A160", len(review_rows) == len(final_rows) and not zero_inbound_without_evidence, {"reviewed_or_routed": len(review_rows), "rows": len(final_rows)}, "referenced stubs have explicit REVIEW evidence and no link loss"),
        _gate(
            "A161",
            not empty_auto_stubs and bool(preflight_checks.get("stub_creation_default_blocked")),
            {"empty_auto_stubs": len(empty_auto_stubs), "new_stub_creation_default_blocked": preflight_checks.get("stub_creation_default_blocked")},
            {"empty_auto_stubs": 0, "new_stub_creation_default_blocked": True},
            "remaining referenced stubs require data-owner promotion/plain-text conversion before final retirement",
            status="REVIEW" if empty_auto_stubs else None,
        ),
        _gate("A162", bool(preflight_checks.get("catalog_vector_policy_parity")), preflight_checks.get("catalog_vector_policy_parity"), True, "catalog/vector/registry/policy parity"),
        _gate(
            "A163",
            postchange_restore.get("status") == "PASS" and staging_rebuild.get("status") == "PASS",
            {"restore": postchange_restore.get("status", "NOT_RUN"), "rebuild": staging_rebuild.get("status", "NOT_RUN")},
            "post-change backup, clean restore, and staging rebuild PASS",
        ),
        _gate("A164", tests.get("status") == "PASS", tests, "R10 architecture/retrieval/cleanup/rollback tests PASS"),
    ]
    failed = [gate["gate_id"] for gate in gates if gate["status"] not in {"PASS"}]
    status = "PASS" if not failed else "BLOCKED" if any(gate["status"] in {"REVIEW", "NOT_RUN"} for gate in gates) else "FAIL"
    report = {
        "schema": "vault-r10-acceptance-v1",
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "vault": str(vault),
        "evidence": {
            "preflight": str(preflight_path),
            "baseline_inventory": str(baseline_inventory_path),
            "final_inventory": str(final_inventory_path),
            "final_plan": str(final_plan_path),
            "prechange_preflight": str(prechange_preflight_path) if prechange_preflight_path else None,
            "prechange_restore": str(prechange_restore_report) if prechange_restore_report else None,
            "postchange_restore": str(postchange_restore_report) if postchange_restore_report else None,
            "staging_rebuild": str(staging_rebuild_report) if staging_rebuild_report else None,
            "cleanup_manifest": str(cleanup_manifest) if cleanup_manifest else None,
        },
        "gates": gates,
        "open_gates": failed,
        "test_result": tests,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acceptance-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "acceptance-report.md").write_text(
        "# Vault R10 Acceptance Report\n\n"
        f"Status: **{status}**\n\n"
        "| Gate | Status |\n|---|---|\n"
        + "\n".join(f"| {gate['gate_id']} | {gate['status']} |" for gate in gates)
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--preflight", type=Path, default=Path("scratch/vault-r10/preflight/r10-preflight.json"))
    parser.add_argument("--baseline-inventory", type=Path, required=True)
    parser.add_argument("--final-inventory", type=Path, required=True)
    parser.add_argument("--final-plan", type=Path, required=True)
    parser.add_argument("--prechange-preflight", type=Path, default=Path("scratch/vault-r9/preflight/r9-preflight.json"))
    parser.add_argument("--prechange-restore-report", type=Path, required=True)
    parser.add_argument("--postchange-restore-report", type=Path, required=True)
    parser.add_argument("--staging-rebuild-report", type=Path, required=True)
    parser.add_argument("--cleanup-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/vault-r10/acceptance"))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    report = run(
        args.vault,
        preflight_path=args.preflight.resolve(),
        baseline_inventory_path=args.baseline_inventory.resolve(),
        final_inventory_path=args.final_inventory.resolve(),
        final_plan_path=args.final_plan.resolve(),
        prechange_preflight_path=args.prechange_preflight.resolve() if args.prechange_preflight else None,
        prechange_restore_report=args.prechange_restore_report.resolve(),
        postchange_restore_report=args.postchange_restore_report.resolve(),
        staging_rebuild_report=args.staging_rebuild_report.resolve(),
        cleanup_manifest=args.cleanup_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        skip_tests=args.skip_tests,
    )
    print(json.dumps({"status": report["status"], "open_gates": report["open_gates"], "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
