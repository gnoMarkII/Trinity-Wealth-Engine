"""Evaluate final R9 architecture gates A123-A144 with explicit evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.scan_vault_writers_r9 import (  # noqa: E402
    APPROVED_ARTIFACT_ROOTS,
    APPROVED_BROKER_ROOTS,
    scan,
)
from tools.archivist.recovery_bundle import capture_runtime_state, restore_recovery_bundle  # noqa: E402
from tools.archivist.runtime_layout import runtime_layout  # noqa: E402
from tools.archivist.schema_registry import load_default_registry  # noqa: E402


R9_TESTS = [
    "tests/architecture",
    "tests/api/test_knowledge_writes_r8.py",
    "tests/application/knowledge",
    "tests/tools/archivist/test_runtime_layout_r9.py",
    "tests/tools/archivist/test_runtime_migration_r9.py",
    "tests/tools/archivist/test_recovery_bundle_r9.py",
    "tests/tools/archivist/test_vault_r8_*.py",
    "tests/integration/test_portfolio_projection_rebuild.py",
    "tests/tools/content/test_notebooklm_vault_migration.py",
    "tests/tools/content/test_briefing_evidence.py",
    "tests/tools/test_news_funnel_pipeline.py",
    "tests/tools/market/test_quant_history.py",
]


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _tree_fingerprint(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = 0
    markdown = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{relative}\0{path.stat().st_size}\0{file_hash}\n".encode("utf-8"))
        files += 1
        markdown += int(path.suffix.lower() == ".md")
    return {"sha256": digest.hexdigest(), "file_count": files, "markdown_count": markdown}


def _gate(gate_id: str, condition: bool, actual: Any, expected: Any, reason: str = "") -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "status": "PASS" if condition else "FAIL",
        "actual": actual,
        "expected": expected,
        "reason": reason,
    }


def _not_run(gate_id: str, actual: Any, expected: Any, reason: str) -> dict[str, Any]:
    return {"gate_id": gate_id, "status": "NOT_RUN", "actual": actual, "expected": expected, "reason": reason}


def _normalize_runtime_state(value: dict[str, Any]) -> dict[str, Any]:
    def clean(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                key: clean(subvalue)
                for key, subvalue in item.items()
                if key not in {"captured_at", "path", "manifest_path", "vault_root", "runtime_root", "db_path", "runtime_layout", "broker_db", "portfolio_db", "catalog_root", "vector_root", "reconciliation_root", "logs_root", "checkpoints_root"}
            }
        if isinstance(item, list):
            return [clean(subvalue) for subvalue in item]
        return item

    return clean(value)


def _snapshot_exact(bundle_root: Path, restored_vault: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    snapshot = bundle_root / Path(str((manifest.get("vault_snapshot") or {}).get("path") or ""))
    if not snapshot.is_file():
        return {"status": "FAIL", "reason": "snapshot missing"}
    with zipfile.ZipFile(snapshot) as archive:
        members = sorted(item.filename for item in archive.infolist() if not item.is_dir())
        actual = sorted(path.relative_to(restored_vault).as_posix() for path in restored_vault.rglob("*") if path.is_file())
        if members != actual:
            return {"status": "FAIL", "reason": "file set mismatch", "archived": len(members), "restored": len(actual)}
        mismatches: list[str] = []
        for relative in members:
            expected_hash = hashlib.sha256(archive.read(relative)).hexdigest()
            actual_hash = hashlib.sha256((restored_vault / relative).read_bytes()).hexdigest()
            if expected_hash != actual_hash:
                mismatches.append(relative)
                if len(mismatches) >= 20:
                    break
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "archived_file_count": len(members),
        "restored_file_count": len(actual),
        "hash_mismatch_count": len(mismatches),
        "hash_mismatches": mismatches,
    }


def _latest_bundle(root: Path) -> Path | None:
    paths = sorted((root / "scratch" / "vault-r9" / "recovery").glob("*/bundle-manifest.json"))
    if not paths:
        return None
    return max(paths, key=lambda path: (str((_read_json(path) or {}).get("created_at") or ""), path.name)).parent


def _restore_proof(vault: Path, bundle_root: Path, output_dir: Path, restore_root: Path | None = None) -> dict[str, Any]:
    manifest = _read_json(bundle_root / "bundle-manifest.json")
    if manifest is None:
        return {"status": "FAIL", "reason": "bundle manifest unreadable"}
    restore_root = restore_root.resolve() if restore_root is not None else output_dir / "restore-r9"
    existing = _read_json(restore_root / "restore-report.json")
    result = existing
    if result is None:
        if restore_root.exists():
            return {"status": "FAIL", "reason": f"restore destination exists without a verified report: {restore_root}"}
        result = restore_recovery_bundle(bundle_root=bundle_root, restore_root=restore_root)
    restored_vault = Path(str(result.get("restored_vault") or restore_root / "vault"))
    exact = _snapshot_exact(bundle_root, restored_vault, manifest)
    restored_runtime = Path(str(result.get("restored_runtime") or restore_root / "runtime")) / str(manifest.get("vault_id") or "")
    production_state = capture_runtime_state(vault, runtime_root=runtime_layout(vault, vault.parent / "data" / "vault_runtime").root)
    restored_state = capture_runtime_state(restored_vault, runtime_root=restored_runtime)
    semantic = _normalize_runtime_state(production_state) == _normalize_runtime_state(restored_state)
    return {
        "status": "PASS" if result.get("status") == "PASS" and exact.get("status") == "PASS" and semantic else "FAIL",
        "restore": result,
        "snapshot_exact": exact,
        "runtime_semantic_parity": semantic,
        "production_runtime_state": production_state,
        "restored_runtime_state": restored_state,
    }


def _run_tests(root: Path, output: Path, *, skip: bool) -> dict[str, Any]:
    if skip:
        return {"status": "NOT_RUN", "reason": "--skip-tests"}
    expanded_tests: list[str] = []
    for item in R9_TESTS:
        if any(char in item for char in "*?["):
            expanded_tests.extend(str(path.relative_to(root)) for path in sorted(root.glob(item)))
        else:
            expanded_tests.append(item)
    command = [sys.executable, "-m", "pytest", *expanded_tests, "-q", "--no-cov"]
    test_env = os.environ.copy()
    # Acceptance may be invoked from a workstation that has a live producer
    # environment. Force all news-funnel persistence used by tests into a
    # unique temp file so the suite cannot touch production data/news state.
    test_env["NEWS_FUNNEL_STORE_PATH"] = str(Path(tempfile.gettempdir()) / f"invest_agents_r9_test_store_{uuid.uuid4().hex}.json")
    completed = subprocess.run(command, cwd=root, env=test_env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
    return {"status": "PASS" if completed.returncode == 0 else "FAIL", "returncode": completed.returncode, "command": command, "output": str(output)}


def _cycle_evidence(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        payload = _read_json(path)
        result.append({"path": str(path), "status": (payload or {}).get("status", "NOT_RUN"), "cycle_type": (payload or {}).get("cycle_type")})
    return result


def _branch_protection_evidence(root: Path, evidence: Path | None) -> dict[str, Any]:
    if evidence is not None:
        payload = _read_json(evidence)
        required = "Vault architecture and write-boundary contracts"
        if payload and payload.get("protected") is True and required in str(payload.get("required_check") or ""):
            return {"status": "PASS", "source": str(evidence), "evidence": payload}
        return {"status": "FAIL", "source": str(evidence), "evidence": payload, "reason": "provided evidence does not prove required protection"}
    remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return {
        "status": "REVIEW",
        "source": "repository host not queried by local acceptance",
        "remote": remote.stdout.strip() if remote.returncode == 0 else None,
        "reason": "branch protection is an external setting; workflow YAML alone is not evidence",
    }


def run(
    vault: Path,
    *,
    runtime_base: Path | None,
    output_dir: Path,
    bundle_root: Path | None,
    restore: bool,
    observation: Path | None,
    cycles: list[Path],
    branch_protection: Path | None,
    restore_root: Path | None,
    skip_tests: bool,
) -> dict[str, Any]:
    root = vault.parent.resolve()
    vault = vault.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_default_registry()
    layout = runtime_layout(vault, runtime_base, create=False)
    runtime_state = capture_runtime_state(vault, runtime_root=layout.root)
    inventory = scan(root)
    preflight_path = root / "scratch/vault-r9/preflight/r9-preflight-after-derived.json"
    preflight = _read_json(preflight_path)
    migration = _read_json(root / "scratch/vault-r9/runtime-migration-apply-2.json")
    bundle_root = bundle_root.resolve() if bundle_root else _latest_bundle(root)
    manifest = _read_json(bundle_root / "bundle-manifest.json") if bundle_root else None
    restore_proof = _restore_proof(vault, bundle_root, output_dir, restore_root) if restore and bundle_root else None
    test_result = _run_tests(root, output_dir / "r9-pytest.txt", skip=skip_tests)
    crash_proof: dict[str, Any]
    if skip_tests:
        crash_proof = {"status": "NOT_RUN", "reason": "--skip-tests"}
    else:
        try:
            from scripts.run_vault_r8_acceptance import _crash_recovery_proof

            crash_proof = _crash_recovery_proof()
        except Exception as exc:  # noqa: BLE001 - acceptance evidence boundary
            crash_proof = {"status": "FAIL", "reason": str(exc)}

    no_runtime_sqlite = not [
        path for path in vault.rglob("*")
        if path.is_file() and (path.suffix.lower() in {".sqlite", ".sqlite3", ".db"} or path.name.endswith(("-wal", "-shm", "-journal")))
    ]
    direct_broker_outside = [
        row for row in inventory["rows"]
        if row.get("operation") == "KnowledgeWriteBroker.construct"
        and not str(row.get("source_file", "")).startswith("scripts/")
        and row.get("source_file") not in APPROVED_BROKER_ROOTS
    ]
    direct_artifact_outside = [
        row for row in inventory["rows"]
        if row.get("operation") == "ArtifactWriter.construct"
        and not str(row.get("source_file", "")).startswith("scripts/")
        and row.get("source_file") not in APPROVED_ARTIFACT_ROOTS
    ]
    api_schema = (root / "api/schemas/knowledge_writes.py").read_text(encoding="utf-8")
    api_router = (root / "api/routers/knowledge_writes.py").read_text(encoding="utf-8")
    api_security = "client-supplied target_path is not accepted" in api_schema and "dependencies=[Depends(require_session)]" in api_router
    bundle_complete = bool(
        manifest
        and manifest.get("bundle_version") == 1
        and manifest.get("vault_snapshot", {}).get("sha256")
        and manifest.get("runtime_state")
        and manifest.get("contract", {}).get("registry_digest") == registry.digest()
        and manifest.get("contract", {}).get("policy_digest") == registry.policy_digest()
        and manifest.get("derived_rebuild", {}).get("catalog")
        and manifest.get("derived_rebuild", {}).get("vector")
        and (manifest.get("runtime", {}).get("canonical", {}).get("present") is True)
    )
    observed = _read_json(observation)
    observation_pass = bool(observed and observed.get("status") == "PASS" and float(observed.get("duration_seconds", 0)) >= 3600 and all(item.get("status") == "PASS" for item in observed.get("samples", [])))
    rehearsal_path = root / "scratch/vault-r9/operations/runbook-rehearsal.json"
    rehearsal = _read_json(rehearsal_path)
    rehearsal_pass = bool(rehearsal and rehearsal.get("status") == "PASS" and rehearsal.get("mutation_mode") == "staging_only")
    cycle_paths = cycles or [Path(item) for item in sorted((root / "scratch/vault-r9/operations").glob("cycle-*.json"))]
    cycle_records = _cycle_evidence(root, cycle_paths)
    cycles_pass = len(cycle_records) >= 2 and all(item.get("status") == "PASS" for item in cycle_records)
    backup_records = [
        {"path": str(path.parent), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted((root / "scratch/vault-r9/recovery").glob("*/bundle-manifest.json"))
    ]
    state_catalog = runtime_state.get("catalog") or {}
    state_vector = runtime_state.get("vector") or {}
    catalog_vector_parity = (
        state_catalog.get("present") is True
        and state_catalog.get("integrity_check") == "ok"
        and state_catalog.get("registry_digest") == registry.digest()
        and state_catalog.get("policy_digest") == registry.policy_digest()
        and int(state_catalog.get("eligible_note_count") or 0) > 0
        and state_vector.get("present") is True
        and state_vector.get("registry_digest") == registry.digest()
        and state_vector.get("policy_digest") == registry.policy_digest()
        and int(state_vector.get("eligible_note_count") or 0) > 0
        and int(state_catalog.get("eligible_note_count") or 0) == int(state_vector.get("eligible_note_count") or 0)
    )
    gates: list[dict[str, Any]] = [
        _gate("A123", str(runtime_state["runtime_layout"]["runtime_root"]) == str(layout.root), runtime_state["runtime_layout"], layout.as_dict()),
        _gate("A124", no_runtime_sqlite, "no SQLite/WAL/SHM/journal under Vault" if no_runtime_sqlite else "runtime database found in Vault", 0),
        _gate("A125", bool(migration and migration.get("status") == "PASS" and migration.get("applied") is True), migration or "migration evidence missing", "applied migration with no conflict"),
        _gate("A126", not [row for row in inventory["rows"] if row.get("operation") == "compatibility_import"], 0, 0),
        _gate("A127", not direct_broker_outside, direct_broker_outside, 0),
        _gate("A128", not direct_artifact_outside, direct_artifact_outside, 0),
        _gate("A129", inventory["counts"].get("broad_allowlist", 0) == 0, inventory["counts"].get("broad_allowlist", 0), 0),
        _gate("A130", all(inventory["counts"].get(key, 0) == 0 for key in ("unresolved", "review", "expired", "parse_error")), inventory["counts"], "all scanner debt zero"),
        _gate("A131", bool(preflight and preflight.get("status") == "PASS" and preflight.get("active_notes", {}).get("invalid_count") == 0), preflight or "preflight evidence missing", "active notes invalid_count=0"),
        _gate("A132", bundle_complete, {"bundle": str(bundle_root) if bundle_root else None, "backup_count": len(backup_records)}, "complete full recovery bundle"),
        _gate("A133", bool(restore_proof and restore_proof.get("status") == "PASS"), restore_proof or "restore not run", "clean staging exact and semantic restore"),
        {"gate_id": "A134", "status": crash_proof.get("status", "FAIL"), "actual": crash_proof, "expected": "pending receipt replay without duplicate revision"},
        _gate("A135", bool(restore_proof and restore_proof.get("runtime_semantic_parity")), restore_proof.get("runtime_semantic_parity") if restore_proof else None, True),
        _gate("A136", catalog_vector_parity, {"catalog": state_catalog, "vector": state_vector}, "catalog/vector policy and eligible count parity"),
        _gate("A137", api_security and test_result.get("status") == "PASS", {"static": api_security, "tests": test_result.get("status")}, "authenticated path-free transport"),
        _gate("A138", cycles_pass and len(backup_records) >= 2, {"cycles": cycle_records, "backups": backup_records}, "two passing scheduled cycles and backups"),
        _gate("A139", observation_pass, {"path": str(observation) if observation else None, "status": observed.get("status") if observed else "NOT_RUN"}, "SLOs pass during observation"),
        _gate("A140", test_result.get("status") == "PASS", test_result, "R9 architecture/regression/migration/recovery/API tests"),
        _branch_protection_evidence(root, branch_protection) | {"gate_id": "A141"},
        _gate("A142", observation_pass, {"duration_seconds": observed.get("duration_seconds") if observed else None}, ">=3600 second observation"),
        _gate("A143", rehearsal_pass, rehearsal or {"path": str(rehearsal_path), "status": "NOT_RUN"}, "runbook rehearsal PASS with staging-only mutation mode"),
    ]
    all_prior_pass = all(gate["status"] == "PASS" for gate in gates)
    gates.append(_gate("A144", all_prior_pass, {"failed_or_open": [gate["gate_id"] for gate in gates if gate["status"] != "PASS"]}, "zero remaining P0 and zero waiver"))
    for gate in gates:
        gate.setdefault("evidence", [])
    report = {
        "schema": "vault-r9-acceptance-v1",
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vault": str(vault),
        "release_candidate": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip() or None,
        "runtime_layout": layout.as_dict(),
        "preflight": str(preflight_path) if preflight else None,
        "bundle": str(bundle_root) if bundle_root else None,
        "bundle_manifest_sha256": hashlib.sha256((bundle_root / "bundle-manifest.json").read_bytes()).hexdigest() if bundle_root and (bundle_root / "bundle-manifest.json").is_file() else None,
        "tree_fingerprint": _tree_fingerprint(vault),
        "writer_inventory": inventory["counts"],
        "test_result": test_result,
        "crash_recovery_proof": crash_proof,
        "observation": str(observation) if observation else None,
        "runbook_rehearsal": str(rehearsal_path) if rehearsal else None,
        "cycles": cycle_records,
        "backups": backup_records,
        "gates": gates,
        "status": "PASS" if all_prior_pass else "BLOCKED" if any(gate["status"] in {"NOT_RUN", "REVIEW"} for gate in gates) else "FAIL",
    }
    _write_json(output_dir / "acceptance-report.json", report)
    lines = ["# Vault R9 Acceptance Report", "", f"Status: **{report['status']}**", "", "| Gate | Status |", "|---|---|"]
    lines.extend(f"| {gate['gate_id']} | {gate['status']} |" for gate in gates)
    (output_dir / "acceptance-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=Path("memories"))
    parser.add_argument("--runtime-base", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/vault-r9/acceptance"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--restore-root", type=Path)
    parser.add_argument("--no-restore", action="store_true")
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--cycle", type=Path, action="append", default=[])
    parser.add_argument("--branch-protection-evidence", type=Path)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    report = run(
        args.vault,
        runtime_base=args.runtime_base,
        output_dir=args.output_dir.resolve(),
        bundle_root=args.bundle,
        restore=not args.no_restore,
        observation=args.observation,
        cycles=[path.resolve() for path in args.cycle],
        branch_protection=args.branch_protection_evidence,
        restore_root=args.restore_root,
        skip_tests=args.skip_tests,
    )
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
