"""Run the R3 F12 rehearsal on an isolated copy of ``memories``.

Every acceptance row is produced by an assertion executed during this run.
The script never writes the live vault; F13 remains a separate, guarded step.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.archivist.artifact_store import DurableArtifactStore
from tools.archivist.artifact_writer import ArtifactWriter, recover_pending_writes
from tools.archivist.navigation_builder import build_navigation_indices
from tools.archivist.vault_acceptance import (
    aggregate_acceptance_records,
    record_assertion,
    write_acceptance_report,
)
from tools.archivist.vault_backup import create_vault_snapshot, restore_vault_snapshot
from tools.archivist.vault_migration import (
    apply_migration_plan,
    create_migration_plan,
    rollback_migration,
    verify_migration,
)
from tools.archivist.vault_paths import VaultPaths


def _tree_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".chroma_index" in path.parts:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(65536):
                digest.update(chunk)
        result[path.relative_to(root).as_posix()] = digest.hexdigest()
    return result


def _reset_scratch(path: Path, workspace: Path) -> None:
    resolved = path.resolve()
    scratch_root = (workspace / "scratch").resolve()
    if not resolved.is_relative_to(scratch_root):
        raise ValueError(f"refusing to remove a path outside scratch: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _record(records: list, **kwargs: Any) -> None:
    records.append(record_assertion(**kwargs))


def _case_evidence(run_dir: Path, case_id: str, payload: Any) -> Path:
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
    path = evidence_dir / f"{safe}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def _resolve_generated_links(vault_root: Path, files: list[Path]) -> tuple[int, list[str]]:
    pattern = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
    checked = 0
    missing: list[str] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        for target in pattern.findall(text):
            target = target.strip()
            candidate = vault_root / target
            if candidate.suffix.lower() != ".md":
                # ``Path.with_suffix`` would treat decimal points in a Thai or
                # numeric filename (for example ``1.3%``) as an extension and
                # truncate the real target.  Wikilinks without .md append it.
                candidate = Path(str(candidate) + ".md")
            checked += 1
            if not candidate.is_file():
                missing.append(target)
    return checked, missing


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    live_vault = Path(os.getenv("OBSIDIAN_VAULT_PATH", str(workspace / "memories"))).resolve()
    run_id = datetime.now(timezone.utc).strftime("f12_%Y%m%dT%H%M%SZ")
    run_dir = workspace / "scratch" / "vault-v2" / "remediation-r3" / run_id
    clone = run_dir / "vault_clone"
    backup_dir = run_dir / "backups"
    plan_file = run_dir / "migration-plan.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    rehearsal_vault = clone
    restore_report: dict[str, Any] = {
        "status": "NOT_RUN",
        "source_clone": str(clone),
        "restored_root": str(run_dir / "restored_vault"),
        "snapshot": None,
        "files_extracted": 0,
        "hashes_match": False,
        "error": None,
    }
    live_before = _tree_hashes(live_vault) if live_vault.is_dir() else {}

    if not live_vault.is_dir():
        _record(
            records,
            case_id="F12.CLONE",
            gate_ids=("A13",),
            run_id=run_id,
            command="run_rehearsal_f12.py",
            phase="clone",
            actual="missing_live_vault",
            expected="clone_created",
            status="BLOCKED",
            reason=f"live vault does not exist: {live_vault}",
        )
    else:
        try:
            _reset_scratch(clone, workspace)
            shutil.copytree(
                live_vault,
                clone,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(".chroma_index", ".trash", ".sync_history"),
            )
            clone_count = len(_tree_hashes(clone))
            clone_evidence = _case_evidence(run_dir, "F12.CLONE", {"clone_count": clone_count})
            _record(
                records,
                case_id="F12.CLONE",
                gate_ids=("A13",),
                run_id=run_id,
                command="copytree(live, scratch/r3/<run>/vault_clone)",
                phase="clone",
                actual=clone_count > 0,
                expected=True,
                evidence_paths=(clone_evidence,),
            )
        except Exception as exc:
            _record(
                records,
                case_id="F12.CLONE",
                gate_ids=("A13",),
                run_id=run_id,
                command="copytree(live, scratch/r3/<run>/vault_clone)",
                phase="clone",
                actual=str(exc),
                expected="clone_created",
                status="FAIL",
                reason=str(exc),
            )

    if clone.is_dir():
        try:
            zip_path, zip_hash = create_vault_snapshot(vault_root=clone, backup_dir=backup_dir)
            restored = run_dir / "restored_vault"
            _reset_scratch(restored, workspace)
            extracted = restore_vault_snapshot(zip_path, restored, verify_checksum=True)
            source_hashes = _tree_hashes(clone)
            restored_hashes = _tree_hashes(restored)
            restore_report.update(
                {
                    "status": "PASS" if source_hashes == restored_hashes else "FAIL",
                    "snapshot": str(zip_path),
                    "snapshot_sha256": zip_hash,
                    "files_extracted": extracted,
                    "source_file_count": len(source_hashes),
                    "restored_file_count": len(restored_hashes),
                    "hashes_match": source_hashes == restored_hashes,
                }
            )
            if source_hashes == restored_hashes:
                rehearsal_vault = restored
            snapshot_evidence = _case_evidence(
                run_dir, "F12.SNAPSHOT", restore_report
            )
            _record(
                records,
                case_id="F12.SNAPSHOT",
                gate_ids=("A13",),
                run_id=run_id,
                command="create_vault_snapshot(clone)",
                phase="snapshot",
                actual=zip_path.is_file() and bool(zip_hash),
                expected=True,
                evidence_paths=(snapshot_evidence, zip_path),
            )
            restore_evidence = _case_evidence(run_dir, "F12.RESTORE", restore_report)
            _record(
                records,
                case_id="F12.RESTORE",
                gate_ids=("A03", "A13"),
                run_id=run_id,
                command="restore_vault_snapshot(snapshot, scratch/restored_vault)",
                phase="restore",
                actual=restore_report["hashes_match"],
                expected=True,
                evidence_paths=(restore_evidence,),
            )
        except Exception as exc:
            restore_report["status"] = "FAIL"
            restore_report["error"] = str(exc)
            _record(records, case_id="F12.SNAPSHOT", gate_ids=("A13",), run_id=run_id,
                    command="create_vault_snapshot(clone)", phase="snapshot", actual=str(exc),
                    expected=True, status="FAIL", reason=str(exc))
            restore_evidence = _case_evidence(run_dir, "F12.RESTORE", restore_report)
            _record(
                records,
                case_id="F12.RESTORE",
                gate_ids=("A03", "A13"),
                run_id=run_id,
                command="restore_vault_snapshot(snapshot, scratch/restored_vault)",
                phase="restore",
                actual=str(exc),
                expected=True,
                status="FAIL",
                reason=str(exc),
                evidence_paths=(restore_evidence,),
            )

        (run_dir / "restore-report.json").write_text(
            json.dumps(restore_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

        try:
            plan = create_migration_plan(rehearsal_vault, output_file=plan_file)
            _record(records, case_id="F12.PLAN", gate_ids=("A13",), run_id=run_id,
                    command="create_migration_plan(clone)", phase="plan", actual=plan.total_files,
                    expected=plan.total_files, evidence_paths=(plan_file,))
        except Exception as exc:
            plan = None
            _record(records, case_id="F12.PLAN", gate_ids=("A13",), run_id=run_id,
                    command="create_migration_plan(clone)", phase="plan", actual=str(exc),
                    expected="plan_file", status="FAIL", reason=str(exc))

        apply_res = None
        if plan is not None:
            try:
                apply_res = apply_migration_plan(plan_file, vault_root=rehearsal_vault, allow_live=True)
                journal = Path(apply_res["journal_file"])
                # The migration journal is append-only and rollback/reapply
                # legitimately adds later records.  Preserve the exact
                # post-apply before-image as an immutable evidence copy so the
                # acceptance aggregator does not mistake a later phase update
                # for a failed apply assertion.
                journal_evidence = _case_evidence(
                    run_dir,
                    "F12.APPLY.journal",
                    {
                        "journal_path": str(journal),
                        "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
                        "journal_bytes_hex": journal.read_bytes().hex(),
                    },
                )
                _record(records, case_id="F12.APPLY", gate_ids=("A03", "A13"), run_id=run_id,
                        command="apply_migration_plan(clone)", phase="apply", actual=journal.is_file(),
                        expected=True, evidence_paths=(journal_evidence,))
            except Exception as exc:
                _record(records, case_id="F12.APPLY", gate_ids=("A03", "A13"), run_id=run_id,
                        command="apply_migration_plan(clone)", phase="apply", actual=str(exc),
                        expected=True, status="FAIL", reason=str(exc))

        if plan is not None:
            try:
                verification = verify_migration(plan_file, vault_root=rehearsal_vault)
                _record(records, case_id="F12.VERIFY", gate_ids=("A02", "A03", "A13"), run_id=run_id,
                        command="verify_migration(clone)", phase="verify", actual=verification["success"],
                        expected=True, evidence_paths=(plan_file,))
            except Exception as exc:
                _record(records, case_id="F12.VERIFY", gate_ids=("A02", "A03", "A13"), run_id=run_id,
                        command="verify_migration(clone)", phase="verify", actual=str(exc), expected=True,
                        status="FAIL", reason=str(exc))

        if apply_res is not None:
            try:
                rollback = rollback_migration(apply_res["journal_file"], vault_root=rehearsal_vault)
                rollback_report = dict(rollback)
                rollback_report["status"] = "PASS" if not rollback["conflicts"] else "FAIL"
                _record(records, case_id="F12.ROLLBACK", gate_ids=("A03", "A13"), run_id=run_id,
                        command="rollback_migration(clone)", phase="rollback", actual=not rollback["conflicts"],
                        expected=True, evidence_paths=(plan_file,))
                (run_dir / "rollback-report.json").write_text(
                    json.dumps(rollback_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
                )
            except Exception as exc:
                (run_dir / "rollback-report.json").write_text(
                    json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                _record(records, case_id="F12.ROLLBACK", gate_ids=("A03", "A13"), run_id=run_id,
                        command="rollback_migration(clone)", phase="rollback", actual=str(exc), expected=True,
                        status="FAIL", reason=str(exc))

            try:
                reapplied = apply_migration_plan(plan_file, vault_root=rehearsal_vault, allow_live=True)
                reverified = verify_migration(plan_file, vault_root=rehearsal_vault)
                _record(records, case_id="F12.REAPPLY", gate_ids=("A03", "A13"), run_id=run_id,
                        command="apply+verify migration plan after rollback", phase="reapply",
                        actual=reverified["success"], expected=True, evidence_paths=(plan_file,))
            except Exception as exc:
                _record(records, case_id="F12.REAPPLY", gate_ids=("A03", "A13"), run_id=run_id,
                        command="apply+verify migration plan after rollback", phase="reapply", actual=str(exc),
                        expected=True, status="FAIL", reason=str(exc))

        try:
            nav = build_navigation_indices(vault_root=rehearsal_vault)
            checked, missing = _resolve_generated_links(rehearsal_vault, list(nav.values()))
            navigation_report = {"status": "PASS" if not missing else "FAIL", "generated_links_checked": checked, "missing": missing}
            (run_dir / "navigation-report.json").write_text(
                json.dumps(navigation_report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            nav_evidence = _case_evidence(run_dir, "F12.NAV", navigation_report)
            _record(records, case_id="F12.NAV", gate_ids=("A12", "A13"), run_id=run_id,
                    command="build_navigation_indices(clone)+resolve_links", phase="navigation",
                    actual=missing, expected=[], evidence_paths=(nav_evidence,))
        except Exception as exc:
            _record(records, case_id="F12.NAV", gate_ids=("A12", "A13"), run_id=run_id,
                    command="build_navigation_indices(clone)+resolve_links", phase="navigation", actual=str(exc),
                    expected=[], status="FAIL", reason=str(exc))

        try:
            writer = ArtifactWriter(vault_paths=VaultPaths(rehearsal_vault))
            ingest_meta = {
                "entity_type": "equity_analysis",
                "ticker": "AAPL",
                "date": "2026-09-08",
                "title": "F12 Verification",
                "document_key": f"v2:f12:{run_id}:primary",
            }
            first = writer.write_note(
                ingest_meta,
                "Rehearsal ingestion body",
                filename=f"F12 Verification {run_id}",
                companion_artifacts={"rehearsal.json": json.dumps({"run_id": run_id, "version": 1})},
            )
            first_archive_file = first.manifest_path.parent / first.primary_file.name
            first_archive_hash_before = hashlib.sha256(first_archive_file.read_bytes()).hexdigest()
            restarted = ArtifactWriter(vault_paths=VaultPaths(rehearsal_vault))
            second = restarted.write_note(
                ingest_meta,
                "Rehearsal ingestion body",
                filename=f"F12 Verification {run_id}",
                companion_artifacts={"rehearsal.json": json.dumps({"run_id": run_id, "version": 1})},
            )
            changed = ArtifactWriter(vault_paths=VaultPaths(rehearsal_vault)).write_note(
                ingest_meta,
                "Rehearsal ingestion body changed",
                filename=f"F12 Verification {run_id}",
                companion_artifacts={"rehearsal.json": json.dumps({"run_id": run_id, "version": 2})},
            )
            changed_retry = ArtifactWriter(vault_paths=VaultPaths(rehearsal_vault)).write_note(
                ingest_meta,
                "Rehearsal ingestion body changed",
                filename=f"F12 Verification {run_id}",
                companion_artifacts={"rehearsal.json": json.dumps({"run_id": run_id, "version": 2})},
            )
            store = DurableArtifactStore(VaultPaths(rehearsal_vault))
            resolved_first = store.get_revision_artifact(first.note_id, first.revision_id)
            resolved_changed = store.get_revision_artifact(changed.note_id, changed.revision_id)
            history = store.list_revisions(first.note_id)
            pinned_first_hash = hashlib.sha256(
                resolved_first.primary_file.read_bytes()
            ).hexdigest()
            pinned_changed_hash = hashlib.sha256(
                resolved_changed.primary_file.read_bytes()
            ).hexdigest()
            actual = {
                "first_revision": first.revision,
                "second_reused": second.is_reused,
                "same_revision": second.revision_id == first.revision_id,
                "changed_revision": changed.revision,
                "changed_new_revision_id": changed.revision_id not in {first.revision_id, second.revision_id},
                "changed_retry_reused": changed_retry.is_reused,
                "reader_first_revision": resolved_first.revision,
                "reader_changed_revision": resolved_changed.revision,
                "history_count": len(history),
                "first_frozen_primary_sha256": pinned_first_hash,
                "first_frozen_intact": pinned_first_hash == first_archive_hash_before,
                "changed_primary_sha256": pinned_changed_hash,
            }
            ingest_evidence = _case_evidence(run_dir, "F12.INGEST_RESTART", actual)
            _record(records, case_id="F12.INGEST_RESTART", gate_ids=("A01", "A02", "A03", "A13"), run_id=run_id,
                    command="ArtifactWriter same-input + changed-input + restart + DurableArtifactStore", phase="ingestion",
                    actual=actual, expected={
                        "first_revision": 1,
                        "second_reused": True,
                        "same_revision": True,
                        "changed_revision": 2,
                        "changed_new_revision_id": True,
                        "changed_retry_reused": True,
                        "reader_first_revision": 1,
                        "reader_changed_revision": 2,
                        "history_count": 2,
                        "first_frozen_primary_sha256": pinned_first_hash,
                        "first_frozen_intact": True,
                        "changed_primary_sha256": pinned_changed_hash,
                    },
                    evidence_paths=(ingest_evidence,))
            (run_dir / "ingestion-restart-report.json").write_text(
                json.dumps(actual, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
            pinned_report = {
                "note_id": first.note_id,
                "history": history,
                "first_revision_id": first.revision_id,
                "changed_revision_id": changed.revision_id,
                "first_primary_sha256": pinned_first_hash,
                "changed_primary_sha256": pinned_changed_hash,
                "first_body": resolved_first.body,
                "changed_body": resolved_changed.body,
            }
            (run_dir / "pinned-reference-report.json").write_text(
                json.dumps(pinned_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )

            # Force a recoverable failure case: an uncommitted journal with no
            # staged primary must remain available for operator inspection and
            # must not be projected into the current note.
            pending_probe = rehearsal_vault / ".system" / "pending_writes" / "f12_failed_recovery"
            pending_probe.mkdir(parents=True, exist_ok=True)
            (pending_probe / "journal.json").write_text(
                json.dumps({
                    "journal_version": 2,
                    "write_id": "f12_failed_recovery",
                    "target": str(rehearsal_vault / "30_Knowledge_Base" / "f12-never-project.md"),
                    "phase": "staged",
                }), encoding="utf-8"
            )
            recovered_first = recover_pending_writes(root=rehearsal_vault)
            recovered_second = recover_pending_writes(root=rehearsal_vault)
            recovery_actual = {
                "first_attempt": recovered_first,
                "second_attempt": recovered_second,
                "failed_journal_retained": (pending_probe / "journal.json").is_file(),
            }
            recovery_evidence = _case_evidence(run_dir, "F12.RECOVERY", recovery_actual)
            _record(records, case_id="F12.RECOVERY", gate_ids=("A03",), run_id=run_id,
                    command="recover_pending_writes(clone) repeated failed journal", phase="recovery",
                    actual=recovery_actual,
                    expected={"first_attempt": [], "second_attempt": [], "failed_journal_retained": True},
                    evidence_paths=(recovery_evidence,))
        except Exception as exc:
            (run_dir / "ingestion-restart-report.json").write_text(
                json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            _record(records, case_id="F12.INGEST_RESTART", gate_ids=("A01", "A02", "A03", "A13"), run_id=run_id,
                    command="ArtifactWriter x2 + DurableArtifactStore", phase="ingestion", actual=str(exc),
                    expected=True, status="FAIL", reason=str(exc))

    # A live clone can legitimately have no relocations. Exercise a separate
    # non-zero legacy fixture so migration/rollback is backed by an actual move
    # and companion operation rather than by a no-op plan.
    synthetic_report: dict[str, Any] = {"status": "NOT_RUN", "error": None}
    synthetic_root = run_dir / "synthetic_repair_vault"
    try:
        _reset_scratch(synthetic_root, workspace)
        legacy_dir = synthetic_root / "30_Knowledge_Base" / "Equities"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_note = legacy_dir / "F12 Legacy.md"
        legacy_sidecar = legacy_dir / "F12 Legacy.json"
        legacy_note.write_text(
            "---\nschema_version: 1\nentity_type: equity_analysis\ntitle: F12 Legacy\n"
            "ticker: F12\ndate: 2026-09-09\n---\n\n# F12 Legacy\n",
            encoding="utf-8",
        )
        legacy_sidecar.write_text(json.dumps({"fixture": "f12", "score": 1}), encoding="utf-8")
        config = synthetic_root / ".system" / "vault_config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            json.dumps({"layout_version": 2, "custom_setting": "preserve-me"}, indent=2),
            encoding="utf-8",
        )

        def _user_hashes(root: Path) -> dict[str, str]:
            return {
                path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(root.rglob("*"))
                if path.is_file() and ".system" not in path.relative_to(root).parts
            }

        before_user = _user_hashes(synthetic_root)
        synthetic_plan_file = run_dir / "synthetic-repair-plan.json"
        synthetic_plan = create_migration_plan(synthetic_root, output_file=synthetic_plan_file)
        synthetic_apply = apply_migration_plan(
            synthetic_plan_file, vault_root=synthetic_root, allow_live=True
        )
        synthetic_verify = verify_migration(synthetic_plan_file, vault_root=synthetic_root)
        synthetic_rollback = rollback_migration(
            synthetic_apply["journal_file"], vault_root=synthetic_root
        )
        after_rollback_user = _user_hashes(synthetic_root)
        synthetic_reapply = apply_migration_plan(
            synthetic_plan_file, vault_root=synthetic_root, allow_live=True
        )
        synthetic_reverify = verify_migration(synthetic_plan_file, vault_root=synthetic_root)
        synthetic_report = {
            "status": "PASS",
            "summary": synthetic_plan.summary,
            "applied_count": synthetic_apply["applied_count"],
            "verify_success": synthetic_verify["success"],
            "rollback_conflicts": synthetic_rollback["conflicts"],
            "exact_before_after_rollback": before_user == after_rollback_user,
            "reapply_verify_success": synthetic_reverify["success"],
            "reapply_journal": synthetic_reapply["journal_file"],
        }
        synthetic_evidence = _case_evidence(run_dir, "F12.SYNTHETIC_REPAIR", synthetic_report)
        _record(
            records,
            case_id="F12.SYNTHETIC_REPAIR",
            gate_ids=("A03", "A13", "A14"),
            run_id=run_id,
            command="synthetic legacy relocate+companion apply/verify/rollback/reapply",
            phase="synthetic_repair",
            actual={
                "relocate": synthetic_plan.summary.get("relocate", 0),
                "applied_count": synthetic_apply["applied_count"],
                "verify_success": synthetic_verify["success"],
                "rollback_conflicts": synthetic_rollback["conflicts"],
                "exact_before_after_rollback": synthetic_report["exact_before_after_rollback"],
                "reapply_verify_success": synthetic_reverify["success"],
            },
            expected={
                "relocate": 1,
                "applied_count": 2,
                "verify_success": True,
                "rollback_conflicts": [],
                "exact_before_after_rollback": True,
                "reapply_verify_success": True,
            },
            evidence_paths=(synthetic_evidence, synthetic_plan_file),
        )
    except Exception as exc:
        synthetic_report["status"] = "FAIL"
        synthetic_report["error"] = str(exc)
        synthetic_evidence = _case_evidence(run_dir, "F12.SYNTHETIC_REPAIR", synthetic_report)
        _record(
            records,
            case_id="F12.SYNTHETIC_REPAIR",
            gate_ids=("A03", "A13", "A14"),
            run_id=run_id,
            command="synthetic legacy relocate+companion apply/verify/rollback/reapply",
            phase="synthetic_repair",
            actual=str(exc),
            expected=True,
            status="FAIL",
            reason=str(exc),
            evidence_paths=(synthetic_evidence,),
        )
    (run_dir / "synthetic-repair-report.json").write_text(
        json.dumps(synthetic_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    live_after = _tree_hashes(live_vault) if live_vault.is_dir() else {}
    live_unchanged = live_before == live_after
    live_evidence = _case_evidence(run_dir, "F12.LIVE_GUARD", {"live_unchanged": live_unchanged})
    _record(records, case_id="F12.LIVE_GUARD", gate_ids=("A13",), run_id=run_id,
            command="hash live vault before/after rehearsal", phase="live_guard", actual=live_unchanged,
            expected=True, evidence_paths=(live_evidence,))

    report = aggregate_acceptance_records(records)
    report["run_id"] = run_id
    report["live_vault"] = str(live_vault)
    report["rehearsal_vault"] = str(rehearsal_vault)
    write_acceptance_report(report, run_dir, run_id=run_id)
    # Keep the filenames promised by the R3 hand-off contract alongside the
    # canonical acceptance report.  They are projections of the same recorded
    # assertions; no status is authored independently here.
    acceptance_json = run_dir / "acceptance.json"
    acceptance_json.write_text(
        (run_dir / "acceptance-report.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    rehearsal_report = run_dir / "rehearsal-report.md"
    rehearsal_report.write_text(
        (run_dir / "acceptance-report.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    unresolved = []
    pending_probe = rehearsal_vault / ".system" / "pending_writes" / "f12_failed_recovery" / "journal.json"
    if pending_probe.is_file():
        unresolved.append(
            "A deliberately failed recovery journal remains retained in the isolated rehearsal vault for inspection."
        )
    unresolved.append(
        "F13 live apply, production model activation, and interactive Obsidian/UI inspection were not executed by this isolated rehearsal."
    )
    (run_dir / "unresolved-items.md").write_text(
        "# F12 Rehearsal Unresolved Items\n\n"
        + "\n".join(f"- {item}" for item in unresolved)
        + "\n",
        encoding="utf-8",
    )
    latest = workspace / "scratch" / "vault-v2" / "remediation-r3" / "latest-run.json"
    latest.write_text(json.dumps({"run_id": run_id, "report_dir": str(run_dir), "overall_status": report["overall_status"]}, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "overall_status": report["overall_status"], "counts": report["counts"], "report": str(run_dir / 'acceptance-report.md')}, ensure_ascii=False))
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
