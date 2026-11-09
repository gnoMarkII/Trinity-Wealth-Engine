"""Command Line Interface for Obsidian Vault V2 Operations.

Provides entry points for audit, synthetic benchmarking, and future migration commands.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from tools.archivist.vault_audit import scan_vault, write_audit_report
from tools.archivist.vault_benchmark import generate_corpus


def _handle_audit(args: argparse.Namespace) -> int:
    vault_path = Path(args.vault).resolve()
    output_path = Path(args.output).resolve()

    print(f"[AUDIT] Scanning Obsidian Vault at: {vault_path}")
    result = scan_vault(vault_path)

    print(f"[AUDIT] Writing inventory and report to: {output_path}")
    inv_file, audit_file = write_audit_report(result, output_path)

    print(f"[OK] Scan completed successfully.")
    print(f"  Total files: {result.total_files}")
    print(f"  Active files: {result.active_files_count}")
    print(f"  Excluded files: {result.excluded_files_count}")
    print(f"  Issues found: {result.stats.get('total_issues', 0)}")
    print(f"  Inventory: {inv_file}")
    print(f"  Audit Report: {audit_file}")
    return 0


def _handle_benchmark(args: argparse.Namespace) -> int:
    from tools.archivist.vault_benchmark import run_scale_benchmarks
    output_root = Path(args.output).resolve()
    sizes = args.sizes or [1000]
    seed = args.seed

    print(f"[BENCHMARK] Running scale benchmarks for sizes {sizes} in: {output_root}")
    results = run_scale_benchmarks(sizes, output_root, seed=seed)
    print(f"[OK] Scale benchmarks completed. Reports saved to {output_root / 'scale-report.md'}")
    return 0


def _handle_plan(args: argparse.Namespace) -> int:
    from tools.archivist.vault_migration import create_migration_plan
    vault = Path(args.vault).resolve()
    out = Path(args.output).resolve()
    print(f"[PLAN] Creating migration plan for {vault}...")
    plan = create_migration_plan(vault, output_file=out)
    print(f"[OK] Migration plan {plan.plan_id} created: {out}")
    print(f"  Total notes: {plan.total_files}")
    print(f"  Relocations needed: {plan.summary.get('relocate', 0)}")
    print(f"  Already canonical (no-op): {plan.summary.get('no_op', 0)}")
    return 0


def _handle_apply(args: argparse.Namespace) -> int:
    from tools.archivist.vault_migration import apply_migration_plan
    plan_file = Path(args.plan).resolve()
    vault = Path(args.vault).resolve() if args.vault else None
    print(f"[APPLY] Applying migration plan {plan_file}...")
    res = apply_migration_plan(plan_file, vault_root=vault, allow_live=args.allow_live)
    print(f"[OK] Applied {res['applied_count']} file movements.")
    print(f"  Journal: {res['journal_file']}")
    return 0


def _handle_verify(args: argparse.Namespace) -> int:
    from tools.archivist.vault_migration import verify_migration
    plan_file = Path(args.plan).resolve()
    vault = Path(args.vault).resolve() if args.vault else None
    print(f"[VERIFY] Verifying migration plan {plan_file}...")
    res = verify_migration(plan_file, vault_root=vault)
    if res["success"]:
        print(f"[OK] All {res['verified_count']} files verified successfully.")
        return 0
    else:
        print(f"[ERROR] Verification failed:")
        print(f"  Missing: {res['missing_targets']}")
        print(f"  Hash mismatches: {res['hash_mismatches']}")
        return 1


def _handle_rollback(args: argparse.Namespace) -> int:
    from tools.archivist.vault_migration import rollback_migration
    journal_file = Path(args.journal).resolve()
    vault = Path(args.vault).resolve() if args.vault else None
    print(f"[ROLLBACK] Rolling back migration using journal {journal_file}...")
    res = rollback_migration(journal_file, vault_root=vault)
    print(f"[OK] Rolled back {res['rolled_back_count']} file operations.")
    return 0


def _handle_snapshot(args: argparse.Namespace) -> int:
    from tools.archivist.vault_backup import create_vault_snapshot
    vault = Path(args.vault).resolve()
    out = Path(args.output).resolve() if args.output else None
    print(f"[SNAPSHOT] Creating compressed backup of vault {vault}...")
    zip_path, checksum = create_vault_snapshot(vault_root=vault, backup_dir=out)
    print(f"[OK] Snapshot created: {zip_path}")
    print(f"  SHA256: {checksum}")
    return 0


def _handle_maintain(args: argparse.Namespace) -> int:
    from tools.archivist.vault_maintenance import detect_orphaned_sidecars, run_catalog_maintenance
    vault = Path(args.vault).resolve()
    maint = run_catalog_maintenance(vault_root=vault, vacuum=not args.no_vacuum)
    orphans = detect_orphaned_sidecars(vault_root=vault)
    print(json.dumps({"maintenance": maint, "orphaned_sidecars": orphans}, ensure_ascii=False, indent=2))
    return 0


def _handle_archive(args: argparse.Namespace) -> int:
    from tools.archivist.vault_archival_policy import apply_archival_policy
    vault = Path(args.vault).resolve()
    res = apply_archival_policy(
        vault_root=vault,
        max_age_days=args.max_age_days,
        dry_run=not args.apply,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _handle_nav_index(args: argparse.Namespace) -> int:
    from tools.archivist.navigation_builder import build_navigation_indices
    vault = Path(args.vault).resolve()
    generated = build_navigation_indices(vault_root=vault)
    print(json.dumps({k: str(v) for k, v in generated.items()}, ensure_ascii=False, indent=2))
    return 0


def _handle_reconcile(args: argparse.Namespace) -> int:
    from tools.archivist.composition import build_knowledge_write_port
    from tools.archivist.human_edit_reconciler import HumanEditReconciler
    from tools.archivist.vault_paths import VaultPaths

    vault = Path(args.vault).resolve()
    vp = VaultPaths(vault)
    runtime_base = args.runtime_base or None
    broker = build_knowledge_write_port(vault_paths=vp, runtime_base=runtime_base) if args.apply else None
    reconciler = HumanEditReconciler(vault_paths=vp, broker=broker, runtime_base=runtime_base)
    report = reconciler.reconcile_once(write_enabled=args.apply)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _handle_concepts_cleanup(args: argparse.Namespace) -> int:
    from tools.archivist.concepts_cleanup_executor import apply_cleanup, rollback_cleanup

    vault = Path(args.vault).resolve()
    if args.rollback:
        if not args.manifest:
            print("[ERROR] --manifest is required when using --rollback", file=sys.stderr)
            return 2
        res = rollback_cleanup(vault, Path(args.manifest).resolve(), owner=args.owner)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if not args.plan or not args.quarantine or not args.run_id:
        print("[ERROR] --plan, --quarantine, and --run-id are required for cleanup apply", file=sys.stderr)
        return 2
    res = apply_cleanup(
        vault,
        Path(args.plan).resolve(),
        Path(args.quarantine).resolve(),
        run_id=args.run_id,
        owner=args.owner,
        policy_only=args.policy_only,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def _handle_runtime_migrate(args: argparse.Namespace) -> int:
    from tools.archivist.runtime_migration import migrate_runtime_layout

    vault = Path(args.vault).resolve()
    res = migrate_runtime_layout(
        vault_root=vault,
        runtime_base=args.runtime_base or None,
        apply=args.apply,
        output=args.output or None,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("status") == "PASS" else 1


def _handle_projection_rebuild(args: argparse.Namespace) -> int:
    from tools.archivist.composition import build_knowledge_write_port
    from tools.archivist.vault_paths import VaultPaths
    from tools.portfolio.projection_boundary import PortfolioProjectionBoundary

    vault = Path(args.vault).resolve()
    payload = json.loads(Path(args.input_json).resolve().read_text(encoding="utf-8"))
    projections = payload.get("projections") or []
    source_state = payload.get("source_state") or {}
    writer = build_knowledge_write_port(vault_paths=VaultPaths(vault), runtime_base=args.runtime_base or None)
    receipts = PortfolioProjectionBoundary(writer).rebuild(
        portfolio_id=args.portfolio_id,
        projections=projections,
        source_state=source_state,
    )
    print(json.dumps([r.to_dict() for r in receipts], ensure_ascii=False, indent=2))
    return 0


def _handle_query_notes(args: argparse.Namespace) -> int:
    from application.knowledge.query_service import KnowledgeQueryService
    from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
    from tools.archivist.catalog_runtime import resolve_catalog_path
    from tools.archivist.link_resolver import VaultLinkResolver

    vault = Path(args.vault).resolve()
    try:
        resolve_catalog_path(vault, require_exists=True)
        read_only = True
    except FileNotFoundError:
        read_only = False

    catalog = SqliteNoteCatalogAdapter(vault_root=vault, read_only=read_only)
    resolver = VaultLinkResolver(catalog=catalog, vault_root=vault)
    service = KnowledgeQueryService(catalog=catalog, link_resolver=resolver)

    output: dict[str, object] = {}
    if args.note_id:
        entry = service.get_note(args.note_id)
        output["note"] = dataclasses.asdict(entry) if entry else None
    if args.ticker:
        entries = service.find_notes_for_ticker(args.ticker, limit=args.limit)
        output["ticker_notes"] = [dataclasses.asdict(e) for e in entries]
    if args.entity_type:
        entries = service.find_notes_by_type(args.entity_type, limit=args.limit)
        output["entity_notes"] = [dataclasses.asdict(e) for e in entries]
    if args.wikilink:
        output["resolved_wikilink"] = service.resolve_wikilink(
            args.wikilink,
            context_path=args.context_path or None,
        )

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _handle_readiness_audit(args: argparse.Namespace) -> int:
    from application.resilience.readiness_audit_service import InstitutionalReadinessAuditService

    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve() if args.output else None
    svc = InstitutionalReadinessAuditService(repo_root=repo_root)
    report = svc.run_readiness_audit(
        write_artifact=not args.no_write,
        artifact_path=output_path,
    )
    print(
        json.dumps(
            {
                "as_of_date": report.as_of_date,
                "decision": report.decision.value,
                "passed": sum(1 for e in report.evaluations if e.verdict.value == "PASS"),
                "total": len(report.evaluations),
                "sha256_checksum": report.sha256_checksum,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.archivist.vault_v2_cli",
        description="Obsidian Vault V2 Management & Migration Tooling",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. audit subcommand
    audit_parser = subparsers.add_parser("audit", help="Read-only scan and audit of vault")
    audit_parser.add_argument(
        "--vault",
        type=str,
        default="./memories",
        help="Path to Obsidian vault root (default: ./memories)",
    )
    audit_parser.add_argument(
        "--output",
        type=str,
        default="./scratch/vault-v2/baseline",
        help="Directory to save inventory.json and audit.md (default: ./scratch/vault-v2/baseline)",
    )

    # 2. benchmark subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Generate synthetic test corpora for scaling validation")
    bench_parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000],
        help="List of corpus sizes to generate (e.g. 1000 10000 50000)",
    )
    bench_parser.add_argument(
        "--seed",
        type=int,
        default=20260906,
        help="Random seed for deterministic generation",
    )
    bench_parser.add_argument(
        "--output",
        type=str,
        default="./scratch/vault-v2/benchmark",
        help="Directory to output benchmark corpora",
    )
    bench_parser.add_argument(
        "--embedding-mode",
        type=str,
        default="fake",
        choices=["fake", "local"],
        help="Embedding mode for future benchmark runner",
    )

    # 3. plan subcommand
    plan_parser = subparsers.add_parser("plan", help="Generate dry-run migration plan")
    plan_parser.add_argument("--vault", type=str, default="./memories", help="Path to vault root")
    plan_parser.add_argument("--output", type=str, default="./scratch/vault-v2/migration-plan.json", help="Path to output plan JSON")

    # 4. apply subcommand
    apply_parser = subparsers.add_parser("apply", help="Execute migration plan")
    apply_parser.add_argument("--plan", type=str, required=True, help="Path to migration plan JSON")
    apply_parser.add_argument("--vault", type=str, default="", help="Override vault root")
    apply_parser.add_argument("--allow-live", action="store_true", help="Explicit confirmation to migrate live memories vault")

    # 5. verify subcommand
    verify_parser = subparsers.add_parser("verify", help="Verify applied migration against plan")
    verify_parser.add_argument("--plan", type=str, required=True, help="Path to migration plan JSON")
    verify_parser.add_argument("--vault", type=str, default="", help="Override vault root")

    # 6. rollback subcommand
    rollback_parser = subparsers.add_parser("rollback", help="Roll back migration using journal")
    rollback_parser.add_argument("--journal", type=str, required=True, help="Path to migration_journal.jsonl")
    rollback_parser.add_argument("--vault", type=str, default="", help="Override vault root")

    # 7. snapshot subcommand
    snapshot_parser = subparsers.add_parser("snapshot", help="Create a compressed zip backup snapshot of the vault")
    snapshot_parser.add_argument("--vault", type=str, default="./memories", help="Path to vault root (default: ./memories)")
    snapshot_parser.add_argument("--output", type=str, default="", help="Directory to save snapshot zip")

    # 8. maintain subcommand
    maintain_parser = subparsers.add_parser("maintain", help="Run SQLite catalog maintenance and orphan sidecar scan")
    maintain_parser.add_argument("--vault", type=str, default="./memories", help="Path to vault root")
    maintain_parser.add_argument("--no-vacuum", action="store_true", help="Skip SQLite VACUUM during maintenance")

    # 9. archive subcommand
    archive_parser = subparsers.add_parser("archive", help="Apply Data Lifecycle archival policy to aged news notes")
    archive_parser.add_argument("--vault", type=str, default="./memories", help="Path to vault root")
    archive_parser.add_argument("--max-age-days", type=int, default=90, help="Retention age threshold in days")
    archive_parser.add_argument("--apply", action="store_true", help="Execute file moves instead of dry-run")

    # 10. nav-index subcommand
    nav_parser = subparsers.add_parser("nav-index", help="Build V2 navigation index notes under 00_Index")
    nav_parser.add_argument("--vault", type=str, default="./memories", help="Path to vault root")

    # 11. reconcile subcommand
    reconcile_parser = subparsers.add_parser("reconcile", help="Scan or import safe human edits via KnowledgeWritePort")
    reconcile_parser.add_argument("--vault", type=str, default="./memories", help="Path to vault root")
    reconcile_parser.add_argument("--apply", action="store_true", help="Submit safe edits through write broker")
    reconcile_parser.add_argument("--runtime-base", type=str, default="", help="External runtime root override")

    # 12. concepts-cleanup subcommand
    cleanup_parser = subparsers.add_parser("concepts-cleanup", help="Apply or rollback R10 Concepts cleanup plan")
    cleanup_parser.add_argument("--vault", type=str, default="./memories", help="Path to vault root")
    cleanup_parser.add_argument("--owner", type=str, default="codex-r10", help="Maintenance lease owner")
    cleanup_parser.add_argument("--plan", type=str, default="", help="Path to R10 cleanup plan JSON")
    cleanup_parser.add_argument("--quarantine", type=str, default="", help="External quarantine root directory")
    cleanup_parser.add_argument("--run-id", type=str, default="", help="Cleanup execution run identifier")
    cleanup_parser.add_argument("--policy-only", action="store_true", help="Repair stub policy metadata only")
    cleanup_parser.add_argument("--rollback", action="store_true", help="Rollback an APPLIED cleanup run")
    cleanup_parser.add_argument("--manifest", type=str, default="", help="Path to manifest.json for rollback")

    # 13. runtime-migrate subcommand
    migrate_rt_parser = subparsers.add_parser("runtime-migrate", help="Plan or apply idempotent R9 runtime layout migration")
    migrate_rt_parser.add_argument("--vault", type=str, default="./memories", help="Path to vault root")
    migrate_rt_parser.add_argument("--runtime-base", type=str, default="", help="Canonical external runtime base override")
    migrate_rt_parser.add_argument("--apply", action="store_true", help="Copy files into the R9 layout")
    migrate_rt_parser.add_argument("--output", type=str, default="", help="Optional JSON report output path")

    # 14. projection-rebuild subcommand
    proj_parser = subparsers.add_parser("projection-rebuild", help="Rebuild deterministic portfolio Markdown projections")
    proj_parser.add_argument("--vault", type=str, default="./memories", help="Path to vault root")
    proj_parser.add_argument("--portfolio-id", type=str, default="default", help="Target portfolio identifier")
    proj_parser.add_argument("--input-json", type=str, required=True, help="JSON file with 'projections' and 'source_state'")
    proj_parser.add_argument("--runtime-base", type=str, default="", help="External runtime base override")

    # 15. query-notes subcommand
    query_parser = subparsers.add_parser("query-notes", help="Query SQLite note catalog and resolve wikilinks")
    query_parser.add_argument("--vault", type=str, default="./memories", help="Path to vault root")
    query_parser.add_argument("--note-id", type=str, default="", help="Look up single note by note_id")
    query_parser.add_argument("--ticker", type=str, default="", help="Filter catalog notes by ticker")
    query_parser.add_argument("--entity-type", type=str, default="", help="Filter catalog notes by entity_type")
    query_parser.add_argument("--wikilink", type=str, default="", help="Resolve wikilink or relative link target")
    query_parser.add_argument("--context-path", type=str, default="", help="Optional context relative path for link resolution")
    query_parser.add_argument("--limit", type=int, default=50, help="Maximum result count")

    # 16. readiness-audit subcommand
    readiness_parser = subparsers.add_parser("readiness-audit", help="Run the 10-question Institutional Readiness Audit")
    readiness_parser.add_argument("--repo-root", type=str, default=".", help="Repository root path")
    readiness_parser.add_argument("--no-write", action="store_true", help="Do not write readiness-decision.md artifact")
    readiness_parser.add_argument("--output", type=str, default="", help="Custom artifact output path")

    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    handlers = {
        "audit": _handle_audit,
        "benchmark": _handle_benchmark,
        "plan": _handle_plan,
        "apply": _handle_apply,
        "verify": _handle_verify,
        "rollback": _handle_rollback,
        "snapshot": _handle_snapshot,
        "maintain": _handle_maintain,
        "archive": _handle_archive,
        "nav-index": _handle_nav_index,
        "reconcile": _handle_reconcile,
        "concepts-cleanup": _handle_concepts_cleanup,
        "runtime-migrate": _handle_runtime_migrate,
        "projection-rebuild": _handle_projection_rebuild,
        "query-notes": _handle_query_notes,
        "readiness-audit": _handle_readiness_audit,
    }
    handler = handlers.get(args.subcommand)
    if handler is not None:
        return handler(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())

