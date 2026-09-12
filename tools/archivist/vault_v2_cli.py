"""Command Line Interface for Obsidian Vault V2 Operations.

Provides entry points for audit, synthetic benchmarking, and future migration commands.
"""
from __future__ import annotations

import argparse
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

    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "audit":
        return _handle_audit(args)
    elif args.subcommand == "benchmark":
        return _handle_benchmark(args)
    elif args.subcommand == "plan":
        return _handle_plan(args)
    elif args.subcommand == "apply":
        return _handle_apply(args)
    elif args.subcommand == "verify":
        return _handle_verify(args)
    elif args.subcommand == "rollback":
        return _handle_rollback(args)
    elif args.subcommand == "snapshot":
        return _handle_snapshot(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())

