"""Institutional Readiness Audit CLI Runner (Appendix J).

Usage:
    .venv\\Scripts\\python scripts/institutional_readiness_audit.py

Reference:
Claude Code for Hedge Funds, Appendix J (pp. 136-139):
"A team is ready for the next stage only if it can answer yes to all 10 questions.
One 'no' identifies the next repair. It is not a reason to write a more persuasive prompt."
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from application.resilience.readiness_audit_service import InstitutionalReadinessAuditService
from core.resilience.readiness_audit import AuditVerdict, ReadinessDecision


def _check_sqlite_db_integrity(db_path: Path | None = None) -> tuple[bool, str, str | None]:
    import sqlite3
    from api.config import get_state_db_path

    if db_path is None:
        configured = Path(get_state_db_path())
        db_file = configured if configured.is_absolute() else _PROJECT_ROOT / configured
    else:
        db_file = Path(db_path)

    if not db_file.exists():
        return False, f"Configured database does not exist: {db_file}", "Initialize the configured state database."
    required_columns = {
        "jobs": {"job_id", "thread_id", "status", "created_at", "updated_at"},
        "kanban_cards": {"card_id", "title", "column_name", "created_at", "updated_at", "is_verified"},
        "thesis_claims": {
            "claim_id", "ticker", "version", "as_of_utc", "falsifiers_json",
            "catalysts_json", "status", "evidence_tier", "checksum_sha256",
        },
        "pm_review_signoffs": {
            "signoff_id", "portfolio_id", "review_date", "signed_off_by",
            "signed_off_at_utc", "acknowledged_exceptions_count", "packet_sha256",
            "verified_sample_ids_json",
        },
    }
    required_indexes = {
        "idx_thesis_claims_ticker_status": ("thesis_claims", ("ticker", "status")),
        "idx_pm_signoff_port_packet": ("pm_review_signoffs", ("portfolio_id", "packet_sha256")),
    }
    try:
        db_uri = f"file:{db_file.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            check_rows = conn.execute("PRAGMA quick_check").fetchall()
            if not check_rows or any(row[0] != "ok" for row in check_rows):
                return False, f"SQLite quick_check failed: {[row[0] for row in check_rows]}", "Restore or rebuild the database."
            fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
            if fk_errors:
                return False, f"SQLite foreign_key_check found {len(fk_errors)} violation(s).", "Repair referential integrity."
            tables = {
                row["name"] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            missing_tables = sorted(set(required_columns) - tables)
            if missing_tables:
                return False, f"Database is missing required tables: {missing_tables}", "Apply the approved database migration."
            for table, required in required_columns.items():
                actual = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
                missing_columns = sorted(required - actual)
                if missing_columns:
                    return False, f"{table} is missing required columns: {missing_columns}", "Apply the approved database migration."
                projection = ", ".join(sorted(required))
                conn.execute(f"SELECT {projection} FROM {table} LIMIT 0")

            for index_name, (table, expected_columns) in required_indexes.items():
                index_row = conn.execute(
                    "SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND name=?",
                    (index_name,),
                ).fetchone()
                if index_row is None or index_row["tbl_name"] != table:
                    return False, f"Required index {index_name} on {table} is missing.", "Apply the approved database migration."
                actual_columns = tuple(
                    row["name"] for row in conn.execute(f"PRAGMA index_info({index_name})")
                )
                if actual_columns != expected_columns:
                    return False, f"Index {index_name} has columns {actual_columns}; expected {expected_columns}.", "Repair the index using the approved migration."
                index_list = conn.execute(f"PRAGMA index_list({table})").fetchall()
                index_meta = next((row for row in index_list if row["name"] == index_name), None)
                if index_name == "idx_pm_signoff_port_packet" and (index_meta is None or not index_meta["unique"]):
                    return False, f"Index {index_name} must be unique for canonical approval idempotency.", "Repair the unique approval index."
            user_version = conn.execute("PRAGMA user_version").fetchone()[0]
            return True, (
                f"Read-only SQLite quick_check and foreign_key_check passed; required table projections and indexes verified "
                f"(PRAGMA user_version={user_version})."
            ), None
        finally:
            conn.close()
    except Exception as exc:
        return False, f"Read-only SQLite schema/integrity check failed: {exc}", "Restore or rebuild the configured database."


class _ReadOnlySignoffRepo:
    """Read-only repository adapter for audit-time packet rebuild verification."""

    def __init__(self, db_file: Path) -> None:
        self._db_file = db_file

    def get_signoff(self, review_date: str, portfolio_id: str = "default", packet_sha256: str | None = None):
        import sqlite3

        db_uri = f"file:{self._db_file.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            if packet_sha256:
                row = conn.execute(
                    "SELECT signoff_id, portfolio_id, review_date, signed_off_by, signed_off_at_utc, "
                    "acknowledged_exceptions_count, notes, packet_sha256 "
                    "FROM pm_review_signoffs WHERE portfolio_id = ? AND packet_sha256 = ? LIMIT 1",
                    (portfolio_id, packet_sha256),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT signoff_id, portfolio_id, review_date, signed_off_by, signed_off_at_utc, "
                    "acknowledged_exceptions_count, notes, packet_sha256 "
                    "FROM pm_review_signoffs WHERE portfolio_id = ? AND review_date = ? "
                    "ORDER BY signed_off_at_utc DESC LIMIT 1",
                    (portfolio_id, review_date),
                ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_signoffs(self, limit: int = 30, portfolio_id: str | None = None):
        return []


def _check_packet_rebuild(db_path: Path | None = None) -> tuple[bool, str]:
    import json
    import sqlite3
    from api.config import get_state_db_path
    from application.macro.vintage_service import MacroVintageApplicationService
    from application.manager.pm_review_service import PMReviewApplicationService
    from tools.portfolio.core import get_structured_portfolio_state

    ok_db, db_notes, _ = _check_sqlite_db_integrity(db_path)
    if not ok_db:
        return False, f"Cannot rebuild packet when authoritative state DB is unhealthy: {db_notes}"

    if db_path is None:
        configured = Path(get_state_db_path())
        db_file = configured if configured.is_absolute() else _PROJECT_ROOT / configured
    else:
        db_file = Path(db_path)

    db_uri = f"file:{db_file.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        has_packets_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pm_review_packets'"
        ).fetchone()
        if has_packets_table:
            row = conn.execute(
                "SELECT packet_id, portfolio_id, review_date, packet_sha256, payload_json "
                "FROM pm_review_packets ORDER BY created_at_utc DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                payload = json.loads(row["payload_json"])
                svc = PMReviewApplicationService(signoff_repo=_ReadOnlySignoffRepo(db_file))
                recomputed = svc._compute_packet_checksum(
                    review_date=payload["review_date"],
                    portfolio_delta=payload["portfolio_delta"],
                    macro_delta=payload["macro_delta"],
                    thesis_exceptions=payload["thesis_exceptions"],
                    pending_decisions=payload["pending_decisions"],
                    thesis_coverage=payload.get("thesis_coverage"),
                    quarantined_holdings=payload.get("quarantined_holdings"),
                    review_sample=payload.get("review_sample"),
                    run_quality=payload.get("run_quality", "ON_TIME"),
                    drift_minutes=int(payload.get("drift_minutes", 0)),
                )
                if recomputed == row["packet_sha256"]:
                    return True, (
                        f"Persisted packet {row['packet_id']} ({row['review_date']}) rebuilt from frozen payload "
                        f"and matched SHA-256 ({recomputed[:12]}...)."
                    )
                return False, (
                    f"Persisted packet {row['packet_id']} checksum mismatch: stored {row['packet_sha256']} vs recomputed {recomputed}."
                )
    finally:
        conn.close()

    # Fallback: deterministic read-only rebuild across two independent service instances from frozen state
    ro_repo = _ReadOnlySignoffRepo(db_file)
    fixed_date = "2026-11-05"
    svc_a = PMReviewApplicationService(
        signoff_repo=ro_repo,
        portfolio_reader=lambda portfolio_id="default", refresh_prices=False: get_structured_portfolio_state(
            portfolio_id=portfolio_id, refresh_prices=False
        ),
        macro_vintage_service=MacroVintageApplicationService(),
    )
    svc_b = PMReviewApplicationService(
        signoff_repo=ro_repo,
        portfolio_reader=lambda portfolio_id="default", refresh_prices=False: get_structured_portfolio_state(
            portfolio_id=portfolio_id, refresh_prices=False
        ),
        macro_vintage_service=MacroVintageApplicationService(),
    )
    pkt_a = svc_a.build_morning_packet(portfolio_id="default", review_date=fixed_date)
    pkt_b = svc_b.build_morning_packet(portfolio_id="default", review_date=fixed_date)
    if (
        pkt_a.packet_sha256
        and pkt_a.packet_sha256 == pkt_b.packet_sha256
        and bool(pkt_a.macro_delta.get("vintage_walk"))
    ):
        return True, (
            f"Read-only frozen-state rebuild verified across independent instances for {fixed_date} "
            f"(SHA-256: {pkt_a.packet_sha256[:12]}...)."
        )
    return False, "Read-only packet rebuild failed determinism or frozen-input completeness check."


def main() -> int:
    print("=" * 80)
    print("🏛️  TRINITY WEALTH ENGINE — INSTITUTIONAL READINESS AUDIT (APPENDIX J)")
    print("=" * 80)

    service = InstitutionalReadinessAuditService(
        repo_root=_PROJECT_ROOT,
        db_integrity_checker=_check_sqlite_db_integrity,
        packet_rebuild_checker=_check_packet_rebuild,
    )
    report = service.run_readiness_audit(
        signatory="PORTFOLIO_MANAGER_HERMES",
        scope="RESEARCH_PREPARATION_AND_ADVISORY",
        write_artifact=True,
    )

    print(f"As-Of Date: {report.as_of_date}")
    print(f"Scope:      {report.scope}")
    print(f"Signatory:  {report.signatory}")
    print("-" * 80)
    print(f"{'#':<4} {'Verdict':<10} {'Controlling Layer':<25} {'Question Text'}")
    print("-" * 80)

    for ev in report.evaluations:
        badge = " [PASS] " if ev.verdict == AuditVerdict.PASS else " [FAIL] "
        print(f"Q{ev.question_number:<3} {badge:<10} {ev.controlling_layer:<25} {ev.question_text}")
        if ev.notes:
            print(f"     Artifact: {ev.evidence_artifact}")
            print(f"     Note:     {ev.notes}")
        if ev.smallest_repair:
            print(f"     REPAIR:   {ev.smallest_repair}")

    print("=" * 80)
    print(f"Summary: {report.pass_count}/{len(report.evaluations)} Checks Passed")
    print(f"Decision Record: {report.decision.value}")
    print(f"Integrity SHA-256: {report.sha256_checksum}")
    print(f"Saved Decision Record: review/signoffs/readiness-decision.md")
    print("=" * 80)

    if report.decision == ReadinessDecision.READY_FOR_NAMED_STAGE:
        print("✅ READY FOR STAGE: Institutional research and advisory pipeline fully verified.")
        return 0
    elif report.decision == ReadinessDecision.REMEDIATE:
        print("⚠️  REMEDIATE: Minor non-critical gaps detected. Bounded repairs required.")
        return 1
    else:
        print("❌ STOP: Critical integrity, security, or math invariants failed.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
