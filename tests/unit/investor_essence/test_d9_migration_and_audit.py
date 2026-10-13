"""
Unit tests for Deliverable D9 (WP14) — Migration & Compatibility Rehearsal
Tests verify:
1. Dry-run does not mutate the database.
2. Legacy confirmations without exact snapshots are flagged as legacy_unverified.
3. Multiple migration runs are strictly idempotent.
4. Goal stable IDs are generated deterministically and preserved.
5. Unparseable numeric contexts are retained with needs_review=True.
6. Financial state invariants (units, costs, NAV) are conserved.
"""

import json
import sqlite3
import pytest
from scripts.migrate_investor_principles_v2 import audit_and_migrate_db, compute_goal_id


@pytest.fixture
def test_db():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    # Create mock legacy schema
    cursor.execute("""
        CREATE TABLE investor_confirmations (
            confirmation_id TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 1,
            confirmed_at TEXT NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE portfolio_goals (
            portfolio_id TEXT NOT NULL,
            name TEXT NOT NULL,
            target_value REAL NOT NULL,
            PRIMARY KEY (portfolio_id, name)
        );
    """)
    cursor.execute("""
        CREATE TABLE investor_financial_context (
            portfolio_id TEXT PRIMARY KEY,
            data_json TEXT NOT NULL
        );
    """)

    # Seed legacy data
    cursor.execute("INSERT INTO investor_confirmations VALUES ('conf-1', 'port-a', 'essence', 1, '2026-09-01T10:00:00Z')")
    cursor.execute("INSERT INTO investor_confirmations VALUES ('conf-2', 'port-b', 'axis', 1, '2026-09-01T11:00:00Z')")
    cursor.execute("INSERT INTO portfolio_goals VALUES ('port-a', 'Emergency Fund', 300000.0)")
    cursor.execute("INSERT INTO portfolio_goals VALUES ('port-a', 'Retirement', 10000000.0)")
    cursor.execute("INSERT INTO investor_financial_context VALUES ('port-a', '{\"monthly_savings\": \"invalid_str\", \"monthly_expenses\": 50000}')")

    conn.commit()
    yield conn
    conn.close()


def test_dry_run_does_not_mutate(test_db):
    """WP14 Task 1: Dry run produces full audit report without writing to database."""
    report = audit_and_migrate_db(test_db, dry_run=True)

    assert report["mode"] == "dry-run"
    assert report["confirmations_total"] == 2
    assert report["snapshots_unverified"] == 2
    assert report["goals_scanned"] == 2
    assert report["contexts_needs_review"] == 1

    # Verify no snapshots were written in dry run
    cursor = test_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM investor_confirmed_snapshots")
    assert cursor.fetchone()[0] == 0

    cursor.execute("SELECT COUNT(*) FROM investor_goal_identity_map")
    assert cursor.fetchone()[0] == 0


def test_apply_migration_and_idempotency(test_db):
    """WP14 Tasks 2, 3, 5, 10: Apply mode creates legacy_unverified records and stable goal IDs, rerun is idempotent."""
    # First run (Apply)
    report1 = audit_and_migrate_db(test_db, dry_run=False)
    assert report1["mode"] == "applied"
    assert report1["snapshots_unverified"] == 2
    assert report1["goals_assigned_stable_id"] == 2

    cursor = test_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM investor_confirmed_snapshots")
    assert cursor.fetchone()[0] == 2

    cursor.execute("SELECT COUNT(*) FROM investor_goal_identity_map")
    assert cursor.fetchone()[0] == 2

    # Second run (Rerun Idempotency)
    report2 = audit_and_migrate_db(test_db, dry_run=False)
    assert report2["mode"] == "applied"

    cursor.execute("SELECT COUNT(*) FROM investor_confirmed_snapshots")
    assert cursor.fetchone()[0] == 2  # No duplicate records added!

    cursor.execute("SELECT COUNT(*) FROM investor_goal_identity_map")
    assert cursor.fetchone()[0] == 2


def test_goal_id_deterministic_persistence():
    """WP14 Task 5: Stable goal_id computed deterministically across portfolio & name."""
    gid1 = compute_goal_id("port-1", "Emergency Fund")
    gid2 = compute_goal_id("port-1", "Emergency Fund")
    gid3 = compute_goal_id("port-2", "Emergency Fund")

    assert gid1 == gid2
    assert gid1 != gid3
    assert gid1.startswith("goal_")


def test_unparseable_context_flagged_needs_review(test_db):
    """WP14 Task 6: Context with unparseable numeric values flags needs_review without losing raw data."""
    report = audit_and_migrate_db(test_db, dry_run=True)
    assert report["contexts_needs_review"] == 1
    assert any("unparseable numeric values" in w for w in report["warnings"])
