"""Unit tests for SQLite Thesis DAO."""
from decimal import Decimal
import sqlite3
import pytest

from api.db.connection import migrate_round5_storage
from core.thesis.evidence_taxonomy import EvidenceTier
from core.thesis.models import (
    CatalystEvent,
    FalsifierRule,
    ThesisClaimRecord,
    ThesisExceptionRow,
)
from tools.thesis.thesis_sqlite_dao import (
    acknowledge_exception,
    get_active_claims,
    get_claim_by_id,
    get_claims_as_of,
    get_signoff,
    insert_exception,
    insert_signoff,
    insert_thesis_claim,
    list_pending_exceptions,
    list_signoffs,
)


@pytest.fixture
def memory_db():
    conn = sqlite3.connect(":memory:")
    # Initialize schema
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS thesis_claims (
            claim_id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            version INTEGER NOT NULL,
            prior_claim_id TEXT,
            as_of_utc TEXT NOT NULL,
            claim_statement TEXT NOT NULL,
            falsifiers_json TEXT NOT NULL DEFAULT '[]',
            catalysts_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            evidence_tier TEXT NOT NULL DEFAULT 'DOCUMENTED',
            author_role TEXT NOT NULL DEFAULT 'HUMAN_PM',
            checksum_sha256 TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS thesis_exceptions (
            exception_id TEXT PRIMARY KEY,
            claim_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            detected_at_utc TEXT NOT NULL,
            exception_type TEXT NOT NULL,
            metric_name TEXT,
            expected_value TEXT,
            observed_value TEXT,
            source_citation TEXT,
            human_action TEXT NOT NULL DEFAULT 'REVIEW_REQUIRED',
            status TEXT NOT NULL DEFAULT 'PENDING'
        );
        CREATE TABLE IF NOT EXISTS pm_review_signoffs (
            signoff_id TEXT PRIMARY KEY,
            portfolio_id TEXT NOT NULL DEFAULT 'default',
            review_date TEXT NOT NULL,
            signed_off_by TEXT NOT NULL,
            signed_off_at_utc TEXT NOT NULL,
            acknowledged_exceptions_count INTEGER NOT NULL,
            notes TEXT,
            packet_sha256 TEXT NOT NULL,
            verified_sample_ids_json TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_pm_signoff_port_packet ON pm_review_signoffs(portfolio_id, packet_sha256);
        CREATE INDEX IF NOT EXISTS idx_pm_signoff_port_date ON pm_review_signoffs(portfolio_id, review_date);
        """
    )
    migrate_round5_storage(conn)
    yield conn
    conn.close()


def test_insert_and_get_claim(memory_db):
    falsifier = FalsifierRule(
        falsifier_id="FAL-01",
        metric_name="gross_margin",
        operator="<",
        threshold_value=Decimal("70.0"),
    )
    catalyst = CatalystEvent(
        catalyst_id="CAT-01",
        event_name="Earnings Call",
        expected_date_str="2026-11-20",
    )
    claim = ThesisClaimRecord.create(
        claim_id="TH-NVDA-v1",
        ticker="NVDA",
        version=1,
        as_of_utc="2026-11-01T12:00:00Z",
        claim_statement="Data center margin floor test.",
        falsifiers=[falsifier],
        linked_catalysts=[catalyst],
    )

    insert_thesis_claim(memory_db, claim)

    fetched = get_claim_by_id(memory_db, "TH-NVDA-v1")
    assert fetched is not None
    assert fetched.ticker == "NVDA"
    assert fetched.version == 1
    assert len(fetched.falsifiers) == 1
    assert fetched.falsifiers[0].metric_name == "gross_margin"
    assert fetched.falsifiers[0].threshold_value == Decimal("70.0")
    assert fetched.verify_integrity() is True

    # Active claims list
    active = get_active_claims(memory_db, "NVDA")
    assert len(active) == 1
    assert active[0].claim_id == "TH-NVDA-v1"


def test_time_travel_as_of_query(memory_db):
    claim_v1 = ThesisClaimRecord.create(
        claim_id="TH-NVDA-v1",
        ticker="NVDA",
        version=1,
        as_of_utc="2026-06-01T00:00:00Z",
        claim_statement="Original NVDA thesis.",
    )
    claim_v2 = ThesisClaimRecord.create(
        claim_id="TH-NVDA-v2",
        ticker="NVDA",
        version=2,
        as_of_utc="2026-10-01T00:00:00Z",
        claim_statement="Revised NVDA thesis after Q2 earnings.",
        prior_claim_id="TH-NVDA-v1",
    )
    insert_thesis_claim(memory_db, claim_v1)
    insert_thesis_claim(memory_db, claim_v2)

    # Query as of July 2026 (v2 did not exist yet) -> must return v1
    as_of_july = get_claims_as_of(memory_db, "NVDA", "2026-07-15T00:00:00Z")
    assert len(as_of_july) == 1
    assert as_of_july[0].claim_id == "TH-NVDA-v1"

    # Query as of November 2026 -> must return latest v2
    as_of_nov = get_claims_as_of(memory_db, "NVDA", "2026-11-01T00:00:00Z")
    assert len(as_of_nov) == 1
    assert as_of_nov[0].claim_id == "TH-NVDA-v2"


def test_exceptions_and_acknowledgement(memory_db):
    exc = ThesisExceptionRow(
        exception_id="EXC-100",
        claim_id="TH-NVDA-v1",
        ticker="NVDA",
        detected_at_utc="2026-11-02T10:00:00Z",
        exception_type="FALSIFIER_BREACHED",
        metric_name="gross_margin",
        expected_value="< 70.0 PERCENT",
        observed_value="68.0 PERCENT",
    )
    insert_exception(memory_db, exc)

    pending = list_pending_exceptions(memory_db, "NVDA")
    assert len(pending) == 1
    assert pending[0].exception_id == "EXC-100"
    assert pending[0].status == "PENDING"

    # Acknowledge
    acknowledge_exception(memory_db, "EXC-100")
    assert len(list_pending_exceptions(memory_db, "NVDA")) == 0


def test_pm_review_signoff_flow(memory_db):
    insert_signoff(
        conn=memory_db,
        signoff_id="SIG-20261102",
        review_date="2026-11-02",
        signed_off_by="CHINO_PM",
        signed_off_at_utc="2026-11-02T16:00:00Z",
        acknowledged_exceptions_count=2,
        notes="All margins re-checked; approved holding position.",
        packet_sha256="abc123sha256",
    )

    signoff = get_signoff(memory_db, "2026-11-02")
    assert signoff is not None
    assert signoff["signed_off_by"] == "CHINO_PM"
    assert signoff["acknowledged_exceptions_count"] == 2
    assert signoff["packet_sha256"] == "abc123sha256"

    # Non-existent date
    assert get_signoff(memory_db, "2026-11-03") is None


def test_concurrent_signoff_preserves_canonical_signer(memory_db):
    """P1 Gate (J02): Duplicate/concurrent sign-off preserves initial canonical signer and returns existing row."""
    res1 = insert_signoff(
        conn=memory_db,
        signoff_id="SIG-INITIAL-001",
        review_date="2026-11-06",
        signed_off_by="REVIEWER_A",
        signed_off_at_utc="2026-11-06T08:00:00Z",
        acknowledged_exceptions_count=1,
        notes="First approval",
        packet_sha256="packet_sha_12345",
        portfolio_id="default",
        verified_sample_ids=["SMP-001"],
    )
    assert res1["signoff_id"] == "SIG-INITIAL-001"
    assert res1["signed_off_by"] == "REVIEWER_A"

    # Second concurrent call with different ID and signer for the exact same (portfolio, packet_sha)
    res2 = insert_signoff(
        conn=memory_db,
        signoff_id="SIG-CONCURRENT-002",
        review_date="2026-11-06",
        signed_off_by="REVIEWER_B",
        signed_off_at_utc="2026-11-06T08:00:01Z",
        acknowledged_exceptions_count=1,
        notes="Second approval attempt",
        packet_sha256="packet_sha_12345",
        portfolio_id="default",
        verified_sample_ids=["SMP-001"],
    )
    # Must preserve canonical row: returns SIG-INITIAL-001 and REVIEWER_A
    assert res2["signoff_id"] == "SIG-INITIAL-001"
    assert res2["signed_off_by"] == "REVIEWER_A"
    assert res2["notes"] == "First approval"

    # DB contains exactly 1 row
    cursor = memory_db.execute("SELECT count(*) FROM pm_review_signoffs WHERE packet_sha256 = 'packet_sha_12345'")
    assert cursor.fetchone()[0] == 1


