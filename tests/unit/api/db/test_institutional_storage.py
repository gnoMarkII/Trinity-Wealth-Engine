import sqlite3

import pytest

from api.db.adapters import SqliteInstitutionalLedgerRepositoryAdapter, SqlitePMReviewSignoffRepositoryAdapter
from api.db.connection import init_schema
from tools.thesis import thesis_sqlite_dao


def test_round5_migration_is_idempotent_and_preserves_legacy_signoffs(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE pm_review_signoffs (
            signoff_id TEXT PRIMARY KEY,
            review_date TEXT NOT NULL,
            signed_off_by TEXT NOT NULL,
            signed_off_at_utc TEXT NOT NULL,
            acknowledged_exceptions_count INTEGER NOT NULL,
            notes TEXT,
            packet_sha256 TEXT NOT NULL
        )"""
    )
    legacy = ("SIG-OLD", "2026-10-01", "pm-1", "2026-10-01T09:00:00Z", 2, "legacy", "old-hash")
    conn.execute(
        "INSERT INTO pm_review_signoffs VALUES (?, ?, ?, ?, ?, ?, ?)", legacy
    )
    init_schema(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(pm_review_signoffs)")}
    assert {
        "portfolio_id", "packet_id", "verified_sample_ids_json",
        "acknowledged_exception_ids_json", "request_digest",
        "principal_subject_id", "principal_session_id",
    } <= columns
    row = conn.execute(
        "SELECT signoff_id, signed_off_by, notes, packet_sha256, "
        "verified_sample_ids_json, acknowledged_exception_ids_json "
        "FROM pm_review_signoffs WHERE signoff_id='SIG-OLD'"
    ).fetchone()
    assert row == ("SIG-OLD", "pm-1", "legacy", "old-hash", None, None)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 6
    assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 1
    init_schema(conn)
    assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 1
    conn.close()


def test_packet_snapshot_and_signoff_are_scoped_and_read_back_exactly(tmp_path):
    db_path = str(tmp_path / "pm.sqlite")
    repo = SqlitePMReviewSignoffRepositoryAdapter(db_path=db_path)
    payload = {
        "review_date": "2026-11-07",
        "review_sample": [{"sample_id": "SMP-1"}],
        "thesis_exceptions": [{"exception_id": "EX-1", "status": "PENDING"}],
        "packet_sha256": "hash-1",
    }
    packet = repo.save_packet_snapshot(
        packet_id="PKT-1",
        portfolio_id="growth",
        review_date="2026-11-07",
        cutoff_utc="2026-11-07T02:00:00Z",
        input_fingerprint_sha256="fp-1",
        policy_version="v1",
        packet_sha256="hash-1",
        payload=payload,
    )
    reread = repo.get_packet_snapshot("growth", "hash-1")
    assert packet["revision"] == 1
    assert reread["payload"] == payload

    signoff = repo.save_signoff(
        signoff_id="SIG-1",
        review_date="2026-11-07",
        signed_off_by="principal-1",
        signed_off_at_utc="2026-11-07T02:01:00Z",
        acknowledged_exceptions_count=1,
        notes="reviewed",
        packet_sha256="hash-1",
        portfolio_id="growth",
        verified_sample_ids=["SMP-1"],
        acknowledged_exception_ids=["EX-1"],
        packet_id="PKT-1",
        request_digest="req-digest",
        principal_subject_id="principal-1",
    )
    assert signoff["acknowledged_exception_ids"] == ["EX-1"]
    assert signoff["verified_sample_ids"] == ["SMP-1"]
    assert signoff["request_digest"] == "req-digest"
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with repo._connection(write=True) as conn:
            conn.execute("UPDATE pm_review_packets SET revision=2 WHERE packet_id='PKT-1'")
    with pytest.raises(ValueError, match="same portfolio/date/hash"):
        repo.save_signoff(
            signoff_id="SIG-2",
            review_date="2026-11-07",
            signed_off_by="principal-2",
            signed_off_at_utc="2026-11-07T02:02:00Z",
            acknowledged_exceptions_count=0,
            notes=None,
            packet_sha256="hash-1",
            portfolio_id="other",
            verified_sample_ids=[],
            acknowledged_exception_ids=[],
            packet_id="PKT-1",
        )


def test_institutional_repository_persists_immutable_ledgers_and_sources(tmp_path):
    repo = SqliteInstitutionalLedgerRepositoryAdapter(
        db_path=str(tmp_path / "institutional.sqlite"),
        artifact_root=tmp_path / "artifacts",
    )
    first = repo.append_routine_event(
        "RUN-1", "IN_PROGRESS",
        {"execution_id": "RUN-1", "routine_id": "ROUTINE-AM", "status": "IN_PROGRESS"},
    )
    second = repo.append_routine_event(
        "RUN-1", "ON_TIME",
        {"execution_id": "RUN-1", "routine_id": "ROUTINE-AM", "status": "ON_TIME"},
    )
    assert [first["event_sequence"], second["event_sequence"]] == [1, 2]
    assert len(repo.list_routine_events()) == 2

    catalyst = {
        "catalyst_id": "CAT-1", "entity_id": "NVDA", "event_name": "Earnings",
        "expected_date": "2026-11-20", "source_id": "SRC-1",
    }
    assert repo.save_catalyst_revision(catalyst)["revision"] == 1
    assert repo.save_catalyst_revision(catalyst)["revision"] == 1
    changed = {**catalyst, "expected_date": "2026-11-21"}
    assert repo.save_catalyst_revision(changed)["revision"] == 2
    assert repo.list_latest_catalysts()[0]["payload"]["expected_date"] == "2026-11-21"

    capture = {
        "source_id": "SRC-1", "entity_id": "NVDA", "source_type": "SEC_FILING",
        "captured_at_utc": "2026-11-07T01:00:00Z", "effective_period": "2026Q3",
        "source_locator": "https://example.test/filing", "content_snippet": "verified filing excerpt",
        "revision": "r1",
    }
    stored_capture = repo.capture_source(capture)
    assert stored_capture["content_sha256"]
    assert repo.list_source_captures()[0]["content_snippet"] == "verified filing excerpt"
    with pytest.raises(ValueError, match="hash"):
        repo.capture_source({**capture, "content_snippet": "different body"})

    estimate = {
        "estimate_id": "EST-1", "ticker": "NVDA", "period": "2026Q3",
        "metric_name": "revenue", "estimate_type": "CONSENSUS", "value": "100.00",
        "unit": "USD_M", "as_of_utc": "2026-11-07T01:00:00Z",
    }
    assert repo.save_estimate_record(estimate)["payload"]["value"] == "100.00"
    assert repo.list_estimate_records()[0]["payload"]["estimate_id"] == "EST-1"

    assessment = {
        "idempotency_key": "assessment-key", "catalyst_id": "CAT-1",
        "exception_fingerprint": "exception-fp", "reviewer_subject_id": "pm-1",
        "assessment_text": "reviewed", "payload": {"exception_id": "EX-1"},
    }
    assert repo.save_catalyst_assessment(assessment)["idempotency_key"] == "assessment-key"
    repo.save_catalyst_assessment(assessment)
    assert len(repo.list_catalyst_assessments()) == 1


def test_legacy_import_is_idempotent_and_rejects_same_source_conflicts(tmp_path):
    repo = SqliteInstitutionalLedgerRepositoryAdapter(db_path=str(tmp_path / "import.sqlite"))
    records = [{
        "estimate_id": "EST-LEGACY-1", "ticker": "NVDA", "period": "2026Q3",
        "metric_name": "revenue", "estimate_type": "CONSENSUS", "value": "100",
        "unit": "USD_M", "as_of_utc": "2026-10-01T00:00:00Z",
    }]
    assert repo.import_legacy_records("source-hash", "estimate", records) == {"imported": 1, "skipped": 0}
    assert repo.import_legacy_records("source-hash", "estimate", records) == {"imported": 0, "skipped": 1}
    with pytest.raises(ValueError, match="import conflict"):
        repo.import_legacy_records(
            "source-hash", "estimate", [{**records[0], "value": "101"}]
        )
    assert len(repo.list_estimate_records()) == 1
