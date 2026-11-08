"""Unit tests for The 10-Question Institutional Readiness Audit (Appendix J)."""
from pathlib import Path
import pytest

from core.resilience.readiness_audit import (
    AuditQuestionSpec,
    AuditVerdict,
    CANONICAL_QUESTIONS,
    InstitutionalReadinessReport,
    QuestionEvaluation,
    ReadinessDecision,
)
from application.resilience.readiness_audit_service import InstitutionalReadinessAuditService


def test_canonical_questions_count():
    assert len(CANONICAL_QUESTIONS) == 10
    nums = [q.question_number for q in CANONICAL_QUESTIONS]
    assert nums == list(range(1, 11))


def test_render_markdown_and_decision():
    evals = [
        QuestionEvaluation(
            question_number=i,
            question_text=f"Question {i}",
            verdict=AuditVerdict.PASS,
            evidence_artifact=f"artifact_{i}",
            notes=f"Note {i}",
            controlling_layer="Core",
        )
        for i in range(1, 11)
    ]

    report = InstitutionalReadinessReport(
        as_of_date="2026-11-05 06:00:00Z",
        decision=ReadinessDecision.READY_FOR_NAMED_STAGE,
        evaluations=tuple(evals),
        sha256_checksum="testchecksum1234567890",
    )

    assert report.pass_count == 10
    assert report.fail_count == 0
    assert report.is_fully_ready is True

    md = report.render_markdown()
    assert "# Institutional Readiness Decision Record" in md
    assert "READY_FOR_NAMED_STAGE" in md
    assert "✅ PASS" in md
    assert "testchecksum1234567890" in md


def test_remediate_decision_on_single_non_critical_failure():
    evals = [
        QuestionEvaluation(
            question_number=i,
            question_text=f"Question {i}",
            verdict=AuditVerdict.PASS,
            evidence_artifact=f"artifact_{i}",
            controlling_layer="Core",
        )
        for i in range(1, 10)
    ]
    # Q10 fails (non-critical)
    evals.append(
        QuestionEvaluation(
            question_number=10,
            question_text="Question 10",
            verdict=AuditVerdict.FAIL,
            evidence_artifact="artifact_10",
            smallest_repair="Fix log sink",
            controlling_layer="Logging",
        )
    )

    report = InstitutionalReadinessReport(
        as_of_date="2026-11-05 06:00:00Z",
        decision=ReadinessDecision.REMEDIATE,
        evaluations=tuple(evals),
    )
    assert report.pass_count == 9
    assert report.fail_count == 1
    assert report.is_fully_ready is False


def test_service_run_readiness_audit_live():
    service = InstitutionalReadinessAuditService()
    report = service.run_readiness_audit(write_artifact=False)
    assert len(report.evaluations) == 10
    assert report.decision != ReadinessDecision.READY_FOR_NAMED_STAGE
    assert next(e for e in report.evaluations if e.question_number == 1).verdict == AuditVerdict.FAIL
    assert next(e for e in report.evaluations if e.question_number == 8).verdict == AuditVerdict.FAIL
    assert len(report.sha256_checksum) == 64

def test_readiness_rejects_dummy_16byte_db(tmp_path):
    from scripts.institutional_readiness_audit import _check_sqlite_db_integrity

    dummy_db = tmp_path / "dummy.sqlite"
    dummy_db.write_bytes(b"SQLite format 3\x00")
    service = InstitutionalReadinessAuditService(
        repo_root=tmp_path,
        db_path=dummy_db,
        db_integrity_checker=lambda: _check_sqlite_db_integrity(dummy_db),
    )
    report = service.run_readiness_audit(write_artifact=False)
    q1 = next(e for e in report.evaluations if e.question_number == 1)
    assert q1.verdict == AuditVerdict.FAIL
    assert "integrity" in q1.notes.lower() or "quick_check" in q1.notes.lower()
    assert report.decision != ReadinessDecision.READY_FOR_NAMED_STAGE

def test_readiness_rejects_empty_manifest(tmp_path):
    import json
    fixtures_dir = tmp_path / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    manifest = fixtures_dir / "manifest_portfolio.json"
    manifest.write_text(json.dumps({
        "state_id": "test_state",
        "cutoff_utc": "2026-11-05T00:00:00Z",
        "immutable_files": [],
        "checksums": {},
    }), encoding="utf-8")
    service = InstitutionalReadinessAuditService(repo_root=tmp_path)
    report = service.run_readiness_audit(write_artifact=False)
    q5 = next(e for e in report.evaluations if e.question_number == 5)
    assert q5.verdict == AuditVerdict.FAIL
    assert "contains zero" in q5.notes or "EMPTY_MANIFEST" in q5.notes
    assert report.decision != ReadinessDecision.READY_FOR_NAMED_STAGE


def test_readiness_rejects_padded_corrupt_db(tmp_path):
    from scripts.institutional_readiness_audit import _check_sqlite_db_integrity

    padded_db = tmp_path / "padded.sqlite"
    padded_db.write_bytes(b"SQLite format 3\x00" + b"\x00" * 4096)
    service = InstitutionalReadinessAuditService(
        repo_root=tmp_path,
        db_path=padded_db,
        db_integrity_checker=lambda: _check_sqlite_db_integrity(padded_db),
    )
    report = service.run_readiness_audit(write_artifact=False)
    q1 = next(e for e in report.evaluations if e.question_number == 1)
    assert q1.verdict == AuditVerdict.FAIL
    assert report.decision != ReadinessDecision.READY_FOR_NAMED_STAGE

def test_readiness_rejects_arbitrary_dict_contract(tmp_path):
    import json
    memories_dir = tmp_path / "memories" / ".system"
    memories_dir.mkdir(parents=True, exist_ok=True)
    contract = memories_dir / "storage_contract.json"
    contract.write_text(json.dumps({"arbitrary": True}), encoding="utf-8")
    service = InstitutionalReadinessAuditService(repo_root=tmp_path)
    report = service.run_readiness_audit(write_artifact=False)
    q5 = next(e for e in report.evaluations if e.question_number == 5)
    assert q5.verdict == AuditVerdict.FAIL
    assert report.decision != ReadinessDecision.READY_FOR_NAMED_STAGE


def test_readiness_rejects_zero_cell_sqlite_db(tmp_path):
    from scripts.institutional_readiness_audit import _check_sqlite_db_integrity

    db_file = tmp_path / "zero_cells.sqlite"
    content = bytearray(4096)
    content[:16] = b"SQLite format 3\x00"
    content[16:18] = (4096).to_bytes(2, "big")
    content[100] = 0x0d
    content[103:105] = (0).to_bytes(2, "big")
    content[105:107] = (4096).to_bytes(2, "big")
    db_file.write_bytes(content)

    service = InstitutionalReadinessAuditService(
        repo_root=tmp_path,
        db_path=db_file,
        db_integrity_checker=lambda: _check_sqlite_db_integrity(db_file),
    )
    report = service.run_readiness_audit(write_artifact=False)
    q1 = next(e for e in report.evaluations if e.question_number == 1)
    assert q1.verdict == AuditVerdict.FAIL
    assert "integrity" in q1.notes.lower() or "quick_check" in q1.notes.lower()

def test_readiness_rejects_required_tables_with_incomplete_columns(tmp_path):
    import sqlite3
    from scripts.institutional_readiness_audit import _check_sqlite_db_integrity

    db_file = tmp_path / "wrong_schema.sqlite"
    conn = sqlite3.connect(db_file)
    try:
        for table in ("jobs", "kanban_cards", "thesis_claims", "pm_review_signoffs"):
            conn.execute(f"CREATE TABLE {table} (id INTEGER)")
        conn.commit()
    finally:
        conn.close()

    ok, notes, _repair = _check_sqlite_db_integrity(db_file)
    assert not ok
    assert "missing required columns" in notes

def test_readiness_requires_frozen_input_rebuild_for_q8():
    service = InstitutionalReadinessAuditService()
    q8 = service._eval_q8_conversation_independent()
    assert q8.verdict == AuditVerdict.FAIL
    assert "frozen-input" in q8.notes

    verified = InstitutionalReadinessAuditService(
        packet_rebuild_checker=lambda: (True, "Persisted frozen inputs rebuilt and packet checksum matched.")
    )
    q8_verified = verified._eval_q8_conversation_independent()
    assert q8_verified.verdict == AuditVerdict.PASS
    assert "checksum matched" in q8_verified.notes

