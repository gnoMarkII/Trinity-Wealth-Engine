"""Unit tests for Scheduled Routine Ledger & Late Run Detection (Chapter 10 & Lab 10)."""
from decimal import Decimal
import pytest

from core.resilience.routine_ledger import (
    RoutineCadence,
    RoutineExecutionRecord,
    RoutineExecutionStatus,
    RoutineSchedule,
    detect_missed_run,
    evaluate_routine_run,
)


def test_chapter10_on_time_routine_execution():
    """Verify on-time routine execution within grace period (e.g. 15 min delay with 30 min grace)."""
    schedule = RoutineSchedule(
        routine_id="ROUTINE-AM-REVIEW",
        name="Morning PM Review Synthesis",
        cadence=RoutineCadence.DAILY_MORNING,
        expected_time_utc="06:00:00Z",
        grace_period_minutes=30,
        target_artifacts=("review/morning_packet.json",),
    )
    rec = evaluate_routine_run(
        execution_id="RUN-001",
        schedule=schedule,
        scheduled_utc="2026-08-06T06:00:00Z",
        actual_start_utc="2026-08-06T06:15:00Z",
        completed_utc="2026-08-06T06:17:30Z",
    )
    assert rec.status == RoutineExecutionStatus.ON_TIME
    assert rec.delay_minutes == Decimal("15.0")
    assert rec.elapsed_seconds == Decimal("150.00")
    assert rec.is_transcript_free is True


def test_chapter10_late_run_flagged_prominently():
    """Book Chapter 10 (p. 70):
    Execution started at 06:45:00Z with 30 min grace period -> LATE_RUN.
    """
    schedule = RoutineSchedule(
        routine_id="ROUTINE-AM-REVIEW",
        name="Morning PM Review Synthesis",
        cadence=RoutineCadence.DAILY_MORNING,
        expected_time_utc="06:00:00Z",
        grace_period_minutes=30,
    )
    rec = evaluate_routine_run(
        execution_id="RUN-002",
        schedule=schedule,
        scheduled_utc="2026-08-06T06:00:00Z",
        actual_start_utc="2026-08-06T06:45:00Z",
        completed_utc="2026-08-06T06:47:00Z",
    )
    assert rec.status == RoutineExecutionStatus.LATE_RUN
    assert rec.delay_minutes == Decimal("45.0")


def test_chapter10_missed_run_detection():
    """Current time is 08:00:00Z for an 06:00:00Z routine with 30 min grace period and no run -> MISSED_RUN."""
    schedule = RoutineSchedule(
        routine_id="ROUTINE-FRED-INGEST",
        name="Macro FRED Ingest",
        cadence=RoutineCadence.DAILY_MORNING,
        expected_time_utc="06:00:00Z",
        grace_period_minutes=30,
    )
    status = detect_missed_run(
        schedule=schedule,
        scheduled_utc="2026-08-06T06:00:00Z",
        as_of_utc="2026-08-06T08:00:00Z",
        last_run=None,
    )
    assert status == RoutineExecutionStatus.MISSED_RUN


def test_routine_run_failure():
    schedule = RoutineSchedule(
        routine_id="ROUTINE-AM-REVIEW",
        name="Morning Review",
        cadence=RoutineCadence.DAILY_MORNING,
        expected_time_utc="06:00:00Z",
    )
    rec = evaluate_routine_run(
        execution_id="RUN-003",
        schedule=schedule,
        scheduled_utc="2026-08-06T06:00:00Z",
        actual_start_utc="2026-08-06T06:05:00Z",
        error_message="Network partition to market data provider",
    )
    assert rec.status == RoutineExecutionStatus.FAILED
    assert rec.error_message is not None

