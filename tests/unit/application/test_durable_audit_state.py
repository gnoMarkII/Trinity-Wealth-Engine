"""Unit tests for Durable State Persistence across In-Memory Services (J10).

Validates that:
- RoutineMonitorApplicationService persists heartbeats and reloads across restarts.
- CatalystApplicationService persists reviewed exceptions, catalysts, and captures across restarts.
- EstimateLedgerApplicationService persists estimate revisions across restarts.
"""
from decimal import Decimal
from pathlib import Path
import pytest

from application.resilience.routine_monitor_service import RoutineMonitorApplicationService
from application.thesis.catalyst_service import CatalystApplicationService
from application.equity.estimate_service import EstimateLedgerApplicationService
from core.equity_analytics.estimate_ledger import (
    EstimateRecord,
    EstimateType,
    RevisionDriver,
)
from core.thesis.catalyst_queue import (
    CatalystClassification,
    CatalystRecord,
    SourceCaptureRecord,
)


def test_routine_monitor_durable_persistence(tmp_path: Path):
    storage_file = tmp_path / "routine_history.json"
    svc1 = RoutineMonitorApplicationService(storage_path=storage_file)

    # Record heartbeat
    rec = svc1.record_run_heartbeat(
        execution_id="EXEC-TEST-001",
        routine_id="ROUTINE-AM-REVIEW",
        scheduled_utc="2026-11-06T06:00:00Z",
        actual_start_utc="2026-11-06T06:05:00Z",
        completed_utc="2026-11-06T06:07:00Z",
    )
    assert rec.status.value == "ON_TIME"
    assert storage_file.exists()

    # Re-instantiate fresh service simulating restart
    svc2 = RoutineMonitorApplicationService(storage_path=storage_file)
    assert len(svc2._execution_history) == 1
    assert svc2._execution_history[0].execution_id == "EXEC-TEST-001"
    assert svc2._execution_history[0].routine_id == "ROUTINE-AM-REVIEW"
    assert svc2._execution_history[0].delay_minutes == Decimal("5.0")


def test_catalyst_service_durable_persistence(tmp_path: Path):
    storage_file = tmp_path / "catalyst_state.json"
    svc1 = CatalystApplicationService(storage_path=storage_file)

    # Register custom capture triggering an exception
    cap = SourceCaptureRecord(
        source_id="TEST-SEC-01",
        entity_id="SYN-NSTAR",
        source_type="SEC_FILING",
        captured_at_utc="2026-08-05T12:00:00Z",
        effective_period="2026Q3",
        source_locator="https://sec.gov/filing123",
        content_snippet="Q3 earnings rescheduled to 2026-11-05",
    )
    svc1.register_capture(cap)

    # Scan and find exception
    exceptions = svc1.scan_exceptions(date_overrides={"CAT-001": "2026-11-05"})
    assert len(exceptions) >= 1
    target_ex = next(e for e in exceptions if e.catalyst_id == "CAT-001")

    # Review and acknowledge exception
    reviewed = svc1.review_exception(
        exception_id=target_ex.exception_id,
        human_assessment="Reschedule acknowledged; negligible NAV impact.",
        reviewer="CHINO_PM",
        date_overrides={"CAT-001": "2026-11-05"},
    )
    assert reviewed is not None
    assert reviewed.status == "ACKNOWLEDGED"
    assert storage_file.exists()

    # Re-instantiate fresh service simulating restart
    svc2 = CatalystApplicationService(storage_path=storage_file)
    reloaded_exceptions = svc2.scan_exceptions(date_overrides={"CAT-001": "2026-11-05"})
    reloaded_target = next(e for e in reloaded_exceptions if e.catalyst_id == "CAT-001")
    assert reloaded_target.status == "ACKNOWLEDGED"
    assert "[CHINO_PM] Reschedule acknowledged" in (reloaded_target.human_assessment or "")


def test_estimate_ledger_durable_persistence(tmp_path: Path):
    storage_file = tmp_path / "estimate_ledger.json"
    svc1 = EstimateLedgerApplicationService(storage_path=storage_file)

    # Append revision
    rev = EstimateRecord(
        estimate_id="EST-0002",
        ticker="SYN-NSTAR",
        period="2026Q3",
        metric_name="revenue",
        estimate_type=EstimateType.FUND_THESIS,
        value=Decimal("1100.0"),
        unit="USD_M",
        as_of_utc="2026-08-10T14:00:00Z",
        reason_code="channel_checks_beat",
        prior_record_id="EST-0001",
        owner="analyst_alpha",
        approval_status="APPROVED",
        evidence_label="SYNTHETIC",
    )
    drivers = [
        RevisionDriver(name="Pricing", delta=Decimal("30.0")),
        RevisionDriver(name="Volume", delta=Decimal("20.0")),
    ]
    bridge = svc1.record_revision(revised_record=rev, drivers=drivers)
    assert bridge.is_mathematically_balanced is True
    assert storage_file.exists()

    # Re-instantiate fresh service simulating restart
    svc2 = EstimateLedgerApplicationService(storage_path=storage_file)
    trail = svc2.list_estimates_for_ticker("SYN-NSTAR")
    assert any(r.estimate_id == "EST-0002" for r in trail)
    reloaded_rev = next(r for r in trail if r.estimate_id == "EST-0002")
    assert reloaded_rev.value == Decimal("1100.0")
    assert reloaded_rev.owner == "analyst_alpha"
