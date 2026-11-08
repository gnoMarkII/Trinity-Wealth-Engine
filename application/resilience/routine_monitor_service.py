"""Application service for Scheduled Routine Monitoring & Late Run Detection (Ch. 10 & Lab 10).

Architecture Rule (Layer 2 - Application):
Coordinates domain models (RoutineSchedule, RoutineExecutionRecord, RoutineExecutionStatus).
Provides freshness and late-run metrics to the Morning PM Review packet.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from application.institutional.ports import InstitutionalLedgerPort
from core.resilience.routine_ledger import (
    RoutineCadence,
    RoutineExecutionRecord,
    RoutineExecutionStatus,
    RoutineSchedule,
    detect_missed_run,
    evaluate_routine_run,
)

log = logging.getLogger(__name__)


def _atomic_save_json(path: Path, data: Any) -> None:
    """Atomically persists JSON data using temp file swap to avoid corrupted writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp_{os.getpid()}")
    try:
        content = json.dumps(data, indent=2, ensure_ascii=False)
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        log.warning("Failed to atomically save state to %s: %s", path, exc)


class RoutineMonitorApplicationService:
    """Monitors scheduled routines, records heartbeats, and flags late/missed runs."""

    def __init__(
        self,
        schedules: Optional[Sequence[RoutineSchedule]] = None,
        storage_path: Optional[Path] = None,
        ledger_repository: Optional[InstitutionalLedgerPort] = None,
    ) -> None:
        self._schedules: Dict[str, RoutineSchedule] = {}
        self._execution_history: List[RoutineExecutionRecord] = []
        self._storage_path = Path(storage_path) if storage_path else None
        self._ledger_repository = ledger_repository

        if schedules:
            for s in schedules:
                self._schedules[s.routine_id] = s
        else:
            self._seed_default_schedules()

        if self._ledger_repository is not None:
            self._refresh_repository_history()
        elif self._storage_path and self._storage_path.exists():
            self._load_from_storage()

    @staticmethod
    def _record_from_dict(row: Dict[str, Any]) -> RoutineExecutionRecord:
        return RoutineExecutionRecord(
            execution_id=row["execution_id"],
            routine_id=row["routine_id"],
            name=row["name"],
            cadence=RoutineCadence(row["cadence"]),
            scheduled_utc=row["scheduled_utc"],
            actual_start_utc=row["actual_start_utc"],
            completed_utc=row.get("completed_utc"),
            elapsed_seconds=Decimal(str(row.get("elapsed_seconds", "0.0"))),
            delay_minutes=Decimal(str(row.get("delay_minutes", "0.0"))),
            status=RoutineExecutionStatus(row["status"]),
            target_artifacts=tuple(row.get("target_artifacts", ())),
            error_message=row.get("error_message"),
            is_transcript_free=bool(row.get("is_transcript_free", True)),
        )

    def _refresh_repository_history(self) -> None:
        if self._ledger_repository is None:
            return
        self._execution_history = [
            self._record_from_dict(event["payload"])
            for event in self._ledger_repository.list_routine_events()
            if isinstance(event.get("payload"), dict)
        ]

    def _load_from_storage(self) -> None:
        """Loads historical routine executions from durable storage."""
        if not self._storage_path or not self._storage_path.exists():
            return
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                loaded = []
                for row in raw:
                    if isinstance(row, dict) and "execution_id" in row:
                        loaded.append(
                            RoutineExecutionRecord(
                                execution_id=row["execution_id"],
                                routine_id=row["routine_id"],
                                name=row["name"],
                                cadence=RoutineCadence(row["cadence"]),
                                scheduled_utc=row["scheduled_utc"],
                                actual_start_utc=row["actual_start_utc"],
                                completed_utc=row.get("completed_utc"),
                                elapsed_seconds=Decimal(str(row.get("elapsed_seconds", "0.0"))),
                                delay_minutes=Decimal(str(row.get("delay_minutes", "0.0"))),
                                status=RoutineExecutionStatus(row["status"]),
                                target_artifacts=tuple(row.get("target_artifacts", ())),
                                error_message=row.get("error_message"),
                                is_transcript_free=bool(row.get("is_transcript_free", True)),
                            )
                        )
                self._execution_history = loaded
        except Exception as exc:
            log.warning("Could not load routine history from %s: %s", self._storage_path, exc)

    def _persist_to_storage(self) -> None:
        """Flushes execution history to durable storage."""
        if self._ledger_repository is not None or not self._storage_path:
            return
        data = [r.to_dict() for r in self._execution_history]
        _atomic_save_json(self._storage_path, data)

    def _seed_default_schedules(self) -> None:
        """Seed default daily routines for the Trinity hedge fund engine."""
        defaults = [
            RoutineSchedule(
                routine_id="ROUTINE-AM-REVIEW",
                name="Morning PM Review Synthesis",
                cadence=RoutineCadence.DAILY_MORNING,
                expected_time_utc="06:00:00Z",
                grace_period_minutes=30,
                target_artifacts=("review/morning_packet.json",),
            ),
            RoutineSchedule(
                routine_id="ROUTINE-FRED-MACRO",
                name="Overnight FRED Macro Data Ingest",
                cadence=RoutineCadence.DAILY_MORNING,
                expected_time_utc="05:30:00Z",
                grace_period_minutes=45,
                target_artifacts=("data/macro/macro_vintages.csv",),
            ),
        ]
        for s in defaults:
            self._schedules[s.routine_id] = s

    def record_run_heartbeat(
        self,
        execution_id: str,
        routine_id: str,
        scheduled_utc: str,
        actual_start_utc: str,
        completed_utc: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> RoutineExecutionRecord:
        """Records a routine execution heartbeat into the in-memory ledger."""
        schedule = self._schedules.get(routine_id)
        if not schedule:
            schedule = RoutineSchedule(
                routine_id=routine_id,
                name=routine_id,
                cadence=RoutineCadence.DAILY_MORNING,
                expected_time_utc=scheduled_utc,
            )
            self._schedules[routine_id] = schedule

        record = evaluate_routine_run(
            execution_id=execution_id,
            schedule=schedule,
            scheduled_utc=scheduled_utc,
            actual_start_utc=actual_start_utc,
            completed_utc=completed_utc,
            error_message=error_message,
        )
        if self._ledger_repository is not None:
            event = self._ledger_repository.append_routine_event(
                execution_id=record.execution_id, event_type=record.status.value, payload=record.to_dict()
            )
            record = self._record_from_dict(event["payload"])
        self._execution_history.append(record)
        self._persist_to_storage()
        return record

    def get_morning_routine_freshness(self, target_date_str: str, as_of_utc: Optional[str] = None) -> Dict[str, Any]:
        """Evaluates morning review freshness status for the PM review packet."""
        routine_id = "ROUTINE-AM-REVIEW"
        schedule = self._schedules.get(routine_id) or RoutineSchedule(
            routine_id=routine_id,
            name="Morning PM Review Synthesis",
            cadence=RoutineCadence.DAILY_MORNING,
            expected_time_utc=f"{target_date_str}T06:00:00Z",
            grace_period_minutes=30,
        )

        scheduled_utc = f"{target_date_str}T06:00:00Z"
        curr_utc = as_of_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Find latest completed or in-progress run for this routine and date
        self._refresh_repository_history()
        matching_runs = [
            r for r in self._execution_history
            if r.routine_id == routine_id and r.scheduled_utc.startswith(target_date_str)
        ]
        latest_run = matching_runs[-1] if matching_runs else None

        if latest_run:
            status = latest_run.status
            delay_min = float(latest_run.delay_minutes)
        else:
            status = detect_missed_run(schedule, scheduled_utc, curr_utc)
            delay_min = 0.0

        is_late = status == RoutineExecutionStatus.LATE_RUN
        is_missed = status == RoutineExecutionStatus.MISSED_RUN

        return {
            "routine_id": routine_id,
            "scheduled_utc": scheduled_utc,
            "status": status.value,
            "is_late_run": is_late,
            "is_missed_run": is_missed,
            "delay_minutes": delay_min,
            "grace_period_minutes": schedule.grace_period_minutes,
            "freshness_label": "FRESH" if status == RoutineExecutionStatus.ON_TIME else status.value,
        }

    def list_all_routine_records(self) -> List[RoutineExecutionRecord]:
        """Returns the chronological audit log of all routine executions."""
        self._refresh_repository_history()
        return list(self._execution_history)

