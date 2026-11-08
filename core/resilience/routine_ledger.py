"""Scheduled Routine Ledger & Missed/Late Run Detection (Chapter 10 & Lab 10).

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies per Hexagonal
Architecture AST Domain Layer Isolation rules. Deterministic math using Decimal only.

Reference:
Claude Code for Hedge Funds, Chapter 10 (pp. 67-73), Laboratory 10 (pp. 92-93):
"Schedule the Preparation, Keep the Decision Human.
Design for missed and late runs:
If a scheduled morning workflow was delayed or missed, flag it prominently rather
than presenting stale data as fresh. A fresh analyst must be able to verify freshness
and rebuild from files alone without the LLM conversation transcript."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class RoutineCadence(str, Enum):
    """The operating cadence of an institutional automation routine."""
    HOURLY = "HOURLY"
    DAILY_MORNING = "DAILY_MORNING"
    PRE_MARKET = "PRE_MARKET"
    POST_CLOSE = "POST_CLOSE"
    WEEKLY = "WEEKLY"


class RoutineExecutionStatus(str, Enum):
    """The four operational routine states plus in-progress."""
    ON_TIME = "ON_TIME"
    LATE_RUN = "LATE_RUN"
    MISSED_RUN = "MISSED_RUN"
    FAILED = "FAILED"
    IN_PROGRESS = "IN_PROGRESS"


@dataclass(frozen=True)
class RoutineSchedule:
    """Operating contract for a scheduled preparation workflow."""
    routine_id: str
    name: str
    cadence: RoutineCadence
    expected_time_utc: str                 # e.g. "06:00:00Z" (HH:MM:SSZ) or full ISO
    grace_period_minutes: int = 30
    target_artifacts: Tuple[str, ...] = ()
    owner: str = "portfolio_manager_hermes"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routine_id": self.routine_id,
            "name": self.name,
            "cadence": self.cadence.value,
            "expected_time_utc": self.expected_time_utc,
            "grace_period_minutes": self.grace_period_minutes,
            "target_artifacts": list(self.target_artifacts),
            "owner": self.owner,
        }


@dataclass(frozen=True)
class RoutineExecutionRecord:
    """Historical record of an executed routine in an append-only routine ledger."""
    execution_id: str
    routine_id: str
    name: str
    cadence: RoutineCadence
    scheduled_utc: str
    actual_start_utc: str
    completed_utc: Optional[str]
    elapsed_seconds: Decimal
    delay_minutes: Decimal
    status: RoutineExecutionStatus
    target_artifacts: Tuple[str, ...]
    error_message: Optional[str] = None
    is_transcript_free: bool = True        # Invariant: Reconstructable without chat logs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "routine_id": self.routine_id,
            "name": self.name,
            "cadence": self.cadence.value,
            "scheduled_utc": self.scheduled_utc,
            "actual_start_utc": self.actual_start_utc,
            "completed_utc": self.completed_utc,
            "elapsed_seconds": str(self.elapsed_seconds),
            "delay_minutes": str(self.delay_minutes),
            "status": self.status.value,
            "target_artifacts": list(self.target_artifacts),
            "error_message": self.error_message,
            "is_transcript_free": self.is_transcript_free,
        }


def _parse_utc_timestamp(ts_str: str) -> Optional[datetime]:
    """Helper to parse standard UTC timestamps."""
    if not ts_str:
        return None
    clean = ts_str.strip().replace("Z", "+00:00")
    if len(clean) == 10 and clean[4] == "-" and clean[7] == "-":
        clean += "T00:00:00+00:00"
    try:
        dt = datetime.fromisoformat(clean)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def evaluate_routine_run(
    execution_id: str,
    schedule: RoutineSchedule,
    scheduled_utc: str,
    actual_start_utc: str,
    completed_utc: Optional[str] = None,
    error_message: Optional[str] = None,
) -> RoutineExecutionRecord:
    """Evaluates an executed run against the schedule and grace period.
    
    Status logic:
    - If error_message is present -> FAILED.
    - If actual_start_utc <= scheduled_utc + grace_period -> ON_TIME.
    - If actual_start_utc > scheduled_utc + grace_period -> LATE_RUN.
    """
    sched_dt = _parse_utc_timestamp(scheduled_utc)
    start_dt = _parse_utc_timestamp(actual_start_utc)
    comp_dt = _parse_utc_timestamp(completed_utc) if completed_utc else None

    # Calculate delay in minutes
    if sched_dt and start_dt:
        delay_sec = (start_dt - sched_dt).total_seconds()
        delay_min = Decimal(str(max(0.0, delay_sec / 60.0))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    else:
        delay_min = Decimal("0.0")

    # Calculate elapsed seconds
    if start_dt and comp_dt:
        elapsed_sec = Decimal(str(max(0.0, (comp_dt - start_dt).total_seconds()))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        elapsed_sec = Decimal("0.00")

    # Determine status
    if error_message:
        status = RoutineExecutionStatus.FAILED
    elif not completed_utc:
        status = RoutineExecutionStatus.IN_PROGRESS
    elif delay_min > Decimal(str(schedule.grace_period_minutes)):
        status = RoutineExecutionStatus.LATE_RUN
    else:
        status = RoutineExecutionStatus.ON_TIME

    return RoutineExecutionRecord(
        execution_id=execution_id,
        routine_id=schedule.routine_id,
        name=schedule.name,
        cadence=schedule.cadence,
        scheduled_utc=scheduled_utc,
        actual_start_utc=actual_start_utc,
        completed_utc=completed_utc,
        elapsed_seconds=elapsed_sec,
        delay_minutes=delay_min,
        status=status,
        target_artifacts=schedule.target_artifacts,
        error_message=error_message,
        is_transcript_free=True,
    )


def detect_missed_run(
    schedule: RoutineSchedule,
    scheduled_utc: str,
    as_of_utc: str,
    last_run: Optional[RoutineExecutionRecord] = None,
) -> RoutineExecutionStatus:
    """Checks whether a scheduled delivery has been missed entirely."""
    sched_dt = _parse_utc_timestamp(scheduled_utc)
    as_of_dt = _parse_utc_timestamp(as_of_utc)

    if not sched_dt or not as_of_dt:
        return RoutineExecutionStatus.ON_TIME

    if last_run and last_run.status in (RoutineExecutionStatus.ON_TIME, RoutineExecutionStatus.LATE_RUN):
        return last_run.status

    # If current time is past scheduled time + 2x grace period with no completed run -> MISSED_RUN
    missed_deadline_seconds = schedule.grace_period_minutes * 60 * 2
    if (as_of_dt - sched_dt).total_seconds() > missed_deadline_seconds:
        return RoutineExecutionStatus.MISSED_RUN

    return RoutineExecutionStatus.IN_PROGRESS

