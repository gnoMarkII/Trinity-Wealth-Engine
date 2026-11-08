"""Pydantic schemas and DTOs for Scheduled Routine Monitoring & Late Run Detection (Chapter 10)."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RoutineStatusReportDTO(BaseModel):
    """Freshness status of an institutional routine."""
    routine_id: str
    scheduled_utc: str
    status: str                          # "ON_TIME", "LATE_RUN", "MISSED_RUN", "FAILED", "IN_PROGRESS"
    is_late_run: bool
    is_missed_run: bool
    delay_minutes: float
    grace_period_minutes: int
    freshness_label: str                 # "FRESH", "LATE_RUN", "MISSED_RUN"


class RoutineHeartbeatRequestDTO(BaseModel):
    """Payload to record an automated background worker heartbeat."""
    execution_id: str = Field(..., description="Unique run ID, e.g. 'RUN-2026-08-06-01'")
    routine_id: str = Field(..., description="Schedule ID, e.g. 'ROUTINE-AM-REVIEW'")
    scheduled_utc: str = Field(..., description="Scheduled target time in UTC")
    actual_start_utc: str = Field(..., description="Actual execution start time in UTC")
    completed_utc: Optional[str] = None
    error_message: Optional[str] = None


class RoutineExecutionRecordDTO(BaseModel):
    """Immutable audit record of an executed routine run."""
    execution_id: str
    routine_id: str
    name: str
    cadence: str
    scheduled_utc: str
    actual_start_utc: str
    completed_utc: Optional[str] = None
    elapsed_seconds: str
    delay_minutes: str
    status: str
    target_artifacts: List[str] = []
    error_message: Optional[str] = None
    is_transcript_free: bool = True

