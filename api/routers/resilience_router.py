"""REST API router for Scheduled Routine Monitoring & Late Run Detection (Ch. 10)."""
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, Query

from api.auth import require_session
from api.dependencies import get_routine_monitor_service
from api.schemas.resilience import (
    RoutineExecutionRecordDTO,
    RoutineHeartbeatRequestDTO,
    RoutineStatusReportDTO,
)
from application.resilience.routine_monitor_service import RoutineMonitorApplicationService

router = APIRouter(prefix="/api/resilience", tags=["resilience-routines"])


@router.get("/routine-status", response_model=RoutineStatusReportDTO)
def get_routine_status(
    target_date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    user: dict = Depends(require_session),
    service: RoutineMonitorApplicationService = Depends(get_routine_monitor_service),
) -> RoutineStatusReportDTO:
    """Returns freshness report and late/missed run status for the morning review."""
    today_str = target_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report = service.get_morning_routine_freshness(target_date_str=today_str)
    return RoutineStatusReportDTO.model_validate(report)


@router.post("/heartbeat", response_model=RoutineExecutionRecordDTO)
def record_heartbeat(
    payload: RoutineHeartbeatRequestDTO,
    user: dict = Depends(require_session),
    service: RoutineMonitorApplicationService = Depends(get_routine_monitor_service),
) -> RoutineExecutionRecordDTO:
    """Records an automated worker routine execution heartbeat."""
    record = service.record_run_heartbeat(
        execution_id=payload.execution_id,
        routine_id=payload.routine_id,
        scheduled_utc=payload.scheduled_utc,
        actual_start_utc=payload.actual_start_utc,
        completed_utc=payload.completed_utc,
        error_message=payload.error_message,
    )
    return RoutineExecutionRecordDTO.model_validate(record.to_dict())


@router.get("/routine-history", response_model=List[RoutineExecutionRecordDTO])
def get_routine_history(
    user: dict = Depends(require_session),
    service: RoutineMonitorApplicationService = Depends(get_routine_monitor_service),
) -> List[RoutineExecutionRecordDTO]:
    """Retrieves full audit log of all scheduled routine runs."""
    records = service.list_all_routine_records()
    return [RoutineExecutionRecordDTO.model_validate(r.to_dict()) for r in records]
