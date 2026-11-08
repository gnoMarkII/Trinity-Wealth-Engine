"""REST API router for Six-Condition Task Routing Governance (Chapter 1 & Lab 1)."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_session
from api.dependencies import get_task_routing_service
from api.schemas.governance import TaskRouteRecordDTO, TaskRouteRequestDTO
from application.governance.routing_service import TaskRoutingGovernanceService
from core.governance.task_router import TaskRoutingContext, WorkShape

router = APIRouter(prefix="/api/governance", tags=["governance-routing"])


@router.post("/route-task", response_model=TaskRouteRecordDTO)
def route_task(
    payload: TaskRouteRequestDTO,
    user: dict = Depends(require_session),
    service: TaskRoutingGovernanceService = Depends(get_task_routing_service),
) -> TaskRouteRecordDTO:
    """Evaluates the Six-Condition Test and selects the execution surface."""
    try:
        shape = WorkShape(payload.work_shape)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid work_shape: '{payload.work_shape}'. Must be a valid WorkShape enum value.",
        )

    context = TaskRoutingContext(
        has_authoritative_calculator=payload.has_authoritative_calculator,
        requires_reconstructability=payload.requires_reconstructability,
        has_rejection_test_gate=payload.has_rejection_test_gate,
        touches_capital_or_policy=payload.touches_capital_or_policy,
        is_data_approved_and_sanitized=payload.is_data_approved_and_sanitized,
        contains_mnpi_risk=payload.contains_mnpi_risk,
        user_role=str(user.get("role", "analyst") if user and hasattr(user, "get") else "analyst"),
    )

    record = service.route_task(
        task_id=payload.task_id,
        task_description=payload.task_description,
        work_shape=shape,
        context=context,
    )
    return TaskRouteRecordDTO.model_validate(record.to_dict())


@router.get("/routing-ledger", response_model=List[TaskRouteRecordDTO])
def get_routing_ledger(
    user: dict = Depends(require_session),
    service: TaskRoutingGovernanceService = Depends(get_task_routing_service),
) -> List[TaskRouteRecordDTO]:
    """Retrieves immutable audit history of all routed tasks."""
    history = service.list_routing_history()
    return [TaskRouteRecordDTO.model_validate(r.to_dict()) for r in history]

