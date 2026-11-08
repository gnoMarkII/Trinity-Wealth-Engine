"""Pydantic schemas and DTOs for Six-Condition Task Routing Governance (Chapter 1)."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskRouteRequestDTO(BaseModel):
    """Payload to evaluate pre-flight task execution surface."""
    task_id: str = Field(..., description="Unique task identifier, e.g. 'TASK-101'")
    task_description: str = Field(..., min_length=3, description="Summary of the proposed task")
    work_shape: str = Field(..., description="Canonical shape, e.g. 'LIVE_FINANCIAL_CALCULATION', 'PORTFOLIO_EXPOSURE_AND_HEDGING'")
    has_authoritative_calculator: bool = False
    requires_reconstructability: bool = False
    has_rejection_test_gate: bool = False
    touches_capital_or_policy: bool = False
    is_data_approved_and_sanitized: bool = True
    contains_mnpi_risk: bool = False


class TaskRouteRecordDTO(BaseModel):
    """Immutable audit record justifying the selected execution surface."""
    task_id: str
    task_description: str
    work_shape: str
    selected_surface: str
    authoritative_system: str
    reconstructable_artifact: str
    rejection_gate: str
    reversal_condition: str
    rationale: str
    status: str
    evaluated_at_utc: str

