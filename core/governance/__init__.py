"""Core governance and routing domain package."""
from core.governance.task_router import (
    ExecutionSurface,
    TaskRouteRecord,
    TaskRoutingContext,
    WorkShape,
    evaluate_task_route,
)

__all__ = [
    "WorkShape",
    "ExecutionSurface",
    "TaskRoutingContext",
    "TaskRouteRecord",
    "evaluate_task_route",
]

