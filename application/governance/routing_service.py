"""Application service for Six-Condition Task Routing Governance (Chapter 1 & Lab 1).

Architecture Rule (Layer 2 - Application):
Coordinates domain models (WorkShape, ExecutionSurface, evaluate_task_route).
Writes reconstructable routing decision records using atomic file write.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.governance.task_router import (
    ExecutionSurface,
    TaskRouteRecord,
    TaskRoutingContext,
    WorkShape,
    evaluate_task_route,
)

log = logging.getLogger(__name__)


class TaskRoutingGovernanceService:
    """Pre-flight governance service evaluating tasks before dispatch to AI, tools, or PM."""

    def __init__(self, routing_audit_dir: Optional[Path] = None) -> None:
        self._audit_dir = routing_audit_dir or Path("review/routing")
        self._in_memory_records: List[TaskRouteRecord] = []
        self._load_persisted_artifacts()

    def _load_persisted_artifacts(self) -> None:
        """Reloads persisted routing records from disk on service startup."""
        if not self._audit_dir.exists():
            return
        for file in sorted(self._audit_dir.glob("task_route_*.json")):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                rec = TaskRouteRecord(
                    task_id=data["task_id"],
                    task_description=data["task_description"],
                    work_shape=WorkShape(data["work_shape"]),
                    selected_surface=ExecutionSurface(data["selected_surface"]),
                    authoritative_system=data["authoritative_system"],
                    reconstructable_artifact=data["reconstructable_artifact"],
                    rejection_gate=data["rejection_gate"],
                    reversal_condition=data["reversal_condition"],
                    rationale=data["rationale"],
                    status=data.get("status", "STOP"),
                    evaluated_at_utc=data.get("evaluated_at_utc", ""),
                )
                self._in_memory_records.append(rec)
            except Exception as e:
                log.warning("Could not load routing artifact %s: %s", file, e)

    def route_task(
        self,
        task_id: str,
        task_description: str,
        work_shape: WorkShape,
        context: Optional[TaskRoutingContext] = None,
        as_of_utc: Optional[str] = None,
        write_artifact: bool = True,
    ) -> TaskRouteRecord:
        """Evaluates the Six-Condition Test and records an immutable audit trail."""
        effective_ctx = context or TaskRoutingContext()
        record = evaluate_task_route(
            task_id=task_id,
            task_description=task_description,
            work_shape=work_shape,
            context=effective_ctx,
            as_of_utc=as_of_utc,
        )
        self._in_memory_records.append(record)

        if write_artifact:
            self._persist_routing_artifact(record)

        return record

    def list_routing_history(self) -> List[TaskRouteRecord]:
        """Returns the audit log of routed tasks."""
        return list(self._in_memory_records)

    def _persist_routing_artifact(self, record: TaskRouteRecord) -> None:
        """Atomically persists the routing decision record to disk."""
        try:
            self._audit_dir.mkdir(parents=True, exist_ok=True)
            target = self._audit_dir / f"task_route_{record.task_id}.json"
            temp_path = self._audit_dir / f"task_route_{record.task_id}.tmp"
            temp_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
            temp_path.replace(target)
        except Exception as err:
            log.warning("Could not persist routing artifact for %s: %s", record.task_id, err)
