"""Compatibility facade for the Agent HTTP router.

The implementation lives in :mod:`api.routers.agents_router`.  This module
keeps the historical import path and the small row-to-DTO helper used by
legacy tests/integrations.
"""
from api.routers.agents_router import router
from api.db.adapters import SqliteJobRepositoryAdapter
from application.jobs.service import JobApplicationService
from api.schemas import JobStatusDTO


def _job_to_dto(conn, job) -> JobStatusDTO:
    """Legacy helper backed by the application job service."""
    service = JobApplicationService(repo=SqliteJobRepositoryAdapter(conn=conn))
    status = service.get_status(job["job_id"])
    if status is None:
        raise ValueError(f"job not found: {job['job_id']}")
    return JobStatusDTO(
        job_id=status["job_id"],
        status=status["status"],
        card_id=status.get("card_id"),
        error_message=status.get("error_message"),
        current_node=status.get("current_node"),
        interrupt_payload=status.get("interrupt_payload"),
        log_count=status.get("log_count", 0),
        created_at=status.get("created_at", 0.0),
        updated_at=status.get("updated_at", 0.0),
    )


__all__ = ["router", "_job_to_dto"]
