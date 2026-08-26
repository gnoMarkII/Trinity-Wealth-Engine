"""Composition root for Jobs Application Context."""
from application.jobs.ports import JobRepositoryPort
from application.jobs.service import JobApplicationService


def build_job_service(
    repo: JobRepositoryPort,
) -> JobApplicationService:
    """Build and wire JobApplicationService with dependencies."""
    return JobApplicationService(repo=repo)
