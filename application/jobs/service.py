"""Application Layer Service for Background Jobs & Execution Context."""
from typing import Optional, List, Dict, Any

from application.jobs.ports import JobRepositoryPort
from application.jobs.dto import JobOutputsDTO, SpecialistOutputDTO, JobStatusDTO

_SUMMARY_NODES = ("manager_summary", "supervisor", "synthesize_notebooklm")


class JobApplicationService:
    """Application service for job lifecycle, logging, and status tracking."""

    def __init__(self, repo: JobRepositoryPort):
        self._repo = repo

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._repo.get_job(job_id)

    def get_job_outputs(self, job_id: str) -> Optional[JobOutputsDTO]:
        job = self._repo.get_job(job_id)
        if job is None:
            return None
        reply_logs = self._repo.get_job_reply_logs(job_id)
        summary_row = next((row for row in reversed(reply_logs) if row.get("node_name") == "manager_summary"), None)
        if summary_row is None:
            summary_row = next((row for row in reversed(reply_logs) if row.get("node_name") == "supervisor"), None)
        if summary_row is None:
            summary_row = next(
                (row for row in reversed(reply_logs) if row.get("node_name") in ("synthesize", "synthesize_notebooklm")),
                None,
            )

        latest_by_node = {}
        for row in reply_logs:
            node_name = row.get("node_name") or ""
            if node_name in _SUMMARY_NODES or node_name.startswith(("post_", "prepare_")):
                continue
            latest_by_node[node_name] = row

        specialists = [
            SpecialistOutputDTO(
                node_name=row.get("node_name") or "Specialist",
                label=row.get("label") or row.get("node_name") or "Specialist",
                content=row.get("content") or "",
                seq=row.get("seq", 0),
                created_at=row.get("created_at", 0.0),
            )
            for row in sorted(latest_by_node.values(), key=lambda r: r.get("seq", 0))
        ]

        return JobOutputsDTO(
            job_id=job["job_id"],
            status=job["status"],
            executive_summary=summary_row["content"] if summary_row else None,
            executive_summary_created_at=summary_row["created_at"] if summary_row else None,
            specialists=specialists,
            last_seq=reply_logs[-1]["seq"] if reply_logs else 0,
            error_message=job.get("error_message"),
        )

    def get_job_logs_after(self, job_id: str, after_seq: int) -> List[Dict[str, Any]]:
        return self._repo.get_job_logs_since(job_id, after_seq)

    def claim_resume(
        self,
        job_id: str,
        resume_value_json: str,
        token_uses: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._repo.claim_job_resume(
            job_id=job_id,
            resume_value_json=resume_value_json,
            token_uses=token_uses,
        )

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return the complete status view needed by inbound adapters."""
        job = self._repo.get_job(job_id)
        if job is None:
            return None
        current_node = (
            self._repo.get_latest_job_log_node(job_id)
            if job.get("status") == "running"
            else None
        )
        interrupt_payload = None
        raw_payload = job.get("interrupt_payload")
        if raw_payload:
            import json

            try:
                interrupt_payload = json.loads(raw_payload)
            except (TypeError, ValueError):
                interrupt_payload = None
        return {
            **job,
            "current_node": current_node,
            "interrupt_payload": interrupt_payload,
            "log_count": self._repo.get_job_log_count(job_id),
        }

    def list_statuses(self, statuses: List[str], flows: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self._repo.list_jobs_by_status(statuses, flows=flows)
