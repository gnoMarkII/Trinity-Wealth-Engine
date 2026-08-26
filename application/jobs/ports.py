"""Outbound Ports for Background Jobs & Execution Context."""
from typing import Protocol, Optional, List, Dict, Any


class JobRepositoryPort(Protocol):
    """Abstract port for accessing and modifying Jobs persistence."""

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        ...

    def find_job_by_idempotency_key(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        ...

    def get_job_reply_logs(self, job_id: str) -> List[Dict[str, Any]]:
        ...

    def get_job_logs_since(self, job_id: str, after_seq: int) -> List[Dict[str, Any]]:
        ...

    def claim_job_resume(
        self,
        job_id: str,
        resume_value_json: str,
        token_uses: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        ...

    def get_latest_job_log_node(self, job_id: str) -> Optional[str]:
        ...

    def get_job_log_count(self, job_id: str) -> int:
        ...

    def list_jobs_by_status(self, statuses: List[str], flows: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        ...

    # Lifecycle operations used by the worker application service/UoW.  A
    # connection-bound adapter implements these without committing so a job
    # transition and its Kanban move can share one transaction.
    def create_job(
        self,
        job_id: str,
        thread_id: str,
        card_id: Optional[str],
        idempotency_key: str,
        instruction: str,
        status: str = "queued",
        flow: str = "manager",
        scope: str = "both",
    ) -> None:
        ...

    def update_job_status(self, job_id: str, status: str, error_message: Optional[str] = None) -> None:
        ...

    def cas_job_status(self, job_id: str, old_status: str, new_status: str) -> bool:
        ...

    def set_job_awaiting_approval(self, job_id: str, interrupt_payload_json: str) -> None:
        ...

    def clear_job_resume_value(self, job_id: str) -> None:
        ...

    def append_job_log(
        self,
        job_id: str,
        node_name: str,
        content: str,
        role: str = "reply",
        label: Optional[str] = None,
    ) -> None:
        ...
