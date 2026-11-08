"""Hexagonal Ports for Manager Chat & Hermes Sandbox."""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional, Protocol, Sequence
from core.sandbox.models import (
    AdvisorChatMessage,
    AdvisorSessionSummary,
    HermesSyncStatus,
    SandboxExecutionResult,
    SandboxStateSnapshot,
)


class SandboxAdapterPort(Protocol):
    """Port for interacting with manager_sandbox/ file boundaries."""
    def get_snapshot(self) -> SandboxStateSnapshot: ...
    def execute_scratch_script(self, filename: str, code: str) -> SandboxExecutionResult: ...
    def update_memory(self, target: str, markdown_content: str) -> None: ...
    def save_skill(self, skill_name: str, skill_content: str) -> bool: ...
    def promote_quarantined_skill(self, skill_name: str) -> bool: ...


class AdvisorSessionRepoPort(Protocol):
    """Port for persisting chat sessions and turns."""
    def list_sessions(self, limit: int = 30) -> list[AdvisorSessionSummary]: ...
    def create_session(self, title: str = "บทสนทนาใหม่", session_id: Optional[str] = None) -> AdvisorSessionSummary: ...
    def update_session_title(self, session_id: str, title: str) -> None: ...
    def get_session(self, session_id: str) -> Optional[AdvisorSessionSummary]: ...
    def get_messages(self, session_id: str) -> list[AdvisorChatMessage]: ...
    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Sequence[str] = (),
    ) -> AdvisorChatMessage: ...


class AdvisorStreamEnginePort(Protocol):
    """Port for streaming advisor model inference."""
    def stream_reply(
        self,
        session_id: str,
        user_message: str,
        context_refs: Sequence[str] = (),
        history: Sequence[AdvisorChatMessage] = (),
    ) -> AsyncIterator[str]: ...


class HermesSyncPort(Protocol):
    """Port for managing NousResearch/hermes-agent upstream repository sync."""
    def get_sync_status(self) -> HermesSyncStatus: ...
    def trigger_sync(self) -> tuple[bool, str]: ...


class PMReviewSignoffRepoPort(Protocol):
    """Port for immutable packet snapshots and daily PM sign-off audit rows."""

    def save_packet_snapshot(
        self, *, packet_id: str, portfolio_id: str, review_date: str,
        cutoff_utc: str, input_fingerprint_sha256: str, policy_version: str,
        packet_sha256: str, payload: Dict[str, Any],
    ) -> Dict[str, Any]: ...
    def get_packet_snapshot(
        self, portfolio_id: str, packet_sha256: str,
    ) -> Optional[Dict[str, Any]]: ...
    def get_signoff(
        self,
        review_date: str,
        portfolio_id: str = "default",
        packet_sha256: Optional[str] = None,
    ) -> Optional[dict[str, Any]]: ...
    def save_signoff(
        self,
        signoff_id: str,
        review_date: str,
        signed_off_by: str,
        signed_off_at_utc: str,
        acknowledged_exceptions_count: int,
        notes: Optional[str],
        packet_sha256: str,
        portfolio_id: str = "default",
        verified_sample_ids: Optional[Sequence[str]] = None,
        acknowledged_exception_ids: Optional[Sequence[str]] = None,
        packet_id: Optional[str] = None,
        request_digest: Optional[str] = None,
        principal_subject_id: Optional[str] = None,
        principal_session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]: ...
    def list_signoffs(self, limit: int = 30, portfolio_id: Optional[str] = None) -> list[dict[str, Any]]: ...
