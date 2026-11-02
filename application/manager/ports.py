"""Hexagonal Ports for Manager Chat & Hermes Sandbox."""
from __future__ import annotations

from typing import AsyncIterator, Optional, Protocol, Sequence
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

