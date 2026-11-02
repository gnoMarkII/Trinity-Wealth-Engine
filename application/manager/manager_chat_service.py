"""Application service for Manager Chat orchestration and session lifecycles."""
from __future__ import annotations

import logging
from typing import AsyncIterator, Optional, Sequence
from core.sandbox.models import (
    AdvisorChatMessage,
    AdvisorSessionSummary,
    SandboxStateSnapshot,
)
from application.manager.ports import (
    AdvisorSessionRepoPort,
    AdvisorStreamEnginePort,
    SandboxAdapterPort,
)

log = logging.getLogger(__name__)


class ManagerChatService:
    """Orchestrates conversations with Trinity AI Advisor."""

    def __init__(
        self,
        session_repo: AdvisorSessionRepoPort,
        sandbox_adapter: SandboxAdapterPort,
        stream_engine: AdvisorStreamEnginePort,
    ) -> None:
        self._session_repo = session_repo
        self._sandbox_adapter = sandbox_adapter
        self._stream_engine = stream_engine

    def list_sessions(self, limit: int = 30) -> list[AdvisorSessionSummary]:
        return self._session_repo.list_sessions(limit=limit)

    def create_session(
        self, title: str = "บทสนทนาใหม่", session_id: Optional[str] = None
    ) -> AdvisorSessionSummary:
        return self._session_repo.create_session(title=title, session_id=session_id)

    def get_session(self, session_id: str) -> Optional[AdvisorSessionSummary]:
        return self._session_repo.get_session(session_id)

    def get_messages(self, session_id: str) -> list[AdvisorChatMessage]:
        return self._session_repo.get_messages(session_id)

    async def stream_turn(
        self,
        session_id: str,
        user_message: str,
        context_refs: Sequence[str] = (),
    ) -> AsyncIterator[str]:
        """Record user message, stream model response, and record assistant message on completion."""
        session = self._session_repo.get_session(session_id)
        first_line = user_message.strip().splitlines()[0][:40] if user_message.strip() else "คำถามใหม่"
        if session is None:
            # Auto-create session if it doesn't exist, using requested session_id
            self._session_repo.create_session(title=first_line, session_id=session_id)
        elif session.title == "บทสนทนาใหม่" and first_line:
            # Auto-rename session on first message
            self._session_repo.update_session_title(session_id=session_id, title=first_line)

        # 1. Fetch prior history before recording the new turn
        history = self._session_repo.get_messages(session_id)

        # 2. Record user message
        self._session_repo.append_message(
            session_id=session_id,
            role="user",
            content=user_message,
            citations=context_refs,
        )

        # 3. Stream tokens from inference engine with history and context
        full_response_chunks: list[str] = []
        try:
            async for chunk in self._stream_engine.stream_reply(
                session_id=session_id,
                user_message=user_message,
                context_refs=context_refs,
                history=history,
            ):
                full_response_chunks.append(chunk)
                yield chunk
        finally:
            full_response = "".join(full_response_chunks).strip()
            if full_response:
                self._session_repo.append_message(
                    session_id=session_id,
                    role="assistant",
                    content=full_response,
                    citations=context_refs,
                )

    def get_last_message(self, session_id: str) -> Optional[AdvisorChatMessage]:
        messages = self._session_repo.get_messages(session_id)
        return messages[-1] if messages else None

    def pop_dispatched_jobs(self, session_id: str) -> list[dict[str, str]]:
        if hasattr(self._stream_engine, "pop_dispatched_jobs"):
            return self._stream_engine.pop_dispatched_jobs(session_id)
        return []

    def get_sandbox_state(self) -> SandboxStateSnapshot:
        return self._sandbox_adapter.get_snapshot()

    def promote_skill(self, skill_name: str) -> bool:
        """Promote a quarantined skill to active status."""
        return self._sandbox_adapter.promote_quarantined_skill(skill_name)

