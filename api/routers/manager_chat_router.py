"""REST & Clean SSE Streaming Router for Trinity AI Advisor (Manager Chat)."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.auth import require_session
from application.manager.hermes_sync_service import HermesSyncService
from application.manager.manager_chat_service import ManagerChatService
from api.dependencies import get_hermes_sync_service, get_manager_chat_service

router = APIRouter(dependencies=[Depends(require_session)])


class CreateSessionRequest(BaseModel):
    title: Optional[str] = Field(default="บทสนทนาใหม่", description="Session title")


class AdvisorSessionDTO(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class AdvisorChatMessageDTO(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    citations: List[str]
    created_at: str


class AdvisorChatPromptRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt message")
    context_refs: List[str] = Field(default_factory=list, description="Referenced context items (@Portfolio, $Ticker, etc.)")


class SkillManifestDTO(BaseModel):
    name: str
    description: str
    is_quarantined: bool
    path: str
    created_at: str = ""


class SandboxStateSnapshotDTO(BaseModel):
    memory_files: List[str]
    active_skills: List[SkillManifestDTO]
    quarantined_skills: List[SkillManifestDTO]
    scratch_files_count: int
    last_audit_entry: Optional[str] = None


class HermesSyncStatusDTO(BaseModel):
    is_installed: bool
    current_commit: str
    upstream_commit: str
    has_updates: bool
    last_synced_at: Optional[str] = None


@router.get("/sessions", response_model=List[AdvisorSessionDTO])
def list_sessions(
    limit: int = 30,
    chat_service: ManagerChatService = Depends(get_manager_chat_service),
) -> List[AdvisorSessionDTO]:
    """Retrieve chat sessions ordered by latest activity."""
    sessions = chat_service.list_sessions(limit=limit)
    return [
        AdvisorSessionDTO(
            id=s.id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=s.message_count,
        )
        for s in sessions
    ]


@router.post("/sessions", response_model=AdvisorSessionDTO, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: CreateSessionRequest = CreateSessionRequest(),
    chat_service: ManagerChatService = Depends(get_manager_chat_service),
) -> AdvisorSessionDTO:
    """Create a new chat session."""
    session = chat_service.create_session(title=payload.title or "บทสนทนาใหม่")
    return AdvisorSessionDTO(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=session.message_count,
    )


@router.get("/sessions/{session_id}/messages", response_model=List[AdvisorChatMessageDTO])
def get_messages(
    session_id: str,
    chat_service: ManagerChatService = Depends(get_manager_chat_service),
) -> List[AdvisorChatMessageDTO]:
    """Retrieve chat history for a session."""
    session = chat_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = chat_service.get_messages(session_id)
    return [
        AdvisorChatMessageDTO(
            id=m.id,
            session_id=m.session_id,
            role=m.role,
            content=m.content,
            citations=list(m.citations),
            created_at=m.created_at,
        )
        for m in messages
    ]


@router.post("/sessions/{session_id}/stream")
async def stream_chat(
    session_id: str,
    payload: AdvisorChatPromptRequest,
    chat_service: ManagerChatService = Depends(get_manager_chat_service),
) -> StreamingResponse:
    """Stream advisor response turns via clean SSE events without technical telemetry noise."""

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for token_chunk in chat_service.stream_turn(
                session_id=session_id,
                user_message=payload.message,
                context_refs=payload.context_refs,
            ):
                data = json.dumps({"delta": token_chunk}, ensure_ascii=False)
                yield f"event: text_delta\ndata: {data}\n\n"

            # If any research jobs were dispatched during the turn, emit job_dispatched events
            for job in chat_service.pop_dispatched_jobs(session_id):
                job_data = json.dumps(job, ensure_ascii=False)
                yield f"event: job_dispatched\ndata: {job_data}\n\n"

            last_msg = chat_service.get_last_message(session_id)
            done_payload = json.dumps({
                "session_id": session_id,
                "message_id": last_msg.id if last_msg else "",
                "citations": list(last_msg.citations) if last_msg else payload.context_refs,
                "status": "completed",
            }, ensure_ascii=False)
            yield f"event: done\ndata: {done_payload}\n\n"
        except Exception as exc:
            err_data = json.dumps({"error": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {err_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sandbox/state", response_model=SandboxStateSnapshotDTO)
def get_sandbox_state(
    chat_service: ManagerChatService = Depends(get_manager_chat_service),
) -> SandboxStateSnapshotDTO:
    """Retrieve current state of files, skills, and memory in manager_sandbox/."""
    snapshot = chat_service.get_sandbox_state()
    return SandboxStateSnapshotDTO(
        memory_files=list(snapshot.memory_files),
        active_skills=[
            SkillManifestDTO(
                name=s.name,
                description=s.description,
                is_quarantined=s.is_quarantined,
                path=s.path,
                created_at=s.created_at,
            )
            for s in snapshot.active_skills
        ],
        quarantined_skills=[
            SkillManifestDTO(
                name=s.name,
                description=s.description,
                is_quarantined=s.is_quarantined,
                path=s.path,
                created_at=s.created_at,
            )
            for s in snapshot.quarantined_skills
        ],
        scratch_files_count=snapshot.scratch_files_count,
        last_audit_entry=snapshot.last_audit_entry,
    )


@router.get("/hermes/sync-status", response_model=HermesSyncStatusDTO)
def get_hermes_sync_status(
    sync_service: HermesSyncService = Depends(get_hermes_sync_service),
) -> HermesSyncStatusDTO:
    """Check git status of upstream NousResearch/hermes-agent."""
    status_obj = sync_service.get_sync_status()
    return HermesSyncStatusDTO(
        is_installed=status_obj.is_installed,
        current_commit=status_obj.current_commit,
        upstream_commit=status_obj.upstream_commit,
        has_updates=status_obj.has_updates,
        last_synced_at=status_obj.last_synced_at,
    )


@router.post("/hermes/sync")
def trigger_hermes_sync(
    sync_service: HermesSyncService = Depends(get_hermes_sync_service),
) -> dict[str, Any]:
    """Pull upstream updates for NousResearch/hermes-agent."""
    success, output = sync_service.trigger_sync()
    return {"success": success, "output": output}


@router.post("/sandbox/skills/{skill_name}/promote")
def promote_skill(
    skill_name: str,
    chat_service: ManagerChatService = Depends(get_manager_chat_service),
) -> dict[str, Any]:
    """Promote a verified quarantined skill to active status."""
    success = chat_service.promote_skill(skill_name)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot promote skill '{skill_name}': not found in quarantine or failed safety check",
        )
    return {"success": True, "message": f"Skill '{skill_name}' successfully promoted to active"}

