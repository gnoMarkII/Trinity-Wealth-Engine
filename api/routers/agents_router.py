"""HTTP adapter for durable Agent jobs.

All persistence is accessed through application services.  The queue object on
``app.state`` is an execution port; it is not used as a database service
locator.
"""
import asyncio
import hashlib
import json
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.auth import require_session
from api.dependencies import get_job_service, get_kanban_service
from api.schemas import ActiveAgentStatusDTO, JobOutputsDTO, JobStatusDTO, SpecialistOutputDTO, UnverifiedDraftSelection
from application.jobs.service import JobApplicationService
from application.kanban.service import KanbanApplicationService

router = APIRouter(dependencies=[Depends(require_session)])

_POLL_INTERVAL_SECONDS = 1.0
_TERMINAL_STATUSES = ("done", "done_with_warnings", "done_with_errors", "error", "awaiting_approval")


class DispatchRequest(BaseModel):
    instruction: str
    card_id: Optional[str] = None
    flow: str = "manager"
    scope: str = "both"


class ResumeRequest(BaseModel):
    approved_news_links: list[str] = []
    approved_youtube_links: list[str] = []
    approved_event_ids: Optional[list[str]] = None
    approved_pitch_ids: Optional[list[str]] = None
    unverified_draft_selections: Optional[list[UnverifiedDraftSelection]] = None
    pitch_presentation_styles: dict[str, str] = {}
    action: Literal["approve", "refresh_sources"] = "approve"


def _status_to_dto(status: dict[str, Any]) -> JobStatusDTO:
    return JobStatusDTO(
        job_id=status["job_id"],
        status=status["status"],
        card_id=status.get("card_id"),
        error_message=status.get("error_message"),
        current_node=status.get("current_node"),
        interrupt_payload=status.get("interrupt_payload"),
        log_count=status.get("log_count", 0),
        created_at=status["created_at"],
        updated_at=status["updated_at"],
    )


def _outputs_to_dto(outputs) -> JobOutputsDTO:
    return JobOutputsDTO(
        job_id=outputs.job_id,
        status=outputs.status,
        executive_summary=outputs.executive_summary,
        executive_summary_created_at=outputs.executive_summary_created_at,
        specialists=[
            SpecialistOutputDTO(
                node_name=item.node_name,
                label=item.label,
                content=item.content,
                seq=item.seq,
                created_at=item.created_at,
            )
            for item in outputs.specialists
        ],
        last_seq=outputs.last_seq,
        error_message=outputs.error_message,
    )


@router.post("/api/agents/dispatch", response_model=JobStatusDTO)
def dispatch_job(
    payload: DispatchRequest,
    request: Request,
    job_service: JobApplicationService = Depends(get_job_service),
    kanban_service: KanbanApplicationService = Depends(get_kanban_service),
) -> JobStatusDTO:
    if not payload.instruction.strip():
        raise HTTPException(status_code=400, detail="instruction ว่างเปล่า")
    job_id = request.app.state.job_queue.dispatch(
        payload.instruction, payload.card_id, flow=payload.flow, scope=payload.scope
    )
    if payload.card_id is not None and kanban_service.get_card(payload.card_id) is not None:
        kanban_service.move_card(payload.card_id, "executing", job_id=job_id)
    status = job_service.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="ไม่พบ job นี้")
    return _status_to_dto(status)


@router.get("/api/agents/jobs/{job_id}", response_model=JobStatusDTO)
def get_job_status(
    job_id: str,
    job_service: JobApplicationService = Depends(get_job_service),
) -> JobStatusDTO:
    status = job_service.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="ไม่พบ job นี้")
    return _status_to_dto(status)


@router.get("/api/agents/jobs/{job_id}/outputs", response_model=JobOutputsDTO)
def get_job_outputs(
    job_id: str,
    job_service: JobApplicationService = Depends(get_job_service),
) -> JobOutputsDTO:
    outputs = job_service.get_job_outputs(job_id)
    if outputs is None:
        raise HTTPException(status_code=404, detail="ไม่พบ job นี้")
    return _outputs_to_dto(outputs)


@router.get("/api/agents/active", response_model=ActiveAgentStatusDTO)
def get_active_agent_status(
    job_service: JobApplicationService = Depends(get_job_service),
) -> ActiveAgentStatusDTO:
    running_jobs = job_service.list_statuses(["running"])
    if not running_jobs:
        return ActiveAgentStatusDTO(running=False)
    job = running_jobs[0]
    return ActiveAgentStatusDTO(
        running=True,
        flow=job.get("flow"),
        node=job_service.get_status(job["job_id"]).get("current_node"),
        job_id=job["job_id"],
    )


@router.post("/api/agents/jobs/{job_id}/resume", response_model=JobStatusDTO)
def resume_job(
    job_id: str,
    payload: ResumeRequest,
    request: Request,
    job_service: JobApplicationService = Depends(get_job_service),
) -> JobStatusDTO:
    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] != "awaiting_approval":
        raise HTTPException(status_code=409, detail="job is no longer awaiting approval")

    interrupt_payload = json.loads(job["interrupt_payload"]) if job.get("interrupt_payload") else {}
    pitches = interrupt_payload.get("pitches") if isinstance(interrupt_payload, dict) else None
    approval_revision = interrupt_payload.get("approval_revision") if isinstance(interrupt_payload, dict) else None
    pitch_by_id = {str(pitch.get("pitch_id")): pitch for pitch in (pitches or []) if isinstance(pitch, dict) and pitch.get("pitch_id")}
    approved_ids = payload.approved_pitch_ids or []
    if len(approved_ids) != len(set(approved_ids)):
        raise HTTPException(status_code=400, detail="duplicate publishable pitch IDs")
    for pitch_id in approved_ids:
        pitch = pitch_by_id.get(pitch_id)
        if not pitch:
            raise HTTPException(status_code=400, detail=f"approved pitch {pitch_id} not found")
        if pitch.get("source_readiness") != "ready":
            raise HTTPException(status_code=400, detail=f"pitch {pitch_id} is not source-ready")

    draft_selections = payload.unverified_draft_selections or []
    draft_ids = [selection.pitch_id for selection in draft_selections]
    if len(draft_ids) != len(set(draft_ids)):
        raise HTTPException(status_code=400, detail="duplicate Unverified Draft pitch IDs")
    if set(approved_ids).intersection(draft_ids):
        raise HTTPException(status_code=400, detail="a pitch cannot be both publishable and an Unverified Draft")
    if payload.action == "refresh_sources":
        if draft_selections or approved_ids:
            raise HTTPException(status_code=400, detail="refresh_sources cannot include pitch selections")
        if int(interrupt_payload.get("source_refresh_attempts", 0) or 0) >= 1:
            raise HTTPException(status_code=409, detail="refresh_sources already attempted")

    token_uses: list[dict[str, str | int]] = []
    if draft_selections:
        from tools.content.provenance_enrichment import verify_eligibility_token

        if not isinstance(approval_revision, int) or approval_revision <= 0:
            raise HTTPException(status_code=409, detail="approval checkpoint has no current Draft revision metadata")
        for selection in draft_selections:
            pitch = pitch_by_id.get(selection.pitch_id)
            if not pitch or not pitch.get("unverified_draft_eligible"):
                raise HTTPException(status_code=400, detail=f"pitch {selection.pitch_id} is not Draft-eligible")
            expected_token = str(pitch.get("unverified_draft_eligibility_token") or "")
            if not expected_token or selection.ack.eligibility_token != expected_token:
                raise HTTPException(status_code=409, detail=f"stale or mismatched token for pitch {selection.pitch_id}")
            claims = verify_eligibility_token(
                selection.ack.eligibility_token,
                expected_job_id=job_id,
                expected_thread_id=str(job["thread_id"]),
                expected_pitch_id=selection.pitch_id,
                expected_revision=approval_revision,
                expected_issue_codes=list(pitch.get("unverified_draft_issue_codes") or []),
            )
            if claims is None:
                raise HTTPException(status_code=400, detail=f"invalid, expired, or mismatched token for pitch {selection.pitch_id}")
            token_uses.append({
                "token_hash": hashlib.sha256(selection.ack.eligibility_token.encode("utf-8")).hexdigest(),
                "jti": claims.jti,
                "thread_id": claims.thread_id,
                "pitch_id": claims.pitch_id,
                "approval_revision": claims.approval_revision,
            })

    resume_value = {
        "approved_news_links": payload.approved_news_links or [],
        "approved_youtube_links": payload.approved_youtube_links or [],
        "approved_event_ids": payload.approved_event_ids or [],
        "approved_pitch_ids": payload.approved_pitch_ids or [],
        "unverified_draft_selections": [selection.model_dump() for selection in draft_selections] or [],
        "pitch_presentation_styles": payload.pitch_presentation_styles or {},
        "action": payload.action,
    }
    try:
        job_service.claim_resume(
            job_id,
            json.dumps(resume_value, ensure_ascii=False),
            token_uses=token_uses,
        )
    except ValueError as exc:
        detail = str(exc)
        status_code = 409 if detail in {"approval_already_claimed", "eligibility_token_already_used"} else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc

    status = job_service.get_status(job_id)
    request.app.state.job_queue.enqueue(job_id)
    return _status_to_dto(status)


@router.get("/api/agents/stream/{job_id}")
async def stream_job(
    job_id: str,
    job_service: JobApplicationService = Depends(get_job_service),
) -> StreamingResponse:
    def read_next(after_seq: int):
        job = job_service.get_job(job_id)
        if job is None:
            return None, []
        return job, job_service.get_job_logs_after(job_id, after_seq)

    async def event_generator():
        after_seq = 0
        while True:
            job, logs = await asyncio.to_thread(read_next, after_seq)
            if job is None:
                yield "event: error\ndata: {\"detail\": \"job not found\"}\n\n"
                return
            for row in logs:
                after_seq = row["seq"]
                yield f"data: {json.dumps({'node': row.get('node_name'), 'content': row.get('content'), 'role': row.get('role'), 'label': row.get('label')}, ensure_ascii=False)}\n\n"
            if job["status"] in _TERMINAL_STATUSES:
                status = await asyncio.to_thread(job_service.get_status, job_id)
                yield f"event: {job['status']}\ndata: {_status_to_dto(status).model_dump_json()}\n\n"
                return
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


__all__ = ["router", "DispatchRequest", "ResumeRequest"]
