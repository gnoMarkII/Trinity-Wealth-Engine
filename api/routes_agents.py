"""POST /api/agents/dispatch, GET /api/agents/stream/{job_id}, POST /api/agents/jobs/{job_id}/resume

SSE ไม่ผูกกับ live run โดยตรง — tail จาก job_logs table เสมอ (Rev.3/5 ข้อ 5)
ปิด/เปิด tab ใหม่ หรือรีเฟรชกลางคัน ยังเห็น log ที่พลาดไปได้ครบ เพราะ log ถูกเขียนลง DB
ก่อนแล้วค่อย stream ออกไป ไม่ใช่ push สดจาก generator ที่หายไปพร้อม connection
"""
import asyncio
import json
from contextlib import closing
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api import state_db
from api.auth import require_session
from api.schemas import ActiveAgentStatusDTO, JobOutputsDTO, JobStatusDTO, SpecialistOutputDTO

router = APIRouter(dependencies=[Depends(require_session)])

_POLL_INTERVAL_SECONDS = 1.0
_TERMINAL_STATUSES = ("done", "error", "awaiting_approval")
_SUMMARY_NODES = ("manager_summary", "supervisor")


def _job_outputs_to_dto(conn, job) -> JobOutputsDTO:
    reply_logs = state_db.get_job_reply_logs(conn, job["job_id"])
    summary_row = next((row for row in reversed(reply_logs) if row["node_name"] == "manager_summary"), None)
    if summary_row is None:
        summary_row = next((row for row in reversed(reply_logs) if row["node_name"] == "supervisor"), None)

    latest_by_node = {}
    for row in reply_logs:
        node_name = row["node_name"] or ""
        if node_name in _SUMMARY_NODES or node_name.startswith(("post_", "prepare_")):
            continue
        latest_by_node[node_name] = row

    specialists = [
        SpecialistOutputDTO(
            node_name=row["node_name"] or "Specialist",
            label=row["label"] or row["node_name"] or "Specialist",
            content=row["content"] or "",
            seq=row["seq"],
            created_at=row["created_at"],
        )
        for row in sorted(latest_by_node.values(), key=lambda row: row["seq"])
    ]

    return JobOutputsDTO(
        job_id=job["job_id"],
        status=job["status"],
        executive_summary=summary_row["content"] if summary_row else None,
        executive_summary_created_at=summary_row["created_at"] if summary_row else None,
        specialists=specialists,
        last_seq=reply_logs[-1]["seq"] if reply_logs else 0,
        error_message=job["error_message"],
    )


class DispatchRequest(BaseModel):
    instruction: str
    card_id: Optional[str] = None
    flow: str = "manager"
    scope: str = "both"


class ResumeRequest(BaseModel):
    approved_news_links: list[str] = []
    approved_youtube_links: list[str] = []


def _job_to_dto(conn, job) -> JobStatusDTO:
    current_node = state_db.get_latest_job_log_node(conn, job["job_id"]) if job["status"] == "running" else None
    interrupt_payload = json.loads(job["interrupt_payload"]) if job["interrupt_payload"] else None
    return JobStatusDTO(
        job_id=job["job_id"],
        status=job["status"],
        card_id=job["card_id"],
        error_message=job["error_message"],
        current_node=current_node,
        interrupt_payload=interrupt_payload,
        log_count=state_db.get_job_log_count(conn, job["job_id"]),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


@router.post("/api/agents/dispatch", response_model=JobStatusDTO)
def dispatch_job(payload: DispatchRequest, request: Request) -> JobStatusDTO:
    if not payload.instruction.strip():
        raise HTTPException(status_code=400, detail="instruction ว่างเปล่า")

    job_queue = request.app.state.job_queue
    job_id = job_queue.dispatch(payload.instruction, payload.card_id, flow=payload.flow, scope=payload.scope)

    with closing(state_db.get_connection()) as conn:
        # ย้ายการ์ดเป็น executing + ผูก job_id ในคำขอเดียวกับ dispatch — กันเคสที่ frontend
        # ยิง PUT /api/kanban/move ตามหลังแล้วล้มเหลว ทำให้การ์ดค้าง backlog ทั้งที่ job รันอยู่จริง
        if payload.card_id is not None:
            existing_card = state_db.get_kanban_card(conn, payload.card_id)
            if existing_card is not None:
                state_db.move_kanban_card(conn, payload.card_id, "executing", job_id=job_id)
        job = state_db.get_job(conn, job_id)
        return _job_to_dto(conn, job)


@router.get("/api/agents/jobs/{job_id}", response_model=JobStatusDTO)
def get_job_status(job_id: str) -> JobStatusDTO:
    with closing(state_db.get_connection()) as conn:
        job = state_db.get_job(conn, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ไม่พบ job นี้")
        return _job_to_dto(conn, job)


@router.get("/api/agents/jobs/{job_id}/outputs", response_model=JobOutputsDTO)
def get_job_outputs(job_id: str) -> JobOutputsDTO:
    with closing(state_db.get_connection()) as conn:
        job = state_db.get_job(conn, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ไม่พบ job นี้")
        return _job_outputs_to_dto(conn, job)


@router.get("/api/agents/active", response_model=ActiveAgentStatusDTO)
def get_active_agent_status() -> ActiveAgentStatusDTO:
    # single-worker queue (api/jobs.py) — จะมี job สถานะ running พร้อมกันได้แค่ 0 หรือ 1 รายการเสมอ
    with closing(state_db.get_connection()) as conn:
        running_jobs = state_db.list_jobs_by_status(conn, ["running"])
        if not running_jobs:
            return ActiveAgentStatusDTO(running=False)
        job = running_jobs[0]
        return ActiveAgentStatusDTO(
            running=True,
            flow=job["flow"],
            node=state_db.get_latest_job_log_node(conn, job["job_id"]),
            job_id=job["job_id"],
        )


@router.post("/api/agents/jobs/{job_id}/resume", response_model=JobStatusDTO)
def resume_job(job_id: str, payload: ResumeRequest, request: Request) -> JobStatusDTO:
    with closing(state_db.get_connection()) as conn:
        job = state_db.get_job(conn, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ไม่พบ job นี้")
        if job["status"] != "awaiting_approval":
            raise HTTPException(status_code=400, detail=f"job นี้ไม่ได้อยู่ในสถานะรอ approve (สถานะปัจจุบัน: {job['status']})")

    resume_value: dict[str, Any] = {
        "approved_news_links": payload.approved_news_links,
        "approved_youtube_links": payload.approved_youtube_links,
    }
    job_queue = request.app.state.job_queue
    job_queue.resume(job_id, resume_value)

    with closing(state_db.get_connection()) as conn:
        job = state_db.get_job(conn, job_id)
        return _job_to_dto(conn, job)


@router.get("/api/agents/stream/{job_id}")
async def stream_job(job_id: str) -> StreamingResponse:
    def _read_next_batch(after_seq: int):
        with closing(state_db.get_connection()) as conn:
            job = state_db.get_job(conn, job_id)
            if job is None:
                return None, []
            logs = state_db.get_job_logs_since(conn, job_id, after_seq)
            return job, logs

    def _read_final_dto():
        with closing(state_db.get_connection()) as conn:
            job = state_db.get_job(conn, job_id)
            return _job_to_dto(conn, job)

    async def event_generator():
        after_seq = 0
        while True:
            job, logs = await asyncio.to_thread(_read_next_batch, after_seq)
            if job is None:
                yield f"event: error\ndata: {json.dumps({'detail': 'job not found'})}\n\n"
                return

            for row in logs:
                after_seq = row["seq"]
                payload = {
                    "node": row["node_name"],
                    "content": row["content"],
                    "role": row["role"],
                    "label": row["label"],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            if job["status"] in _TERMINAL_STATUSES:
                dto = await asyncio.to_thread(_read_final_dto)
                yield f"event: {job['status']}\ndata: {dto.model_dump_json()}\n\n"
                return

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
