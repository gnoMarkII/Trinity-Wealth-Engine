"""GET/POST /api/kanban/cards, PUT /api/kanban/move — state เก็บใน SQLite ของ Web UI เอง
ไม่สร้างไฟล์ลง Obsidian Vault (ดู Rev.2 1.1 — Vault ต้องคงความสะอาดเป็น institutional archive)
"""
import uuid
from contextlib import closing
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api import state_db
from api.auth import require_session
from api.schemas import KanbanCardDTO

router = APIRouter(dependencies=[Depends(require_session)])

_VALID_COLUMNS = {"backlog", "approval", "executing", "done"}


class CreateCardRequest(BaseModel):
    title: str
    flow: str = "manager"
    prompt: Optional[str] = None
    scope: str = "both"


class UpdateCardRequest(BaseModel):
    title: str
    prompt: Optional[str] = None
    flow: str
    scope: str = "both"


class MoveCardRequest(BaseModel):
    card_id: str
    column_name: str
    job_id: Optional[str] = None


class CreateCardResponse(BaseModel):
    card: KanbanCardDTO
    created: bool


def _card_to_dto(row) -> KanbanCardDTO:
    return KanbanCardDTO(
        card_id=row["card_id"],
        title=row["title"],
        prompt=row["prompt"],
        column_name=row["column_name"],
        job_id=row["job_id"],
        flow=row["flow"],
        scope=row["scope"],
        display_seq=row["display_seq"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get("/api/kanban/cards", response_model=list[KanbanCardDTO])
def list_cards() -> list[KanbanCardDTO]:
    with closing(state_db.get_connection()) as conn:
        rows = state_db.list_kanban_cards(conn)
    return [_card_to_dto(r) for r in rows]


@router.post("/api/kanban/cards", response_model=CreateCardResponse)
def create_card(payload: CreateCardRequest) -> CreateCardResponse:
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title ว่างเปล่า")
    prompt = (payload.prompt or "").strip() or None

    with closing(state_db.get_connection()) as conn:
        # กันการ์ดซ้ำใน Backlog — ถ้ามีการ์ดชื่อ+prompt เดียวกันอยู่แล้วให้คืนอันเดิม ไม่สร้างซ้ำ
        # (idiom เดียวกับ idempotency ของ JobQueue.dispatch())
        existing = state_db.find_kanban_card_by_title_in_column(conn, title, "backlog", prompt=prompt)
        if existing is not None:
            return CreateCardResponse(card=_card_to_dto(existing), created=False)

        card_id = str(uuid.uuid4())
        state_db.create_kanban_card(
            conn, card_id, title, column_name="backlog", flow=payload.flow, prompt=prompt, scope=payload.scope
        )
        row = state_db.get_kanban_card(conn, card_id)
    return CreateCardResponse(card=_card_to_dto(row), created=True)


@router.patch("/api/kanban/cards/{card_id}", response_model=KanbanCardDTO)
def update_card(card_id: str, payload: UpdateCardRequest) -> KanbanCardDTO:
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title ว่างเปล่า")
    prompt = (payload.prompt or "").strip() or None

    with closing(state_db.get_connection()) as conn:
        existing = state_db.get_kanban_card(conn, card_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="ไม่พบการ์ดนี้")
        if existing["column_name"] != "backlog":
            raise HTTPException(status_code=400, detail="แก้ไขได้เฉพาะการ์ดที่ยังอยู่ใน Backlog เท่านั้น")
        state_db.update_kanban_card(conn, card_id, title=title, prompt=prompt, flow=payload.flow, scope=payload.scope)
        row = state_db.get_kanban_card(conn, card_id)
    return _card_to_dto(row)


@router.put("/api/kanban/move", response_model=KanbanCardDTO)
def move_card(payload: MoveCardRequest) -> KanbanCardDTO:
    if payload.column_name not in _VALID_COLUMNS:
        raise HTTPException(status_code=400, detail=f"column_name ต้องเป็นหนึ่งใน {sorted(_VALID_COLUMNS)}")
    with closing(state_db.get_connection()) as conn:
        existing = state_db.get_kanban_card(conn, payload.card_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="ไม่พบการ์ดนี้")
        state_db.move_kanban_card(conn, payload.card_id, payload.column_name, job_id=payload.job_id)
        row = state_db.get_kanban_card(conn, payload.card_id)
    return _card_to_dto(row)


@router.delete("/api/kanban/cards/{card_id}")
def delete_card(card_id: str) -> dict:
    with closing(state_db.get_connection()) as conn:
        existing = state_db.get_kanban_card(conn, card_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="ไม่พบการ์ดนี้")
        state_db.delete_kanban_card(conn, card_id)
    return {"ok": True}
