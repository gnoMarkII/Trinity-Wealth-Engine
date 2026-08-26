"""GET/POST /api/kanban/cards, PUT /api/kanban/move — state เก็บใน SQLite ของ Web UI เอง
ไม่สร้างไฟล์ลง Obsidian Vault (ดู Rev.2 1.1 — Vault ต้องคงความสะอาดเป็น institutional archive)
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_session
from api.dependencies import get_kanban_service
from application.kanban.service import KanbanApplicationService
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


class ToggleDiscordRequest(BaseModel):
    enabled: bool


class CreateCardResponse(BaseModel):
    card: KanbanCardDTO
    created: bool


def _to_api_card(card) -> KanbanCardDTO:
    """Map application DTO to the transport DTO (separate bounded contexts)."""
    return KanbanCardDTO.model_validate(card.__dict__ if hasattr(card, "__dict__") else card)


@router.get("/api/kanban/cards", response_model=list[KanbanCardDTO])
def list_cards(
    service: KanbanApplicationService = Depends(get_kanban_service),
) -> list[KanbanCardDTO]:
    return [_to_api_card(card) for card in service.list_cards()]


@router.post("/api/kanban/cards", response_model=CreateCardResponse)
def create_card(
    payload: CreateCardRequest,
    service: KanbanApplicationService = Depends(get_kanban_service),
) -> CreateCardResponse:
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title ว่างเปล่า")

    card_dto, created = service.create_card(
        title=title,
        flow=payload.flow,
        prompt=payload.prompt,
        scope=payload.scope,
    )
    return CreateCardResponse(card=_to_api_card(card_dto), created=created)


@router.patch("/api/kanban/cards/{card_id}", response_model=KanbanCardDTO)
def update_card(
    card_id: str,
    payload: UpdateCardRequest,
    service: KanbanApplicationService = Depends(get_kanban_service),
) -> KanbanCardDTO:
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="title ว่างเปล่า")

    existing = service.get_card(card_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="ไม่พบการ์ดนี้")
    if existing.column_name != "backlog":
        raise HTTPException(status_code=400, detail="แก้ไขได้เฉพาะการ์ดที่ยังอยู่ใน Backlog เท่านั้น")

    updated = service.update_card(
        card_id=card_id,
        title=title,
        flow=payload.flow,
        prompt=payload.prompt,
        scope=payload.scope,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="ไม่พบการ์ดนี้")
    return _to_api_card(updated)


@router.put("/api/kanban/move", response_model=KanbanCardDTO)
def move_card(
    payload: MoveCardRequest,
    service: KanbanApplicationService = Depends(get_kanban_service),
) -> KanbanCardDTO:
    if payload.column_name not in _VALID_COLUMNS:
        raise HTTPException(status_code=400, detail=f"column_name ต้องเป็นหนึ่งใน {sorted(_VALID_COLUMNS)}")

    updated = service.move_card(card_id=payload.card_id, target_column=payload.column_name, job_id=payload.job_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="ไม่พบการ์ดนี้")
    return _to_api_card(updated)


@router.patch("/api/kanban/cards/{card_id}/discord", response_model=KanbanCardDTO)
def toggle_card_discord(
    card_id: str,
    payload: ToggleDiscordRequest,
    service: KanbanApplicationService = Depends(get_kanban_service),
) -> KanbanCardDTO:
    updated = service.toggle_discord(card_id=card_id, enabled=payload.enabled)
    if updated is None:
        raise HTTPException(status_code=404, detail="ไม่พบการ์ดนี้")
    return _to_api_card(updated)


@router.delete("/api/kanban/cards/{card_id}")
def delete_card(
    card_id: str,
    service: KanbanApplicationService = Depends(get_kanban_service),
) -> dict:
    deleted = service.delete_card(card_id=card_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="ไม่พบการ์ดนี้")
    return {"ok": True}
