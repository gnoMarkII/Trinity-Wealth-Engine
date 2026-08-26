"""Application DTOs for Kanban Context."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KanbanCardDTO:
    card_id: str
    title: str
    column_name: str
    flow: str
    created_at: float
    updated_at: float
    prompt: Optional[str] = None
    job_id: Optional[str] = None
    source_key: Optional[str] = None
    scope: str = "both"
    display_seq: Optional[int] = None
    discord_notify: bool = True
    is_verified: bool = True
