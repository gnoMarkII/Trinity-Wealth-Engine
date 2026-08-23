"""Kanban and Agent Job API Schemas."""
from typing import Any, Optional
from pydantic import BaseModel

class JobStatusDTO(BaseModel):
    job_id: str
    status: str
    card_id: Optional[str] = None
    error_message: Optional[str] = None
    current_node: Optional[str] = None
    interrupt_payload: Optional[dict] = None
    log_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0


class ActiveAgentStatusDTO(BaseModel):
    running: bool
    flow: Optional[str] = None
    node: Optional[str] = None
    job_id: Optional[str] = None


class JobLogEntryDTO(BaseModel):
    seq: int
    node_name: Optional[str] = None
    content: str
    role: str = "reply"
    label: Optional[str] = None


class SpecialistOutputDTO(BaseModel):
    node_name: str
    label: str
    content: str
    seq: int
    created_at: float


class JobOutputsDTO(BaseModel):
    job_id: str
    status: str
    executive_summary: Optional[str] = None
    executive_summary_created_at: Optional[float] = None
    specialists: list[SpecialistOutputDTO] = []
    last_seq: int = 0
    error_message: Optional[str] = None


class KanbanCardDTO(BaseModel):
    card_id: str
    title: str
    prompt: Optional[str] = None
    column_name: str
    job_id: Optional[str] = None
    flow: str = "manager"
    scope: str = "both"
    display_seq: Optional[int] = None
    discord_notify: bool = True
    is_verified: bool = True
    created_at: float
    updated_at: float


# ---------------------------------------------------------
# Actual Portfolio Hub DTOs (Phase 1 & Phase 2)
# ---------------------------------------------------------



class NewsFunnelPendingItemDTO(BaseModel):
    event_id: str
    canonical_title: str
    comprehensive_summary: str = ""
    macro_impact_score: int = 0
    asset_impact_score: int = 0
    extracted_tickers: list[str] = []
    extracted_themes: list[str] = []
    primary_tags: list[str] = []
    links: list[str] = []
    triage_source: Optional[str] = None
    triage_fallback_reason: Optional[str] = None




class NewsFunnelFilteredItemDTO(BaseModel):
    event_id: str
    canonical_title: str
    comprehensive_summary: str = ""
    macro_impact_score: int = 0
    asset_impact_score: int = 0
    extracted_tickers: list[str] = []
    extracted_themes: list[str] = []
    primary_tags: list[str] = []
    links: list[str] = []
    triage_source: Optional[str] = None
    triage_fallback_reason: Optional[str] = None
    status: str
    triage_reasoning: Optional[str] = None
    error_msg: Optional[str] = None
    ingested_at: Optional[str] = None


