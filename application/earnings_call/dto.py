"""Application Layer DTOs for Earnings Call Context."""
from dataclasses import dataclass
from typing import Optional

from application.earnings_call.workflow import (
    EarningsCallRunStatus,
    EarningsCallKanbanStatus,
)


@dataclass(frozen=True)
class EarningsCallSummarizeRequestDTO:
    ticker: str
    period: str
    transcript: str


@dataclass(frozen=True)
class LeaseDTO:
    lease_token: str
    lease_expires_at: float


@dataclass(frozen=True)
class KanbanCardResultDTO:
    card_id: str
    created: bool
    title: str
    status: str = "ok"


@dataclass(frozen=True)
class EarningsCallRunDTO:
    run_id: str
    source_key: str
    ticker: str
    period: str
    transcript_hash: str
    prompt_version: str
    status: EarningsCallRunStatus
    kanban_status: EarningsCallKanbanStatus = EarningsCallKanbanStatus.NONE
    highlights: Optional[str] = None
    vault_path: Optional[str] = None
    kanban_card_id: Optional[str] = None
    execution_token: Optional[str] = None
    execution_expires_at: Optional[float] = None
    attempt_count: int = 0
    last_error_code: Optional[str] = None
    reused_existing_run: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def is_idempotent_replay(self) -> bool:
        """Deprecated alias for reused_existing_run, maintained for backwards compatibility."""
        return self.reused_existing_run


@dataclass(frozen=True)
class ClaimDTO:
    run: EarningsCallRunDTO
    owns_execution: bool
    execution_token: Optional[str] = None
    execution_expires_at: Optional[float] = None


@dataclass(frozen=True)
class EarningsCallOutboxEventDTO:
    event_id: str
    run_id: str
    source_key: str
    event_type: str
    status: str
    attempts: int
    available_at: float
    last_error: Optional[str] = None
    lease_token: Optional[str] = None
    lease_expires_at: Optional[float] = None
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(frozen=True)
class EarningsCallNoteDTO:
    title: str
    ticker: str
    period: str
    vault_path: str
    highlights: str
    date: str
    last_updated: str
    has_full_transcript: bool = True
