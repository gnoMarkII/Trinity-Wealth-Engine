"""Outbound Ports for Earnings Call Context."""
from typing import Optional, Protocol

from application.earnings_call.dto import (
    ClaimDTO,
    EarningsCallNoteDTO,
    EarningsCallWriteResultDTO,
    EarningsCallOutboxEventDTO,
    EarningsCallRunDTO,
    KanbanCardResultDTO,
    LeaseDTO,
)


class EarningsCallLlmPort(Protocol):
    """Abstract port for summarizing earnings call transcripts."""

    def summarize(self, ticker: str, period: str, transcript: str) -> str:
        ...


class EarningsCallNoteWriterPort(Protocol):
    """Abstract port for writing and reading earnings call markdown notes to/from Obsidian vault."""

    def write_note(
        self, ticker: str, period: str, transcript: str, highlights: str
    ) -> str:
        """Writes note containing both highlights and raw transcript; returns vault-relative path."""
        ...

    def write_note_result(
        self, ticker: str, period: str, transcript: str, highlights: str
    ) -> EarningsCallWriteResultDTO:
        """Optional richer boundary carrying the committed revision reference."""
        ...

    def list_notes_for_ticker(self, ticker: str) -> list[EarningsCallNoteDTO]:
        """Lists all existing earnings call notes for a ticker with parsed highlights."""
        ...


class EarningsCallKanbanPort(Protocol):
    """Abstract port for ensuring a Kanban card exists for an earnings call."""

    def ensure_card(
        self, ticker: str, period: str, vault_relative_path: str, source_key: str
    ) -> KanbanCardResultDTO:
        """Creates or retrieves existing Kanban card; returns typed result."""
        ...


class EarningsCallWorkflowPort(Protocol):
    """Abstract port for managing the Earnings Call Saga state machine & Transactional Outbox."""

    def claim_or_resume(
        self,
        source_key: str,
        ticker: str,
        period: str,
        transcript_hash: str,
        prompt_version: str,
        lease_seconds: int = 60,
    ) -> ClaimDTO:
        """Atomically claims execution lease for a new/existing run."""
        ...

    def renew_execution_lease(
        self, run_id: str, execution_token: str, extension_seconds: int = 60
    ) -> Optional[ClaimDTO]:
        """Extends execution lease before long-running tasks."""
        ...

    def save_summary(
        self, run_id: str, execution_token: str, highlights: str
    ) -> EarningsCallRunDTO:
        """Saves LLM highlights under valid execution lease."""
        ...

    def record_note_and_enqueue(
        self,
        run_id: str,
        execution_token: str,
        vault_path: str,
        revision_ref: Optional[str] = None,
        content_sha256: Optional[str] = None,
        outbox_lease_seconds: int = 60,
    ) -> tuple[EarningsCallRunDTO, EarningsCallOutboxEventDTO, LeaseDTO]:
        """Marks note written and enqueues transactional outbox event with initial lease."""
        ...

    def complete_kanban_delivery(
        self,
        run_id: str,
        event_id: str,
        lease_token: str,
        card_id: str,
        is_existing: bool,
    ) -> EarningsCallRunDTO:
        """Atomically marks run as COMPLETED and outbox event as completed."""
        ...

    def schedule_kanban_retry(
        self,
        run_id: str,
        event_id: str,
        lease_token: str,
        error_code: str,
        retry_delay_seconds: int,
    ) -> EarningsCallRunDTO:
        """Marks run as KANBAN_PENDING and schedules outbox retry with backoff."""
        ...

    def mark_terminal_failure(
        self,
        run_id: str,
        event_id: str,
        lease_token: str,
        error_code: str,
    ) -> EarningsCallRunDTO:
        """Marks run as FAILED and outbox event as dead_letter."""
        ...

    def get_run(self, run_id: str) -> Optional[EarningsCallRunDTO]:
        """Retrieves run by run_id."""
        ...

    def list_pending_outbox(self, limit: int = 10) -> list[EarningsCallOutboxEventDTO]:
        """Fetches pending outbox events or expired leased events."""
        ...

    def lease_outbox_event(
        self, event_id: str, lease_seconds: int = 60
    ) -> Optional[LeaseDTO]:
        """Acquires fencing lease on a pending outbox event."""
        ...

    def reset_run_for_manual_retry(
        self, run_id: str, lease_seconds: int = 60
    ) -> tuple[EarningsCallRunDTO, EarningsCallOutboxEventDTO, LeaseDTO]:
        """Resets dead-lettered outbox event for explicit user-triggered retry."""
        ...
