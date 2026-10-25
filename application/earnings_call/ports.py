"""Outbound Ports for Earnings Call Context."""
from typing import Any, Optional, Protocol

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


class PromiseExtractorPort(Protocol):
    """Abstract port for extracting and resolving forward-looking earnings commitments."""

    def extract_commitments_from_transcript(
        self,
        company: str,
        ticker: str,
        fiscal_quarter: str,
        call_date: str,
        fy_end_month: str,
        open_promises: list[dict],
        transcript_text: str,
    ) -> dict:
        """Extract forward-looking commitments, reiterations, and restatements."""
        ...

    def resolve_promise(
        self,
        promise: dict,
        resolving_quarter: str,
        grace_quarter: Optional[str],
        transcript_text: str,
    ) -> dict:
        """Resolve a specific commitment against subsequent transcript text."""
        ...




    def load_transcript(self, ticker: str, quarter: str) -> Optional[str]:
        ...

    def list_cached(self, ticker: str) -> list[str]:
        ...

    def save_ledger(self, ticker: str, ledger: dict) -> str:
        ...

    def load_ledger(self, ticker: str) -> Optional[dict]:
        ...

class TranscriptScraperPort(Protocol):
    """Abstract port for discovering and fetching earnings call transcripts."""

    def discover_calls(self, ticker: str, limit: int = 8) -> list[dict]:
        ...

    def fetch_transcript(self, call: Any) -> tuple[str, int]:
        ...


class TranscriptVaultPort(Protocol):
    """Abstract port for caching transcripts and storing promise ledgers in Vault."""

    def save_transcript(self, ticker: str, quarter: str, text: str) -> str:
        ...

    def load_transcript(self, ticker: str, quarter: str) -> Optional[str]:
        ...

    def list_cached(self, ticker: str) -> list[str]:
        ...

    def save_ledger(self, ticker: str, ledger: dict) -> str:
        ...

    def load_ledger(self, ticker: str) -> Optional[dict]:
        ...
