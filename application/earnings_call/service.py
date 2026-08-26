"""Application Layer Service for Earnings Call Saga Orchestration & Outbox Delivery."""
from dataclasses import replace
import re
from typing import Optional

from core.logger import get_logger
from application.earnings_call.dto import (
    EarningsCallSummarizeRequestDTO,
    EarningsCallRunDTO,
    EarningsCallOutboxEventDTO,
    EarningsCallNoteDTO,
    LeaseDTO,
)
from application.earnings_call.errors import (
    EarningsCallValidationError,
    EarningsCallProcessingError,
    EarningsCallRunNotFoundError,
    EarningsCallTickerMismatchError,
)
from application.earnings_call.ports import (
    EarningsCallLlmPort,
    EarningsCallNoteWriterPort,
    EarningsCallKanbanPort,
    EarningsCallWorkflowPort,
)
from application.earnings_call.workflow import (
    EarningsCallRunStatus,
    compute_source_key,
)

log = get_logger(__name__)

_TICKER_RE = re.compile(r"^[A-Za-z0-9\.\-]{1,10}$")
_MAX_TRANSCRIPT_LENGTH = 120_000
_MIN_TRANSCRIPT_LENGTH = 20


class EarningsCallApplicationService:
    """Coordinates earnings call summarization, storage in Obsidian, and resilient Kanban delivery."""

    def __init__(
        self,
        llm_port: EarningsCallLlmPort,
        writer_port: EarningsCallNoteWriterPort,
        workflow_port: EarningsCallWorkflowPort,
        kanban_port: EarningsCallKanbanPort,
        prompt_version: str = "v1",
        max_attempts: int = 5,
    ) -> None:
        self._llm = llm_port
        self._writer = writer_port
        self._workflow = workflow_port
        self._kanban = kanban_port
        self._prompt_version = prompt_version
        self._max_attempts = max_attempts

    def validate_and_normalize(
        self, dto: EarningsCallSummarizeRequestDTO
    ) -> tuple[str, str, str]:
        ticker = (dto.ticker or "").strip().upper()
        if not ticker or not _TICKER_RE.match(ticker):
            raise EarningsCallValidationError(f"Invalid ticker symbol: '{dto.ticker}'")

        period = (dto.period or "").strip()
        if not period or len(period) > 20:
            raise EarningsCallValidationError(f"Invalid period format: '{dto.period}'")

        transcript = (dto.transcript or "").strip()
        if len(transcript) < _MIN_TRANSCRIPT_LENGTH:
            raise EarningsCallValidationError(
                f"Transcript is too short (min {_MIN_TRANSCRIPT_LENGTH} characters required)"
            )
        if len(transcript) > _MAX_TRANSCRIPT_LENGTH:
            raise EarningsCallValidationError(
                f"Transcript exceeds maximum allowed length of {_MAX_TRANSCRIPT_LENGTH:,} characters"
            )

        return ticker, period, transcript

    def summarize_and_store(
        self, request_dto: EarningsCallSummarizeRequestDTO
    ) -> EarningsCallRunDTO:
        ticker, period, transcript = self.validate_and_normalize(request_dto)
        source_key, transcript_hash = compute_source_key(
            canonical_ticker=ticker,
            canonical_period=period,
            transcript=transcript,
            prompt_version=self._prompt_version,
        )

        # 1. Claim run execution lease
        claim = self._workflow.claim_or_resume(
            source_key=source_key,
            ticker=ticker,
            period=period,
            transcript_hash=transcript_hash,
            prompt_version=self._prompt_version,
            lease_seconds=90,
        )

        run = claim.run
        if run.status == EarningsCallRunStatus.COMPLETED:
            log.info("Idempotent replay: Run %s already completed for %s", run.run_id, source_key)
            return replace(run, reused_existing_run=True)

        if not claim.owns_execution:
            log.info("Run %s is in-progress by another execution owner; returning 202 status", run.run_id)
            return run

        execution_token = claim.execution_token or "initial_owner"

        # 2. Generate Highlights via LLM if not yet done
        highlights = run.highlights
        if not highlights:
            self._workflow.renew_execution_lease(run.run_id, execution_token, extension_seconds=90)
            highlights = self._llm.summarize(ticker=ticker, period=period, transcript=transcript)
            if not highlights or not highlights.strip():
                raise EarningsCallProcessingError("LLM returned empty highlights")
            run = self._workflow.save_summary(run.run_id, execution_token, highlights)

        # 3. Write note to Obsidian and enqueue Outbox event
        vault_path = run.vault_path
        if not vault_path:
            vault_path = self._writer.write_note(
                ticker=ticker,
                period=period,
                transcript=transcript,
                highlights=highlights,
            )
            run, event, outbox_lease = self._workflow.record_note_and_enqueue(
                run_id=run.run_id,
                execution_token=execution_token,
                vault_path=vault_path,
                outbox_lease_seconds=60,
            )
        else:
            # Note already written in previous attempt, grab pending outbox event if available
            events = self._workflow.list_pending_outbox(limit=10)
            event = next((e for e in events if e.run_id == run.run_id), None)
            outbox_lease = self._workflow.lease_outbox_event(event.event_id, lease_seconds=60) if event else None

        # 4. Immediate initial delivery via Outbox Lease
        if event and outbox_lease:
            run = self._deliver_outbox_event(run=run, event=event, lease=outbox_lease)

        return run

    def _deliver_outbox_event(
        self, run: EarningsCallRunDTO, event: EarningsCallOutboxEventDTO, lease: LeaseDTO
    ) -> EarningsCallRunDTO:
        """Consumes an outbox event to create/ensure Kanban card with atomicity."""
        try:
            card_res = self._kanban.ensure_card(
                ticker=run.ticker,
                period=run.period,
                vault_relative_path=run.vault_path or "",
                source_key=run.source_key,
            )
            return self._workflow.complete_kanban_delivery(
                run_id=run.run_id,
                event_id=event.event_id,
                lease_token=lease.lease_token,
                card_id=card_res.card_id,
                is_existing=not card_res.created,
            )
        except Exception as exc:
            log.warning("Initial Kanban delivery failed for run %s: %s", run.run_id, exc)
            error_code = "ERR_KANBAN_DELIVERY_FAILED"
            if event.attempts + 1 >= self._max_attempts:
                return self._workflow.mark_terminal_failure(
                    run_id=run.run_id,
                    event_id=event.event_id,
                    lease_token=lease.lease_token,
                    error_code=error_code,
                )
            else:
                retry_delay = min(300, 5 * (2 ** event.attempts))
                return self._workflow.schedule_kanban_retry(
                    run_id=run.run_id,
                    event_id=event.event_id,
                    lease_token=lease.lease_token,
                    error_code=error_code,
                    retry_delay_seconds=retry_delay,
                )

    def get_run_for_ticker(self, ticker: str, run_id: str) -> EarningsCallRunDTO:
        clean_ticker = (ticker or "").strip().upper()
        run = self._workflow.get_run(run_id)
        if not run:
            raise EarningsCallRunNotFoundError(f"Earnings call run '{run_id}' not found")
        if run.ticker != clean_ticker:
            raise EarningsCallTickerMismatchError(
                f"Run '{run_id}' belongs to ticker '{run.ticker}', not '{clean_ticker}'"
            )
        return run

    def retry_run_for_ticker(self, ticker: str, run_id: str) -> EarningsCallRunDTO:
        run = self.get_run_for_ticker(ticker=ticker, run_id=run_id)
        if run.status == EarningsCallRunStatus.COMPLETED:
            return run

        # Reset dead-lettered / pending outbox event for manual retry
        run, event, lease = self._workflow.reset_run_for_manual_retry(run_id=run_id, lease_seconds=60)
        return self._deliver_outbox_event(run=run, event=event, lease=lease)

    def process_outbox_batch(self, limit: int = 10) -> int:
        """Worker batch polling loop: claims expired/pending outbox events and executes delivery."""
        pending_events = self._workflow.list_pending_outbox(limit=limit)
        processed_count = 0

        for event in pending_events:
            lease = self._workflow.lease_outbox_event(event.event_id, lease_seconds=60)
            if not lease:
                continue

            run = self._workflow.get_run(event.run_id)
            if not run:
                continue

            self._deliver_outbox_event(run=run, event=event, lease=lease)
            processed_count += 1

        return processed_count

    def list_earnings_calls(self, ticker: str) -> list[EarningsCallNoteDTO]:
        """Lists all existing earnings call notes and highlights for a ticker from the Obsidian vault."""
        clean_ticker = (ticker or "").strip().upper()
        if not clean_ticker or not _TICKER_RE.match(clean_ticker):
            raise EarningsCallValidationError(f"Invalid ticker symbol: '{ticker}'")
        return self._writer.list_notes_for_ticker(clean_ticker)
