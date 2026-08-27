"""Unit tests for EarningsCallApplicationService with Saga Orchestration & Outbox."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock
import pytest

from application.earnings_call.dto import (
    ClaimDTO,
    EarningsCallOutboxEventDTO,
    EarningsCallRunDTO,
    EarningsCallSummarizeRequestDTO,
    KanbanCardResultDTO,
    LeaseDTO,
)
from application.earnings_call.errors import (
    EarningsCallLeaseExpiredError,
    EarningsCallRunInProgressError,
    EarningsCallProcessingError,
    EarningsCallRunNotFoundError,
    EarningsCallRunNotReadyError,
    EarningsCallTickerMismatchError,
    EarningsCallValidationError,
)
from application.earnings_call.service import (
    EarningsCallApplicationService,
)
from application.earnings_call.workflow import (
    EarningsCallKanbanStatus,
    EarningsCallRunStatus,
)


def _make_run(
    run_id="run-1",
    ticker="TSM",
    period="Q4 2024",
    status=EarningsCallRunStatus.NEW,
    kanban_status=EarningsCallKanbanStatus.NONE,
    highlights=None,
    vault_path=None,
    kanban_card_id=None,
    execution_token="exec-tok-1",
    execution_expires_at=9999999999.0,
    attempt_count=1,
    source_key="source-key-1",
):
    return EarningsCallRunDTO(
        run_id=run_id,
        source_key=source_key,
        ticker=ticker,
        period=period,
        transcript_hash="hash-1",
        prompt_version="v1",
        status=status,
        kanban_status=kanban_status,
        highlights=highlights,
        vault_path=vault_path,
        kanban_card_id=kanban_card_id,
        execution_token=execution_token,
        execution_expires_at=execution_expires_at,
        attempt_count=attempt_count,
        created_at=100.0,
        updated_at=100.0,
    )


def test_validate_and_normalize_success():
    service = EarningsCallApplicationService(
        llm_port=MagicMock(),
        writer_port=MagicMock(),
        workflow_port=MagicMock(),
        kanban_port=MagicMock(),
    )
    dto = EarningsCallSummarizeRequestDTO(
        ticker="tsm",
        period=" Q4 2024 ",
        transcript="This is a valid earnings call transcript for TSM with enough length.",
    )
    ticker, period, transcript = service.validate_and_normalize(dto)
    assert ticker == "TSM"
    assert period == "Q4 2024"
    assert "TSM with enough length" in transcript


def test_validate_and_normalize_canonicalizes_period_for_idempotency():
    service = EarningsCallApplicationService(
        llm_port=MagicMock(),
        writer_port=MagicMock(),
        workflow_port=MagicMock(),
        kanban_port=MagicMock(),
    )
    dto = EarningsCallSummarizeRequestDTO(
        ticker="tsm",
        period="  q4   2024 ",
        transcript="This is a valid earnings call transcript for TSM with enough length.",
    )

    ticker, period, transcript = service.validate_and_normalize(dto)

    assert ticker == "TSM"
    assert period == "Q4 2024"
    assert transcript.startswith("This is a valid")


@pytest.mark.parametrize(
    "invalid_ticker",
    ["", "   ", "TOOLONGTICKERNAME", "TSM/INVALID", "TSM$BAD"],
)
def test_validate_invalid_ticker(invalid_ticker):
    service = EarningsCallApplicationService(
        llm_port=MagicMock(),
        writer_port=MagicMock(),
        workflow_port=MagicMock(),
        kanban_port=MagicMock(),
    )
    dto = EarningsCallSummarizeRequestDTO(
        ticker=invalid_ticker,
        period="Q4 2024",
        transcript="This is a valid earnings call transcript for TSM with enough length.",
    )
    with pytest.raises(EarningsCallValidationError, match="Invalid ticker"):
        service.validate_and_normalize(dto)


def test_summarize_and_store_success_path():
    llm_port = MagicMock()
    llm_port.summarize.return_value = "### 1. Financial Highlights\nRevenue beat."

    writer_port = MagicMock()
    writer_port.write_note.return_value = "30_Knowledge_Base/Earnings_Calls/TSM/Q4_2024_TSM_Earnings_Call.md"

    kanban_port = MagicMock()
    kanban_port.ensure_card.return_value = KanbanCardResultDTO(
        card_id="card-123",
        created=True,
        title="[TSM] Earnings Call Q4 2024",
        status="ok",
    )

    initial_run = _make_run(status=EarningsCallRunStatus.NEW)
    summarized_run = _make_run(status=EarningsCallRunStatus.SUMMARIZED, highlights="### 1. Financial Highlights\nRevenue beat.")
    note_written_run = _make_run(
        status=EarningsCallRunStatus.NOTE_WRITTEN,
        highlights="### 1. Financial Highlights\nRevenue beat.",
        vault_path="30_Knowledge_Base/Earnings_Calls/TSM/Q4_2024_TSM_Earnings_Call.md",
    )
    completed_run = _make_run(
        status=EarningsCallRunStatus.COMPLETED,
        kanban_status=EarningsCallKanbanStatus.CREATED,
        highlights="### 1. Financial Highlights\nRevenue beat.",
        vault_path="30_Knowledge_Base/Earnings_Calls/TSM/Q4_2024_TSM_Earnings_Call.md",
        kanban_card_id="card-123",
    )

    outbox_event = EarningsCallOutboxEventDTO(
        event_id="ev-1",
        run_id="run-1",
        source_key="source-key-1",
        event_type="deliver_kanban",
        status="leased",
        attempts=1,
        available_at=100.0,
    )
    lease = LeaseDTO(lease_token="lease-tok-1", lease_expires_at=999999.0)

    workflow_port = MagicMock()
    workflow_port.claim_or_resume.return_value = ClaimDTO(
        run=initial_run,
        owns_execution=True,
        execution_token="exec-tok-1",
        execution_expires_at=999999.0,
    )
    workflow_port.save_summary.return_value = summarized_run
    workflow_port.record_note_and_enqueue.return_value = (note_written_run, outbox_event, lease)
    workflow_port.complete_kanban_delivery.return_value = completed_run

    service = EarningsCallApplicationService(
        llm_port=llm_port,
        writer_port=writer_port,
        workflow_port=workflow_port,
        kanban_port=kanban_port,
    )

    request_dto = EarningsCallSummarizeRequestDTO(
        ticker="tsm",
        period="Q4 2024",
        transcript="Welcome to TSMC fourth quarter 2024 earnings conference call.",
    )

    result = service.summarize_and_store(request_dto)

    assert result.status == EarningsCallRunStatus.COMPLETED
    assert result.kanban_status == EarningsCallKanbanStatus.CREATED
    assert result.kanban_card_id == "card-123"
    assert result.reused_existing_run is False

    llm_port.summarize.assert_called_once()
    writer_port.write_note.assert_called_once()
    kanban_port.ensure_card.assert_called_once()
    workflow_port.complete_kanban_delivery.assert_called_once_with(
        run_id="run-1",
        event_id="ev-1",
        lease_token="lease-tok-1",
        card_id="card-123",
        is_existing=False,
    )


def test_summarize_and_store_idempotent_replay():
    completed_run = _make_run(
        status=EarningsCallRunStatus.COMPLETED,
        kanban_status=EarningsCallKanbanStatus.CREATED,
        highlights="Cached highlights",
        vault_path="30_Knowledge_Base/Earnings_Calls/TSM/Q4_2024_TSM_Earnings_Call.md",
        kanban_card_id="card-existing",
    )

    workflow_port = MagicMock()
    workflow_port.claim_or_resume.return_value = ClaimDTO(
        run=completed_run,
        owns_execution=False,
    )

    llm_port = MagicMock()
    writer_port = MagicMock()
    kanban_port = MagicMock()

    service = EarningsCallApplicationService(
        llm_port=llm_port,
        writer_port=writer_port,
        workflow_port=workflow_port,
        kanban_port=kanban_port,
    )

    request_dto = EarningsCallSummarizeRequestDTO(
        ticker="TSM",
        period="Q4 2024",
        transcript="Welcome to TSMC fourth quarter 2024 earnings conference call.",
    )

    result = service.summarize_and_store(request_dto)

    assert result.status == EarningsCallRunStatus.COMPLETED
    assert result.reused_existing_run is True
    assert result.is_idempotent_replay is True

    # Critical: No LLM, Note Writer, or Kanban side effects on idempotent replay!
    llm_port.summarize.assert_not_called()
    writer_port.write_note.assert_not_called()
    kanban_port.ensure_card.assert_not_called()


def test_summarize_and_store_concurrent_caller_not_owner():
    in_progress_run = _make_run(status=EarningsCallRunStatus.NEW)

    workflow_port = MagicMock()
    workflow_port.claim_or_resume.return_value = ClaimDTO(
        run=in_progress_run,
        owns_execution=False,  # Another thread owns it
    )

    llm_port = MagicMock()
    writer_port = MagicMock()
    kanban_port = MagicMock()

    service = EarningsCallApplicationService(
        llm_port=llm_port,
        writer_port=writer_port,
        workflow_port=workflow_port,
        kanban_port=kanban_port,
    )

    request_dto = EarningsCallSummarizeRequestDTO(
        ticker="TSM",
        period="Q4 2024",
        transcript="Welcome to TSMC fourth quarter 2024 earnings conference call.",
    )

    result = service.summarize_and_store(request_dto)

    assert result.status == EarningsCallRunStatus.NEW
    assert result.reused_existing_run is False
    llm_port.summarize.assert_not_called()


def test_summarize_and_store_kanban_failure_schedules_retry():
    llm_port = MagicMock()
    llm_port.summarize.return_value = "Highlights."

    writer_port = MagicMock()
    writer_port.write_note.return_value = "path/to/note.md"

    kanban_port = MagicMock()
    kanban_port.ensure_card.side_effect = RuntimeError("Kanban SQLite locked")

    initial_run = _make_run(status=EarningsCallRunStatus.NEW)
    summarized_run = _make_run(status=EarningsCallRunStatus.SUMMARIZED, highlights="Highlights.")
    note_written_run = _make_run(status=EarningsCallRunStatus.NOTE_WRITTEN, highlights="Highlights.", vault_path="path/to/note.md")
    pending_run = _make_run(
        status=EarningsCallRunStatus.KANBAN_PENDING,
        kanban_status=EarningsCallKanbanStatus.PENDING,
        highlights="Highlights.",
        vault_path="path/to/note.md",
    )

    outbox_event = EarningsCallOutboxEventDTO(
        event_id="ev-1",
        run_id="run-1",
        source_key="source-key-1",
        event_type="deliver_kanban",
        status="leased",
        attempts=1,
        available_at=100.0,
    )
    lease = LeaseDTO(lease_token="lease-tok-1", lease_expires_at=999999.0)

    workflow_port = MagicMock()
    workflow_port.claim_or_resume.return_value = ClaimDTO(run=initial_run, owns_execution=True, execution_token="exec-1")
    workflow_port.save_summary.return_value = summarized_run
    workflow_port.record_note_and_enqueue.return_value = (note_written_run, outbox_event, lease)
    workflow_port.schedule_kanban_retry.return_value = pending_run

    service = EarningsCallApplicationService(
        llm_port=llm_port,
        writer_port=writer_port,
        workflow_port=workflow_port,
        kanban_port=kanban_port,
    )

    request_dto = EarningsCallSummarizeRequestDTO(
        ticker="TSM",
        period="Q4 2024",
        transcript="Welcome to TSMC fourth quarter 2024 earnings conference call.",
    )

    result = service.summarize_and_store(request_dto)

    assert result.status == EarningsCallRunStatus.KANBAN_PENDING
    assert result.kanban_status == EarningsCallKanbanStatus.PENDING
    workflow_port.schedule_kanban_retry.assert_called_once()


def test_empty_llm_output_raises_processing_error():
    llm_port = MagicMock()
    llm_port.summarize.return_value = "   "  # Empty output

    workflow_port = MagicMock()
    workflow_port.claim_or_resume.return_value = ClaimDTO(
        run=_make_run(status=EarningsCallRunStatus.NEW),
        owns_execution=True,
        execution_token="exec-1",
    )

    service = EarningsCallApplicationService(
        llm_port=llm_port,
        writer_port=MagicMock(),
        workflow_port=workflow_port,
        kanban_port=MagicMock(),
    )

    request_dto = EarningsCallSummarizeRequestDTO(
        ticker="TSM",
        period="Q4 2024",
        transcript="Welcome to TSMC fourth quarter 2024 earnings conference call.",
    )

    with pytest.raises(EarningsCallProcessingError, match="empty highlights"):
        service.summarize_and_store(request_dto)


def test_execution_lease_loss_blocks_external_work():
    """A caller that loses its execution lease must not invoke LLM or writer."""
    workflow_port = MagicMock()
    workflow_port.claim_or_resume.return_value = ClaimDTO(
        run=_make_run(status=EarningsCallRunStatus.NEW),
        owns_execution=True,
        execution_token="exec-1",
    )
    workflow_port.renew_execution_lease.return_value = None

    llm_port = MagicMock()
    writer_port = MagicMock()
    service = EarningsCallApplicationService(
        llm_port=llm_port,
        writer_port=writer_port,
        workflow_port=workflow_port,
        kanban_port=MagicMock(),
    )

    request_dto = EarningsCallSummarizeRequestDTO(
        ticker="TSM",
        period="Q4 2024",
        transcript="Welcome to TSMC fourth quarter 2024 earnings conference call.",
    )

    with pytest.raises(EarningsCallRunInProgressError) as exc_info:
        service.summarize_and_store(request_dto)

    assert exc_info.value.run_id == "run-1"

    llm_port.summarize.assert_not_called()
    writer_port.write_note.assert_not_called()
    workflow_port.save_summary.assert_not_called()


def test_manual_retry_cannot_bypass_note_written_state():
    """Retry is a Kanban delivery operation, never a shortcut around the Saga."""
    workflow_port = MagicMock()
    workflow_port.get_run.return_value = _make_run(
        status=EarningsCallRunStatus.NEW,
        execution_expires_at=0.0,
    )
    service = EarningsCallApplicationService(
        llm_port=MagicMock(),
        writer_port=MagicMock(),
        workflow_port=workflow_port,
        kanban_port=MagicMock(),
    )

    with pytest.raises(EarningsCallRunNotReadyError):
        service.retry_run_for_ticker(ticker="TSM", run_id="run-1")

    workflow_port.reset_run_for_manual_retry.assert_not_called()


def test_manual_retry_returns_active_execution_without_duplicate_delivery():
    workflow_port = MagicMock()
    workflow_port.get_run.return_value = _make_run(
        status=EarningsCallRunStatus.NOTE_WRITTEN,
        vault_path="path/to/note.md",
        execution_expires_at=9999999999.0,
    )
    kanban_port = MagicMock()
    service = EarningsCallApplicationService(
        llm_port=MagicMock(),
        writer_port=MagicMock(),
        workflow_port=workflow_port,
        kanban_port=kanban_port,
    )

    result = service.retry_run_for_ticker(ticker="TSM", run_id="run-1")

    assert result.status == EarningsCallRunStatus.NOTE_WRITTEN
    workflow_port.reset_run_for_manual_retry.assert_not_called()
    kanban_port.ensure_card.assert_not_called()


def test_stale_outbox_fence_is_not_reclassified_as_delivery_failure():
    workflow_port = MagicMock()
    workflow_port.complete_kanban_delivery.side_effect = EarningsCallLeaseExpiredError(
        "stale fence"
    )
    kanban_port = MagicMock()
    kanban_port.ensure_card.return_value = KanbanCardResultDTO(
        card_id="card-1",
        created=True,
        title="[TSM] Earnings Call Q4 2024",
    )
    service = EarningsCallApplicationService(
        llm_port=MagicMock(),
        writer_port=MagicMock(),
        workflow_port=workflow_port,
        kanban_port=kanban_port,
    )
    event = EarningsCallOutboxEventDTO(
        event_id="event-1",
        run_id="run-1",
        source_key="source-key-1",
        event_type="deliver_kanban",
        status="leased",
        attempts=1,
        available_at=0.0,
    )
    lease = LeaseDTO(lease_token="lease-1", lease_expires_at=9999999999.0)

    with pytest.raises(EarningsCallRunInProgressError) as exc_info:
        service._deliver_outbox_event(
            run=_make_run(status=EarningsCallRunStatus.NOTE_WRITTEN, vault_path="path/to/note.md"),
            event=event,
            lease=lease,
        )

    assert exc_info.value.run_id == "run-1"

    workflow_port.schedule_kanban_retry.assert_not_called()
    workflow_port.mark_terminal_failure.assert_not_called()


def test_outbox_attempt_limit_counts_current_delivery():
    workflow_port = MagicMock()
    workflow_port.mark_terminal_failure.return_value = _make_run(
        status=EarningsCallRunStatus.FAILED,
        kanban_status=EarningsCallKanbanStatus.FAILED,
        vault_path="path/to/note.md",
    )
    kanban_port = MagicMock()
    kanban_port.ensure_card.side_effect = RuntimeError("temporary Kanban failure")
    service = EarningsCallApplicationService(
        llm_port=MagicMock(),
        writer_port=MagicMock(),
        workflow_port=workflow_port,
        kanban_port=kanban_port,
        max_attempts=2,
    )
    event = EarningsCallOutboxEventDTO(
        event_id="event-limit",
        run_id="run-1",
        source_key="source-key-1",
        event_type="deliver_kanban",
        status="leased",
        attempts=1,
        available_at=0.0,
    )
    lease = LeaseDTO(lease_token="lease-limit", lease_expires_at=9999999999.0)

    result = service._deliver_outbox_event(
        run=_make_run(status=EarningsCallRunStatus.NOTE_WRITTEN, vault_path="path/to/note.md"),
        event=event,
        lease=lease,
    )

    assert result.status == EarningsCallRunStatus.FAILED
    workflow_port.mark_terminal_failure.assert_called_once()
    workflow_port.schedule_kanban_retry.assert_not_called()


def test_get_run_ticker_ownership_check():
    workflow_port = MagicMock()
    workflow_port.get_run.return_value = _make_run(run_id="run-1", ticker="AAPL")

    service = EarningsCallApplicationService(
        llm_port=MagicMock(),
        writer_port=MagicMock(),
        workflow_port=workflow_port,
        kanban_port=MagicMock(),
    )

    # Matching ticker succeeds
    run = service.get_run_for_ticker(ticker="AAPL", run_id="run-1")
    assert run.ticker == "AAPL"

    # Mismatched ticker raises
    with pytest.raises(EarningsCallTickerMismatchError, match="belongs to ticker 'AAPL', not 'TSM'"):
        service.get_run_for_ticker(ticker="TSM", run_id="run-1")

    # Not found raises
    workflow_port.get_run.return_value = None
    with pytest.raises(EarningsCallRunNotFoundError):
        service.get_run_for_ticker(ticker="AAPL", run_id="run-nonexistent")
