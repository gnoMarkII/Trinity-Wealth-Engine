"""Unit tests for SQLite DAO repositories & adapters for Earnings Call Workflow & Outbox."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import pytest

from api.db.connection import get_connection, init_schema
from api.db.repositories import (
    earnings_call_repository as run_dao,
    earnings_call_outbox_repository as outbox_dao,
    kanban_repository as kanban_dao,
)
from api.db.adapters import SqliteEarningsCallWorkflowAdapter
from application.earnings_call.workflow import EarningsCallRunStatus, EarningsCallKanbanStatus


@pytest.fixture
def db_conn(tmp_path):
    db_file = tmp_path / "test_workflow.sqlite"
    conn = get_connection(str(db_file))
    yield conn
    conn.close()


def test_claim_or_resume_atomic_and_concurrent(db_conn, tmp_path):
    source_key = "test-source-key-1"
    ticker = "TSM"
    period = "Q4 2024"
    transcript_hash = "abc123hash"
    prompt_version = "v1"
    db_file = str(tmp_path / "test_workflow.sqlite")

    # Simulate 2 concurrent threads claiming the same source_key via workflow adapter
    def _claim(thread_id):
        adapter = SqliteEarningsCallWorkflowAdapter(db_path=db_file)
        return adapter.claim_or_resume(
            source_key=source_key,
            ticker=ticker,
            period=period,
            transcript_hash=transcript_hash,
            prompt_version=prompt_version,
            lease_seconds=60,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_claim, 1)
        f2 = executor.submit(_claim, 2)
        c1 = f1.result()
        c2 = f2.result()

    # Exactly ONE thread must own the execution lease
    assert c1.run.run_id == c2.run.run_id
    assert c1.run.source_key == source_key
    assert (c1.owns_execution and not c2.owns_execution) or (c2.owns_execution and not c1.owns_execution)


def test_workflow_state_transitions(db_conn):
    source_key = "test-source-key-flow"
    claim = run_dao.claim_or_resume(
        conn=db_conn,
        source_key=source_key,
        ticker="AAPL",
        period="Q1 2025",
        transcript_hash="hash-flow",
        prompt_version="v1",
        lease_seconds=60,
    )
    run_id = claim.run.run_id
    exec_token = claim.execution_token

    assert claim.owns_execution is True
    assert claim.run.status == EarningsCallRunStatus.NEW

    # 1. Save summary
    run_after_summary = run_dao.save_summary(
        conn=db_conn,
        run_id=run_id,
        execution_token=exec_token,
        highlights="Summary text",
    )
    assert run_after_summary.status == EarningsCallRunStatus.SUMMARIZED
    assert run_after_summary.highlights == "Summary text"

    # 2. Mark note written
    run_after_note = run_dao.mark_note_written(
        conn=db_conn,
        run_id=run_id,
        execution_token=exec_token,
        vault_path="30_Knowledge_Base/Earnings_Calls/AAPL/Q1_2025_AAPL.md",
    )
    assert run_after_note.status == EarningsCallRunStatus.NOTE_WRITTEN
    assert run_after_note.vault_path == "30_Knowledge_Base/Earnings_Calls/AAPL/Q1_2025_AAPL.md"

    # 3. Outbox enqueue & delivery
    event_dto, lease_dto = outbox_dao.enqueue_event(
        conn=db_conn,
        run_id=run_id,
        source_key=source_key,
        event_type="deliver_kanban",
        outbox_lease_seconds=60,
    )
    assert event_dto.run_id == run_id
    assert event_dto.status == "leased"

    # Complete Kanban delivery
    run_completed = run_dao.complete_kanban_delivery(
        conn=db_conn,
        run_id=run_id,
        card_id="card-111",
    )
    assert run_completed.status == EarningsCallRunStatus.COMPLETED
    assert run_completed.kanban_status == EarningsCallKanbanStatus.CREATED
    assert run_completed.kanban_card_id == "card-111"

    outbox_completed = outbox_dao.complete_event(
        conn=db_conn,
        event_id=event_dto.event_id,
        lease_token=lease_dto.lease_token,
    )
    assert outbox_completed is True


def test_outbox_lease_expiry_recovery(db_conn):
    run_id = "run-outbox-test"
    source_key = "source-outbox-key"

    # Insert event that was leased in the past (lease_expires_at = 0)
    db_conn.execute(
        "INSERT INTO earnings_call_outbox ("
        "  event_id, run_id, source_key, event_type, status, attempts, available_at, lease_token, lease_expires_at, created_at, updated_at"
        ") VALUES ('ev-expired', ?, ?, 'deliver_kanban', 'leased', 1, 0, 'old-token', 0, 100, 100)",
        (run_id, source_key),
    )

    pending = outbox_dao.list_pending(conn=db_conn, limit=10)
    assert any(e.event_id == "ev-expired" for e in pending)

    # Lease event with fencing token
    new_lease = outbox_dao.lease_event(conn=db_conn, event_id="ev-expired", lease_seconds=60)
    assert new_lease is not None
    assert new_lease.lease_token != "old-token"


def test_kanban_repository_source_key_dedupe(db_conn):
    kanban_dao.create_kanban_card(
        conn=db_conn,
        card_id="card-1",
        title="[NVDA] Earnings Call Q3",
        column_name="backlog",
        source_key="nvda-source-key",
    )

    # Find by source_key
    found = kanban_dao.find_kanban_card_by_source_key(db_conn, "nvda-source-key")
    assert found is not None
    assert found["card_id"] == "card-1"

    # Move to 'done' column — should still be found!
    kanban_dao.move_kanban_card(db_conn, "card-1", "done")
    found_in_done = kanban_dao.find_kanban_card_by_source_key(db_conn, "nvda-source-key")
    assert found_in_done is not None
    assert found_in_done["column_name"] == "done"
