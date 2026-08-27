"""Unit tests for DbUnitOfWork and SQLite raw DAO transaction isolation."""
import sqlite3
import pytest
from pathlib import Path

from api.db.connection import get_connection, init_schema
from api.db.uow import DbUnitOfWork
from api.db.repositories.kanban_repository import list_kanban_cards, create_kanban_card
from api.db.repositories.job_repository import create_job, get_job, list_jobs_by_status
from api.db.repositories.dcf_repository import record_dcf_evaluation, get_latest_dcf_evaluation


@pytest.fixture
def test_db_path(tmp_path: Path) -> str:
    db_file = str(tmp_path / "uow_test.db")
    conn = get_connection(db_file)
    init_schema(conn)
    conn.close()
    return db_file


def test_uow_automatic_commit_on_success(test_db_path: str):
    """Test that DbUnitOfWork automatically commits modifications when exiting normally."""
    with DbUnitOfWork(db_path=test_db_path) as uow:
        create_kanban_card(
            conn=uow.conn,
            card_id="card_alpha",
            title="Card Alpha",
            column_name="backlog",
        )

    # Verify committed in separate connection
    conn2 = get_connection(test_db_path)
    cards = list_kanban_cards(conn2)
    conn2.close()

    assert len(cards) == 1
    assert cards[0]["card_id"] == "card_alpha"
    assert cards[0]["title"] == "Card Alpha"


def test_uow_automatic_rollback_on_exception(test_db_path: str):
    """Test that DbUnitOfWork rolls back all modifications when an exception is raised."""
    try:
        with DbUnitOfWork(db_path=test_db_path) as uow:
            create_job(
                conn=uow.conn,
                job_id="job_rollback_1",
                thread_id="thread_1",
                card_id=None,
                idempotency_key="idem_1",
                instruction="Test rollback",
            )
            # Verify job is visible in active uncommitted transaction
            job_in_tx = get_job(uow.conn, "job_rollback_1")
            assert job_in_tx is not None

            # Raise simulated error
            raise RuntimeError("Simulated transaction failure")
    except RuntimeError:
        pass

    # Verify not committed in separate connection
    conn2 = get_connection(test_db_path)
    job_after = get_job(conn2, "job_rollback_1")
    conn2.close()

    assert job_after is None


def test_uow_multi_repository_atomic_coordination(test_db_path: str):
    """Test coordinating multiple repositories in a single atomic transaction."""
    with DbUnitOfWork(db_path=test_db_path) as uow:
        create_kanban_card(
            conn=uow.conn,
            card_id="card_beta",
            title="Card Beta",
            column_name="backlog",
        )
        record_dcf_evaluation(
            conn=uow.conn,
            evaluation_id="eval_100",
            ticker="NVDA",
            market="US",
            evaluated_at="2026-08-23",
            scenarios={"base": {"fair_value": 150.0}},
        )

    conn2 = get_connection(test_db_path)
    cards = list_kanban_cards(conn2)
    dcf = get_latest_dcf_evaluation(conn2, "NVDA")
    conn2.close()

    assert len(cards) == 1
    assert dcf is not None
    assert dcf["ticker"] == "NVDA"


def test_connection_bound_adapters_commit_as_one_unit(test_db_path: str):
    """Adapters bound to one UoW must observe and commit the same transaction."""
    with DbUnitOfWork(db_path=test_db_path) as uow:
        create_job(
            conn=uow.conn,
            job_id="job_bound_1",
            thread_id="thread_bound",
            card_id="card_bound",
            idempotency_key="idem_bound_1",
            instruction="bound transaction",
            status="awaiting_approval",
        )
        uow.kanban.create_kanban_card(
            card_id="card_bound",
            title="Bound card",
            column_name="approval",
        )
        uow.jobs.claim_job_resume("job_bound_1", '{"ok": true}')
        assert uow.jobs.get_job("job_bound_1")["status"] == "queued"
        assert uow.kanban.get_kanban_card("card_bound")["title"] == "Bound card"

    conn = get_connection(test_db_path)
    assert get_job(conn, "job_bound_1")["status"] == "queued"
    assert conn.execute("SELECT 1 FROM kanban_cards WHERE card_id = 'card_bound'").fetchone()
    conn.close()


def test_connection_bound_adapters_roll_back_as_one_unit(test_db_path: str):
    with pytest.raises(RuntimeError):
        with DbUnitOfWork(db_path=test_db_path) as uow:
            create_job(
                conn=uow.conn,
                job_id="job_bound_rollback",
                thread_id="thread_bound",
                card_id=None,
                idempotency_key="idem_bound_rollback",
                instruction="rollback transaction",
            )
            uow.kanban.create_kanban_card(
                card_id="card_bound_rollback",
                title="Rollback card",
                column_name="backlog",
            )
            raise RuntimeError("rollback")

    conn = get_connection(test_db_path)
    assert get_job(conn, "job_bound_rollback") is None
    assert conn.execute(
        "SELECT 1 FROM kanban_cards WHERE card_id = 'card_bound_rollback'"
    ).fetchone() is None
    conn.close()


def test_standalone_claim_resume_is_persisted(test_db_path: str):
    conn = get_connection(test_db_path)
    create_job(
        conn=conn,
        job_id="job_claim_persisted",
        thread_id="thread_claim",
        card_id=None,
        idempotency_key="idem_claim_persisted",
        instruction="claim",
        status="awaiting_approval",
    )
    conn.commit()
    conn.close()

    from api.db.adapters import SqliteJobRepositoryAdapter

    SqliteJobRepositoryAdapter(db_path=test_db_path).claim_job_resume(
        "job_claim_persisted", '{"approved": true}'
    )
    conn = get_connection(test_db_path)
    assert get_job(conn, "job_claim_persisted")["status"] == "queued"
    conn.close()


def test_uow_exposes_connection_bound_earnings_call_daos(test_db_path: str):
    """Earnings Call workflow adapter composes through the same UoW transaction."""
    with DbUnitOfWork(db_path=test_db_path) as uow:
        workflow = uow.earnings_call_workflow
        assert workflow is uow.earnings_call_workflow
        claim = workflow.claim_or_resume(
            source_key="uow-earnings-source",
            ticker="TSM",
            period="Q4 2024",
            transcript_hash="hash-uow",
            prompt_version="v1",
            lease_seconds=60,
        )
        assert claim.owns_execution is True
        assert workflow.get_run(claim.run.run_id).run_id == claim.run.run_id

        # The typed workflow boundary owns both run and outbox DAOs.  A run
        # with a leased event is not returned as pending until its lease ends.
        run = workflow.record_note_and_enqueue(
            run_id=claim.run.run_id,
            execution_token=claim.execution_token,
            vault_path="note.md",
            outbox_lease_seconds=60,
        )
        assert run[1].event_id
        assert run[2].lease_token
        assert workflow.list_pending_outbox(limit=10) == []

    conn = get_connection(test_db_path)
    assert conn.execute(
        "SELECT 1 FROM earnings_call_outbox WHERE event_id = ?", (run[1].event_id,)
    ).fetchone()
    conn.close()
