"""Unit tests for JobApplicationService."""
import pytest
from pathlib import Path
from api.db.connection import get_connection, init_schema
from api.db.repositories.job_repository import create_job, append_job_log
from api.db.adapters import SqliteJobRepositoryAdapter
from application.jobs.service import JobApplicationService


@pytest.fixture
def test_db(tmp_path: Path) -> str:
    db_file = str(tmp_path / "test_job_service.db")
    conn = get_connection(db_file)
    init_schema(conn)
    conn.close()
    return db_file


def test_job_service_get_outputs_and_specialists(test_db: str):
    conn = get_connection(test_db)
    create_job(
        conn=conn,
        job_id="job_app_1",
        thread_id="th_1",
        card_id="card_1",
        idempotency_key="idem_app_1",
        instruction="Analyze AAPL",
        status="done",
    )
    append_job_log(conn, job_id="job_app_1", node_name="valuation_specialist", content="Fair value $220", role="reply", label="Valuation Specialist")
    append_job_log(conn, job_id="job_app_1", node_name="manager_summary", content="Buy recommendation", role="reply", label="Manager")
    conn.commit()
    conn.close()

    service = JobApplicationService(repo=SqliteJobRepositoryAdapter(db_path=test_db))
    outputs = service.get_job_outputs("job_app_1")

    assert outputs is not None
    assert outputs.job_id == "job_app_1"
    assert outputs.status == "done"
    assert outputs.executive_summary == "Buy recommendation"
    assert len(outputs.specialists) == 1
    assert outputs.specialists[0].node_name == "valuation_specialist"
    assert outputs.specialists[0].content == "Fair value $220"


def test_job_service_get_logs_after(test_db: str):
    conn = get_connection(test_db)
    create_job(conn=conn, job_id="job_app_2", thread_id="th_2", card_id=None, idempotency_key="idem_2", instruction="Log test")
    append_job_log(conn, job_id="job_app_2", node_name="node_1", content="Step 1")
    append_job_log(conn, job_id="job_app_2", node_name="node_2", content="Step 2")
    append_job_log(conn, job_id="job_app_2", node_name="node_3", content="Step 3")
    conn.commit()
    conn.close()

    service = JobApplicationService(repo=SqliteJobRepositoryAdapter(db_path=test_db))
    logs = service.get_job_logs_after("job_app_2", after_seq=1)
    assert len(logs) == 2
    assert logs[0]["seq"] == 2
    assert logs[1]["seq"] == 3
