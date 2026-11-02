from api import state_db
from api.jobs import _append_manager_summary


def test_get_job_outputs_returns_manager_summary_and_latest_specialists(authed_client):
    conn = state_db.get_connection()
    state_db.create_job(conn, "job-output-1", "thread-1", "card-1", "key-1", "analyse", status="done")
    state_db.append_job_log(conn, "job-output-1", "archivist", "initial archivist", role="reply", label="Archivist")
    state_db.append_job_log(conn, "job-output-1", "post_archivist", "duplicate wrapper result", role="reply", label="post")
    state_db.append_job_log(conn, "job-output-1", "archivist", "latest archivist", role="reply", label="Archivist")
    state_db.append_job_log(conn, "job-output-1", "macro_quant", "quant result", role="reply", label="Macro Quant")
    state_db.append_job_log(conn, "job-output-1", "manager_summary", "# Manager summary", role="reply", label="Manager Summary")
    conn.close()

    response = authed_client.get("/api/agents/jobs/job-output-1/outputs")

    assert response.status_code == 200
    body = response.json()
    assert body["executive_summary"] == "# Manager summary"
    assert body["last_seq"] == 5
    assert [(item["node_name"], item["content"]) for item in body["specialists"]] == [
        ("archivist", "latest archivist"),
        ("macro_quant", "quant result"),
    ]


def test_get_job_outputs_returns_404_for_unknown_job(authed_client):
    response = authed_client.get("/api/agents/jobs/missing/outputs")

    assert response.status_code == 404


def test_append_manager_summary_persists_generated_content(tmp_path, monkeypatch):
    captured = {}

    def fake_summary(instruction, deliverables):
        captured["instruction"] = instruction
        captured["deliverables"] = deliverables
        return "# Fresh manager summary"

    monkeypatch.setattr("agents.research_swarm_flow.generate_manager_summary", fake_summary)
    db_path = str(tmp_path / "state.sqlite")
    conn = state_db.get_connection(db_path)
    state_db.create_job(conn, "job-summary-1", "thread-1", None, "key-1", "original task", status="running")
    state_db.append_job_log(conn, "job-summary-1", "archivist", "archivist result", role="reply")
    state_db.append_job_log(conn, "job-summary-1", "post_archivist", "duplicate result", role="reply")

    _append_manager_summary(conn, "job-summary-1", "original task")

    logs = state_db.get_job_reply_logs(conn, "job-summary-1")
    conn.close()
    assert captured == {"instruction": "original task", "deliverables": [("archivist", "archivist result")]}
    assert logs[-1]["node_name"] == "manager_summary"
    assert logs[-1]["content"] == "# Fresh manager summary"


def test_append_manager_summary_skips_specialized_flows(tmp_path, monkeypatch):
    called = False

    def fake_summary(instruction, deliverables):
        nonlocal called
        called = True
        return "# Should not be generated"

    monkeypatch.setattr("agents.research_swarm_flow.generate_manager_summary", fake_summary)
    db_path = str(tmp_path / "state.sqlite")
    conn = state_db.get_connection(db_path)
    state_db.create_job(conn, "job-summary-2", "thread-2", None, "key-2", "original task", flow="youtube_pitch", status="running")
    state_db.append_job_log(conn, "job-summary-2", "generate_pitch", "pitch results", role="reply")

    _append_manager_summary(conn, "job-summary-2", "original task", flow="youtube_pitch")

    logs = state_db.get_job_reply_logs(conn, "job-summary-2")
    conn.close()
    assert not called
    assert not any(row["node_name"] == "manager_summary" for row in logs)
