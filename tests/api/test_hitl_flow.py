"""ทดสอบ human-in-the-loop approval flow (news/YouTube) แบบ end-to-end ผ่าน default_run_fn จริง
+ agents.news_youtube_flow graph จริง — mock แค่ตัวดึง RSS/network เท่านั้น ไม่ mock LangGraph
"""
import json
from contextlib import closing

from api import state_db
from api.jobs import JobQueue, default_run_fn


def _fake_news_candidates():
    return [{"title": "ข่าว A", "link": "http://a.test", "source": "Test", "age_hours": 1, "is_stale": False, "is_fetched": False}]


def _fake_youtube_candidates():
    return [{"channel": "Test Channel", "title": "คลิป B", "link": "http://b.test", "video_id": "abc123", "published": "2026-07-01", "is_fetched": False}]


def test_default_run_fn_news_youtube_flow_pauses_for_approval(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.macro.news_radar.get_news_candidates", lambda max_items=15: _fake_news_candidates())
    monkeypatch.setattr("tools.knowledge.youtube_monitor.get_youtube_candidates", lambda lookback_days=30: _fake_youtube_candidates())
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("WEBUI_STATE_DB_PATH", str(tmp_path / "webui_state.sqlite"))

    conn = state_db.get_connection(str(tmp_path / "webui_state.sqlite"))
    state_db.create_job(conn, "job-hitl-1", "thread-hitl-1", "card-1", "key-1", "ดึงข่าวล่าสุด", status="running", flow="news_youtube")
    conn.close()

    default_run_fn(job_id="job-hitl-1", thread_id="thread-hitl-1", instruction="ดึงข่าวล่าสุด", flow="news_youtube", resume_value=None)

    conn = state_db.get_connection(str(tmp_path / "webui_state.sqlite"))
    job = state_db.get_job(conn, "job-hitl-1")
    conn.close()

    assert job["status"] == "awaiting_approval"
    payload = json.loads(job["interrupt_payload"])
    assert payload["type"] == "news_youtube_approval"
    assert payload["news_candidates"][0]["title"] == "ข่าว A"
    assert payload["youtube_candidates"][0]["title"] == "คลิป B"


def test_default_run_fn_news_youtube_flow_resumes_and_completes(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.macro.news_radar.get_news_candidates", lambda max_items=15: _fake_news_candidates())
    monkeypatch.setattr("tools.knowledge.youtube_monitor.get_youtube_candidates", lambda lookback_days=30: _fake_youtube_candidates())
    monkeypatch.setattr("agents.news_youtube_flow._save_ingested_content", lambda content: "saved OK")
    # กัน jobs.py เรียก LLM จริงตอน job จบ (_append_manager_summary) — เดิม test นี้
    # order-dependent: รันเดี่ยวผ่านเพราะ .env ยังไม่ถูกโหลด (LLM fail → None) แต่รันทั้ง
    # suite แล้ว test อื่น import api.main → load_dotenv() → มี API key จริง → ยิง LLM จริง
    # ทุกครั้ง (เปลืองเงิน) แล้ว summary มาต่อท้าย log ทำให้ logs[-1] ไม่ใช่บรรทัด ingest
    monkeypatch.setattr("agents.manager_agent.generate_manager_summary", lambda instruction, deliverables: None)

    class _FakeIngestTool:
        def invoke(self, args):
            return "---\ntitle: x\nentity_type: article_note\n---\ncontent"

    monkeypatch.setattr("tools.knowledge.article.ingest_article_url", _FakeIngestTool())
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("WEBUI_STATE_DB_PATH", str(tmp_path / "webui_state.sqlite"))

    conn = state_db.get_connection(str(tmp_path / "webui_state.sqlite"))
    state_db.create_job(conn, "job-hitl-2", "thread-hitl-2", "card-2", "key-2", "ดึงข่าวล่าสุด", status="running", flow="news_youtube")
    conn.close()

    # รอบแรก — ต้อง pause
    default_run_fn(job_id="job-hitl-2", thread_id="thread-hitl-2", instruction="ดึงข่าวล่าสุด", flow="news_youtube", resume_value=None)

    # resume ด้วยการเลือกอนุมัติแค่ข่าว A
    default_run_fn(
        job_id="job-hitl-2",
        thread_id="thread-hitl-2",
        instruction="ดึงข่าวล่าสุด",
        flow="news_youtube",
        resume_value={"approved_news_links": ["http://a.test"], "approved_youtube_links": []},
    )

    conn = state_db.get_connection(str(tmp_path / "webui_state.sqlite"))
    logs = state_db.get_job_logs_since(conn, "job-hitl-2", after_seq=0)
    conn.close()

    assert len(logs) >= 1
    assert "saved OK" in logs[-1]["content"]


def test_ingest_node_skips_saving_when_tool_returns_inline_error(tmp_path, monkeypatch):
    """เจอจริงจาก live test: ingest_article_url ไม่ raise แต่คืน 'ERROR: ...' string ตรงๆ
    (เช่น LLM RESOURCE_EXHAUSTED) — ต้องไม่บันทึกข้อความ error นั้นเป็นไฟล์ขยะลง Vault
    """
    from agents.news_youtube_flow import ingest_node

    class _FakeErrorTool:
        def invoke(self, args):
            return "ERROR: LLM Extraction ล้มเหลว: 429 RESOURCE_EXHAUSTED"

    saved_calls = []
    monkeypatch.setattr("tools.knowledge.article.ingest_article_url", _FakeErrorTool())
    monkeypatch.setattr(
        "agents.news_youtube_flow._save_ingested_content",
        lambda content: saved_calls.append(content) or "should-not-be-called",
    )

    result = ingest_node({"approved_news_links": ["http://a.test"], "approved_youtube_links": []})

    assert saved_calls == []  # ไม่เรียก save เลยเพราะเป็น error string
    assert "ERROR" in result["result_summary"]


def test_job_queue_resume_transitions_status_and_reruns(tmp_path, monkeypatch):
    """ทดสอบผ่าน JobQueue.resume() โดยตรง (ไม่ผ่าน HTTP) — จำลอง run_fn แบบง่ายที่ตั้ง
    awaiting_approval รอบแรก แล้ว complete รอบ resume
    """
    db_path = str(tmp_path / "state.sqlite")
    calls = []

    def _fake_run_fn(job_id, thread_id, instruction, flow="manager", resume_value=None, **kwargs):
        calls.append(resume_value)
        conn = state_db.get_connection(db_path)
        if resume_value is None:
            state_db.set_job_awaiting_approval(conn, job_id, json.dumps({"type": "news_youtube_approval"}))
        else:
            state_db.append_job_log(conn, job_id, "ingest", "done", role="reply", label="ingest")
        conn.close()

    queue = JobQueue(run_fn=_fake_run_fn, db_path=db_path)
    job_id = queue.dispatch("ดึงข่าวล่าสุด", card_id="card-x", flow="news_youtube")

    import asyncio

    async def _drive():
        await queue._run_job(job_id)

    asyncio.run(_drive())

    conn = state_db.get_connection(db_path)
    job = state_db.get_job(conn, job_id)
    conn.close()
    assert job["status"] == "awaiting_approval"

    queue.resume(job_id, {"approved_news_links": ["http://a.test"], "approved_youtube_links": []})

    conn = state_db.get_connection(db_path)
    job = state_db.get_job(conn, job_id)
    conn.close()
    assert job["status"] == "running"

    asyncio.run(_drive())

    conn = state_db.get_connection(db_path)
    job = state_db.get_job(conn, job_id)
    conn.close()
    assert job["status"] == "done"
    assert calls == [None, {"approved_news_links": ["http://a.test"], "approved_youtube_links": []}]
