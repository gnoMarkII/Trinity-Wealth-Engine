"""Unit test ตรง JobQueue — ไม่ผ่าน HTTP layer เพื่อกัน timing flakiness ของ async worker"""
from api import state_db
from api.db.adapters import SqliteJobRepositoryAdapter
from api.jobs import JobQueue
from api.routers.agents_router import _status_to_dto
from api.schemas import JobStatusDTO
from application.jobs.service import JobApplicationService


def _job_to_dto(conn, job) -> JobStatusDTO:
    service = JobApplicationService(repo=SqliteJobRepositoryAdapter(conn=conn))
    status = service.get_status(job["job_id"])
    assert status is not None
    return _status_to_dto(status)


def _noop_run_fn(**kwargs) -> None:
    return None


def test_get_connection_migrates_old_schema_missing_columns(tmp_path):
    """จำลองไฟล์ .sqlite เก่าที่ถูกสร้างก่อนเพิ่ม flow/interrupt_payload/resume_value/role/label —
    เจอบั๊กจริง: dispatch งานจาก Kanban พังด้วย 'table jobs has no column named flow'
    เพราะ CREATE TABLE IF NOT EXISTS ไม่แก้ตารางเก่าที่มีอยู่แล้ว
    """
    import sqlite3

    db_path = str(tmp_path / "old_schema.sqlite")
    old_conn = sqlite3.connect(db_path)
    old_conn.executescript("""
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            card_id TEXT,
            idempotency_key TEXT UNIQUE,
            instruction TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            error_message TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE job_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            node_name TEXT,
            content TEXT,
            created_at REAL NOT NULL
        );
    """)
    old_conn.commit()
    old_conn.close()

    conn = state_db.get_connection(db_path)

    job_cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
    assert {"flow", "interrupt_payload", "resume_value"} <= job_cols

    log_cols = {r["name"] for r in conn.execute("PRAGMA table_info(job_logs)")}
    assert {"role", "label"} <= log_cols

    # ต้อง insert ได้จริงหลัง migrate ไม่ error แบบที่เจอตอน dispatch จริง
    state_db.create_job(conn, "j1", "t1", None, "k1", "instr", flow="news_youtube")
    state_db.append_job_log(conn, "j1", "gate", "hello", role="instruction", label="manager → gate")
    conn.close()


def test_dispatch_same_card_and_instruction_reuses_job_id(tmp_path):
    db_path = str(tmp_path / "state.sqlite")
    queue = JobQueue(run_fn=_noop_run_fn, db_path=db_path)

    job_id_1 = queue.dispatch("วิเคราะห์ตลาด", card_id="card-1")
    job_id_2 = queue.dispatch("วิเคราะห์ตลาด", card_id="card-1")

    assert job_id_1 == job_id_2

    conn = state_db.get_connection(db_path)
    count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    conn.close()
    assert count == 1


def test_dispatch_different_card_creates_separate_job(tmp_path):
    db_path = str(tmp_path / "state.sqlite")
    queue = JobQueue(run_fn=_noop_run_fn, db_path=db_path)

    job_id_1 = queue.dispatch("วิเคราะห์ตลาด", card_id="card-1")
    job_id_2 = queue.dispatch("วิเคราะห์ตลาด", card_id="card-2")

    assert job_id_1 != job_id_2


def test_dispatch_generates_fresh_thread_id_per_job(tmp_path):
    """thread_id ต้องเป็น per-job เสมอ ไม่ผูก session (Rev.5 ข้อ 2) — กัน AgentState รั่วข้ามงาน"""
    db_path = str(tmp_path / "state.sqlite")
    queue = JobQueue(run_fn=_noop_run_fn, db_path=db_path)

    job_id_1 = queue.dispatch("งานที่หนึ่ง", card_id="card-1")
    job_id_2 = queue.dispatch("งานที่สอง", card_id="card-2")

    conn = state_db.get_connection(db_path)
    thread_1 = state_db.get_job(conn, job_id_1)["thread_id"]
    thread_2 = state_db.get_job(conn, job_id_2)["thread_id"]
    conn.close()

    assert thread_1 != thread_2


def test_reenqueue_pending_marks_stale_running_job_as_error(tmp_path):
    """งานที่ค้าง status=running ตอน process ตายกลางคัน ต้องไม่หายเงียบ (Rev.5 ข้อ 3)"""
    db_path = str(tmp_path / "state.sqlite")
    conn = state_db.get_connection(db_path)
    state_db.create_job(conn, "job-x", "thread-x", "card-x", "key-x", "instr", status="running")
    conn.close()

    queue = JobQueue(run_fn=_noop_run_fn, db_path=db_path)
    queue.reenqueue_pending()

    conn = state_db.get_connection(db_path)
    job = state_db.get_job(conn, "job-x")
    conn.close()

    assert job["status"] == "error"
    assert "restart" in job["error_message"]


def test_reenqueue_pending_requeues_stale_queued_job(tmp_path):
    """งานที่ยัง queued (ไม่ทันเริ่มรัน) ปลอดภัยที่จะ re-push เข้า queue ใหม่"""
    db_path = str(tmp_path / "state.sqlite")
    conn = state_db.get_connection(db_path)
    state_db.create_job(conn, "job-y", "thread-y", "card-y", "key-y", "instr", status="queued")
    conn.close()

    queue = JobQueue(run_fn=_noop_run_fn, db_path=db_path)
    queue.reenqueue_pending()

    assert queue._queue.qsize() == 1


def test_reenqueue_pending_with_flows_filter_ignores_other_flows_running_job(tmp_path):
    """queue ที่แชร์ DB กับ queue อื่น (เช่น notebooklm_job_queue กับคิวหลัก) ต้องไม่ไปแตะงาน
    flow อื่นตอน reenqueue_pending — ไม่งั้นจะ mark งาน manager/youtube_pitch ที่กำลังรันอยู่จริง
    เป็น error ผิดๆ เพียงเพราะ queue คนละตัวมาสแกนเจอ (เกิดจาก list_jobs_by_status ไม่ filter flow)
    """
    db_path = str(tmp_path / "state.sqlite")
    conn = state_db.get_connection(db_path)
    state_db.create_job(conn, "job-manager", "thread-1", None, "key-manager", "instr", status="running", flow="manager")
    state_db.create_job(conn, "job-notebooklm", "thread-2", None, "key-notebooklm", "instr", status="running", flow="notebooklm")
    conn.close()

    queue = JobQueue(run_fn=_noop_run_fn, db_path=db_path, flows={"notebooklm"})
    queue.reenqueue_pending()

    conn = state_db.get_connection(db_path)
    manager_job = state_db.get_job(conn, "job-manager")
    notebooklm_job = state_db.get_job(conn, "job-notebooklm")
    conn.close()

    assert manager_job["status"] == "running"  # ไม่ถูกแตะ
    assert notebooklm_job["status"] == "error"  # ถูก mark ตามปกติ


def test_reenqueue_pending_with_flows_filter_only_requeues_matching_flow(tmp_path):
    db_path = str(tmp_path / "state.sqlite")
    conn = state_db.get_connection(db_path)
    state_db.create_job(conn, "job-manager", "thread-1", None, "key-manager", "instr", status="queued", flow="manager")
    state_db.create_job(conn, "job-notebooklm", "thread-2", None, "key-notebooklm", "instr", status="queued", flow="notebooklm")
    conn.close()

    queue = JobQueue(run_fn=_noop_run_fn, db_path=db_path, flows={"notebooklm"})
    queue.reenqueue_pending()

    assert queue._queue.qsize() == 1


def test_reenqueue_pending_without_flows_filter_processes_all_flows_as_before(tmp_path):
    """flows=None (default) ต้องพฤติกรรมเดิมเป๊ะ — กัน regression กับคิวหลักที่มีอยู่แล้ว"""
    db_path = str(tmp_path / "state.sqlite")
    conn = state_db.get_connection(db_path)
    state_db.create_job(conn, "job-manager", "thread-1", None, "key-manager", "instr", status="running", flow="manager")
    state_db.create_job(conn, "job-notebooklm", "thread-2", None, "key-notebooklm", "instr", status="running", flow="notebooklm")
    conn.close()

    queue = JobQueue(run_fn=_noop_run_fn, db_path=db_path)
    queue.reenqueue_pending()

    conn = state_db.get_connection(db_path)
    manager_job = state_db.get_job(conn, "job-manager")
    notebooklm_job = state_db.get_job(conn, "job-notebooklm")
    conn.close()

    assert manager_job["status"] == "error"
    assert notebooklm_job["status"] == "error"


def test_job_log_role_label_and_current_node(tmp_path):
    """log ต้องแยก instruction (manager→worker) กับ reply (worker เอง) พร้อม label ทิศทาง
    และ current_node ต้องดึงจาก log แถวล่าสุดได้ (สำหรับ badge 'Workers Executing')"""
    db_path = str(tmp_path / "state.sqlite")
    conn = state_db.get_connection(db_path)
    state_db.create_job(conn, "job-z", "thread-z", "card-z", "key-z", "instr", status="running")

    state_db.append_job_log(conn, "job-z", "archivist", "อ่านพอร์ตมา", role="instruction", label="manager → archivist")
    state_db.append_job_log(conn, "job-z", "archivist", "อ่านพอร์ตเสร็จแล้ว", role="reply", label="archivist")

    logs = state_db.get_job_logs_since(conn, "job-z", after_seq=0)
    assert len(logs) == 2
    assert logs[0]["role"] == "instruction"
    assert logs[0]["label"] == "manager → archivist"
    assert logs[1]["role"] == "reply"

    assert state_db.get_latest_job_log_node(conn, "job-z") == "archivist"
    conn.close()


def test_job_to_dto_exposes_current_node_only_while_running(tmp_path):
    db_path = str(tmp_path / "state.sqlite")
    conn = state_db.get_connection(db_path)
    state_db.create_job(conn, "job-w", "thread-w", "card-w", "key-w", "instr", status="running")
    state_db.append_job_log(conn, "job-w", "macro_quant", "กำลังคำนวณ", role="reply", label="macro_quant")

    job = state_db.get_job(conn, "job-w")
    dto = _job_to_dto(conn, job)
    assert dto.current_node == "macro_quant"

    state_db.update_job_status(conn, "job-w", "done")
    job = state_db.get_job(conn, "job-w")
    dto = _job_to_dto(conn, job)
    assert dto.current_node is None
    conn.close()


def test_job_queue_preserves_done_with_errors_terminal_status(tmp_path):
    """A flow that completed one pitch but failed another must not be overwritten as done."""
    import asyncio

    db_path = str(tmp_path / "state.sqlite")

    def _partial_run(job_id, **_kwargs):
        conn = state_db.get_connection(db_path)
        try:
            state_db.update_job_status(conn, job_id, "done_with_errors", error_message="Pitch p-2 failed quality gate")
        finally:
            conn.close()

    queue = JobQueue(run_fn=_partial_run, db_path=db_path)
    job_id = queue.dispatch("สร้าง briefing", card_id="card-1", flow="youtube_pitch")
    asyncio.run(queue._run_job(job_id))

    conn = state_db.get_connection(db_path)
    job = state_db.get_job(conn, job_id)
    conn.close()
    assert job["status"] == "done_with_errors"
    assert "quality gate" in job["error_message"]


def test_job_queue_persists_terminal_error_notification(tmp_path):
    """A failed job must expose the diagnostic in both job status and SSE logs."""
    import asyncio

    db_path = str(tmp_path / "state.sqlite")

    def _failing_run(**_kwargs):
        raise ValueError("macro provider remained stale after forced refresh")

    queue = JobQueue(run_fn=_failing_run, db_path=db_path)
    job_id = queue.dispatch("create briefing", card_id="card-1", flow="youtube_pitch")
    asyncio.run(queue._run_job(job_id))

    conn = state_db.get_connection(db_path)
    job = state_db.get_job(conn, job_id)
    logs = state_db.get_job_logs_since(conn, job_id)
    conn.close()

    assert job["status"] == "error"
    assert job["error_message"] == "macro provider remained stale after forced refresh"
    assert logs[-1]["node_name"] == "system_error"
    assert logs[-1]["label"] == "System Error"
    assert "macro provider remained stale" in logs[-1]["content"]


def test_get_connection_only_initializes_schema_once_per_path(tmp_path, monkeypatch):
    """init_schema (CREATE TABLE + migrate + backfill) ต้องรันแค่ครั้งเดียวต่อ db path —
    ไม่ใช่ทุกครั้งที่เปิด connection ใหม่ (SSE poll ทุก 1s จะเปิด connection ใหม่รัวๆ)
    """
    import api.state_db as state_db_module

    db_path = str(tmp_path / "state.sqlite")
    calls = []
    original_init_schema = state_db_module.init_schema

    def _counting_init_schema(conn):
        calls.append(1)
        original_init_schema(conn)

    monkeypatch.setattr(state_db_module, "init_schema", _counting_init_schema)

    state_db_module.get_connection(db_path)
    state_db_module.get_connection(db_path)
    state_db_module.get_connection(db_path)

    assert len(calls) == 1


def test_migrate_columns_moves_dispatcher_cards_to_backlog(tmp_path):
    """คอลัมน์ 'dispatcher' ไม่มีอยู่ใน UI แล้ว — การ์ดเก่าที่ยังค้างอยู่ต้องถูกย้ายกลับ backlog
    อัตโนมัติตอน init_schema ไม่งั้นจะมองไม่เห็นจากหน้าเว็บเลย
    """
    from api import state_db

    db_path = str(tmp_path / "state.sqlite")
    conn = state_db.get_connection(db_path)
    state_db.create_kanban_card(conn, "card-stuck", "งานค้าง", column_name="backlog", flow="manager")
    conn.execute("UPDATE kanban_cards SET column_name = 'dispatcher' WHERE card_id = ?", ("card-stuck",))
    conn.commit()
    conn.close()

    # เปิด connection ใหม่ผ่าน path เดิม — จำลอง process restart ที่ init_schema ต้องรันใหม่จริง
    import api.state_db as state_db_module
    state_db_module._INITIALIZED_DB_PATHS.discard(db_path)
    conn2 = state_db.get_connection(db_path)
    card = state_db.get_kanban_card(conn2, "card-stuck")
    conn2.close()

    assert card["column_name"] == "backlog"


def test_dispatch_same_card_and_instruction_different_flow_creates_separate_job():
    """งานชื่อเดียวกัน + scope เดียวกัน แต่คนละ flow ต้องไม่ dedup กัน — เดิม idempotency_key
    ไม่รวม flow ทำให้ 'manager' กับ 'news_youtube' สลับ job กันได้ถ้า instruction/scope ตรงกัน
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/state.sqlite"
        queue = JobQueue(run_fn=_noop_run_fn, db_path=db_path)

        job_id_1 = queue.dispatch("งานเดียวกัน", card_id="card-1", flow="manager")
        job_id_2 = queue.dispatch("งานเดียวกัน", card_id="card-1", flow="news_youtube")

        assert job_id_1 != job_id_2


def test_worker_loop_survives_run_job_raising_unexpected_exception(tmp_path):
    """ถ้า _run_job โยน exception ที่ไม่ได้ถูกจับไว้แล้ว (เช่น DB error กลาง commit) worker loop
    ต้องไม่ตายเงียบ — งานถัดไปในคิวต้องยังถูกประมวลผลได้ปกติ
    """
    import asyncio

    db_path = str(tmp_path / "state.sqlite")
    processed = []

    def _run_fn(job_id, **kwargs):
        if job_id == "job-boom":
            raise RuntimeError("boom")
        processed.append(job_id)

    queue = JobQueue(run_fn=_run_fn, db_path=db_path)

    # จำลอง _run_job พังกลางทางแบบไม่คาดคิด (ไม่ใช่ exception จาก run_fn ที่ _run_job ดักไว้แล้ว)
    original_run_job = queue._run_job
    call_count = {"n": 0}

    async def _flaky_run_job(job_id):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("unexpected failure inside _run_job itself")
        await original_run_job(job_id)

    queue._run_job = _flaky_run_job

    # queue.start() ใช้ asyncio.create_task() ต้องมี running loop อยู่แล้ว และ stop() (หลัง
    # แก้ไข) ต้อง await self._worker_task ตัวเดิม — start/dispatch/wait/stop จึงต้องอยู่ใน
    # asyncio.run() ครั้งเดียวกันทั้งหมด (สอง asyncio.run() แยกกันจะคนละ event loop กัน)
    async def _run():
        queue.start()

        job_id_bad = queue.dispatch("งานที่พัง", card_id="card-bad")
        job_id_good = queue.dispatch("งานที่ดี", card_id="card-good")

        for _ in range(50):
            if job_id_good in processed:
                break
            await asyncio.sleep(0.05)

        await queue.stop()
        return job_id_bad, job_id_good

    job_id_bad, job_id_good = asyncio.run(_run())

    assert job_id_good in processed


def test_job_queue_cas_concurrency(tmp_path):
    """ทดสอบ CAS (cas_job_status) ว่าสามารถป้องกัน concurrency ได้จริง"""
    import asyncio
    db_path = str(tmp_path / "state.sqlite")
    queue = JobQueue(run_fn=_noop_run_fn, db_path=db_path)
    job_id = queue.dispatch("ทดสอบ CAS", card_id="card-cas")

    conn = state_db.get_connection(db_path)
    state_db.update_job_status(conn, job_id, "running")
    conn.close()

    calls = []
    def _mocked_run_fn(*args, **kwargs):
        calls.append(True)
    queue._run_fn = _mocked_run_fn

    asyncio.run(queue._run_job(job_id))
    assert len(calls) == 0


def test_job_queue_persistence_failure(tmp_path):
    """ทดสอบกรณี _run_fn โยน exception ระหว่าง _run_job ต้อง mark job เป็น error"""
    import asyncio
    db_path = str(tmp_path / "state.sqlite")

    def _faulty_run_fn(*args, **kwargs):
        raise RuntimeError("simulated persistence failure")

    queue = JobQueue(run_fn=_faulty_run_fn, db_path=db_path)
    job_id = queue.dispatch("ทดสอบ persistence failure", card_id="card-fail")

    asyncio.run(queue._run_job(job_id))

    conn = state_db.get_connection(db_path)
    job = state_db.get_job(conn, job_id)
    conn.close()

    assert job["status"] == "error"
    assert "simulated persistence failure" in str(job["error_message"])
