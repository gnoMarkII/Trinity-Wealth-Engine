"""Single-worker job queue — durable (Rev.5 ข้อ 3) + idempotent (software-design review ข้อ 6)
+ human-in-the-loop resume (LangGraph interrupt/Command(resume=...))

Design notes:
- `thread_id` เป็น per-job เสมอ ไม่ผูกกับ session cookie (Rev.5 ข้อ 2) — กัน AgentState
  (task_queue/replan_count/quant_raw ฯลฯ) จากงานหนึ่งรั่วไปงานอื่นที่ไม่เกี่ยวข้องกัน
- ไม่ถือ sqlite3.Connection เดียวข้ามเธรด — แต่ละหน่วยงานเปิด connection ของตัวเองสั้นๆ
  แล้วปิด (WAL mode รองรับหลาย connection พร้อมกันอยู่แล้ว ปลอดภัยกว่าแชร์ Connection object
  เดียวข้าม asyncio.to_thread)
- `flow` เลือกว่าจะรัน graph ไหน: "manager" (pipeline หลักเดิม ไม่แตะเลย) หรือ
  "news_youtube" (กราฟแยกใหม่ที่มี interrupt() รอ human approve ก่อนเจาะลึกข่าว/คลิป)
"""
import asyncio
import json
import sqlite3
import uuid
from contextlib import closing
from typing import Any, Callable, Optional

from api import state_db

# run_fn(job_id=..., thread_id=..., instruction=..., flow=..., scope=..., resume_value=...) -> None
RunFn = Callable[..., None]


class JobQueue:
    def __init__(self, run_fn: RunFn, db_path: Optional[str] = None, flows: Optional[set[str]] = None):
        """flows=None (default) = ประมวลผลทุก flow เหมือนเดิม — ใส่ค่าเมื่อ JobQueue ตัวนี้แชร์ DB
        เดียวกับคิวอื่น (เช่น notebooklm_job_queue แชร์กับคิวหลัก) เพื่อกัน reenqueue_pending()
        กวาดงาน flow อื่นเข้าคิวตัวเองโดยไม่ได้ตั้งใจ
        """
        self._run_fn = run_fn
        self._db_path = db_path
        self._flows = list(flows) if flows is not None else None
        self._queue: "asyncio.Queue[str]" = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_task: Optional[asyncio.Task] = None

    def _conn(self) -> sqlite3.Connection:
        return state_db.get_connection(self._db_path)

    def start(self) -> None:
        if self._worker_task is None:
            self._loop = asyncio.get_running_loop()
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    def dispatch(self, instruction: str, card_id: Optional[str], flow: str = "manager", scope: str = "both") -> str:
        """คืน job_id — ถ้ามีงานเดียวกัน (flow เดิม + card_id เดิม + instruction เดิม + scope เดิม)
        กำลัง queued/running อยู่แล้ว คืน job_id เดิมแทนการรันซ้ำ (กันลาก/กด dispatch ซ้ำเปลืองงบ LLM)
        flow ต้องอยู่ใน key ด้วย ไม่งั้นงานชื่อเดียวกัน scope เดียวกันแต่คนละ flow จะ dedup ผิดตัว
        """
        idempotency_key = f"{flow}:{card_id or 'nocard'}:{instruction.strip()}:{scope}"
        with closing(self._conn()) as conn:
            existing = state_db.find_job_by_idempotency_key(conn, idempotency_key)
            if existing is not None and existing["status"] in ("queued", "running", "awaiting_approval"):
                return existing["job_id"]

            job_id = str(uuid.uuid4())
            thread_id = str(uuid.uuid4())
            key = idempotency_key
            if existing is not None:
                # เคยมีงานเดิม (done/error) ใช้ key นี้แล้ว — เติม job_id กันชน UNIQUE constraint
                key = f"{idempotency_key}:{job_id}"
            state_db.create_job(conn, job_id, thread_id, card_id, key, instruction, status="queued", flow=flow, scope=scope)

        self.enqueue(job_id)
        return job_id

    def resume(self, job_id: str, resume_value: dict[str, Any]) -> None:
        """Deprecated: Use state_db.claim_job_resume instead"""
        raise NotImplementedError("Use state_db.claim_job_resume directly to ensure atomicity")

    def enqueue(self, job_id: str) -> None:
        """Queue a job whose durable state has already been updated."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, job_id)
        else:
            self._queue.put_nowait(job_id)

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._run_job(job_id)
            except Exception:
                # _run_job เองมี try/except ครอบ run_fn ไว้แล้ว (เขียน status='error' ให้เสมอ) —
                # เผื่อพังนอกเหนือจากนั้น (เช่น sqlite error ระหว่าง update status) ต้องไม่ให้
                # worker loop ตายเงียบทั้งกระบวนการ ไม่งั้นงานถัดไปในคิวจะไม่ถูกประมวลผลเลย
                import logging
                logging.getLogger(__name__).exception("Unexpected error processing job %s", job_id)

    async def _run_job(self, job_id: str) -> None:
        with closing(self._conn()) as conn:
            job = state_db.get_job(conn, job_id)
            if job is None:
                return
            if job["status"] != "queued":
                return
            if not state_db.cas_job_status(conn, job_id, "queued", "running"):
                return

            if job["card_id"]:
                state_db.move_kanban_card(conn, job["card_id"], "executing", job_id)

            thread_id = job["thread_id"]
            instruction = job["instruction"]
            flow = job["flow"]
            scope = job["scope"]
            resume_value_raw = job["resume_value"]
            resume_value = json.loads(resume_value_raw) if resume_value_raw else None

        try:
            await asyncio.to_thread(
                self._run_fn,
                job_id=job_id,
                thread_id=thread_id,
                instruction=instruction,
                flow=flow,
                scope=scope,
                resume_value=resume_value,
            )
            with closing(self._conn()) as conn:
                current = state_db.get_job(conn, job_id)
                if current is not None:
                    if resume_value is not None:
                        state_db.clear_job_resume_value(conn, job_id)

                    if current["status"] == "running":
                        state_db.update_job_status(conn, job_id, "done")
                        if current["card_id"]:
                            state_db.move_kanban_card(conn, current["card_id"], "done", job_id)
                    elif current["status"] in ("done", "done_with_warnings", "done_with_errors", "error"):
                        if current["card_id"]:
                            col = "done" if current["status"] in ("done", "done_with_warnings", "done_with_errors") else "backlog"
                            state_db.move_kanban_card(conn, current["card_id"], col, job_id)
                    elif current["status"] == "awaiting_approval":
                        if current["card_id"]:
                            state_db.move_kanban_card(conn, current["card_id"], "approval", job_id)
        except Exception as e:
            def _extract_msg(ex: BaseException) -> str:
                if isinstance(ex, BaseExceptionGroup) and ex.exceptions:
                    return _extract_msg(ex.exceptions[0])
                return str(ex) or ex.__class__.__name__
            
            error_message = _extract_msg(e)
            with closing(self._conn()) as conn:
                state_db.append_job_log(
                    conn,
                    job_id,
                    "system_error",
                    f"Job failed: {error_message}",
                    role="reply",
                    label="System Error",
                )
                state_db.update_job_status(conn, job_id, "error", error_message=error_message)
                current = state_db.get_job(conn, job_id)
                if current and current["card_id"]:
                    state_db.move_kanban_card(conn, current["card_id"], "backlog", job_id)

    def reenqueue_pending(self) -> None:
        """เรียกตอน FastAPI startup — งานที่ยัง `queued` (ไม่ทันเริ่มรันตอน process ตาย)
        ปลอดภัยที่จะ re-push เข้า queue ใหม่ ส่วนงานที่ค้างสถานะ `running` ตอน process
        ตายกลางคัน จะไม่พยายาม resume เพราะไม่รู้ว่า LangGraph รันไปถึงไหนแล้วจริง —
        mark เป็น error ให้ user เห็นและสั่งงานใหม่เอง (ดู Rev.5 ข้อ 3: ห้ามหายเงียบ)

        งานที่ `awaiting_approval` ไม่ต้องแตะ — checkpoint ถูกบันทึกไว้แล้วตอน interrupt()
        เกิดขึ้น (ต้องมี checkpointer เสมอ) ปลอดภัยที่จะรอ user approve ทีหลังได้แม้ restart
        """
        with closing(self._conn()) as conn:
            for job in state_db.list_jobs_by_status(conn, ["running"], flows=self._flows):
                state_db.update_job_status(
                    conn, job["job_id"], "error",
                    error_message="ถูกขัดจังหวะเพราะ server restart กลางคัน — กรุณาสั่งงานใหม่อีกครั้ง",
                )
                if job["card_id"]:
                    state_db.move_kanban_card(conn, job["card_id"], "backlog", job["job_id"])
            queued = state_db.list_jobs_by_status(conn, ["queued"], flows=self._flows)
        for job in queued:
            self._queue.put_nowait(job["job_id"])


def _log_manager_messages(log_conn, job_id: str, event: dict) -> None:
    from langchain_core.messages import HumanMessage
    from core.utils import normalize_content

    for node_name, node_state in event.items():
        if not isinstance(node_state, dict) or "messages" not in node_state:
            continue
        messages = node_state.get("messages")
        if not messages:
            continue
        if not isinstance(messages, list):
            messages = [messages]
        # log ทุกข้อความใน list ไม่ใช่แค่ตัวสุดท้าย — เดิมอ่านแค่ messages[-1] ทำให้ node ที่คืน
        # หลายข้อความในครั้งเดียว (เช่น ingest_node ที่คืน 1 ข้อความต่อ 1 ไฟล์ที่ประมวลผล)
        # โชว์ใน terminal แค่บรรทัดสุดท้ายบรรทัดเดียว (เจอจริงจาก live test)
        for last in messages:
            content = normalize_content(getattr(last, "content", ""))
            if not content:
                continue
            # แยก instruction ที่ Manager ส่งต่อให้ worker (HumanMessage(name=..)) จากคำตอบของ
            # worker เอง (AIMessage) — ให้ terminal เห็นบทสนทนาจริงระหว่าง agent ไม่ใช่แค่
            # output สุดท้าย (ใช้ตรวจ prompting ที่ส่งไปด้วย)
            if isinstance(last, HumanMessage):
                sender = getattr(last, "name", None) or "manager"
                role = "instruction"
                label = f"{sender} → {node_name}"
            else:
                role = "reply"
                label = node_name
            state_db.append_job_log(log_conn, job_id, node_name, content, role=role, label=label)


def _append_manager_summary(log_conn, job_id: str, instruction: str, flow: str = "manager") -> None:
    if flow != "manager":
        return

    from agents.manager_agent import generate_manager_summary

    reply_logs = state_db.get_job_reply_logs(log_conn, job_id)
    if any(row["node_name"] == "manager_summary" for row in reply_logs):
        return

    excluded_nodes = {"supervisor", "manager_summary"}
    deliverables = [
        (row["node_name"] or "Specialist", row["content"] or "")
        for row in reply_logs
        if row["node_name"] not in excluded_nodes
        and not (row["node_name"] or "").startswith(("post_", "prepare_"))
    ]
    if not deliverables:
        deliverables = [
            (row["node_name"] or "Manager", row["content"] or "")
            for row in reply_logs
            if row["node_name"] == "supervisor"
        ]
    summary = generate_manager_summary(instruction, deliverables)
    if summary:
        state_db.append_job_log(log_conn, job_id, "manager_summary", summary, role="reply", label="Manager Summary")


def default_run_fn(
    job_id: str,
    thread_id: str,
    instruction: str,
    flow: str = "manager",
    scope: str = "both",
    resume_value: Optional[dict[str, Any]] = None,
) -> None:
    """run_fn จริงสำหรับ production — เรียก LangGraph ผ่าน with_retry, สตรีม log ทีละ node
    ลง job_logs (ผูกกับ job_id เดียวกัน) ให้ SSE endpoint tail ได้

    ถ้า stream เจอ __interrupt__ (LangGraph human-in-the-loop) จะตั้งสถานะ job เป็น
    awaiting_approval พร้อมเก็บ payload ไว้ แล้ว return ปกติ (ไม่ raise) — รอ resume ทีหลัง
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    from api.config import get_checkpoint_db_path
    from core.retry import with_retry

    with SqliteSaver.from_conn_string(get_checkpoint_db_path()) as checkpointer:
        terminal_status: Optional[str] = None
        terminal_error: Optional[str] = None

        if flow == "news_youtube":
            from agents.news_youtube_flow import build_news_youtube_graph
            graph = build_news_youtube_graph(checkpointer=checkpointer)
            fresh_inputs: dict = {"scope": scope}
        elif flow == "news_funnel":
            from agents.news_funnel_flow import build_news_funnel_graph
            graph = build_news_funnel_graph(checkpointer=checkpointer)
            fresh_inputs = {}
        elif flow == "youtube_pitch":
            from agents.youtube_pitch_flow import build_youtube_pitch_graph
            graph = build_youtube_pitch_graph(checkpointer=checkpointer)
            fresh_inputs = {"instruction": instruction}
        else:
            from agents.manager_agent import build_graph
            graph = build_graph(checkpointer=checkpointer)
            fresh_inputs = {"messages": [("user", instruction)]}

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 40,
            "tags": ["invest-agents", "web-session", flow],
            "metadata": {"run_type": "chain", "session_source": "web", "job_id": job_id},
        }

        if resume_value is not None:
            from langgraph.types import Command
            stream_input = Command(resume=resume_value)
        else:
            stream_input = fresh_inputs

        def _stream_and_log() -> None:
            nonlocal terminal_status, terminal_error
            with closing(state_db.get_connection()) as log_conn:
                for event in graph.stream(stream_input, config=config, stream_mode="updates"):
                    if "__interrupt__" in event:
                        payload = event["__interrupt__"][0].value
                        state_db.set_job_awaiting_approval(
                            log_conn, job_id, json.dumps(payload, ensure_ascii=False)
                        )
                        return
                    _log_manager_messages(log_conn, job_id, event)
                    if flow == "youtube_pitch":
                        for node_name in ("synthesize_notebooklm", "persist_parking_lot"):
                            node_update = event.get(node_name)
                            if isinstance(node_update, dict):
                                status = node_update.get("synthesis_status")
                                if status == "done_with_errors":
                                    failures = node_update.get("synthesis_failures") or []
                                    terminal_status = "done_with_errors"
                                    terminal_error = "\n".join(str(item) for item in failures) or "Some approved pitches failed"
                                elif status == "done_with_warnings" and terminal_status != "done_with_errors":
                                    warnings = node_update.get("synthesis_warnings") or []
                                    terminal_status = "done_with_warnings"
                                    terminal_error = "\n".join(str(item) for item in warnings) or "Completed with warnings (Unverified Drafts or Parking Lot partial notices)"
                _append_manager_summary(log_conn, job_id, instruction, flow=flow)


        with_retry(_stream_and_log)

        if terminal_status:
            with closing(state_db.get_connection()) as log_conn:
                state_db.update_job_status(log_conn, job_id, terminal_status, error_message=terminal_error)
