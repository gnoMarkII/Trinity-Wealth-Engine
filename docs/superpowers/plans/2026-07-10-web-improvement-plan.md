# Web UI + Backend API Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix correctness bugs, remove duplicated UI code, add CSS-only animation polish, add basic accessibility, and harden the FastAPI backend (`api/`) per `docs/superpowers/specs/2026-07-10-web-improvement-design.md`.

**Architecture:** No new dependencies. Frontend changes stay within `web/src` (React 19 + Vite + Tailwind 3.4, CSS-only animation). Backend changes stay within `api/` (FastAPI + SQLite). Backend tasks land first because two frontend tasks depend on the new atomic-dispatch API contract.

**Tech Stack:** React 19, TypeScript (strict, `noUnusedLocals`/`noUnusedParameters`), Tailwind 3.4, Vite 8, FastAPI, SQLite (stdlib `sqlite3`), pytest, oxlint.

## Global Constraints

- No new npm or pip dependencies.
- Do not touch `agents/`, `tools/`, `schemas/`, `main.py` — backend scope is `api/` only.
- All CSS animation must reuse keyframes already defined in `web/src/index.css` except one new keyframe (`notice-out`, added in Task 8).
- Every animation must end at a normal, readable state and respect the existing `prefers-reduced-motion: reduce` block in `index.css` (already present — no changes needed there).
- Frontend has no test framework. Verification per frontend task is: `npx tsc -b` (run from `web/`) must pass with no errors, plus a concrete manual check described in the step.
- Backend has pytest (`uv run pytest tests/api/`) — write/adjust real tests per backend task, TDD style.
- Commit after every task, using the repo's existing commit style (no `Co-Authored-By` requirement unless the user's global instructions apply).

## File Structure

Backend:
- `api/state_db.py` — SQLite schema/queries. Modify: schema init runs once per path, `dispatcher` column migrated away.
- `api/routes_kanban.py` — Kanban REST routes. Modify: drop `dispatcher` from valid columns.
- `api/jobs.py` — `JobQueue` worker. Modify: idempotency key, worker loop resilience.
- `api/routes_agents.py` — dispatch/stream/resume routes. Modify: atomic dispatch, async-safe SSE.
- `tests/api/*.py` — existing pytest suite. Modify where behavior changes; add new test functions.

Frontend:
- `web/src/lib/agentStatus.ts` — **new**. Shared `TerminalStatus` type + `columnForStatus()` helper.
- `web/src/api/client.ts` — add 401 interception hook.
- `web/src/auth/AuthContext.tsx` — register the 401 hook, force logout.
- `web/src/pages/Kanban.tsx` — dedupe move-on-status logic, fix timer leaks, notice animation, card stagger on initial load.
- `web/src/components/kanban/KanbanColumn.tsx` — thread stagger delay to cards.
- `web/src/components/kanban/KanbanCard.tsx` — accept `style` prop; keyboard accessibility.
- `web/src/components/kanban/KanbanDetailDrawer.tsx` — dedupe move logic, lazy width init.
- `web/src/components/LiveTerminal.tsx` — use shared `TerminalStatus` type.
- `web/src/pages/Macro.tsx` — fix `writing-vertical` class, loading skeleton, `aria-pressed` on filter tabs.
- `web/src/components/MacroReferenceDrawer.tsx` — fix Tailwind v4 classes, default export, dialog a11y, animation.
- `web/src/components/ui/Modal.tsx` — **new**. Shared modal shell (focus trap, escape, aria, animation).
- `web/src/components/ui/SegmentedControl.tsx` — **new**. Shared button-group control.
- `web/src/components/kanban/KanbanCardModal.tsx` — use `Modal` + `SegmentedControl`, label `htmlFor`.
- `web/src/components/kanban/EditTemplateModal.tsx` — use `Modal` + `SegmentedControl`, label `htmlFor`.
- `web/src/pages/Login.tsx` — label `htmlFor` for password field.
- `web/src/components/kanban/AddCardDropdown.tsx` — dropdown animation, `aria-expanded`/`aria-haspopup`.
- `web/src/components/RegimeProbabilityChart.tsx`, `web/src/components/PortfolioStanceBar.tsx` — bar-grow animation.
- `web/src/components/kanban/KanbanHeader.tsx` — `aria-pressed` on filter tabs.
- `web/src/index.css` — add `notice-out` keyframe.
- `web/src/assets/hero.png`, `react.svg`, `vite.svg` — delete (unused).

---

## Task 1: Backend — schema hygiene (init once, drop `dispatcher` column)

**Files:**
- Modify: `api/state_db.py:51-119`
- Modify: `api/routes_kanban.py:17`
- Modify: `tests/api/test_kanban_and_dispatch.py:1-14`
- Test: `tests/api/test_jobs_queue.py` (new test function)

**Interfaces:**
- Consumes: nothing new.
- Produces: `state_db.get_connection(db_path)` no longer re-runs `init_schema` on every call for a path it has already initialized this process. `state_db.init_schema(conn)` still migrates any `kanban_cards` row with `column_name = 'dispatcher'` to `'backlog'`. `routes_kanban._VALID_COLUMNS` no longer contains `"dispatcher"`.

- [ ] **Step 1: Write the failing test for schema-init-once behavior**

Add to `tests/api/test_jobs_queue.py` (append at end of file):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/api/test_jobs_queue.py -k "init_schema_only or dispatcher_cards_to_backlog" -v`
Expected: FAIL — `test_get_connection_only_initializes_schema_once_per_path` fails because `len(calls)` is 3 (init runs every call); `test_migrate_columns_moves_dispatcher_cards_to_backlog` fails with `AttributeError: module 'api.state_db' has no attribute '_INITIALIZED_DB_PATHS'`.

- [ ] **Step 3: Implement init-once tracking and the dispatcher migration in `api/state_db.py`**

Replace lines 51-60 (the `get_connection` function):

```python
_INITIALIZED_DB_PATHS: set[str] = set()


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or get_state_db_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    if path not in _INITIALIZED_DB_PATHS:
        init_schema(conn)
        _INITIALIZED_DB_PATHS.add(path)
    return conn
```

Add a new migration step. Replace the `_migrate_columns` function (currently lines 83-94) with:

```python
def _migrate_columns(conn: sqlite3.Connection) -> None:
    """เพิ่มคอลัมน์ใหม่ให้ตารางเก่าที่มีอยู่แล้วในไฟล์ SQLite จริง — `CREATE TABLE IF NOT EXISTS`
    ไม่แก้ตารางที่มีอยู่แล้ว ถ้า schema เปลี่ยนหลังจากไฟล์ .sqlite ถูกสร้างไปแล้ว (เช่น
    เพิ่ม flow/interrupt_payload ตอนทำ HITL) คอลัมน์ใหม่จะไม่มีอยู่จริง ทำให้ INSERT/SELECT
    พังด้วย "table X has no column named Y" — พบเจอจริงตอน dispatch งานจาก Kanban
    """
    for table, columns in _COLUMN_MIGRATIONS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for col_name, col_def in columns.items():
            if col_name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    conn.commit()


def _migrate_dispatcher_column_cards(conn: sqlite3.Connection) -> None:
    """คอลัมน์ 'dispatcher' ถูกตัดออกจาก UI แล้ว (เหลือ backlog/approval/executing/done) —
    การ์ดเก่าที่ยังค้างอยู่ใน 'dispatcher' ต้องย้ายกลับ backlog ไม่งั้นจะไม่โผล่ในหน้าเว็บเลย
    เพราะ frontend ไม่มีคอลัมน์นั้นให้ render อีกต่อไป
    """
    conn.execute("UPDATE kanban_cards SET column_name = 'backlog' WHERE column_name = 'dispatcher'")
    conn.commit()
```

Update `init_schema` (currently lines 115-119) to call the new migration:

```python
def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()
    _migrate_columns(conn)
    _migrate_dispatcher_column_cards(conn)
    _backfill_kanban_display_seq(conn)
```

- [ ] **Step 4: Remove `dispatcher` from the valid columns in `api/routes_kanban.py`**

Replace line 17:

```python
_VALID_COLUMNS = {"backlog", "approval", "executing", "done"}
```

- [ ] **Step 5: Fix the existing test that relied on `dispatcher` being a valid column**

In `tests/api/test_kanban_and_dispatch.py`, replace `test_create_and_move_kanban_card` (lines 1-14):

```python
def test_create_and_move_kanban_card(authed_client):
    r = authed_client.post("/api/kanban/cards", json={"title": "รันกลยุทธ์วันนี้"})
    assert r.status_code == 200
    body = r.json()
    assert body["created"] is True
    card = body["card"]
    assert card["column_name"] == "backlog"

    r = authed_client.put(
        "/api/kanban/move",
        json={"card_id": card["card_id"], "column_name": "executing"},
    )
    assert r.status_code == 200
    assert r.json()["column_name"] == "executing"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/api/test_jobs_queue.py tests/api/test_kanban_and_dispatch.py -v`
Expected: all PASS, including the two new tests and the fixed `test_create_and_move_kanban_card`.

- [ ] **Step 7: Run the full backend suite to check for regressions**

Run: `uv run pytest tests/api/ -v`
Expected: all PASS (no other test references `"dispatcher"`).

- [ ] **Step 8: Commit**

```bash
git add api/state_db.py api/routes_kanban.py tests/api/test_kanban_and_dispatch.py tests/api/test_jobs_queue.py
git commit -m "fix(api): init schema once per db path, migrate stray 'dispatcher' column cards to backlog"
```

---

## Task 2: Backend — JobQueue robustness (idempotency key includes flow, worker loop survives exceptions)

**Files:**
- Modify: `api/jobs.py:26-73`
- Test: `tests/api/test_jobs_queue.py` (new test functions)

**Interfaces:**
- Consumes: nothing new.
- Produces: `JobQueue.dispatch(instruction, card_id, flow="manager", scope="both")` — idempotency key now includes `flow`, so the same `card_id`+`instruction`+`scope` under a different `flow` creates a separate job. `JobQueue._worker_loop` no longer dies on an unexpected exception from `_run_job`. `JobQueue.stop()` awaits the worker task to actually finish (swallowing `CancelledError`) instead of firing-and-forgetting `cancel()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_jobs_queue.py`:

```python
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
    queue.start()

    job_id_bad = queue.dispatch("งานที่พัง", card_id="card-bad")
    job_id_good = queue.dispatch("งานที่ดี", card_id="card-good")

    async def _wait_for_processing():
        for _ in range(50):
            if job_id_good in processed:
                return
            await asyncio.sleep(0.05)

    asyncio.run(_wait_for_processing())
    asyncio.run(queue.stop())

    assert job_id_good in processed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/api/test_jobs_queue.py -k "different_flow or survives_run_job" -v`
Expected: FAIL — the flow test fails because both dispatches return the same `job_id`; the worker-loop test times out / fails because the loop dies after the first `RuntimeError` and `job_id_good` never gets processed.

- [ ] **Step 3: Implement the fix in `api/jobs.py`**

Replace the `dispatch` method (currently lines 45-64):

```python
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

        self._queue.put_nowait(job_id)
        return job_id
```

Replace `stop` (currently lines 40-43) and `_worker_loop` (currently lines 74-77):

```python
    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
```

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/api/test_jobs_queue.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full backend suite**

Run: `uv run pytest tests/api/ -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add api/jobs.py tests/api/test_jobs_queue.py
git commit -m "fix(api): include flow in job idempotency key, make worker loop resilient to exceptions"
```

---

## Task 3: Backend — atomic dispatch (move card + bind job_id in one request)

**Files:**
- Modify: `api/routes_agents.py:54-64`
- Test: `tests/api/test_kanban_and_dispatch.py` (new test function)

**Interfaces:**
- Consumes: `state_db.move_kanban_card(conn, card_id, column_name, job_id=None)` (already exists in `api/state_db.py:261-273`).
- Produces: `POST /api/agents/dispatch` — when `payload.card_id` is set, the card is moved to `"executing"` and bound to the new `job_id` as part of the same request, before the response is returned. Frontend no longer needs a follow-up `PUT /api/kanban/move` call after dispatch (see Task 6).

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_kanban_and_dispatch.py`:

```python
def test_dispatch_with_card_id_moves_card_to_executing_atomically(authed_client):
    """dispatch ต้องย้ายการ์ดเป็น executing + ผูก job_id ให้เองในคำขอเดียว — ไม่ต้องให้ frontend
    ยิง PUT /api/kanban/move ตามหลัง (ถ้า call ที่สองล้มเหลว การ์ดจะค้าง backlog ทั้งที่ job รันอยู่)
    """
    r = authed_client.post("/api/kanban/cards", json={"title": "งานที่จะ dispatch"})
    card_id = r.json()["card"]["card_id"]

    r = authed_client.post(
        "/api/agents/dispatch",
        json={"instruction": "วิเคราะห์ตลาดวันนี้", "card_id": card_id},
    )
    assert r.status_code == 200
    job = r.json()

    cards = authed_client.get("/api/kanban/cards").json()
    dispatched_card = next(c for c in cards if c["card_id"] == card_id)
    assert dispatched_card["column_name"] == "executing"
    assert dispatched_card["job_id"] == job["job_id"]


def test_dispatch_without_card_id_does_not_touch_kanban(authed_client):
    r = authed_client.post("/api/agents/dispatch", json={"instruction": "งานไม่มีการ์ด"})
    assert r.status_code == 200
    # ไม่ raise, ไม่มีการ์ดให้ move — แค่ยืนยันว่าไม่พังตอนไม่มี card_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/api/test_kanban_and_dispatch.py -k "dispatch_with_card_id or dispatch_without_card_id" -v`
Expected: FAIL — `test_dispatch_with_card_id_moves_card_to_executing_atomically` fails because the card's `column_name` is still `"backlog"` and `job_id` is `None`.

- [ ] **Step 3: Implement in `api/routes_agents.py`**

Replace `dispatch_job` (currently lines 54-64):

```python
@router.post("/api/agents/dispatch", response_model=JobStatusDTO)
def dispatch_job(payload: DispatchRequest, request: Request) -> JobStatusDTO:
    if not payload.instruction.strip():
        raise HTTPException(status_code=400, detail="instruction ว่างเปล่า")

    job_queue = request.app.state.job_queue
    job_id = job_queue.dispatch(payload.instruction, payload.card_id, flow=payload.flow, scope=payload.scope)

    with closing(state_db.get_connection()) as conn:
        # ย้ายการ์ดเป็น executing + ผูก job_id ในคำขอเดียวกับ dispatch — กันเคสที่ frontend
        # ยิง PUT /api/kanban/move ตามหลังแล้วล้มเหลว ทำให้การ์ดค้าง backlog ทั้งที่ job รันอยู่จริง
        if payload.card_id is not None:
            existing_card = state_db.get_kanban_card(conn, payload.card_id)
            if existing_card is not None:
                state_db.move_kanban_card(conn, payload.card_id, "executing", job_id=job_id)
        job = state_db.get_job(conn, job_id)
        return _job_to_dto(conn, job)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/api/test_kanban_and_dispatch.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full backend suite**

Run: `uv run pytest tests/api/ -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add api/routes_agents.py tests/api/test_kanban_and_dispatch.py
git commit -m "feat(api): move kanban card to executing atomically inside dispatch endpoint"
```

---

## Task 4: Backend — SSE stream uses `asyncio.to_thread` for DB reads

**Files:**
- Modify: `api/routes_agents.py:113-143`
- Test: `tests/api/test_kanban_and_dispatch.py` (new test function)

**Interfaces:**
- Consumes: nothing new.
- Produces: `GET /api/agents/stream/{job_id}` — the async generator's blocking SQLite calls run via `asyncio.to_thread` instead of directly in the event loop.

- [ ] **Step 1: Write the failing test**

Append to `tests/api/test_kanban_and_dispatch.py`:

```python
def test_stream_endpoint_completes_for_already_done_job(authed_client):
    """SSE endpoint ต้องทำงานได้ปกติ (ไม่ hang, ไม่ throw) หลังเปลี่ยนมาอ่าน DB ผ่าน
    asyncio.to_thread — ใช้ job ที่ done ไปแล้วเพื่อให้ stream จบทันทีโดยไม่ต้อง sleep รอ
    """
    r = authed_client.post("/api/agents/dispatch", json={"instruction": "งานทดสอบ stream"})
    job_id = r.json()["job_id"]

    import time
    for _ in range(50):
        status = authed_client.get(f"/api/agents/jobs/{job_id}").json()["status"]
        if status in ("done", "error"):
            break
        time.sleep(0.05)

    with authed_client.stream("GET", f"/api/agents/stream/{job_id}") as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes())
    assert b"event: done" in body or b"event: error" in body
```

- [ ] **Step 2: Run test to verify it currently passes (baseline) then confirm behavior after refactor**

Run: `uv run pytest tests/api/test_kanban_and_dispatch.py -k stream_endpoint_completes -v`
Expected: PASS even before the change (this test locks in current behavior — the refactor in Step 3 must keep it passing, not introduce a regression). This is a characterization test, not a red/green TDD test; proceed to Step 3 directly.

- [ ] **Step 3: Implement in `api/routes_agents.py`**

Replace `stream_job` (currently lines 113-143):

```python
@router.get("/api/agents/stream/{job_id}")
async def stream_job(job_id: str) -> StreamingResponse:
    def _read_next_batch(after_seq: int):
        with closing(state_db.get_connection()) as conn:
            job = state_db.get_job(conn, job_id)
            if job is None:
                return None, []
            logs = state_db.get_job_logs_since(conn, job_id, after_seq)
            return job, logs

    def _read_final_dto():
        with closing(state_db.get_connection()) as conn:
            job = state_db.get_job(conn, job_id)
            return _job_to_dto(conn, job)

    async def event_generator():
        after_seq = 0
        while True:
            job, logs = await asyncio.to_thread(_read_next_batch, after_seq)
            if job is None:
                yield f"event: error\ndata: {json.dumps({'detail': 'job not found'})}\n\n"
                return

            for row in logs:
                after_seq = row["seq"]
                payload = {
                    "node": row["node_name"],
                    "content": row["content"],
                    "role": row["role"],
                    "label": row["label"],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            if job["status"] in _TERMINAL_STATUSES:
                dto = await asyncio.to_thread(_read_final_dto)
                yield f"event: {job['status']}\ndata: {dto.model_dump_json()}\n\n"
                return

            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

- [ ] **Step 4: Run the test to verify it still passes**

Run: `uv run pytest tests/api/test_kanban_and_dispatch.py -k stream_endpoint_completes -v`
Expected: PASS.

- [ ] **Step 5: Run the full backend suite**

Run: `uv run pytest tests/api/ -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add api/routes_agents.py tests/api/test_kanban_and_dispatch.py
git commit -m "refactor(api): read SSE log batches via asyncio.to_thread instead of blocking the event loop"
```

---

## Task 5: Frontend — force logout on mid-session 401

**Files:**
- Modify: `web/src/api/client.ts:17-38`
- Modify: `web/src/auth/AuthContext.tsx:12-33`

**Interfaces:**
- Consumes: nothing new.
- Produces: `web/src/api/client.ts` exports `setUnauthorizedHandler(handler: (() => void) | null): void`. When any `request()` call (except `/api/auth/login`) receives a 401, the registered handler is invoked before the `ApiError` is thrown.

- [ ] **Step 1: Add the 401 hook to `web/src/api/client.ts`**

Replace lines 17-38 (the `request` function) and insert the new export right before it:

```typescript
let unauthorizedHandler: (() => void) | null = null

export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch {
      // ignore — ไม่มี JSON body
    }
    // เจอ 401 กลางคัน (session หมดอายุ/ถูกลบ) — ไม่ใช่ตอน login เอง (401 = รหัสผ่านผิด เป็น
    // เรื่องปกติที่ฟอร์ม login จัดการเอง ไม่ใช่สัญญาณว่า session หลุด) → บังคับ logout ไปหน้า login
    if (res.status === 401 && path !== '/api/auth/login') {
      unauthorizedHandler?.()
    }
    throw new ApiError(res.status, detail)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}
```

- [ ] **Step 2: Register the handler in `web/src/auth/AuthContext.tsx`**

Replace the full file:

```typescript
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { api, ApiError, setUnauthorizedHandler } from '../api/client'

interface AuthContextValue {
  status: 'loading' | 'authenticated' | 'unauthenticated'
  login: (password: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthContextValue['status']>('loading')

  useEffect(() => {
    fetch('/api/auth/me', { credentials: 'include' })
      .then((res) => res.json())
      .then((body) => setStatus(body.authenticated ? 'authenticated' : 'unauthenticated'))
      .catch(() => setStatus('unauthenticated'))
  }, [])

  useEffect(() => {
    setUnauthorizedHandler(() => setStatus('unauthenticated'))
    return () => setUnauthorizedHandler(null)
  }, [])

  async function login(password: string) {
    await api.login(password)
    setStatus('authenticated')
  }

  async function logout() {
    await api.logout()
    setStatus('unauthenticated')
  }

  return <AuthContext.Provider value={{ status, login, logout }}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth ต้องถูกเรียกภายใน <AuthProvider>')
  return ctx
}

export { ApiError }
```

- [ ] **Step 3: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 4: Manual verification**

Start the backend (`uvicorn api.main:app --reload`) and frontend (`npm run dev` in `web/`). Log in. In the browser devtools Application tab, delete the `invest_agents_session` cookie. Trigger any API call (e.g. click a filter tab that refetches, or wait for `AgentStatusPanel`'s 4s poll to hit a 401 — note: `getActiveAgentStatus` swallows errors silently per its own design, so instead trigger a user action like creating a kanban card). Confirm the app navigates to `/login` instead of showing a raw error string.

- [ ] **Step 5: Commit**

```bash
git add web/src/api/client.ts web/src/auth/AuthContext.tsx
git commit -m "fix(web): force logout on mid-session 401 instead of showing a bare error"
```

---

## Task 6: Frontend — dedupe move-on-status logic, remove redundant dispatch move call

**Files:**
- Create: `web/src/lib/agentStatus.ts`
- Modify: `web/src/components/LiveTerminal.tsx:1-58`
- Modify: `web/src/pages/Kanban.tsx:200-243`
- Modify: `web/src/components/kanban/KanbanDetailDrawer.tsx:76-90`

**Interfaces:**
- Consumes: backend atomic dispatch from Task 3 (dispatch now moves the card itself).
- Produces: `web/src/lib/agentStatus.ts` exports `TerminalStatus` (type) and `columnForStatus(status: TerminalStatus): string | null`. `LiveTerminal`'s `Props.onStatusChange` and internal state now use the imported `TerminalStatus` type instead of a locally-declared duplicate.

- [ ] **Step 1: Create `web/src/lib/agentStatus.ts`**

```typescript
export type TerminalStatus = 'idle' | 'streaming' | 'done' | 'error' | 'awaiting_approval'

// mapping สถานะ terminal → คอลัมน์ kanban ปลายทาง — เดิมเขียนซ้ำใน Kanban.tsx และ
// KanbanDetailDrawer.tsx แยกกัน ทำให้ทั้งคู่ยิง moveKanbanCard พร้อมกันได้เมื่อ drawer เปิด
// การ์ดที่กำลังรันอยู่ (ดู design spec ส่วนที่ 1 ข้อ 3)
export function columnForStatus(status: TerminalStatus): string | null {
  if (status === 'done') return 'done'
  if (status === 'error') return 'backlog'
  if (status === 'awaiting_approval') return 'approval'
  return null
}
```

- [ ] **Step 2: Use the shared type in `web/src/components/LiveTerminal.tsx`**

Replace line 1-2 (imports):

```typescript
import { useEffect, useRef, useState } from 'react'
import type { JobStatusDTO, NewsYoutubeApprovalPayload } from '../api/types'
import type { TerminalStatus } from '../lib/agentStatus'
import { nodeDisplayName } from '../lib/nodeDisplayNames'
```

Delete line 34 (the local `type TerminalStatus = ...` declaration) — the `LAST_STEP_DOT_CLASS` Record type below it (currently lines 36-42) stays unchanged since it already refers to `TerminalStatus` by name, which now resolves to the import.

- [ ] **Step 3: Typecheck after the LiveTerminal change**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 4: Rewrite `handleTerminalStatusChange` and `dispatchCard` in `web/src/pages/Kanban.tsx`**

Add the import near the top (after the existing `KanbanCardDTO` type import):

```typescript
import { columnForStatus, type TerminalStatus } from '../lib/agentStatus'
```

Replace `dispatchCard` (currently lines 200-216) — remove the now-redundant `moveKanbanCard` call since the backend dispatch endpoint (Task 3) moves the card atomically:

```typescript
  async function dispatchCard(card: KanbanCardDTO, flow: string) {
    setDispatching(true)
    try {
      const instruction = card.prompt?.trim() || card.title
      const job = await api.dispatchJob(instruction, card.card_id, flow, card.scope ?? 'both')
      setActiveDispatches((prev) => ({
        ...prev,
        [card.card_id]: { jobId: job.job_id, dispatchedAt: Date.now() },
      }))
      await refresh()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'สั่งงานไม่สำเร็จ')
    } finally {
      setDispatching(false)
    }
  }
```

Replace `handleTerminalStatusChange` (currently lines 224-243):

```typescript
  // background driver (hideUi) ขับแค่ dispatch → running → (done หรือ awaiting_approval)
  // เท่านั้น — พอถึง awaiting_approval มันปิด connection ตัวเองแล้ว (LiveTerminal design)
  // ส่วนที่เหลือ (approve → resume → done) เป็นหน้าที่ของ terminal ใน KanbanDetailDrawer
  // ที่ผู้ใช้ต้องเปิดดูเพื่อกด approve อยู่แล้ว ผูกกับ card_id ของมันเอง ไม่ชนกัน
  // รับ cardId ตรงๆ (ผูกกับ closure ของ card นั้นตอน render แต่ละ LiveTerminal instance)
  // แทนการอ้าง activeCardId ตัวเดียวส่วนกลาง เพื่อให้ track ได้หลาย job พร้อมกัน
  function handleTerminalStatusChange(cardId: string, status: TerminalStatus) {
    const targetColumn = columnForStatus(status)
    if (targetColumn) {
      // guard: ถ้าการ์ดอยู่คอลัมน์เป้าหมายอยู่แล้ว ไม่ยิง API ซ้ำ — กัน race กับ
      // KanbanDetailDrawer ที่ผูกกับ card_id เดียวกันตอนเปิด drawer ดูงานที่กำลังรันอยู่
      const card = cards.find((c) => c.card_id === cardId)
      if (card && card.column_name !== targetColumn) {
        api
          .moveKanbanCard(cardId, targetColumn)
          .then(refresh)
          .catch((e) => setError(e instanceof ApiError ? e.message : 'อัปเดตสถานะการ์ดไม่สำเร็จ'))
      }
      removeActiveDispatch(cardId)
    }
  }
```

- [ ] **Step 5: Typecheck after the Kanban.tsx change**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 6: Rewrite `handleStatusChange` in `web/src/components/kanban/KanbanDetailDrawer.tsx`**

Add the import near the top (after the `FLOW_TAG` import):

```typescript
import { columnForStatus, type TerminalStatus } from '../../lib/agentStatus'
```

Replace `handleStatusChange` (currently lines 76-90):

```typescript
  function handleStatusChange(status: TerminalStatus) {
    if (!card) return
    const targetColumn = columnForStatus(status)
    if (!targetColumn || card.column_name === targetColumn) return
    api
      .moveKanbanCard(card.card_id, targetColumn)
      .then(onCardTransition)
      .catch((e) => setError(e instanceof ApiError ? e.message : 'อัปเดตสถานะการ์ดไม่สำเร็จ'))
  }
```

- [ ] **Step 7: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 8: Manual verification**

With backend from Task 3 running: create a card, dispatch it, and confirm in the Network tab that only ONE request touches kanban state right after dispatch (the `POST /api/agents/dispatch` call itself — no separate `PUT /api/kanban/move`). Open the card's detail drawer while it's still running, and confirm no duplicate `PUT /api/kanban/move` fires once the job reaches `done` (only one call, from whichever of Kanban.tsx's background driver or the drawer's `LiveTerminal` sees the terminal status first).

- [ ] **Step 9: Commit**

```bash
git add web/src/lib/agentStatus.ts web/src/components/LiveTerminal.tsx web/src/pages/Kanban.tsx web/src/components/kanban/KanbanDetailDrawer.tsx
git commit -m "refactor(web): share terminal-status-to-column mapping, drop redundant move call after dispatch"
```

---

## Task 7: Frontend — fix notice/delete timer leaks, add notice-out animation

**Files:**
- Modify: `web/src/pages/Kanban.tsx:31-113,178-198,267-373`
- Modify: `web/src/index.css` (append new keyframe)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Kanban.tsx` tracks its `notice` `setTimeout` in a ref and clears it before scheduling a new one and on unmount; `deleteCard`'s `setTimeout` ids are tracked in a ref `Set` and cleared on unmount. The notice `<p>` now plays `animate-notice-in` while appearing and `animate-notice-out` during its last 150ms before removal.

- [ ] **Step 1: Add the `notice-out` keyframe to `web/src/index.css`**

Insert immediately after the `.animate-notice-in { ... }` block (after line 70):

```css
@keyframes notice-out {
  from {
    opacity: 1;
    transform: translateY(0);
  }
  to {
    opacity: 0;
    transform: translateY(-6px);
  }
}
.animate-notice-out {
  animation: notice-out 0.15s ease-in forwards;
}
```

- [ ] **Step 2: Add timer constants and refs in `web/src/pages/Kanban.tsx`**

Replace lines 31-33 (constants):

```typescript
const DONE_FADE_AFTER_DAYS = 7
const NOTICE_DISMISS_MS = 2500
const NOTICE_LEAVE_MS = 150
const DELETE_ANIM_MS = 150
```

In the component body, after the existing `const [notice, setNotice] = useState<string | null>(null)` line, add a new state and change the notice-related refs. Replace the state/ref declarations (currently around lines 51-69) with:

```typescript
export default function Kanban() {
  const [cards, setCards] = useState<KanbanCardDTO[]>([])
  const [modalState, setModalState] = useState<ModalState>(null)
  const [modalError, setModalError] = useState<string | null>(null)
  const [quickTemplates, setQuickTemplates] = useState<QuickTemplate[]>(loadQuickTemplates)
  const [editingTemplateIndex, setEditingTemplateIndex] = useState<number | null>(null)
  const [activeDispatches, setActiveDispatches] = useState<Record<string, ActiveDispatch>>({})
  const [liveNode, setLiveNode] = useState<Record<string, string | null>>({})
  const [liveLogCount, setLiveLogCount] = useState<Record<string, number>>({})
  const [nowTick, setNowTick] = useState(Date.now())
  const [dispatching, setDispatching] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [noticeLeaving, setNoticeLeaving] = useState(false)
  const [removingIds, setRemovingIds] = useState<Set<string>>(new Set())
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [flowFilter, setFlowFilter] = useState<FlowFilter>('all')
  const [search, setSearch] = useState('')
  const [selectedCardId, setSelectedCardId] = useState<string | null>(null)
  const tickTimer = useRef<number | null>(null)
  const noticeDismissTimer = useRef<number | null>(null)
  const noticeLeaveTimer = useRef<number | null>(null)
  const deleteTimers = useRef<Set<number>>(new Set())
  const activeCount = Object.keys(activeDispatches).length
```

- [ ] **Step 3: Rewrite `flashNotice` and add a cleanup effect**

Replace `flashNotice` (currently lines 110-113):

```typescript
  function flashNotice(message: string) {
    if (noticeDismissTimer.current) window.clearTimeout(noticeDismissTimer.current)
    if (noticeLeaveTimer.current) window.clearTimeout(noticeLeaveTimer.current)
    setNotice(message)
    setNoticeLeaving(false)
    noticeDismissTimer.current = window.setTimeout(() => {
      setNoticeLeaving(true)
      noticeLeaveTimer.current = window.setTimeout(() => {
        setNotice(null)
        setNoticeLeaving(false)
      }, NOTICE_LEAVE_MS)
    }, NOTICE_DISMISS_MS)
  }
```

Add a cleanup effect right after the existing tick-timer `useEffect` (which watches `activeCount`, currently ending around line 108):

```typescript
  useEffect(() => {
    return () => {
      if (noticeDismissTimer.current) window.clearTimeout(noticeDismissTimer.current)
      if (noticeLeaveTimer.current) window.clearTimeout(noticeLeaveTimer.current)
      deleteTimers.current.forEach((id) => window.clearTimeout(id))
    }
  }, [])
```

- [ ] **Step 4: Track the delete timer id in `deleteCard`**

Replace `deleteCard` (currently lines 178-198):

```typescript
  async function deleteCard(cardId: string) {
    setRemovingIds((prev) => new Set(prev).add(cardId))
    const timerId = window.setTimeout(async () => {
      deleteTimers.current.delete(timerId)
      try {
        await api.deleteKanbanCard(cardId)
        removeActiveDispatch(cardId)
        if (cardId === selectedCardId) setSelectedCardId(null)
        await refresh()
      } catch (e) {
        // เดิมไม่มี catch — ถ้า delete fail การ์ดจะค้าง opacity เฟดครึ่งเดียวตลอดไป
        // เพราะ removingIds ไม่เคยถูกลบออก ตอนนี้ finally ด้านล่างลบให้เสมอไม่ว่าจะสำเร็จหรือพัง
        setError(e instanceof ApiError ? e.message : 'ลบการ์ดไม่สำเร็จ')
      } finally {
        setRemovingIds((prev) => {
          const next = new Set(prev)
          next.delete(cardId)
          return next
        })
      }
    }, DELETE_ANIM_MS)
    deleteTimers.current.add(timerId)
  }
```

- [ ] **Step 5: Wire the leaving animation into the notice `<p>` render**

Replace the notice paragraph (currently lines 294-298):

```tsx
        {notice && (
          <p
            className={`rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800 ${
              noticeLeaving ? 'animate-notice-out' : 'animate-notice-in'
            }`}
          >
            {notice}
          </p>
        )}
```

- [ ] **Step 6: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 7: Manual verification**

Start the dev server. Trigger two duplicate-card notices in quick succession (add the same quick-template card twice within 2.5s). Confirm the second notice's timer isn't cut short by the first (each shown for the full ~2.5s from when it appeared) and that the notice fades/slides out smoothly rather than disappearing abruptly. Delete a card and confirm it still fades out (`animate-card-out`) before being removed from the list.

- [ ] **Step 8: Commit**

```bash
git add web/src/pages/Kanban.tsx web/src/index.css
git commit -m "fix(web): stop notice/delete setTimeout leaks in Kanban, add notice-out animation"
```

---

## Task 8: Frontend — lazy drawer width init (no flash)

**Files:**
- Modify: `web/src/components/kanban/KanbanDetailDrawer.tsx:32-49`

**Interfaces:**
- Consumes: existing `loadStoredWidth()` (unchanged, `web/src/components/kanban/KanbanDetailDrawer.tsx:19-23`).
- Produces: `width` state is initialized from `localStorage` synchronously on first render (no `useEffect` flash).

- [ ] **Step 1: Replace the width state and remove the flash-causing effect**

Replace lines 32-49 (component start through the second `useEffect`):

```typescript
export default function KanbanDetailDrawer({ card, onClose, onCardTransition }: Props) {
  const [approvalPayload, setApprovalPayload] = useState<NewsYoutubeApprovalPayload | null>(null)
  const [approving, setApproving] = useState(false)
  const [terminalKey, setTerminalKey] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [width, setWidth] = useState(loadStoredWidth)
  const [resizing, setResizing] = useState(false)
  const widthRef = useRef(width)

  useEffect(() => {
    setApprovalPayload(null)
    setTerminalKey((k) => k + 1)
    setError(null)
  }, [card?.card_id])
```

- [ ] **Step 2: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0. (`noUnusedLocals` will fail the build if `loadStoredWidth` becomes unused — it's still used here as the lazy initializer, so this should be clean.)

- [ ] **Step 3: Manual verification**

Set a custom drawer width (drag the resize handle), reload the page, and select a card. Confirm the drawer renders at the saved width immediately with no visible snap/flash from the default 384px.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/kanban/KanbanDetailDrawer.tsx
git commit -m "fix(web): initialize kanban drawer width lazily to avoid a resize flash on load"
```

---

## Task 9: Frontend — Macro/MacroReferenceDrawer fixes, a11y, and animation

**Files:**
- Modify: `web/src/pages/Macro.tsx:1-9,394-412`
- Modify: `web/src/components/MacroReferenceDrawer.tsx` (full rewrite)

**Interfaces:**
- Consumes: nothing new.
- Produces: `MacroReferenceDrawer` becomes a default export (`export default function MacroReferenceDrawer(...)`) instead of a named `React.FC` const. `Macro.tsx` updates its import accordingly.

- [ ] **Step 1: Fix the import and the vertical-text class typo in `web/src/pages/Macro.tsx`**

Replace line 8:

```typescript
import MacroReferenceDrawer from '../components/MacroReferenceDrawer'
```

Replace line 401 (inside the floating references button):

```tsx
        <span className="writing-vertical tracking-wide">References</span>
```

- [ ] **Step 2: Rewrite `web/src/components/MacroReferenceDrawer.tsx`**

Replace the full file:

```tsx
import { useState, useEffect, useRef } from 'react'
import type { MacroDashboardDTO } from '../api/types'

interface Props {
  data: MacroDashboardDTO
  isOpen: boolean
  onClose: () => void
}

type TabKey = 'source_files' | 'observables' | 'items'

export default function MacroReferenceDrawer({ data, isOpen, onClose }: Props) {
  const [activeTab, setActiveTab] = useState<TabKey>('source_files')
  const closeButtonRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!isOpen) return
    closeButtonRef.current?.focus()
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  // Extract source files
  const sourceFiles = data.source_files || []

  // Collect all observable refs across items
  const observableSet = new Set<string>()
  data.asset_allocation?.forEach((a) => {
    a.observable_refs?.forEach((r) => observableSet.add(r))
  })
  data.pair_trades?.forEach((pt) => {
    pt.observable_refs?.forEach((r) => observableSet.add(r))
  })
  data.regime_evidence?.forEach((re) => {
    re.observable_refs?.forEach((r) => observableSet.add(r))
  })
  const allObservables = Array.from(observableSet)

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div
        className="animate-fade-in fixed inset-0 bg-black/40 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Slide-over Panel */}
      <div className="fixed inset-y-0 right-0 flex max-w-full pl-10">
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="macro-reference-drawer-title"
          className="animate-drawer-in flex w-screen max-w-lg flex-col bg-white shadow-2xl"
        >
          {/* Drawer Header */}
          <div className="flex items-center justify-between border-b border-zinc-200 bg-zinc-900 px-6 py-4 text-white">
            <div className="flex items-center gap-2.5">
              <span className="text-xl">📚</span>
              <div>
                <h2 id="macro-reference-drawer-title" className="text-base font-semibold">
                  แหล่งอ้างอิงและข้อมูลฐาน (References)
                </h2>
                <p className="text-xs text-zinc-400">
                  ที่มาข้อมูล รายการไฟล์สแนปชอต และตัวชี้วัดเศรษฐกิจ
                </p>
              </div>
            </div>
            <button
              ref={closeButtonRef}
              onClick={onClose}
              className="rounded-lg bg-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-700 hover:text-white"
            >
              ✕ ปิดหน้าต่าง
            </button>
          </div>

          {/* Sub-Tabs Selector */}
          <div className="flex border-b border-zinc-200 bg-zinc-50/80 px-6 pt-3">
            <button
              onClick={() => setActiveTab('source_files')}
              aria-pressed={activeTab === 'source_files'}
              className={`mr-6 border-b-2 pb-3 text-xs font-semibold transition-colors ${
                activeTab === 'source_files'
                  ? 'border-zinc-900 text-zinc-900'
                  : 'border-transparent text-zinc-500 hover:text-zinc-800'
              }`}
            >
              📑 ไฟล์ต้นทาง ({sourceFiles.length})
            </button>
            <button
              onClick={() => setActiveTab('observables')}
              aria-pressed={activeTab === 'observables'}
              className={`mr-6 border-b-2 pb-3 text-xs font-semibold transition-colors ${
                activeTab === 'observables'
                  ? 'border-zinc-900 text-zinc-900'
                  : 'border-transparent text-zinc-500 hover:text-zinc-800'
              }`}
            >
              📊 ตัวชี้วัดตลาด ({allObservables.length})
            </button>
            <button
              onClick={() => setActiveTab('items')}
              aria-pressed={activeTab === 'items'}
              className={`border-b-2 pb-3 text-xs font-semibold transition-colors ${
                activeTab === 'items'
                  ? 'border-zinc-900 text-zinc-900'
                  : 'border-transparent text-zinc-500 hover:text-zinc-800'
              }`}
            >
              🔍 แยกรายกลยุทธ์
            </button>
          </div>

          {/* Tab Contents */}
          <div className="custom-scrollbar flex-1 overflow-y-auto overscroll-contain p-6">
            {/* TAB 1: Source Files */}
            {activeTab === 'source_files' && (
              <div className="space-y-4">
                <div className="rounded-lg bg-zinc-100 p-3 text-xs text-zinc-600">
                  <span className="font-semibold text-zinc-800">ระบบประเมินผล:</span>{' '}
                  {data.generated_by || 'Strategic Allocator / Macro Core Engine'}{' '}
                  <span className="text-zinc-400">({data.evaluated_at})</span>
                </div>

                <h3 className="text-sm font-semibold text-zinc-900">
                  รายการไฟล์ในฐานข้อมูล (Obsidian Vault Knowledge Base)
                </h3>

                {sourceFiles.length === 0 ? (
                  <p className="text-xs italic text-zinc-400">ไม่มีรายการไฟล์ต้นทางระบุไว้</p>
                ) : (
                  <ul className="space-y-2">
                    {sourceFiles.map((file, idx) => {
                      const isPython = file.endsWith('.py')
                      return (
                        <li
                          key={idx}
                          className="flex items-center justify-between rounded-xl border border-zinc-200 bg-white p-3 shadow-sm transition-colors hover:border-zinc-300"
                        >
                          <div className="flex items-center gap-2.5">
                            <span className="text-base">{isPython ? '⚙️' : '📄'}</span>
                            <div>
                              <div className="font-mono text-xs font-semibold text-zinc-800">{file}</div>
                              <div className="text-[11px] text-zinc-400">
                                {isPython ? 'Quantitative Analytics Script' : 'Obsidian PKM Snapshot / Report'}
                              </div>
                            </div>
                          </div>
                          <span className="rounded-md bg-zinc-100 px-2 py-0.5 text-[10px] font-medium text-zinc-600">
                            {isPython ? 'Code Module' : 'Markdown'}
                          </span>
                        </li>
                      )
                    })}
                  </ul>
                )}
              </div>
            )}

            {/* TAB 2: Observables */}
            {activeTab === 'observables' && (
              <div className="space-y-4">
                <div className="rounded-lg bg-blue-50/60 p-3 text-xs text-blue-900">
                  รายการรหัสซีรีส์และตัวชี้วัดเศรษฐกิจมหภาค (Observable Metrics) จาก FRED,
                  ตลาดการเงิน และระบบดัชนีชี้วัดภายใน
                </div>

                {allObservables.length === 0 ? (
                  <p className="text-xs italic text-zinc-400">ไม่มีตัวชี้วัดระบุไว้</p>
                ) : (
                  <div className="grid grid-cols-1 gap-2.5">
                    {allObservables.map((obs, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between rounded-xl border border-zinc-200 bg-zinc-50/60 p-3"
                      >
                        <div className="flex items-center gap-2">
                          <span className="h-2 w-2 rounded-full bg-blue-600" />
                          <span className="font-mono text-xs font-semibold text-zinc-900">{obs}</span>
                        </div>
                        <span className="text-[11px] font-medium text-zinc-500">Economic Indicator</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* TAB 3: By Strategy Item */}
            {activeTab === 'items' && (
              <div className="space-y-5">
                {/* Asset Allocations */}
                <div>
                  <h3 className="mb-2.5 text-xs font-bold uppercase tracking-wider text-zinc-500">
                    Cross-Asset Allocations
                  </h3>
                  <div className="space-y-2">
                    {data.asset_allocation?.map((a, idx) => (
                      <div key={idx} className="rounded-xl border border-zinc-200 bg-white p-3 text-xs">
                        <div className="flex items-center justify-between font-semibold text-zinc-900">
                          <span>{a.asset_class}</span>
                          <span className="text-zinc-500">{a.stance}</span>
                        </div>
                        {(a.source_refs && a.source_refs.length > 0) ||
                        (a.observable_refs && a.observable_refs.length > 0) ? (
                          <div className="mt-2 space-y-1 text-zinc-600">
                            {a.source_refs && a.source_refs.length > 0 && (
                              <div>
                                <span className="font-semibold text-zinc-700">Sources: </span>
                                {a.source_refs.join(', ')}
                              </div>
                            )}
                            {a.observable_refs && a.observable_refs.length > 0 && (
                              <div>
                                <span className="font-semibold text-zinc-700">Observables: </span>
                                {a.observable_refs.join(', ')}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="mt-1 italic text-zinc-400">อ้างอิงจากรายงานหลัก</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Pair Trades */}
                {data.pair_trades && data.pair_trades.length > 0 && (
                  <div>
                    <h3 className="mb-2.5 text-xs font-bold uppercase tracking-wider text-zinc-500">
                      Tactical Pair Trades
                    </h3>
                    <div className="space-y-2">
                      {data.pair_trades.map((pt, idx) => (
                        <div key={idx} className="rounded-xl border border-zinc-200 bg-white p-3 text-xs">
                          <div className="font-semibold text-zinc-900">
                            Long {pt.long_leg} / Short {pt.short_leg}
                          </div>
                          {(pt.source_refs && pt.source_refs.length > 0) ||
                          (pt.observable_refs && pt.observable_refs.length > 0) ? (
                            <div className="mt-2 space-y-1 text-zinc-600">
                              {pt.source_refs && pt.source_refs.length > 0 && (
                                <div>
                                  <span className="font-semibold text-zinc-700">Sources: </span>
                                  {pt.source_refs.join(', ')}
                                </div>
                              )}
                              {pt.observable_refs && pt.observable_refs.length > 0 && (
                                <div>
                                  <span className="font-semibold text-zinc-700">Observables: </span>
                                  {pt.observable_refs.join(', ')}
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="mt-1 italic text-zinc-400">อ้างอิงจากรายงานหลัก</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Drawer Footer */}
          <div className="border-t border-zinc-200 bg-zinc-50 px-6 py-3 text-right">
            <button
              onClick={onClose}
              className="rounded-lg bg-zinc-900 px-4 py-2 text-xs font-semibold text-white hover:bg-zinc-800"
            >
              ปิดหน้าต่างอ้างอิง
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
```

Note: the source-file `<li>` originally used `shadow-xs`, a Tailwind v4-only class (silently a no-op on this project's Tailwind 3.4) — the rewrite above already replaces it with `shadow-sm`.

- [ ] **Step 3: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 4: Manual verification**

Open the Macro page, click "References" (both the header button and the floating side tab). Confirm: the drawer slides in from the right with a visible animation (not an instant snap), the backdrop fades in, the close button is focused immediately (visible focus ring), Escape closes it, and the vertical "References" text on the floating tab renders rotated (not falling back to horizontal — confirms the class name fix). Confirm `backdrop-blur-sm`/`shadow-sm` actually blur/shadow visibly (Tailwind v4-only classes were silently no-op-ing before).

- [ ] **Step 5: Commit**

```bash
git add web/src/pages/Macro.tsx web/src/components/MacroReferenceDrawer.tsx
git commit -m "fix(web): correct Tailwind v3 classes, convert MacroReferenceDrawer to default export, add dialog a11y + slide animation"
```

---

## Task 10: Frontend — shared `Modal` and `SegmentedControl` components

**Files:**
- Create: `web/src/components/ui/Modal.tsx`
- Create: `web/src/components/ui/SegmentedControl.tsx`

**Interfaces:**
- Produces:
  - `Modal({ titleId, onClose, children }: { titleId: string; onClose: () => void; children: ReactNode })` — default export. Renders a backdrop (click-to-close, `animate-fade-in`) and a dialog box (`role="dialog"`, `aria-modal="true"`, `aria-labelledby={titleId}`, `animate-modal-in`) with Escape-to-close, Tab focus trap, autofocus on the first focusable element, and focus restoration to the previously-focused element on unmount.
  - `SegmentedControl({ options, value, onChange }: { options: { key: string; label: string }[]; value: string; onChange: (value: string) => void })` — default export. Renders a row of toggle buttons with `aria-pressed`.

- [ ] **Step 1: Create `web/src/components/ui/Modal.tsx`**

```tsx
import { useEffect, useRef, type ReactNode } from 'react'

interface Props {
  titleId: string
  onClose: () => void
  children: ReactNode
}

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'

export default function Modal({ titleId, onClose, children }: Props) {
  const dialogRef = useRef<HTMLDivElement>(null)
  const previouslyFocused = useRef<HTMLElement | null>(null)

  useEffect(() => {
    previouslyFocused.current = document.activeElement as HTMLElement | null
    const container = dialogRef.current
    const firstField = container?.querySelector<HTMLElement>(FOCUSABLE_SELECTOR)
    firstField?.focus()

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        onClose()
        return
      }
      if (e.key !== 'Tab' || !container) return
      const focusable = container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)
      if (focusable.length === 0) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      previouslyFocused.current?.focus()
    }
  }, [onClose])

  return (
    <div onClick={onClose} className="animate-fade-in fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div
        ref={dialogRef}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="animate-modal-in w-full max-w-lg rounded-xl border border-zinc-200 bg-white p-5 shadow-lg"
      >
        {children}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Create `web/src/components/ui/SegmentedControl.tsx`**

```tsx
interface Option {
  key: string
  label: string
}

interface Props {
  options: Option[]
  value: string
  onChange: (value: string) => void
}

export default function SegmentedControl({ options, value, onChange }: Props) {
  return (
    <div className="flex w-fit gap-1 rounded-lg border border-zinc-200 bg-white p-1">
      {options.map((opt) => (
        <button
          key={opt.key}
          type="button"
          onClick={() => onChange(opt.key)}
          aria-pressed={value === opt.key}
          className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
            value === opt.key ? 'bg-terra/10 text-terra' : 'text-zinc-500 hover:text-zinc-800'
          }`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}
```

- [ ] **Step 3: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0. (Both files are unused so far — `noUnusedLocals`/`noUnusedParameters` apply to locals within a file, not to whole unconsumed exports, so this passes; they'll be wired up in Tasks 11-12.)

- [ ] **Step 4: Commit**

```bash
git add web/src/components/ui/Modal.tsx web/src/components/ui/SegmentedControl.tsx
git commit -m "feat(web): add shared Modal and SegmentedControl components"
```

---

## Task 11: Frontend — `KanbanCardModal` uses `Modal` + `SegmentedControl`

**Files:**
- Modify: `web/src/components/kanban/KanbanCardModal.tsx` (full rewrite)

**Interfaces:**
- Consumes: `Modal` (Task 10), `SegmentedControl` (Task 10).
- Produces: same external `Props` and behavior as before (`mode`, `initialTitle`, `initialPrompt`, `initialFlow`, `initialScope`, `errorMessage`, `onClose`, `onSubmit`) — no change to how `Kanban.tsx` uses this component.

- [ ] **Step 1: Rewrite `web/src/components/kanban/KanbanCardModal.tsx`**

```tsx
import { useState } from 'react'
import Modal from '../ui/Modal'
import SegmentedControl from '../ui/SegmentedControl'
import Button from '../ui/Button'
import TextInput from '../ui/TextInput'

interface Props {
  mode: 'create' | 'edit'
  initialTitle?: string
  initialPrompt?: string
  initialFlow?: string
  initialScope?: string
  errorMessage?: string | null
  onClose: () => void
  onSubmit: (values: { title: string; prompt: string; flow: string; scope: string }) => Promise<void>
}

const FLOW_OPTIONS: { key: string; label: string }[] = [
  { key: 'manager', label: 'Macro' },
  { key: 'news_youtube', label: 'News/YouTube' },
]

const SCOPE_OPTIONS: { key: string; label: string }[] = [
  { key: 'news', label: 'ข่าวเท่านั้น' },
  { key: 'youtube', label: 'YouTube เท่านั้น' },
  { key: 'both', label: 'ทั้งคู่' },
]

export default function KanbanCardModal({
  mode,
  initialTitle = '',
  initialPrompt = '',
  initialFlow = 'manager',
  initialScope = 'both',
  errorMessage,
  onClose,
  onSubmit,
}: Props) {
  const [title, setTitle] = useState(initialTitle)
  const [prompt, setPrompt] = useState(initialPrompt)
  const [flow, setFlow] = useState(initialFlow)
  const [scope, setScope] = useState(initialScope)
  const [submitting, setSubmitting] = useState(false)

  const trimmedTitle = title.trim()

  async function handleSubmit() {
    if (!trimmedTitle || submitting) return
    setSubmitting(true)
    try {
      await onSubmit({ title: trimmedTitle, prompt: prompt.trim(), flow, scope })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Modal titleId="kanban-card-modal-title" onClose={onClose}>
      <h2 id="kanban-card-modal-title" className="mb-4 text-sm font-semibold text-zinc-900">
        {mode === 'create' ? 'เพิ่มการ์ดใหม่' : 'แก้ไขการ์ด'}
      </h2>

      <div className="space-y-4">
        <div>
          <label htmlFor="kanban-card-title" className="mb-1 block text-xs font-medium text-zinc-600">
            ชื่อการ์ด
          </label>
          <TextInput
            id="kanban-card-title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
            placeholder="ชื่อการ์ดสั้นๆ เช่น 'วิเคราะห์พอร์ตวันนี้'"
            className="w-full"
          />
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-600">ประเภทงาน</label>
          <SegmentedControl options={FLOW_OPTIONS} value={flow} onChange={setFlow} />
        </div>

        {flow === 'news_youtube' && (
          <div>
            <label className="mb-1 block text-xs font-medium text-zinc-600">ขอบเขต</label>
            <SegmentedControl options={SCOPE_OPTIONS} value={scope} onChange={setScope} />
          </div>
        )}

        <div>
          <label htmlFor="kanban-card-prompt" className="mb-1 block text-xs font-medium text-zinc-600">
            Prompt สำหรับ Manager (ไม่บังคับ)
          </label>
          <textarea
            id="kanban-card-prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') handleSubmit()
            }}
            rows={7}
            placeholder="อธิบายรายละเอียดงานให้ agent เข้าใจชัดเจน เช่น ขอบเขตการวิเคราะห์ กรอบเวลา สินทรัพย์ที่สนใจ ข้อมูลอ้างอิงพิเศษ ฯลฯ"
            className="w-full resize-none rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 outline-none transition-colors placeholder-zinc-400 focus:border-terra focus:ring-1 focus:ring-terra/30"
          />
          <p className="mt-1 text-xs text-zinc-500">ถ้าเว้นว่างไว้ ระบบจะใช้ชื่อการ์ดเป็นคำสั่งแทน</p>
        </div>
      </div>

      {errorMessage && (
        <p className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">{errorMessage}</p>
      )}

      <div className="mt-5 flex justify-end gap-2">
        <Button type="button" variant="secondary" onClick={onClose}>
          ยกเลิก
        </Button>
        <Button type="button" onClick={handleSubmit} disabled={!trimmedTitle || submitting}>
          {mode === 'create' ? 'เพิ่มการ์ด' : 'บันทึก'}
        </Button>
      </div>
    </Modal>
  )
}
```

- [ ] **Step 2: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 3: Manual verification**

Open "+ เพิ่มการ์ด" → "+ กำหนดเอง...". Confirm: the title field is focused automatically, the box animates in (scale/fade), pressing Tab from the last field (Save button) wraps focus back to the title field (and Shift+Tab from the title field wraps to the last button), Escape closes the modal, and focus returns to the button that opened it. Confirm flow/scope segmented buttons still switch correctly and news_youtube reveals the scope row.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/kanban/KanbanCardModal.tsx
git commit -m "refactor(web): rebuild KanbanCardModal on shared Modal + SegmentedControl"
```

---

## Task 12: Frontend — `EditTemplateModal` uses `Modal` + `SegmentedControl`

**Files:**
- Modify: `web/src/components/kanban/EditTemplateModal.tsx` (full rewrite)

**Interfaces:**
- Consumes: `Modal` (Task 10), `SegmentedControl` (Task 10).
- Produces: same external `Props` and behavior as before (`template`, `onClose`, `onSave`).

- [ ] **Step 1: Rewrite `web/src/components/kanban/EditTemplateModal.tsx`**

```tsx
import { useState } from 'react'
import Modal from '../ui/Modal'
import SegmentedControl from '../ui/SegmentedControl'
import Button from '../ui/Button'
import TextInput from '../ui/TextInput'
import type { QuickTemplate } from '../../lib/quickTemplateStorage'

interface Props {
  template: QuickTemplate
  onClose: () => void
  onSave: (template: QuickTemplate) => void
}

const FLOW_OPTIONS: { key: string; label: string }[] = [
  { key: 'manager', label: 'Macro' },
  { key: 'news_youtube', label: 'News/YouTube' },
]

const SCOPE_OPTIONS: { key: string; label: string }[] = [
  { key: 'news', label: 'ข่าวเท่านั้น' },
  { key: 'youtube', label: 'YouTube เท่านั้น' },
  { key: 'both', label: 'ทั้งคู่' },
]

export default function EditTemplateModal({ template, onClose, onSave }: Props) {
  const [label, setLabel] = useState(template.label)
  const [instruction, setInstruction] = useState(template.instruction)
  const [flow, setFlow] = useState(template.flow)
  const [scope, setScope] = useState(template.scope)

  const trimmedLabel = label.trim()
  const trimmedInstruction = instruction.trim()
  const canSave = trimmedLabel.length > 0 && trimmedInstruction.length > 0

  function handleSave() {
    if (!canSave) return
    onSave({ label: trimmedLabel, instruction: trimmedInstruction, flow, scope })
  }

  return (
    <Modal titleId="edit-template-modal-title" onClose={onClose}>
      <h2 id="edit-template-modal-title" className="mb-4 text-sm font-semibold text-zinc-900">
        แก้ไขปุ่มลัด
      </h2>

      <div className="space-y-4">
        <div>
          <label htmlFor="edit-template-label" className="mb-1 block text-xs font-medium text-zinc-600">
            ชื่อปุ่มลัด
          </label>
          <TextInput
            id="edit-template-label"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSave()}
            placeholder="เช่น 'วิเคราะห์เศรษฐกิจมหภาค'"
            className="w-full"
          />
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-zinc-600">ประเภทงาน</label>
          <SegmentedControl options={FLOW_OPTIONS} value={flow} onChange={setFlow} />
        </div>

        {flow === 'news_youtube' && (
          <div>
            <label className="mb-1 block text-xs font-medium text-zinc-600">ขอบเขต</label>
            <SegmentedControl options={SCOPE_OPTIONS} value={scope} onChange={setScope} />
          </div>
        )}

        <div>
          <label htmlFor="edit-template-instruction" className="mb-1 block text-xs font-medium text-zinc-600">
            คำสั่งเต็มที่จะส่งให้ Manager
          </label>
          <textarea
            id="edit-template-instruction"
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') handleSave()
            }}
            rows={5}
            placeholder="คำสั่งที่จะถูกส่งให้ agent ทันทีเมื่อกดปุ่มลัดนี้"
            className="w-full resize-none rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 outline-none transition-colors placeholder-zinc-400 focus:border-terra focus:ring-1 focus:ring-terra/30"
          />
        </div>
      </div>

      <div className="mt-5 flex justify-end gap-2">
        <Button type="button" variant="secondary" onClick={onClose}>
          ยกเลิก
        </Button>
        <Button type="button" onClick={handleSave} disabled={!canSave}>
          บันทึก
        </Button>
      </div>
    </Modal>
  )
}
```

- [ ] **Step 2: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 3: Manual verification**

Open the "+ เพิ่มการ์ด" dropdown, click the pencil icon next to a quick template. Confirm the same focus-trap/animate/Escape behavior as Task 11, and that saving updates the template (check `localStorage['kanban-quick-templates']` in devtools, or that the dropdown shows the new label).

- [ ] **Step 4: Commit**

```bash
git add web/src/components/kanban/EditTemplateModal.tsx
git commit -m "refactor(web): rebuild EditTemplateModal on shared Modal + SegmentedControl"
```

---

## Task 13: Frontend — Login password field label

**Files:**
- Modify: `web/src/pages/Login.tsx:29-51`

**Interfaces:**
- Consumes: nothing new.
- Produces: no visible layout change — a visually-hidden `<label>` is associated with the password `TextInput` via `htmlFor`/`id`.

- [ ] **Step 1: Add the label**

Replace lines 29-51 (the return statement):

```tsx
  return (
    <div className="flex min-h-full items-center justify-center bg-white">
      <form
        onSubmit={handleSubmit}
        className="animate-page-in w-full max-w-sm space-y-4 rounded-2xl border border-zinc-200 bg-white p-8 shadow-xl shadow-black/5"
      >
        <h1 className="text-lg font-semibold text-zinc-900">Money ReRoute</h1>
        <p className="text-sm text-zinc-500">กรอกรหัสผ่านเพื่อเข้าใช้งาน</p>
        <label htmlFor="login-password" className="sr-only">
          รหัสผ่าน
        </label>
        <TextInput
          id="login-password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="รหัสผ่าน"
          autoFocus
          className="w-full"
        />
        {error && <p className="text-sm text-red-600">{error}</p>}
        <Button type="submit" disabled={submitting || !password} className="w-full">
          {submitting ? 'กำลังเข้าสู่ระบบ...' : 'เข้าสู่ระบบ'}
        </Button>
      </form>
    </div>
  )
}
```

- [ ] **Step 2: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 3: Manual verification**

Load `/login`, inspect the password input in devtools accessibility tree, confirm it now has an accessible name ("รหัสผ่าน") instead of relying solely on the placeholder. Visually the page is unchanged (label is `sr-only`).

- [ ] **Step 4: Commit**

```bash
git add web/src/pages/Login.tsx
git commit -m "fix(web): associate a real label with the login password field"
```

---

## Task 14: Frontend — `AddCardDropdown` animation + a11y

**Files:**
- Modify: `web/src/components/kanban/AddCardDropdown.tsx:63-73`

**Interfaces:**
- Consumes: nothing new.
- Produces: the dropdown menu plays `animate-dropdown-in` when it opens; the trigger button exposes `aria-expanded` and `aria-haspopup="menu"`.

- [ ] **Step 1: Update the trigger button and menu container**

Replace lines 63-73:

```tsx
  return (
    <div ref={containerRef} className="relative">
      <Button
        size="sm"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="menu"
      >
        + เพิ่มการ์ด
      </Button>

      {open && (
        <div
          role="menu"
          className="animate-dropdown-in absolute right-0 top-full z-40 mt-1 w-64 rounded-lg border border-zinc-200 bg-white p-1.5 shadow-lg shadow-black/5"
        >
```

- [ ] **Step 2: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 3: Manual verification**

Click "+ เพิ่มการ์ด" repeatedly. Confirm the menu visibly scales/fades in from the top-right corner each time it opens (not an instant pop), and inspect the trigger button's `aria-expanded` attribute toggling between `"true"`/`"false"` in devtools.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/kanban/AddCardDropdown.tsx
git commit -m "feat(web): animate AddCardDropdown menu, add aria-expanded/aria-haspopup"
```

---

## Task 15: Frontend — bar-grow animation for regime/allocation charts

**Files:**
- Modify: `web/src/components/RegimeProbabilityChart.tsx:26-46`
- Modify: `web/src/components/PortfolioStanceBar.tsx:20-30`

**Interfaces:**
- Consumes: existing `.animate-bar-grow` keyframe (`web/src/index.css:145-156`).
- Produces: no prop/type changes — purely visual.

- [ ] **Step 1: Add stagger animation to `RegimeProbabilityChart.tsx`**

Replace lines 26-46 (the return statement):

```tsx
  return (
    <div className="space-y-3 rounded-xl border border-zinc-200 bg-white p-4 shadow-sm shadow-black/5">
      {names.map((name, i) => {
        const value = probabilities[name] ?? 0
        const pct = Math.round(value * 100)
        return (
          <div key={name} className="flex items-center gap-3">
            <span className="w-28 shrink-0 text-sm text-zinc-700">{name}</span>
            <div className="h-4 flex-1 bg-zinc-100">
              {/* square ที่ baseline (0%), โค้งแค่ data-end (ปลายขวา) ตาม mark spec */}
              <div
                className="animate-bar-grow h-4 rounded-r-full"
                style={{ width: `${pct}%`, backgroundColor: colorFor(name), animationDelay: `${i * 60}ms` }}
              />
            </div>
            <span className="w-12 shrink-0 text-right font-mono text-sm text-zinc-500">{pct}%</span>
          </div>
        )
      })}
    </div>
  )
}
```

- [ ] **Step 2: Add the same treatment to `PortfolioStanceBar.tsx`**

Replace lines 20-30 (the return statement):

```tsx
  return (
    <div className="space-y-2">
      <div className="flex h-3 overflow-hidden rounded-full bg-zinc-100">
        {LEGEND.map((item, i) => (
          <div
            key={item.key}
            className={`animate-bar-grow ${item.barClass}`}
            style={{ flex: counts[item.key] || 0, animationDelay: `${i * 60}ms` }}
          />
        ))}
      </div>
```

(Lines after this, the legend `<div className="flex flex-wrap ...">` block, are unchanged.)

- [ ] **Step 3: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 4: Manual verification**

Load `/macro`. Watch the "Regime Probabilities" bars and the "Cross-Asset Allocation Strategy" overview bar on first paint — each bar should grow from the left edge, staggered by ~60ms per row/segment, instead of appearing at full width instantly.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/RegimeProbabilityChart.tsx web/src/components/PortfolioStanceBar.tsx
git commit -m "feat(web): animate regime/allocation bars growing in with a stagger"
```

---

## Task 16: Frontend — Macro page loading skeleton

**Files:**
- Modify: `web/src/pages/Macro.tsx:62-63`

**Interfaces:**
- Consumes: existing `.animate-shimmer` utility (`web/src/index.css:158-171`).
- Produces: no prop/type changes — replaces the plain "กำลังโหลดรายงาน..." text with a layout-shaped skeleton while `data` is `null`.

- [ ] **Step 1: Replace the loading branch**

Replace lines 62-63:

```tsx
  if (error) return <p className="text-sm text-red-600">{error}</p>
  if (!data) {
    return (
      <div className="animate-page-in space-y-6 pb-10">
        <div className="animate-shimmer h-40 rounded-2xl border border-zinc-200/80" />
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="space-y-6 lg:col-span-5">
            <div className="animate-shimmer h-64 rounded-xl border border-zinc-200/80" />
            <div className="animate-shimmer h-40 rounded-xl border border-zinc-200/80" />
          </div>
          <div className="space-y-6 lg:col-span-7">
            <div className="animate-shimmer h-96 rounded-xl border border-zinc-200/80" />
          </div>
        </div>
      </div>
    )
  }
```

- [ ] **Step 2: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 3: Manual verification**

Throttle the network in devtools (Slow 3G) and reload `/macro`. Confirm a shimmering skeleton layout (roughly matching the real banner + two-column layout) appears instead of a plain loading sentence, and it's replaced by real content once the fetch completes.

- [ ] **Step 4: Commit**

```bash
git add web/src/pages/Macro.tsx
git commit -m "feat(web): show a shimmer skeleton while the macro dashboard loads"
```

---

## Task 17: Frontend — Kanban board initial-load card stagger

**Files:**
- Modify: `web/src/pages/Kanban.tsx` (add ref + prop wiring)
- Modify: `web/src/components/kanban/KanbanColumn.tsx` (full rewrite)
- Modify: `web/src/components/kanban/KanbanCard.tsx:24-44`

**Interfaces:**
- Consumes: existing `.animate-card-in` keyframe (already applied unconditionally).
- Produces: `KanbanColumn`'s `Props` gains `staggerCards: boolean`. `KanbanCard`'s `Props` gains `style?: CSSProperties`, applied to the card's outer `<div>`.

- [ ] **Step 1: Track first-load completion in `web/src/pages/Kanban.tsx`**

Add a ref near the other refs (alongside `tickTimer`, from Task 7's edit):

```typescript
  const hasLoadedOnceRef = useRef(false)
```

Add an effect that flips it after the first non-empty render, placed right after the `useEffect` that calls `refresh()` on mount:

```typescript
  useEffect(() => {
    refresh().catch((e) => setError(e instanceof ApiError ? e.message : 'โหลดการ์ดไม่สำเร็จ'))
  }, [])

  useEffect(() => {
    // ref (ไม่ใช่ state) เพราะต้องอ่านค่า "ก่อนแฟล็กถูกตั้ง" ในรอบ render เดียวกับที่การ์ดจริง
    // ปรากฏครั้งแรก ถ้าใช้ state ทั้งคู่จะถูก batch เข้า render เดียวกันแล้วธงจะกลายเป็น true
    // ไปแล้วตั้งแต่ก่อนการ์ดจะ render จริง ทำให้ stagger ไม่ทำงานเลย
    if (cards.length > 0) hasLoadedOnceRef.current = true
  }, [cards])
```

(This replaces the existing bare `useEffect(() => { refresh()... }, [])` — keep it as-is and add the new effect immediately after it.)

- [ ] **Step 2: Pass `staggerCards` to each `KanbanColumn` in the render**

Find the `<KanbanColumn ... />` usage (in the columns grid) and add one prop:

```tsx
          {visibleColumns.map((col) => (
            <KanbanColumn
              key={col.key}
              column={col}
              cards={cardsForColumn(col.key)}
              isBacklogColumn={col.key === 'backlog'}
              isCardFaded={(c) => col.key === 'done' && daysSince(c.updated_at) > DONE_FADE_AFTER_DAYS}
              removingIds={removingIds}
              selectedCardId={selectedCardId}
              workspacePreviewFor={workspacePreviewFor}
              staggerCards={!hasLoadedOnceRef.current}
              onDeleteCard={deleteCard}
              onCardClick={(c) => setSelectedCardId(c.card_id)}
              onEditCard={openEditModal}
              onDispatchCard={(c) => {
                if (!dispatching) {
                  setError(null)
                  dispatchCard(c, c.flow ?? 'manager')
                }
              }}
            />
          ))}
```

- [ ] **Step 3: Rewrite `web/src/components/kanban/KanbanColumn.tsx`**

```tsx
import type { KanbanCardDTO } from '../../api/types'
import KanbanCard from './KanbanCard'
import type { ColumnDef, WorkspacePreview } from './types'

const STATUS_DOT: Record<string, string> = {
  backlog: 'bg-blue-500',
  approval: 'bg-purple-500',
  executing: 'bg-emerald-500 animate-pulse',
  done: 'bg-emerald-600',
}

const STAGGER_STEP_MS = 30
const STAGGER_CAP_MS = 240

interface Props {
  column: ColumnDef
  cards: KanbanCardDTO[]
  isBacklogColumn: boolean
  isCardFaded: (card: KanbanCardDTO) => boolean
  removingIds: Set<string>
  selectedCardId?: string | null
  workspacePreviewFor: (card: KanbanCardDTO) => WorkspacePreview | undefined
  staggerCards?: boolean
  onDeleteCard: (cardId: string) => void
  onCardClick?: (card: KanbanCardDTO) => void
  onEditCard?: (card: KanbanCardDTO) => void
  onDispatchCard?: (card: KanbanCardDTO) => void
}

export default function KanbanColumn({
  column,
  cards,
  isBacklogColumn,
  isCardFaded,
  removingIds,
  selectedCardId,
  workspacePreviewFor,
  staggerCards,
  onDeleteCard,
  onCardClick,
  onEditCard,
  onDispatchCard,
}: Props) {
  return (
    <div className="flex h-full flex-col rounded-xl border border-zinc-200/80 bg-white p-2.5 transition-colors duration-150">
      <h3 className="mb-2 flex shrink-0 items-center gap-1.5 px-1 text-xs font-semibold text-zinc-600">
        <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[column.key] ?? 'bg-zinc-400'}`} />
        {column.label} <span className="text-zinc-400">({cards.length})</span>
      </h3>
      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto pr-0.5">
        {cards.map((c, i) => (
          <KanbanCard
            key={c.card_id}
            card={c}
            faded={isCardFaded(c)}
            removing={removingIds.has(c.card_id)}
            selected={c.card_id === selectedCardId}
            workspacePreview={workspacePreviewFor(c)}
            onDelete={() => onDeleteCard(c.card_id)}
            onClick={onCardClick ? () => onCardClick(c) : undefined}
            editable={isBacklogColumn}
            onEdit={onEditCard ? () => onEditCard(c) : undefined}
            onDispatch={onDispatchCard ? () => onDispatchCard(c) : undefined}
            style={staggerCards ? { animationDelay: `${Math.min(i * STAGGER_STEP_MS, STAGGER_CAP_MS)}ms` } : undefined}
          />
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Accept and apply `style` in `web/src/components/kanban/KanbanCard.tsx`**

Replace lines 1-44 (imports through the function signature):

```tsx
import type { CSSProperties } from 'react'
import type { KanbanCardDTO } from '../../api/types'
import { nodeDisplayName } from '../../lib/nodeDisplayNames'
import { FLOW_TAG } from '../../lib/flowTags'
import type { WorkspacePreview } from './types'

interface Props {
  card: KanbanCardDTO
  faded: boolean
  removing: boolean
  selected?: boolean
  workspacePreview?: WorkspacePreview
  onDelete: () => void
  onClick?: () => void
  editable?: boolean
  onEdit?: () => void
  onDispatch?: () => void
  style?: CSSProperties
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`
  return `${Math.round(seconds / 60)}m`
}

export default function KanbanCard({
  card,
  faded,
  removing,
  selected,
  workspacePreview,
  onDelete,
  onClick,
  editable,
  onEdit,
  onDispatch,
  style,
}: Props) {
```

Then update the outer `<div>` (currently lines 37-44) to pass `style`:

```tsx
    <div
      onClick={onClick}
      style={style}
      className={`group relative rounded-lg border bg-white p-3 pr-8 text-xs text-zinc-800 shadow-[0_1px_2px_rgba(0,0,0,0.03)] transition-all duration-150 hover:-translate-y-0.5 hover:shadow-md ${
        selected || workspacePreview ? 'border-2 border-terra-light' : 'border-zinc-200 hover:border-zinc-300'
      } ${onClick ? 'cursor-pointer' : 'cursor-default'} ${
        removing ? 'animate-card-out' : 'animate-card-in'
      } ${faded ? 'opacity-40' : ''}`}
    >
```

- [ ] **Step 5: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 6: Manual verification**

Hard-reload the Kanban board with several existing cards in a column. Confirm cards in that column appear with a visible top-to-bottom stagger (slight delay increasing per card) on the very first load. Then create a new card or switch a filter tab — confirm cards do NOT re-stagger on those subsequent updates (they should appear immediately, matching current behavior).

- [ ] **Step 7: Commit**

```bash
git add web/src/pages/Kanban.tsx web/src/components/kanban/KanbanColumn.tsx web/src/components/kanban/KanbanCard.tsx
git commit -m "feat(web): stagger kanban card entrance animation on initial board load only"
```

---

## Task 18: Frontend — `KanbanCard` keyboard accessibility

**Files:**
- Modify: `web/src/components/kanban/KanbanCard.tsx` (the file from Task 17, further edits)

**Interfaces:**
- Consumes: nothing new.
- Produces: when `onClick` is provided, the card's outer `<div>` gets `role="button"`, `tabIndex={0}`, a `focus-visible` ring, and responds to Enter/Space the same as a click.

- [ ] **Step 1: Add keyboard handling**

Add the import for `KeyboardEvent` (extend the existing `type CSSProperties` import from Task 17):

```tsx
import type { CSSProperties, KeyboardEvent } from 'react'
```

Add a handler function right before the `export default function KanbanCard` line:

```tsx
function handleCardKeyDown(e: KeyboardEvent<HTMLDivElement>, onClick?: () => void) {
  if (!onClick) return
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault()
    onClick()
  }
}
```

Update the outer `<div>` (the one modified in Task 17 Step 4) to:

```tsx
    <div
      onClick={onClick}
      onKeyDown={(e) => handleCardKeyDown(e, onClick)}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      style={style}
      className={`group relative rounded-lg border bg-white p-3 pr-8 text-xs text-zinc-800 shadow-[0_1px_2px_rgba(0,0,0,0.03)] transition-all duration-150 hover:-translate-y-0.5 hover:shadow-md focus-visible:outline focus-visible:outline-2 focus-visible:outline-terra ${
        selected || workspacePreview ? 'border-2 border-terra-light' : 'border-zinc-200 hover:border-zinc-300'
      } ${onClick ? 'cursor-pointer' : 'cursor-default'} ${
        removing ? 'animate-card-out' : 'animate-card-in'
      } ${faded ? 'opacity-40' : ''}`}
    >
```

- [ ] **Step 2: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 3: Manual verification**

On the Kanban board, press Tab repeatedly until a card receives focus (visible outline ring). Press Enter — confirm the detail drawer opens for that card, same as a mouse click. Press Space on a focused card — confirm the same (and that Space doesn't scroll the page, since `preventDefault` is called). Confirm the edit/delete/dispatch buttons inside the card (which are separate focusable elements with their own `stopPropagation`) still work independently by mouse and keyboard.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/kanban/KanbanCard.tsx
git commit -m "feat(web): make kanban cards keyboard-operable (Enter/Space, focus ring)"
```

---

## Task 19: Frontend — `aria-pressed` on filter tabs

**Files:**
- Modify: `web/src/components/kanban/KanbanHeader.tsx:29-57`
- Modify: `web/src/pages/Macro.tsx:205-219`

**Interfaces:**
- Consumes: nothing new.
- Produces: no visual change — filter tab buttons expose `aria-pressed` reflecting the active tab.

- [ ] **Step 1: Add `aria-pressed` in `web/src/components/kanban/KanbanHeader.tsx`**

Replace lines 29-57 (the return statement):

```tsx
  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex gap-1 rounded-lg border border-zinc-200 bg-white p-1">
        {STATUS_TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => onStatusFilterChange(t.key)}
            aria-pressed={statusFilter === t.key}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              statusFilter === t.key ? 'bg-surface text-zinc-900' : 'text-zinc-500 hover:text-zinc-800'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="flex gap-1 rounded-lg border border-zinc-200 bg-white p-1">
        {FLOW_TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => onFlowFilterChange(t.key)}
            aria-pressed={flowFilter === t.key}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              flowFilter === t.key ? 'bg-terra/10 text-terra' : 'text-zinc-500 hover:text-zinc-800'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Add `aria-pressed` in `web/src/pages/Macro.tsx`**

Replace the stance filter tab button block (currently lines 205-219, inside the `STANCE_FILTER_TABS.map`):

```tsx
              <div className="flex w-fit gap-1 rounded-lg border border-zinc-200 bg-zinc-50 p-1">
                {STANCE_FILTER_TABS.map((t) => (
                  <button
                    key={t.key}
                    onClick={() => handleStanceFilterChange(t.key)}
                    aria-pressed={stanceFilter === t.key}
                    className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                      stanceFilter === t.key
                        ? 'bg-zinc-900 text-white shadow-sm'
                        : 'text-zinc-600 hover:text-zinc-900'
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
```

- [ ] **Step 3: Typecheck**

Run: `cd web && npx tsc -b`
Expected: no output, exit code 0.

- [ ] **Step 4: Manual verification**

On the Kanban page and the Macro page, inspect the filter tab buttons in devtools and confirm `aria-pressed` toggles `true`/`false` correctly as you click between tabs.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/kanban/KanbanHeader.tsx web/src/pages/Macro.tsx
git commit -m "fix(web): expose aria-pressed on kanban/macro filter tab buttons"
```

---

## Task 20: Frontend — remove unused asset files

**Files:**
- Delete: `web/src/assets/hero.png`
- Delete: `web/src/assets/react.svg`
- Delete: `web/src/assets/vite.svg`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (dead files removed).

- [ ] **Step 1: Confirm nothing imports these files**

Run: `cd web && grep -rn "hero.png\|react.svg\|vite.svg" src/`
Expected: no output (already confirmed during design review, re-verify here before deleting).

- [ ] **Step 2: Delete the files**

```bash
git rm web/src/assets/hero.png web/src/assets/react.svg web/src/assets/vite.svg
```

- [ ] **Step 3: Typecheck and lint**

Run: `cd web && npx tsc -b && npx oxlint`
Expected: no errors from either command.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(web): remove unused asset files"
```

---

## Task 21: Final verification pass

**Files:** none (verification only).

**Interfaces:** none.

- [ ] **Step 1: Run the full backend test suite**

Run: `uv run pytest tests/api/ -v`
Expected: all tests PASS.

- [ ] **Step 2: Run the full frontend build**

Run: `cd web && npm run build`
Expected: completes with no TypeScript errors and no Vite build errors (produces `web/dist/`).

- [ ] **Step 3: Run the linter**

Run: `cd web && npm run lint`
Expected: no errors (warnings from pre-existing code are acceptable; no new errors introduced by this plan's changes).

- [ ] **Step 4: Manual end-to-end smoke test**

With `uvicorn api.main:app --reload` and `npm run dev` (in `web/`) both running:
1. Log in at `/login` — confirm the password field, submit button, and error states all work.
2. On the Kanban board, create a card via a quick template and via "+ กำหนดเอง..." (custom modal) — confirm both work, the modal traps focus, and closes on Escape/backdrop click.
3. Dispatch a card — confirm in the Network tab that only `POST /api/agents/dispatch` fires (no follow-up `PUT /api/kanban/move`), and the card visually moves to "Workers Executing".
4. Open the card's detail drawer while it runs — confirm the terminal streams, and once the job finishes, confirm exactly one `PUT /api/kanban/move` call fires (not two).
5. Delete a card — confirm the fade-out animation plays and the card is removed.
6. Navigate to `/macro` — confirm the loading skeleton appears briefly (throttle network if it loads too fast to see), then confirm bar-grow animations play on the regime/allocation charts, and the References drawer opens/closes with animation and correct focus behavior.
7. Tab through the Kanban board using only the keyboard — confirm cards are focusable and Enter/Space opens the detail drawer.
8. Delete the session cookie mid-use and trigger an action — confirm the app redirects to `/login` instead of showing a raw error.

- [ ] **Step 5: Report results**

If any manual check fails, stop and fix the specific task before proceeding — do not mark this plan complete. If all checks pass, the plan is done; no commit needed for this task (verification only).

---

## Spec Coverage Checklist (self-review)

- ส่วนที่ 1 ข้อ 1 (writing-vertical) → Task 9. ✅
- ส่วนที่ 1 ข้อ 2 (Tailwind v4 classes) → Task 9. ✅
- ส่วนที่ 1 ข้อ 3 (duplicate move logic) → Task 6. ✅
- ส่วนที่ 1 ข้อ 4 (timer leaks) → Task 7. ✅
- ส่วนที่ 1 ข้อ 5 (401 mid-session) → Task 5. ✅
- ส่วนที่ 1 ข้อ 6 (drawer width flash) → Task 8. ✅
- ส่วนที่ 2 (Modal, SegmentedControl, MacroReferenceDrawer export style, unused assets) → Tasks 9, 10, 11, 12, 20. ✅
- ส่วนที่ 3 (all 8 animation rows) → Tasks 7 (notice), 9 (drawer/fade), 10+11+12 (modal), 14 (dropdown), 15 (bar-grow), 16 (shimmer), 17 (card stagger). ✅
- ส่วนที่ 4 (all 6 a11y items) → Tasks 11+12 (modal focus trap/labels), 9 (drawer dialog role), 14 (dropdown aria), 18 (card keyboard), 19 (aria-pressed). ✅
- ส่วนที่ 5 ข้อ 1-6 (backend) → Tasks 1, 2, 3, 4. ✅
- Verification section → Task 21 covers both `pytest` and the manual web checklist. ✅
