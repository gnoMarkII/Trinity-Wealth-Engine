"""SQLite store สำหรับ job log + kanban state — แยกไฟล์จาก LangGraph checkpoint DB โดยตั้งใจ
กัน agent run ที่กำลังรันหนักๆ ไป lock หน้า Kanban/Portfolio ที่ไม่เกี่ยวข้องกัน (ดู Rev.5 ข้อ 6)
"""
import json
import os
import sqlite3
import time

from api.config import get_state_db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    card_id TEXT,
    idempotency_key TEXT UNIQUE,
    instruction TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    error_message TEXT,
    flow TEXT NOT NULL DEFAULT 'manager',
    interrupt_payload TEXT,
    resume_value TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS job_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    node_name TEXT,
    content TEXT,
    role TEXT NOT NULL DEFAULT 'reply',
    label TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_job_logs_job_id ON job_logs(job_id, seq);

CREATE TABLE IF NOT EXISTS used_eligibility_tokens (
    token_hash TEXT PRIMARY KEY,
    jti TEXT NOT NULL UNIQUE,
    job_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    pitch_id TEXT NOT NULL,
    approval_revision INTEGER NOT NULL,
    used_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_used_eligibility_tokens_job ON used_eligibility_tokens(job_id);

CREATE TABLE IF NOT EXISTS kanban_cards (
    card_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    column_name TEXT NOT NULL DEFAULT 'backlog',
    job_id TEXT,
    flow TEXT NOT NULL DEFAULT 'manager',
    display_seq INTEGER,
    is_verified INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS dcf_evaluations_ledger (
    evaluation_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    market TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    model_version TEXT NOT NULL DEFAULT 'dcf_v1.0',
    valuation_price_basis TEXT NOT NULL DEFAULT 'split_adjusted_only',
    current_price_at_eval REAL,
    wacc_pct REAL,
    valuation_verdict TEXT,
    scenarios_json TEXT NOT NULL,
    corporate_action_evidence_json TEXT,
    input_snapshot_json TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dcf_evaluations_ticker ON dcf_evaluations_ledger(ticker, evaluated_at DESC);

CREATE TABLE IF NOT EXISTS sec_form4_raw_ledger (
    accession_number TEXT PRIMARY KEY,
    issuer_cik TEXT NOT NULL,
    ticker TEXT NOT NULL,
    filing_url TEXT NOT NULL,
    filed_at TEXT NOT NULL,
    reporting_owner_cik TEXT,
    reporting_owner_name TEXT,
    is_director INTEGER NOT NULL DEFAULT 0,
    is_officer INTEGER NOT NULL DEFAULT 0,
    is_ten_percent_owner INTEGER NOT NULL DEFAULT 0,
    officer_title TEXT,
    raw_xml_payload TEXT,
    is_amendment INTEGER NOT NULL DEFAULT 0,
    amends_accession_number TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sec_raw_ticker ON sec_form4_raw_ledger(ticker, filed_at DESC);

CREATE TABLE IF NOT EXISTS sec_insider_transactions (
    transaction_id TEXT PRIMARY KEY,
    accession_number TEXT NOT NULL,
    ticker TEXT NOT NULL,
    transaction_date TEXT NOT NULL,
    transaction_code TEXT NOT NULL,
    shares REAL NOT NULL,
    price_per_share REAL NOT NULL,
    acquired_or_disposed TEXT NOT NULL,
    shares_owned_following REAL,
    ownership_nature TEXT,
    is_derivative INTEGER NOT NULL DEFAULT 0,
    normalized_weight REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    FOREIGN KEY (accession_number) REFERENCES sec_form4_raw_ledger(accession_number)
);
CREATE INDEX IF NOT EXISTS idx_insider_tx_ticker ON sec_insider_transactions(ticker, transaction_date DESC);

CREATE TABLE IF NOT EXISTS analyst_context_cache (
    ticker           TEXT PRIMARY KEY,
    provider_symbol  TEXT NOT NULL,
    market           TEXT NOT NULL,
    currency         TEXT NOT NULL,
    exchange_tz      TEXT NOT NULL,
    target_mean      REAL,
    target_high      REAL,
    target_low       REAL,
    num_analysts     INTEGER,
    next_earnings_date TEXT,
    eps_history_json TEXT NOT NULL DEFAULT '[]',
    source_as_of     TEXT,
    data_status      TEXT NOT NULL DEFAULT 'ok',
    synced_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyst_context_ticker ON analyst_context_cache(ticker);

"""


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


_COLUMN_MIGRATIONS: dict[str, dict[str, str]] = {
    "jobs": {
        "flow": "flow TEXT NOT NULL DEFAULT 'manager'",
        "interrupt_payload": "interrupt_payload TEXT",
        "resume_value": "resume_value TEXT",
        "scope": "scope TEXT NOT NULL DEFAULT 'both'",
    },
    "job_logs": {
        "role": "role TEXT NOT NULL DEFAULT 'reply'",
        "label": "label TEXT",
    },
    "kanban_cards": {
        "flow": "flow TEXT NOT NULL DEFAULT 'manager'",
        "display_seq": "display_seq INTEGER",
        "prompt": "prompt TEXT",
        "scope": "scope TEXT NOT NULL DEFAULT 'both'",
        "discord_notify": "discord_notify INTEGER NOT NULL DEFAULT 1",
        "discord_sent_events": "discord_sent_events TEXT",
        "is_verified": "is_verified INTEGER NOT NULL DEFAULT 1",
    },
}


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


def _backfill_kanban_display_seq(conn: sqlite3.Connection) -> None:
    """การ์ดเก่าที่มีอยู่ก่อน Rev.2 (ก่อนมีคอลัมน์ display_seq) จะมีค่า NULL — เติมเลขให้
    ตามลำดับ created_at เพื่อให้ Linear-style #AG-N ID เรียงลำดับสร้างจริง ไม่ใช่เลขสุ่ม
    """
    cur = conn.execute("SELECT COUNT(*) FROM kanban_cards WHERE display_seq IS NULL")
    if cur.fetchone()[0] == 0:
        return
    cur = conn.execute("SELECT COALESCE(MAX(display_seq), 0) FROM kanban_cards")
    next_seq = cur.fetchone()[0] + 1
    rows = conn.execute(
        "SELECT card_id FROM kanban_cards WHERE display_seq IS NULL ORDER BY created_at ASC"
    ).fetchall()
    for row in rows:
        conn.execute("UPDATE kanban_cards SET display_seq = ? WHERE card_id = ?", (next_seq, row["card_id"]))
        next_seq += 1
    conn.commit()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()
    _migrate_columns(conn)
    _migrate_dispatcher_column_cards(conn)
    _backfill_kanban_display_seq(conn)


# --- Jobs ---

def create_job(conn: sqlite3.Connection, job_id: str, thread_id: str, card_id: str | None,
                idempotency_key: str, instruction: str, status: str = "queued", flow: str = "manager",
                scope: str = "both") -> None:
    now = time.time()
    conn.execute(
        "INSERT INTO jobs (job_id, thread_id, card_id, idempotency_key, instruction, status, flow, scope, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (job_id, thread_id, card_id, idempotency_key, instruction, status, flow, scope, now, now),
    )
    conn.commit()


def set_job_awaiting_approval(conn: sqlite3.Connection, job_id: str, interrupt_payload_json: str) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'awaiting_approval', interrupt_payload = ?, updated_at = ? WHERE job_id = ?",
        (interrupt_payload_json, time.time(), job_id),
    )
    conn.commit()


def set_job_resume_value(conn: sqlite3.Connection, job_id: str, resume_value_json: str) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'running', resume_value = ?, interrupt_payload = NULL, updated_at = ? WHERE job_id = ?",
        (resume_value_json, time.time(), job_id),
    )
    conn.commit()


def claim_job_resume(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    resume_value_json: str,
    token_uses: list[dict[str, str | int]] | None = None,
) -> None:
    """Atomically consume Draft tokens and move one approval back to the queue.

    A compare-and-set status check is essential: two browser clicks must not
    enqueue the same LangGraph interrupt twice.
    """
    now = time.time()
    try:
        conn.execute("BEGIN IMMEDIATE")
        job = conn.execute(
            "SELECT job_id, status FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if job is None:
            raise ValueError("job_not_found")
        if job["status"] != "awaiting_approval":
            raise ValueError("approval_already_claimed")
        for token_use in token_uses or []:
            conn.execute(
                "INSERT INTO used_eligibility_tokens "
                "(token_hash, jti, job_id, thread_id, pitch_id, approval_revision, used_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    token_use["token_hash"],
                    token_use["jti"],
                    job_id,
                    token_use["thread_id"],
                    token_use["pitch_id"],
                    token_use["approval_revision"],
                    now,
                ),
            )
        conn.execute(
            "UPDATE jobs SET status = 'queued', resume_value = ?, interrupt_payload = NULL, updated_at = ? "
            "WHERE job_id = ? AND status = 'awaiting_approval'",
            (resume_value_json, now, job_id),
        )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise ValueError("eligibility_token_already_used") from exc
    except Exception:
        conn.rollback()
        raise


def clear_job_resume_value(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute(
        "UPDATE jobs SET resume_value = NULL WHERE job_id = ?",
        (job_id,),
    )
    conn.commit()


def find_job_by_idempotency_key(conn: sqlite3.Connection, idempotency_key: str) -> sqlite3.Row | None:
    cur = conn.execute("SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,))
    return cur.fetchone()


def get_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    return cur.fetchone()


def update_job_status(conn: sqlite3.Connection, job_id: str, status: str, error_message: str | None = None) -> None:
    conn.execute(
        "UPDATE jobs SET status = ?, error_message = ?, updated_at = ? WHERE job_id = ?",
        (status, error_message, time.time(), job_id),
    )
    conn.commit()


def cas_job_status(conn: sqlite3.Connection, job_id: str, old_status: str, new_status: str) -> bool:
    """Compare and Swap job status. Returns True if successful, False if current status was not old_status."""
    cur = conn.execute(
        "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ? AND status = ?",
        (new_status, time.time(), job_id, old_status),
    )
    conn.commit()
    return cur.rowcount > 0


def list_jobs_by_status(conn: sqlite3.Connection, statuses: list[str], flows: list[str] | None = None) -> list[sqlite3.Row]:
    """flows=None (default) = ทุก flow เหมือนเดิมทุกประการ — ใส่ให้ JobQueue ที่แชร์ DB เดียวกันกับ
    คิวอื่น (เช่น notebooklm_job_queue) กรองเฉพาะ flow ของตัวเอง กัน reenqueue_pending() ข้ามคิวไปกวาด
    งานคนละ flow มาประมวลผลผิดที่"""
    placeholders = ",".join("?" for _ in statuses)
    params: list[str] = list(statuses)
    query = f"SELECT * FROM jobs WHERE status IN ({placeholders})"
    if flows is not None:
        flow_placeholders = ",".join("?" for _ in flows)
        query += f" AND flow IN ({flow_placeholders})"
        params += list(flows)
    cur = conn.execute(query, tuple(params))
    return cur.fetchall()


def append_job_log(
    conn: sqlite3.Connection,
    job_id: str,
    node_name: str,
    content: str,
    role: str = "reply",
    label: str | None = None,
) -> int:
    cur = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM job_logs WHERE job_id = ?", (job_id,))
    seq = cur.fetchone()[0]
    conn.execute(
        "INSERT INTO job_logs (job_id, seq, node_name, content, role, label, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (job_id, seq, node_name, content, role, label or node_name, time.time()),
    )
    conn.commit()
    return seq


def get_job_logs_since(conn: sqlite3.Connection, job_id: str, after_seq: int = 0) -> list[sqlite3.Row]:
    cur = conn.execute(
        "SELECT * FROM job_logs WHERE job_id = ? AND seq > ? ORDER BY seq ASC",
        (job_id, after_seq),
    )
    return cur.fetchall()


def get_job_reply_logs(conn: sqlite3.Connection, job_id: str) -> list[sqlite3.Row]:
    cur = conn.execute(
        "SELECT seq, node_name, content, label, created_at FROM job_logs "
        "WHERE job_id = ? AND role = 'reply' ORDER BY seq ASC",
        (job_id,),
    )
    return cur.fetchall()


def get_latest_job_log_node(conn: sqlite3.Connection, job_id: str) -> str | None:
    cur = conn.execute(
        "SELECT node_name FROM job_logs WHERE job_id = ? ORDER BY seq DESC LIMIT 1",
        (job_id,),
    )
    row = cur.fetchone()
    return row["node_name"] if row else None


def get_job_log_count(conn: sqlite3.Connection, job_id: str) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM job_logs WHERE job_id = ?", (job_id,))
    return cur.fetchone()[0]


# --- Kanban ---

def list_kanban_cards(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM kanban_cards ORDER BY created_at ASC")
    return cur.fetchall()


def create_kanban_card(
    conn: sqlite3.Connection,
    card_id: str,
    title: str,
    column_name: str = "backlog",
    flow: str = "manager",
    prompt: str | None = None,
    scope: str = "both",
    is_verified: bool = True,
) -> None:
    now = time.time()
    next_seq = conn.execute("SELECT COALESCE(MAX(display_seq), 0) + 1 FROM kanban_cards").fetchone()[0]
    conn.execute(
        "INSERT INTO kanban_cards (card_id, title, column_name, job_id, flow, display_seq, prompt, scope, is_verified, created_at, updated_at) "
        "VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)",
        (card_id, title, column_name, flow, next_seq, prompt, scope, 1 if is_verified else 0, now, now),
    )
    conn.commit()


def create_parking_lot_cards_atomic(
    ideas: list[str],
    source_pitch_id: str,
    db_path: str | None = None,
) -> int:
    """Atomically create parking lot cards in backlog column with dedicated connection and BEGIN IMMEDIATE.

    Returns the count of newly inserted cards.
    """
    if not ideas:
        return 0
    import hashlib
    import re
    import unicodedata
    from contextlib import closing

    now = time.time()
    created_count = 0
    with closing(get_connection(db_path)) as conn:
        conn.isolation_level = None  # Autocommit mode for explicit transaction control
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur_seq = conn.execute("SELECT COALESCE(MAX(display_seq), 0) FROM kanban_cards").fetchone()[0]
            for raw_idea in ideas[:5]:
                norm = unicodedata.normalize("NFC", str(raw_idea)).strip()
                norm = re.sub(r"\s+", " ", norm)
                if not norm:
                    continue
                card_id = f"parking:{hashlib.sha256(norm.casefold().encode('utf-8')).hexdigest()}"
                prompt = f"ไอเดียต่อยอดจาก YouTube Pitch ({source_pitch_id}): {norm}"
                cur = conn.execute(
                    "INSERT OR IGNORE INTO kanban_cards (card_id, title, column_name, job_id, flow, display_seq, prompt, scope, is_verified, created_at, updated_at) "
                    "VALUES (?, ?, 'backlog', NULL, 'youtube_pitch', ?, ?, 'both', 1, ?, ?)",
                    (card_id, norm, cur_seq + 1, prompt, now, now),
                )
                if cur.rowcount > 0:
                    cur_seq += 1
                    created_count += 1
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return created_count



def update_kanban_card(
    conn: sqlite3.Connection, card_id: str, title: str, prompt: str | None, flow: str, scope: str
) -> None:
    now = time.time()
    conn.execute(
        "UPDATE kanban_cards SET title = ?, prompt = ?, flow = ?, scope = ?, updated_at = ? WHERE card_id = ?",
        (title, prompt, flow, scope, now, card_id),
    )
    conn.commit()


def set_kanban_card_source(conn: sqlite3.Connection, card_id: str, prompt: str, is_verified: bool) -> None:
    """UPDATE เฉพาะ prompt/is_verified — partial patch โดยตั้งใจ ไม่แตะ title/flow/scope (pattern
    เดียวกับ toggle_kanban_card_discord) ใช้ตอนผู้ใช้เลือก Briefing Book ให้การ์ด NotebookLM
    ครั้งแรกใน Drawer ไม่ให้กระทบชื่อการ์ด/flow ที่ผู้ใช้ตั้งไว้ตอนสร้าง
    """
    now = time.time()
    conn.execute(
        "UPDATE kanban_cards SET prompt = ?, is_verified = ?, updated_at = ? WHERE card_id = ?",
        (prompt, 1 if is_verified else 0, now, card_id),
    )
    conn.commit()


def toggle_kanban_card_discord(conn: sqlite3.Connection, card_id: str, enabled: bool) -> None:
    """UPDATE เฉพาะคอลัมน์ discord_notify — partial patch โดยตั้งใจ ไม่แตะ title/prompt/flow/scope
    เพื่อไม่ให้ toggle ถูก reset ทุกครั้งที่ upsert_news_funnel_card เรียก update_kanban_card
    """
    now = time.time()
    conn.execute(
        "UPDATE kanban_cards SET discord_notify = ?, updated_at = ? WHERE card_id = ?",
        (1 if enabled else 0, now, card_id),
    )
    conn.commit()


def mark_discord_events_sent(conn: sqlite3.Connection, card_id: str, event_ids: list[str]) -> None:
    """เพิ่ม event_ids ที่เพิ่งส่ง Discord สำเร็จเข้า discord_sent_events (JSON array) — อ่านค่าเก่า
    มารวมกับใหม่แล้ว UPDATE เฉพาะคอลัมน์นี้ ป้องกันแจ้งซ้ำเมื่อการ์ดถูก upsert รอบถัดไป
    """
    row = conn.execute("SELECT discord_sent_events FROM kanban_cards WHERE card_id = ?", (card_id,)).fetchone()
    if row is None:
        return
    try:
        existing_ids = json.loads(row["discord_sent_events"]) if row["discord_sent_events"] else []
    except (TypeError, ValueError):
        existing_ids = []
    merged_ids = list(dict.fromkeys(existing_ids + list(event_ids)))
    now = time.time()
    conn.execute(
        "UPDATE kanban_cards SET discord_sent_events = ?, updated_at = ? WHERE card_id = ?",
        (json.dumps(merged_ids), now, card_id),
    )
    conn.commit()


def move_kanban_card(conn: sqlite3.Connection, card_id: str, column_name: str, job_id: str | None = None) -> None:
    now = time.time()
    if job_id is not None:
        conn.execute(
            "UPDATE kanban_cards SET column_name = ?, job_id = ?, updated_at = ? WHERE card_id = ?",
            (column_name, job_id, now, card_id),
        )
    else:
        conn.execute(
            "UPDATE kanban_cards SET column_name = ?, updated_at = ? WHERE card_id = ?",
            (column_name, now, card_id),
        )
    conn.commit()


def get_kanban_card(conn: sqlite3.Connection, card_id: str) -> sqlite3.Row | None:
    cur = conn.execute("SELECT * FROM kanban_cards WHERE card_id = ?", (card_id,))
    return cur.fetchone()


def find_kanban_card_by_title_in_column(
    conn: sqlite3.Connection, title: str, column_name: str, prompt: str | None = None
) -> sqlite3.Row | None:
    cur = conn.execute(
        "SELECT * FROM kanban_cards WHERE title = ? AND column_name = ? AND COALESCE(prompt, '') = COALESCE(?, '') "
        "ORDER BY created_at ASC LIMIT 1",
        (title, column_name, prompt),
    )
    return cur.fetchone()


def delete_kanban_card(conn: sqlite3.Connection, card_id: str) -> None:
    conn.execute("DELETE FROM kanban_cards WHERE card_id = ?", (card_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# DCF Evaluations Ledger (Immutable Canonical SSOT)
# ---------------------------------------------------------------------------

def record_dcf_evaluation(
    conn: sqlite3.Connection,
    evaluation_id: str,
    ticker: str,
    market: str,
    evaluated_at: str,
    scenarios: dict,
    model_version: str = "dcf_v1.0",
    valuation_price_basis: str = "split_adjusted_only",
    current_price_at_eval: float | None = None,
    wacc_pct: float | None = None,
    valuation_verdict: str = "unknown",
    corporate_action_evidence: list | None = None,
    input_snapshot: dict | None = None,
) -> None:
    """บันทึกการประเมิน DCF ลง Immutable Ledger แบบ Canonical Record"""
    now = time.time()
    conn.execute(
        """
        INSERT OR REPLACE INTO dcf_evaluations_ledger (
            evaluation_id, ticker, market, evaluated_at, model_version,
            valuation_price_basis, current_price_at_eval, wacc_pct,
            valuation_verdict, scenarios_json, corporate_action_evidence_json,
            input_snapshot_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            evaluation_id,
            ticker.upper(),
            market.upper(),
            evaluated_at,
            model_version,
            valuation_price_basis,
            current_price_at_eval,
            wacc_pct,
            valuation_verdict,
            json.dumps(scenarios),
            json.dumps(corporate_action_evidence or []),
            json.dumps(input_snapshot or {}),
            now,
        ),
    )
    conn.commit()


def get_latest_dcf_evaluation(conn: sqlite3.Connection, ticker: str) -> sqlite3.Row | None:
    """ดึงผลการประเมิน DCF ล่าสุดสำหรับ ticker จาก Canonical Ledger"""
    cur = conn.execute(
        "SELECT * FROM dcf_evaluations_ledger WHERE ticker = ? ORDER BY evaluated_at DESC, created_at DESC LIMIT 1",
        (ticker.upper(),),
    )
    return cur.fetchone()


def get_dcf_evaluation_by_id(conn: sqlite3.Connection, evaluation_id: str) -> sqlite3.Row | None:
    """ดึงผลการประเมิน DCF ตาม evaluation_id ที่เจาะจง"""
    cur = conn.execute(
        "SELECT * FROM dcf_evaluations_ledger WHERE evaluation_id = ?",
        (evaluation_id,),
    )
    return cur.fetchone()


# ---------------------------------------------------------------------------
# SEC Form 4 Ingestion Pipeline & Raw Filing Ledger
# ---------------------------------------------------------------------------

def record_sec_form4_filing(
    conn: sqlite3.Connection,
    accession_number: str,
    issuer_cik: str,
    ticker: str,
    filing_url: str,
    filed_at: str,
    reporting_owner_cik: str | None = None,
    reporting_owner_name: str | None = None,
    is_director: bool = False,
    is_officer: bool = False,
    is_ten_percent_owner: bool = False,
    officer_title: str | None = None,
    raw_xml_payload: str | None = None,
    is_amendment: bool = False,
    amends_accession_number: str | None = None,
) -> None:
    """บันทึก Raw SEC Form 4 Filing ลง Ledger พร้อมจัดการ Form 4/A Amendment Invalidation"""
    now = time.time()
    if is_amendment and amends_accession_number:
        # Form 4/A Amendment: ลบ normalized transactions เดิมของ filing ที่ถูกแก้ไข
        conn.execute("DELETE FROM sec_insider_transactions WHERE accession_number = ?", (amends_accession_number,))

    conn.execute(
        """
        INSERT OR REPLACE INTO sec_form4_raw_ledger (
            accession_number, issuer_cik, ticker, filing_url, filed_at,
            reporting_owner_cik, reporting_owner_name, is_director,
            is_officer, is_ten_percent_owner, officer_title, raw_xml_payload,
            is_amendment, amends_accession_number, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            accession_number,
            issuer_cik,
            ticker.upper(),
            filing_url,
            filed_at,
            reporting_owner_cik,
            reporting_owner_name,
            1 if is_director else 0,
            1 if is_officer else 0,
            1 if is_ten_percent_owner else 0,
            officer_title,
            raw_xml_payload,
            1 if is_amendment else 0,
            amends_accession_number,
            now,
        ),
    )
    conn.commit()


def record_sec_insider_transaction(
    conn: sqlite3.Connection,
    transaction_id: str,
    accession_number: str,
    ticker: str,
    transaction_date: str,
    transaction_code: str,
    shares: float,
    price_per_share: float,
    acquired_or_disposed: str,
    shares_owned_following: float | None = None,
    ownership_nature: str | None = None,
    is_derivative: bool = False,
    normalized_weight: float = 1.0,
) -> None:
    """บันทึก Normalized Transaction ที่สกัดจาก Form 4"""
    now = time.time()
    conn.execute(
        """
        INSERT OR REPLACE INTO sec_insider_transactions (
            transaction_id, accession_number, ticker, transaction_date,
            transaction_code, shares, price_per_share, acquired_or_disposed,
            shares_owned_following, ownership_nature, is_derivative,
            normalized_weight, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            transaction_id,
            accession_number,
            ticker.upper(),
            transaction_date,
            transaction_code.upper(),
            shares,
            price_per_share,
            acquired_or_disposed.upper(),
            shares_owned_following,
            ownership_nature,
            1 if is_derivative else 0,
            normalized_weight,
            now,
        ),
    )
    conn.commit()


def get_sec_insider_filings_and_transactions(
    conn: sqlite3.Connection, ticker: str, since_date: str | None = None
) -> list[dict]:
    """ดึงประวัติ Insider Filings และ Transactions ของ ticker จาก Ledger"""
    query = """
        SELECT 
            t.transaction_id, t.accession_number, t.ticker, t.transaction_date,
            t.transaction_code, t.shares, t.price_per_share, t.acquired_or_disposed,
            t.shares_owned_following, t.ownership_nature, t.is_derivative, t.normalized_weight,
            f.issuer_cik, f.filing_url, f.filed_at, f.reporting_owner_cik,
            f.reporting_owner_name, f.is_director, f.is_officer, f.is_ten_percent_owner,
            f.officer_title, f.is_amendment, f.amends_accession_number
        FROM sec_insider_transactions t
        JOIN sec_form4_raw_ledger f ON t.accession_number = f.accession_number
        WHERE t.ticker = ?
    """
    params = [ticker.upper()]
    if since_date:
        query += " AND t.transaction_date >= ?"
        params.append(since_date)
    query += " ORDER BY t.transaction_date DESC, f.filed_at DESC"

    cur = conn.execute(query, params)
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_analyst_context_cache(conn: sqlite3.Connection, ticker: str) -> dict | None:
    """ดึงข้อมูล Analyst Context จาก SQLite cache พร้อม decode JSON และ format synced_at เป็น ISO string"""
    row = conn.execute(
        "SELECT * FROM analyst_context_cache WHERE ticker = ?", (ticker.upper(),)
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        raw_json = d.pop("eps_history_json", "[]") or "[]"
        decoded = json.loads(raw_json)
        if not isinstance(decoded, list):
            return None  # Invalid JSON structure -> cache miss
        d["earnings_history"] = decoded
    except Exception:
        return None  # Corrupted JSON -> cache miss
    from datetime import datetime, timezone
    d["synced_at"] = datetime.fromtimestamp(d["synced_at"], tz=timezone.utc).isoformat()
    return d


def upsert_analyst_context_cache(conn: sqlite3.Connection, ticker: str, data: dict) -> None:
    """บันทึกข้อมูล Analyst Context ลง SQLite cache เฉพาะสถานะ ok และ partial"""
    conn.execute(
        """
        INSERT INTO analyst_context_cache (
            ticker, provider_symbol, market, currency, exchange_tz,
            target_mean, target_high, target_low, num_analysts,
            next_earnings_date, eps_history_json, source_as_of,
            data_status, synced_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            provider_symbol=excluded.provider_symbol,
            market=excluded.market,
            currency=excluded.currency,
            exchange_tz=excluded.exchange_tz,
            target_mean=excluded.target_mean,
            target_high=excluded.target_high,
            target_low=excluded.target_low,
            num_analysts=excluded.num_analysts,
            next_earnings_date=excluded.next_earnings_date,
            eps_history_json=excluded.eps_history_json,
            source_as_of=excluded.source_as_of,
            data_status=excluded.data_status,
            synced_at=excluded.synced_at
        """,
        (
            ticker.upper(),
            data["provider_symbol"],
            data["market"],
            data["currency"],
            data["exchange_tz"],
            data.get("target_mean"),
            data.get("target_high"),
            data.get("target_low"),
            data.get("num_analysts"),
            data.get("next_earnings_date"),
            json.dumps(data.get("earnings_history", [])),
            data.get("source_as_of"),
            data["data_status"],
            data["synced_at"],
        ),
    )
    conn.commit()


