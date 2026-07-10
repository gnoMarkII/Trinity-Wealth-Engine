"""SQLite store สำหรับ job log + kanban state — แยกไฟล์จาก LangGraph checkpoint DB โดยตั้งใจ
กัน agent run ที่กำลังรันหนักๆ ไป lock หน้า Kanban/Portfolio ที่ไม่เกี่ยวข้องกัน (ดู Rev.5 ข้อ 6)
"""
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

CREATE TABLE IF NOT EXISTS kanban_cards (
    card_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    column_name TEXT NOT NULL DEFAULT 'backlog',
    job_id TEXT,
    flow TEXT NOT NULL DEFAULT 'manager',
    display_seq INTEGER,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
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


def list_jobs_by_status(conn: sqlite3.Connection, statuses: list[str]) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in statuses)
    cur = conn.execute(f"SELECT * FROM jobs WHERE status IN ({placeholders})", tuple(statuses))
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
) -> None:
    now = time.time()
    next_seq = conn.execute("SELECT COALESCE(MAX(display_seq), 0) + 1 FROM kanban_cards").fetchone()[0]
    conn.execute(
        "INSERT INTO kanban_cards (card_id, title, column_name, job_id, flow, display_seq, prompt, scope, created_at, updated_at) "
        "VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)",
        (card_id, title, column_name, flow, next_seq, prompt, scope, now, now),
    )
    conn.commit()


def update_kanban_card(
    conn: sqlite3.Connection, card_id: str, title: str, prompt: str | None, flow: str, scope: str
) -> None:
    now = time.time()
    conn.execute(
        "UPDATE kanban_cards SET title = ?, prompt = ?, flow = ?, scope = ?, updated_at = ? WHERE card_id = ?",
        (title, prompt, flow, scope, now, card_id),
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
