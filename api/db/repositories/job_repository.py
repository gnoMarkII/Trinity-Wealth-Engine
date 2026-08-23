"""Job and Job Log SQLite DAO repository."""
import sqlite3
import time
from typing import Optional, List, Dict


def create_job(
    conn: sqlite3.Connection,
    job_id: str,
    thread_id: str,
    card_id: str | None,
    idempotency_key: str,
    instruction: str,
    status: str = "queued",
    flow: str = "manager",
    scope: str = "both",
) -> None:
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
