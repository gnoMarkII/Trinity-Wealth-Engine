"""Earnings Call Outbox SQLite DAO Repository."""
import sqlite3
import time
import uuid
from typing import Optional, List

from application.earnings_call.dto import EarningsCallOutboxEventDTO, LeaseDTO


def _row_to_outbox_dto(row: sqlite3.Row) -> EarningsCallOutboxEventDTO:
    return EarningsCallOutboxEventDTO(
        event_id=row["event_id"],
        run_id=row["run_id"],
        source_key=row["source_key"],
        event_type=row["event_type"],
        status=row["status"],
        attempts=row["attempts"],
        available_at=row["available_at"],
        last_error=row["last_error"],
        lease_token=row["lease_token"],
        lease_expires_at=row["lease_expires_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def enqueue_event(
    conn: sqlite3.Connection,
    run_id: str,
    source_key: str,
    event_type: str = "deliver_kanban",
    outbox_lease_seconds: int = 60,
) -> tuple[EarningsCallOutboxEventDTO, LeaseDTO]:
    now = time.time()
    event_id = uuid.uuid4().hex
    lease_token = uuid.uuid4().hex
    lease_expires_at = now + outbox_lease_seconds

    conn.execute(
        "INSERT INTO earnings_call_outbox ("
        "  event_id, run_id, source_key, event_type, status, attempts, available_at,"
        "  lease_token, lease_expires_at, created_at, updated_at"
        ") VALUES (?, ?, ?, ?, 'leased', 1, ?, ?, ?, ?, ?) "
        "ON CONFLICT(run_id, event_type) DO UPDATE SET "
        "  source_key = excluded.source_key, status = 'leased', "
        "  attempts = attempts + 1, lease_token = excluded.lease_token,"
        "  lease_expires_at = excluded.lease_expires_at, last_error = NULL, "
        "  updated_at = excluded.updated_at",
        (event_id, run_id, source_key, event_type, now, lease_token, lease_expires_at, now, now),
    )

    cur = conn.execute("SELECT * FROM earnings_call_outbox WHERE run_id = ? AND event_type = ?", (run_id, event_type))
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"Failed to enqueue outbox event for run '{run_id}'")

    dto = _row_to_outbox_dto(row)
    lease = LeaseDTO(lease_token=row["lease_token"], lease_expires_at=row["lease_expires_at"])
    return dto, lease


def list_pending(
    conn: sqlite3.Connection,
    limit: int = 10,
) -> List[EarningsCallOutboxEventDTO]:
    now = time.time()
    cur = conn.execute(
        "SELECT * FROM earnings_call_outbox "
        "WHERE (status = 'pending' AND available_at <= ?) "
        "   OR (status = 'leased' AND (lease_expires_at IS NULL OR lease_expires_at <= ?)) "
        "ORDER BY available_at ASC LIMIT ?",
        (now, now, limit),
    )
    return [_row_to_outbox_dto(r) for r in cur.fetchall()]


def lease_event(
    conn: sqlite3.Connection,
    event_id: str,
    lease_seconds: int = 60,
) -> Optional[LeaseDTO]:
    now = time.time()
    lease_token = uuid.uuid4().hex
    lease_expires_at = now + lease_seconds

    cur = conn.execute(
        "UPDATE earnings_call_outbox SET "
        "  status = 'leased', lease_token = ?, lease_expires_at = ?, attempts = attempts + 1, updated_at = ? "
        "WHERE event_id = ? AND ((status = 'pending' AND available_at <= ?) "
        "OR (status = 'leased' AND (lease_expires_at IS NULL OR lease_expires_at <= ?)))",
        (lease_token, lease_expires_at, now, event_id, now, now),
    )
    if cur.rowcount == 0:
        return None
    return LeaseDTO(lease_token=lease_token, lease_expires_at=lease_expires_at)


def complete_event(
    conn: sqlite3.Connection,
    event_id: str,
    lease_token: str,
) -> bool:
    now = time.time()
    cur = conn.execute(
        "UPDATE earnings_call_outbox SET "
        "  status = 'completed', last_error = NULL, lease_token = NULL, "
        "  lease_expires_at = NULL, updated_at = ? "
        "WHERE event_id = ? AND lease_token = ? "
        "  AND status = 'leased' AND lease_expires_at > ?",
        (now, event_id, lease_token, now),
    )
    return cur.rowcount > 0


def schedule_retry(
    conn: sqlite3.Connection,
    event_id: str,
    lease_token: str,
    error_code: str,
    retry_delay_seconds: int,
) -> bool:
    now = time.time()
    available_at = now + retry_delay_seconds
    cur = conn.execute(
        "UPDATE earnings_call_outbox SET "
        "  status = 'pending', available_at = ?, last_error = ?, lease_token = NULL, lease_expires_at = NULL, updated_at = ? "
        "WHERE event_id = ? AND lease_token = ? "
        "  AND status = 'leased' AND lease_expires_at > ?",
        (available_at, error_code, now, event_id, lease_token, now),
    )
    return cur.rowcount > 0


def mark_dead_letter(
    conn: sqlite3.Connection,
    event_id: str,
    lease_token: str,
    error_code: str,
) -> bool:
    now = time.time()
    cur = conn.execute(
        "UPDATE earnings_call_outbox SET "
        "  status = 'dead_letter', last_error = ?, lease_token = NULL, lease_expires_at = NULL, updated_at = ? "
        "WHERE event_id = ? AND lease_token = ? "
        "  AND status = 'leased' AND lease_expires_at > ?",
        (error_code, now, event_id, lease_token, now),
    )
    return cur.rowcount > 0


def reset_for_manual_retry(
    conn: sqlite3.Connection,
    run_id: str,
    source_key: str,
    lease_seconds: int = 60,
) -> tuple[EarningsCallOutboxEventDTO, LeaseDTO]:
    now = time.time()
    lease_token = uuid.uuid4().hex
    lease_expires_at = now + lease_seconds

    cur = conn.execute(
        "UPDATE earnings_call_outbox SET "
        "  status = 'leased', attempts = 1, available_at = ?, last_error = NULL, "
        "  lease_token = ?, lease_expires_at = ?, updated_at = ? "
        "WHERE run_id = ? AND source_key = ? AND event_type = 'deliver_kanban' "
        "  AND (status IN ('pending', 'dead_letter') "
        "       OR (status = 'leased' AND (lease_expires_at IS NULL OR lease_expires_at <= ?)))",
        (now, lease_token, lease_expires_at, now, run_id, source_key, now),
    )
    if cur.rowcount == 0:
        raise RuntimeError(
            f"Run '{run_id}' has no retryable deliver_kanban outbox event"
        )

    row = conn.execute(
        "SELECT * FROM earnings_call_outbox "
        "WHERE run_id = ? AND source_key = ? AND event_type = 'deliver_kanban'",
        (run_id, source_key),
    ).fetchone()
    dto = _row_to_outbox_dto(row)
    lease = LeaseDTO(lease_token=lease_token, lease_expires_at=lease_expires_at)
    return dto, lease
