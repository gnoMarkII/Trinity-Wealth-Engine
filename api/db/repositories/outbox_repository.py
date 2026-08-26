"""Raw SQLite DAO for durable application notifications.

The DAO intentionally does not commit or rollback.  A caller-owned
``DbUnitOfWork`` decides the transaction boundary so an outbox event can be
written atomically with the state mutation that produced it.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Mapping


def enqueue_event(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    idempotency_key: str,
    aggregate_type: str,
    aggregate_id: str,
    event_type: str,
    payload: Mapping[str, Any],
) -> sqlite3.Row:
    now = time.time()
    conn.execute(
        """INSERT INTO notification_outbox
           (event_id, idempotency_key, aggregate_type, aggregate_id, event_type,
            payload_json, status, attempts, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
           ON CONFLICT(idempotency_key) DO UPDATE SET updated_at = updated_at""",
        (
            event_id,
            idempotency_key,
            aggregate_type,
            aggregate_id,
            event_type,
            json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str),
            now,
            now,
        ),
    )
    row = conn.execute(
        "SELECT * FROM notification_outbox WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()
    if row is None:  # pragma: no cover - defensive guard for a broken DB driver
        raise RuntimeError("Outbox enqueue did not return the persisted event")
    return row


def get_event(conn: sqlite3.Connection, idempotency_key: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM notification_outbox WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()


def list_pending(conn: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM notification_outbox WHERE status IN ('pending', 'failed') "
        "ORDER BY created_at ASC LIMIT ?",
        (max(1, int(limit)),),
    ).fetchall()


def mark_sent(conn: sqlite3.Connection, idempotency_key: str) -> None:
    now = time.time()
    conn.execute(
        "UPDATE notification_outbox SET status = 'sent', sent_at = ?, updated_at = ?, last_error = NULL "
        "WHERE idempotency_key = ?",
        (now, now, idempotency_key),
    )


def mark_failed(conn: sqlite3.Connection, idempotency_key: str, error: str) -> None:
    now = time.time()
    conn.execute(
        "UPDATE notification_outbox SET status = 'failed', attempts = attempts + 1, "
        "last_error = ?, updated_at = ? WHERE idempotency_key = ?",
        (str(error)[:2000], now, idempotency_key),
    )
