"""Kanban cards and Parking Lot SQLite DAO repository."""
import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from typing import Optional, List, Dict

# Compatibility import retained for failure-injection tests and old callers.
# Transaction ownership remains in the facade; the DAO itself never invokes
# this factory from its SQL methods.
from api.db.connection import get_connection


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
    source_key: str | None = None,
    scope: str = "both",
    is_verified: bool = True,
) -> None:
    now = time.time()
    next_seq = conn.execute("SELECT COALESCE(MAX(display_seq), 0) + 1 FROM kanban_cards").fetchone()[0]
    conn.execute(
        "INSERT INTO kanban_cards (card_id, title, column_name, job_id, flow, display_seq, prompt, source_key, scope, is_verified, created_at, updated_at) "
        "VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
        (card_id, title, column_name, flow, next_seq, prompt, source_key, scope, 1 if is_verified else 0, now, now),
    )


def upsert_open_card(conn: sqlite3.Connection, card: Dict[str, object]) -> sqlite3.Row:
    """Atomically create/update the single open approval card for a flow.

    The caller owns the transaction.  Keeping the read and write on the same
    connection prevents two concurrent funnel runs from both creating an
    approval card.
    """
    flow = str(card.get("flow") or "manager")
    title = str(card.get("title") or "")
    prompt = card.get("prompt")
    scope = str(card.get("scope") or "both")
    row = conn.execute(
        "SELECT * FROM kanban_cards "
        "WHERE flow = ? AND column_name IN ('backlog', 'approval') "
        "ORDER BY created_at ASC LIMIT 1",
        (flow,),
    ).fetchone()
    if row is None:
        create_kanban_card(
            conn=conn,
            card_id=str(card.get("card_id") or ""),
            title=title,
            column_name="backlog",
            flow=flow,
            prompt=str(prompt) if prompt is not None else None,
            scope=scope,
            is_verified=bool(card.get("is_verified", True)),
        )
    else:
        update_kanban_card(
            conn=conn,
            card_id=str(row["card_id"]),
            title=title,
            prompt=str(prompt) if prompt is not None else None,
            flow=flow,
            scope=scope,
        )
    result_id = str(row["card_id"]) if row is not None else str(card.get("card_id") or "")
    result = get_kanban_card(conn, result_id)
    if result is None:
        raise RuntimeError("Kanban upsert did not return the persisted card")
    return result


def create_parking_lot_cards(
    conn: sqlite3.Connection,
    ideas: list[str],
    source_pitch_id: str,
) -> int:
    """Insert parking-lot cards on a caller-owned transaction connection.

    This raw DAO deliberately owns no transaction lifecycle.  The
    compatibility facade or a higher-level unit of work decides when to begin,
    commit, or roll back the operation.
    """
    if not ideas:
        return 0

    now = time.time()
    created_count = 0
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
    return created_count


def update_kanban_card(
    conn: sqlite3.Connection, card_id: str, title: str, prompt: str | None, flow: str, scope: str
) -> None:
    now = time.time()
    conn.execute(
        "UPDATE kanban_cards SET title = ?, prompt = ?, flow = ?, scope = ?, updated_at = ? WHERE card_id = ?",
        (title, prompt, flow, scope, now, card_id),
    )


def set_kanban_card_source(conn: sqlite3.Connection, card_id: str, prompt: str, is_verified: bool) -> None:
    """UPDATE เฉพาะ prompt/is_verified — partial patch โดยตั้งใจ ไม่แตะ title/flow/scope."""
    now = time.time()
    conn.execute(
        "UPDATE kanban_cards SET prompt = ?, is_verified = ?, updated_at = ? WHERE card_id = ?",
        (prompt, 1 if is_verified else 0, now, card_id),
    )


def toggle_kanban_card_discord(conn: sqlite3.Connection, card_id: str, enabled: bool) -> None:
    """UPDATE เฉพาะคอลัมน์ discord_notify — partial patch โดยตั้งใจ ไม่แตะ title/prompt/flow/scope."""
    now = time.time()
    conn.execute(
        "UPDATE kanban_cards SET discord_notify = ?, updated_at = ? WHERE card_id = ?",
        (1 if enabled else 0, now, card_id),
    )


def mark_discord_events_sent(conn: sqlite3.Connection, card_id: str, event_ids: list[str]) -> None:
    """เพิ่ม event_ids ที่เพิ่งส่ง Discord สำเร็จเข้า discord_sent_events."""
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


def find_kanban_card_by_source_key(
    conn: sqlite3.Connection, source_key: str
) -> sqlite3.Row | None:
    cur = conn.execute(
        "SELECT * FROM kanban_cards WHERE source_key = ? "
        "ORDER BY created_at ASC LIMIT 1",
        (source_key,),
    )
    return cur.fetchone()


def delete_kanban_card(conn: sqlite3.Connection, card_id: str) -> None:
    conn.execute("DELETE FROM kanban_cards WHERE card_id = ?", (card_id,))
