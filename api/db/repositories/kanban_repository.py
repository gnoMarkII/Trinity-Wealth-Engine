"""Kanban cards and Parking Lot SQLite DAO repository."""
import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from contextlib import closing
from typing import Optional, List, Dict

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
