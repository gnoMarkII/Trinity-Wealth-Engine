"""Advisor Chat SQLite DAO for managing session and message storage.

This module is a pure data access object (DAO) residing in tools/hermes_bridge.
It receives a managed connection from UnitOfWork without initiating connections or calling commit/rollback directly.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Optional, Sequence

from core.sandbox.models import AdvisorChatMessage, AdvisorSessionSummary


def list_sessions(conn: sqlite3.Connection, limit: int = 30) -> list[AdvisorSessionSummary]:
    """Retrieve recent chat sessions with their message counts."""
    cursor = conn.execute(
        """
        SELECT s.session_id, s.title, s.created_at, s.updated_at, COUNT(m.id) as msg_count
        FROM advisor_sessions s
        LEFT JOIN advisor_messages m ON s.session_id = m.session_id
        GROUP BY s.session_id
        ORDER BY s.updated_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    results: list[AdvisorSessionSummary] = []
    for row in rows:
        results.append(
            AdvisorSessionSummary(
                id=row["session_id"],
                title=row["title"],
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                message_count=row["msg_count"],
            )
        )
    return results


def create_session(conn: sqlite3.Connection, session_id: str, title: str) -> AdvisorSessionSummary:
    """Insert a new advisor chat session record."""
    now = time.time()
    conn.execute(
        "INSERT INTO advisor_sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, title, now, now),
    )
    return AdvisorSessionSummary(
        id=session_id,
        title=title,
        created_at=str(now),
        updated_at=str(now),
        message_count=0,
    )


def update_session_title(conn: sqlite3.Connection, session_id: str, title: str) -> None:
    """Update title for an existing chat session."""
    now = time.time()
    conn.execute(
        "UPDATE advisor_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
        (title, now, session_id),
    )


def get_session(conn: sqlite3.Connection, session_id: str) -> Optional[AdvisorSessionSummary]:
    """Fetch session details by session_id."""
    cursor = conn.execute(
        """
        SELECT s.session_id, s.title, s.created_at, s.updated_at, COUNT(m.id) as msg_count
        FROM advisor_sessions s
        LEFT JOIN advisor_messages m ON s.session_id = m.session_id
        WHERE s.session_id = ?
        GROUP BY s.session_id
        """,
        (session_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return AdvisorSessionSummary(
        id=row["session_id"],
        title=row["title"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        message_count=row["msg_count"],
    )


def get_messages(conn: sqlite3.Connection, session_id: str) -> list[AdvisorChatMessage]:
    """Fetch ordered chat messages for a session."""
    cursor = conn.execute(
        "SELECT id, session_id, role, content, citations_json, created_at FROM advisor_messages WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    )
    rows = cursor.fetchall()
    messages: list[AdvisorChatMessage] = []
    for r in rows:
        try:
            citations = tuple(json.loads(r["citations_json"] or "[]"))
        except Exception:
            citations = ()
        messages.append(
            AdvisorChatMessage(
                id=r["id"],
                session_id=r["session_id"],
                role=r["role"],
                content=r["content"],
                citations=citations,
                created_at=str(r["created_at"]),
            )
        )
    return messages


def append_message(
    conn: sqlite3.Connection,
    msg_id: str,
    session_id: str,
    role: str,
    content: str,
    citations_json: str = "[]",
) -> AdvisorChatMessage:
    """Append a new chat turn message into session history."""
    now = time.time()
    conn.execute(
        "INSERT INTO advisor_messages (id, session_id, role, content, citations_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (msg_id, session_id, role, content, citations_json, now),
    )
    conn.execute(
        "UPDATE advisor_sessions SET updated_at = ? WHERE session_id = ?",
        (now, session_id),
    )
    try:
        citations = tuple(json.loads(citations_json or "[]"))
    except Exception:
        citations = ()
    return AdvisorChatMessage(
        id=msg_id,
        session_id=session_id,
        role=role,
        content=content,
        citations=citations,
        created_at=str(now),
    )

