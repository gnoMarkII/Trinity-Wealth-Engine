"""Advisor Chat SQLite DAO repository facade delegating to tools.hermes_bridge.advisor_session_dao."""
from __future__ import annotations

from tools.hermes_bridge.advisor_session_dao import (
    append_message,
    create_session,
    get_messages,
    get_session,
    list_sessions,
    update_session_title,
)

__all__ = [
    "append_message",
    "create_session",
    "get_messages",
    "get_session",
    "list_sessions",
    "update_session_title",
]
