"""Compatibility-driven state-store adapter for background workers.

Worker modules should not import the ``api.state_db`` compatibility facade
directly.  This adapter is the single legacy bridge while the worker use
cases move to connection-bound repository ports.  It resolves the facade
lazily so existing tests and downstream integrations that patch its functions
keep working during the migration.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Optional

from application.macro.ports import NewsFunnelCardPort, NewsFunnelPromptPort
from api.db.uow import DbUnitOfWork
from api.db.repositories import kanban_repository
from api.db.repositories import outbox_repository


def _legacy_state_db():
    from api import state_db

    return state_db


class LegacyStateStoreAdapter:
    """Outbound compatibility adapter for legacy state-store operations."""

    def get_connection(self, db_path: Optional[str] = None):
        return _legacy_state_db().get_connection(db_path)

    def append_job_log(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        node_name: str,
        content: str,
        role: str = "reply",
        label: Optional[str] = None,
    ) -> None:
        _legacy_state_db().append_job_log(conn, job_id, node_name, content, role, label)

    def claim_job_resume(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        resume_value_json: str,
        token_uses: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        _legacy_state_db().claim_job_resume(
            conn, job_id, resume_value_json, token_uses=token_uses
        )

    def get_job_reply_logs(self, conn: sqlite3.Connection, job_id: str):
        return _legacy_state_db().get_job_reply_logs(conn, job_id)

    def get_job(self, conn: sqlite3.Connection, job_id: str):
        return _legacy_state_db().get_job(conn, job_id)

    def get_kanban_card(self, conn: sqlite3.Connection, card_id: str):
        return _legacy_state_db().get_kanban_card(conn, card_id)

    def list_kanban_cards(self, conn: sqlite3.Connection):
        return _legacy_state_db().list_kanban_cards(conn)

    def create_kanban_card(self, conn: sqlite3.Connection, **kwargs: Any) -> None:
        _legacy_state_db().create_kanban_card(conn, **kwargs)

    def update_kanban_card(self, conn: sqlite3.Connection, **kwargs: Any) -> None:
        _legacy_state_db().update_kanban_card(conn, **kwargs)

    def mark_discord_events_sent(
        self, conn: sqlite3.Connection, card_id: str, event_ids: list[str]
    ) -> None:
        _legacy_state_db().mark_discord_events_sent(conn, card_id, event_ids)

    def set_job_awaiting_approval(
        self, conn: sqlite3.Connection, job_id: str, interrupt_payload_json: str
    ) -> None:
        _legacy_state_db().set_job_awaiting_approval(conn, job_id, interrupt_payload_json)

    def update_job_status(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        _legacy_state_db().update_job_status(conn, job_id, status, error_message)


state_store = LegacyStateStoreAdapter()


class LegacyNewsFunnelCardAdapter(NewsFunnelCardPort):
    """Connection-lifecycle adapter for the legacy Kanban facade."""

    def upsert_open_card(self, card: dict[str, Any]) -> dict[str, Any]:
        """Persist the open card in one caller-owned UoW transaction."""
        with DbUnitOfWork() as uow:
            row = kanban_repository.upsert_open_card(uow.conn, card)
            return dict(row)

    def find_open_card(self, flow: str) -> Optional[dict[str, Any]]:
        conn = state_store.get_connection()
        try:
            cards = state_store.list_kanban_cards(conn)
            return next(
                (dict(card) for card in cards if card["flow"] == flow and card["column_name"] in ("backlog", "approval")),
                None,
            )
        finally:
            conn.close()

    def create_card(self, card: dict[str, Any]) -> None:
        conn = state_store.get_connection()
        try:
            state_store.create_kanban_card(
                conn,
                card_id=card["card_id"],
                title=card["title"],
                column_name="backlog",
                flow=card["flow"],
                prompt=card["prompt"],
                scope=card.get("scope", "both"),
            )
        finally:
            conn.close()

    def update_card(self, card_id: str, card: dict[str, Any]) -> None:
        conn = state_store.get_connection()
        try:
            existing = state_store.get_kanban_card(conn, card_id)
            state_store.update_kanban_card(
                conn,
                card_id=card_id,
                title=card["title"],
                prompt=card["prompt"],
                flow=card["flow"],
                scope=(existing["scope"] if existing else card.get("scope", "both")),
            )
        finally:
            conn.close()


class LegacyNotebookLMWorkerStateAdapter:
    """Typed worker state port backed by the compatibility facade."""

    def append_log(self, job_id: str, node: str, message: str) -> None:
        conn = state_store.get_connection()
        try:
            state_store.append_job_log(conn, job_id, node, message, role="reply", label=node)
        finally:
            conn.close()

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        conn = state_store.get_connection()
        try:
            row = state_store.get_job(conn, job_id)
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def get_card(self, card_id: str) -> Optional[dict[str, Any]]:
        conn = state_store.get_connection()
        try:
            row = state_store.get_kanban_card(conn, card_id)
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def mark_discord_events_sent(self, card_id: str, event_ids: list[str]) -> None:
        conn = state_store.get_connection()
        try:
            state_store.mark_discord_events_sent(conn, card_id, event_ids)
        finally:
            conn.close()


class LegacyNotebookLMNotificationOutboxAdapter:
    """Outbox adapter using the same patchable connection factory as workers.

    This bridge is temporary: new workers should receive the connection-bound
    ``SqliteNotificationOutboxAdapter`` from the composition root.  Keeping
    this adapter on the legacy factory preserves existing test/integration
    seams while the worker entry point is migrated.
    """

    def enqueue(self, **kwargs: Any) -> dict[str, Any]:
        conn = state_store.get_connection()
        try:
            row = outbox_repository.enqueue_event(conn, **kwargs)
            conn.commit()
            return dict(row)
        finally:
            conn.close()

    def get(self, idempotency_key: str) -> Optional[dict[str, Any]]:
        conn = state_store.get_connection()
        try:
            row = outbox_repository.get_event(conn, idempotency_key)
            return dict(row) if row is not None else None
        finally:
            conn.close()

    def list_pending(self, limit: int = 100) -> list[dict[str, Any]]:
        conn = state_store.get_connection()
        try:
            return [dict(row) for row in outbox_repository.list_pending(conn, limit=limit)]
        finally:
            conn.close()

    def mark_sent(self, idempotency_key: str) -> None:
        conn = state_store.get_connection()
        try:
            outbox_repository.mark_sent(conn, idempotency_key)
            conn.commit()
        finally:
            conn.close()

    def mark_failed(self, idempotency_key: str, error: str) -> None:
        conn = state_store.get_connection()
        try:
            outbox_repository.mark_failed(conn, idempotency_key, error)
            conn.commit()
        finally:
            conn.close()

class NewsFunnelPromptAdapter(NewsFunnelPromptPort):
    def format_prompt(self, period: str, events: list[dict[str, Any]]) -> str:
        from tools.macro.news_funnel import format_news_funnel_card_prompt

        return format_news_funnel_card_prompt(period, events)

__all__ = [
    "LegacyStateStoreAdapter",
    "LegacyNewsFunnelCardAdapter",
    "LegacyNotebookLMWorkerStateAdapter",
    "LegacyNotebookLMNotificationOutboxAdapter",
    "NewsFunnelPromptAdapter",
    "state_store",
]
