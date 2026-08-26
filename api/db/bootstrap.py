"""Composition hooks for database-adjacent integrations.

The SQLite compatibility facade must not configure unrelated content modules
at import time.  The API composition root calls this function during startup;
CLI callers can opt in explicitly, while the outbox module still has its
standalone fallback for offline use.
"""
from __future__ import annotations


def configure_content_outbox_sync() -> None:
    """Wire parking-lot reconciliation to the compatibility SQLite facade."""
    from tools.content.parking_lot_outbox import set_sync_handler
    from api.state_db import create_parking_lot_cards_atomic

    set_sync_handler(
        lambda ideas, source_pitch_id, db_path=None: create_parking_lot_cards_atomic(
            ideas=ideas,
            source_pitch_id=source_pitch_id,
            db_path=db_path,
        )
    )


__all__ = ["configure_content_outbox_sync"]
