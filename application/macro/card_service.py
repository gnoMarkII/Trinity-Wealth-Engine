"""Application use case for publishing News Funnel approval cards."""
from __future__ import annotations

import uuid
from typing import Any

from application.macro.ports import NewsFunnelCardPort, NewsFunnelPromptPort


class NewsFunnelCardApplicationService:
    def __init__(self, storage: NewsFunnelCardPort, prompt: NewsFunnelPromptPort) -> None:
        self._storage = storage
        self._prompt = prompt

    def upsert(self, period: str, pending_events: list[dict[str, Any]]) -> None:
        title = f"[{period.upper()}] News Funnel High-Impact ({len(pending_events)} items)"
        card_payload = {
            "title": title,
            "flow": "news_funnel",
            "prompt": self._prompt.format_prompt(period, pending_events),
            "scope": "both",
        }
        card = {"card_id": str(uuid.uuid4()), **card_payload}
        # New repository adapters perform this as one transaction.  The
        # fallback is intentionally retained only for older injected test or
        # downstream adapters during the migration window.
        upsert = getattr(self._storage, "upsert_open_card", None)
        if callable(upsert):
            upsert(card)
            return

        existing = self._storage.find_open_card("news_funnel")
        if existing is None:
            self._storage.create_card(card)
        else:
            self._storage.update_card(existing["card_id"], card_payload)


__all__ = ["NewsFunnelCardApplicationService"]
