"""Adapter for the persisted Macro news funnel store."""
from __future__ import annotations

from typing import Any, Callable, Optional

from tools.macro import news_funnel_store
from tools.macro.news_funnel import get_synthesis_period


class NewsFunnelStoreAdapter:
    def __init__(self, card_sync: Optional[Callable[[str, list[dict[str, Any]]], None]] = None) -> None:
        self._card_sync = card_sync

    def pending(self) -> list[dict[str, Any]]:
        return news_funnel_store.get_pending_high_impact_events()

    def filtered(self) -> list[dict[str, Any]]:
        return news_funnel_store.get_filtered_or_rejected_events()

    def reject(self, event_id: str) -> int:
        news_funnel_store.update_events_status(rejected_ids=[event_id])
        remaining = self.pending()
        if self._card_sync is not None:
            self._card_sync(get_synthesis_period(), remaining)
        return len(remaining)
