"""Application Query Service for Note Catalog and Link Resolution."""
from __future__ import annotations

from typing import Optional

from application.knowledge.ports import (
    LinkResolverPort,
    NoteCatalogEntry,
    NoteCatalogPort,
)


class KnowledgeQueryService:
    """Provides high-level queries against the Vault Note Catalog."""

    def __init__(
        self,
        catalog: NoteCatalogPort,
        link_resolver: Optional[LinkResolverPort] = None,
    ) -> None:
        self._catalog = catalog
        self._link_resolver = link_resolver

    def get_note(self, note_id: str) -> Optional[NoteCatalogEntry]:
        return self._catalog.get_by_id(note_id)

    def get_note_by_path(self, relative_path: str) -> Optional[NoteCatalogEntry]:
        return self._catalog.get_by_path(relative_path)

    def find_notes_for_ticker(self, ticker: str, limit: int = 50) -> list[NoteCatalogEntry]:
        return self._catalog.find_notes(ticker=ticker.upper(), limit=limit)

    def find_notes_by_type(self, entity_type: str, limit: int = 100) -> list[NoteCatalogEntry]:
        return self._catalog.find_notes(entity_type=entity_type, limit=limit)

    def resolve_wikilink(self, target: str, context_path: Optional[str] = None) -> Optional[str]:
        if self._link_resolver is None:
            return None
        return self._link_resolver.resolve_target(target, context_path)
