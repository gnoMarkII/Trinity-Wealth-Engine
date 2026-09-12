"""Application ports and data models for Note Catalog and Link Resolution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class NoteCatalogEntry:
    note_id: str
    document_key: str
    relative_path: str
    entity_type: str
    title: str
    date: Optional[str] = None
    ticker: Optional[str] = None
    source_key: Optional[str] = None
    mtime: float = 0.0
    file_size: int = 0
    content_sha256: str = ""
    metadata_json: str = "{}"
    updated_at: str = ""
    current_revision_id: Optional[str] = None
    current_revision: Optional[int] = None
    manifest_digest: Optional[str] = None
    artifact_set_hash: Optional[str] = None
    body_sha256: Optional[str] = None
    record_state: str = "active"
    storage_scope: str = "active"


@runtime_checkable
class NoteCatalogPort(Protocol):
    """Port for querying and managing indexed knowledge notes."""

    def upsert_note(self, entry: NoteCatalogEntry) -> None:
        """Insert or update a note entry in the catalog."""
        ...

    def delete_note(self, note_id: str) -> None:
        """Remove a note entry by its note_id."""
        ...

    def get_by_id(self, note_id: str) -> Optional[NoteCatalogEntry]:
        """Fetch note entry by note_id."""
        ...

    def get_by_path(self, relative_path: str) -> Optional[NoteCatalogEntry]:
        """Fetch note entry by relative path within the vault."""
        ...

    def find_notes(
        self,
        *,
        entity_type: Optional[str] = None,
        ticker: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[NoteCatalogEntry]:
        """Find note entries matching filters."""
        ...

    def count_notes(self, entity_type: Optional[str] = None) -> int:
        """Count total notes matching optional entity_type filter."""
        ...

    def iter_notes(
        self,
        *,
        entity_type: Optional[str] = None,
        ticker: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page_size: int = 500,
    ) -> Any:
        """Iterate notes using a stable keyset cursor without materializing the corpus."""
        ...


@runtime_checkable
class LinkResolverPort(Protocol):
    """Port for resolving wikilinks and note references."""

    def resolve_target(self, link_target: str, context_path: Optional[str] = None) -> Optional[str]:
        """Resolves a link target string (e.g. [[FTNT]] or [[Analysis/2026-09-05]]) to a canonical relative path."""
        ...
