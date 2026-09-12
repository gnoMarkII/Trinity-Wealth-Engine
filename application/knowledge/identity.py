"""Pure Domain and Application Identity Contract for Knowledge Objects.

Defines deterministic document keys, note identity, revision references,
and ports for identity reservation and stable entity mapping.
Strict boundary: No filesystem, SQLite, or external provider imports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


def normalize_source_identity(kind: str, source_identity: str) -> str:
    """Normalizes source_identity depending on kind.
    URLs and video_ids preserve case. Tickers are uppercased.
    """
    clean_kind = str(kind).strip().lower()
    clean_source = str(source_identity).strip()
    if clean_kind in ("youtube_summary", "article", "company_news", "web_page") or "http://" in clean_source or "https://" in clean_source:
        return clean_source.rstrip("/")
    return clean_source.upper()


def build_document_key(
    kind: str,
    source_identity: str,
    role: str = "primary",
    version: int = 1,
    as_of: Optional[str] = None,
    scope: Optional[str] = None,
) -> str:
    """Constructs a deterministic, versioned canonical document_key tuple.

    document_key = logical source + note role according to document type rules.
    It is NOT a filename or a run ID.
    Example: 'v1:stock_hub:FTNT:hub' or 'v1:earnings_call:FTNT:2026-Q2:primary'
    Or with as_of: 'v1:equity_analysis:FTNT:2026-09-01:primary'
    """
    clean_kind = str(kind).strip().lower()
    clean_source = normalize_source_identity(clean_kind, source_identity)
    clean_role = str(role).strip().lower()

    parts = [f"v{version}", clean_kind, clean_source]
    if as_of:
        parts.append(str(as_of).strip())
    if scope:
        parts.append(str(scope).strip().lower())
    parts.append(clean_role)
    return ":".join(parts)


@dataclass(frozen=True)
class NoteIdentity:
    note_id: str
    document_key: str
    entity_id: Optional[str] = None
    created_at: Optional[str] = None


@dataclass(frozen=True)
class RevisionRef:
    note_id: str
    revision_id: str
    revision: int  # Human-readable integer revision counter (1, 2, 3...)
    content_hash: str
    created_at: Optional[str] = None


@runtime_checkable
class IdentityReservationPort(Protocol):
    """Port for reserving and persisting note identities across concurrent processes."""

    def reserve_note_identity(
        self,
        document_key: str,
        entity_id: Optional[str] = None,
    ) -> NoteIdentity:
        """Atomically reserves or retrieves the existing NoteIdentity for document_key."""
        ...

    def import_note_identity(
        self,
        note_id: str,
        document_key: str,
        entity_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> NoteIdentity:
        """Explicitly imports and pins an existing note_id for document_key without allocating new ID."""
        ...

    def get_note_identity(self, document_key: str) -> Optional[NoteIdentity]:
        """Retrieves existing NoteIdentity by document_key if already allocated."""
        ...


@runtime_checkable
class EntityRegistryPort(Protocol):
    """Port for resolving financial assets/tickers to stable entity_ids."""

    def resolve_stable_entity_id(
        self,
        ticker: str,
        market: Optional[str] = None,
    ) -> str:
        """Returns a stable entity_id independent of ticker changes or relistings."""
        ...
