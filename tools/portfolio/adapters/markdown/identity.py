"""Stable identity helpers for Markdown-backed portfolio state projections."""
from __future__ import annotations

from pathlib import Path

from application.knowledge.identity import build_document_key
from tools.archivist.identity_store import DurableIdentityStore


def portfolio_note_identity(
    vault_root: str | Path,
    portfolio_id: str,
    component: str,
    item_key: str | None = None,
) -> tuple[str, str]:
    """Return a durable note_id/document_key pair for one state projection."""
    root = Path(vault_root).resolve()
    if item_key is None:
        document_key = build_document_key(
            kind="portfolio_state",
            source_identity=portfolio_id,
            role=component,
        )
    else:
        document_key = build_document_key(
            kind="portfolio_item",
            source_identity=f"{portfolio_id}:{item_key}",
            role=component,
        )
    identity = DurableIdentityStore(root=root).reserve_note_identity(
        document_key=document_key,
        entity_id=f"portfolio:{portfolio_id}",
    )
    return identity.note_id, document_key

