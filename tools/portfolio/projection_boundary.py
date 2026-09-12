"""Deterministic portfolio projection commands and managed-block rendering."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from application.knowledge.write_models import KnowledgeWriteCommand, KnowledgeWriteReceipt
from application.knowledge.write_ports import KnowledgeWritePort
from tools.archivist.managed_blocks import ManagedBlockError, replace_managed_block
from tools.archivist.metadata import parse_note


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ProjectionCheckpoint:
    portfolio_id: str
    checkpoint_id: str
    projection_version: str
    source_hash: str
    generated_at: str


def checkpoint_for(portfolio_id: str, source_state: Mapping[str, Any], *, projection_version: str = "portfolio-projection-v2") -> ProjectionCheckpoint:
    return ProjectionCheckpoint(
        portfolio_id=str(portfolio_id),
        checkpoint_id=f"checkpoint_{_digest({'portfolio_id': portfolio_id, 'state': source_state})[:24]}",
        projection_version=projection_version,
        source_hash=_digest(source_state),
        generated_at=_utc_now(),
    )


def render_projection_body(
    generated_text: str,
    *,
    existing_body: Optional[str] = None,
    block_id: str = "portfolio-summary",
) -> str:
    """Replace only the managed block, preserving human prose exactly."""
    generated = str(generated_text).strip("\n")
    if existing_body and f"<!-- managed:start {block_id} -->" in existing_body:
        return replace_managed_block(existing_body, block_id, generated)
    if existing_body:
        return f"<!-- managed:start {block_id} -->\n{generated}\n<!-- managed:end {block_id} -->\n\n{existing_body.strip()}\n"
    return f"<!-- managed:start {block_id} -->\n{generated}\n<!-- managed:end {block_id} -->\n\n## Human Notes\n\n"


class PortfolioProjectionBoundary:
    """Build and submit projection updates without exposing filesystem paths."""

    def __init__(self, writer: KnowledgeWritePort) -> None:
        self.writer = writer

    def rebuild(
        self,
        *,
        portfolio_id: str,
        projections: Iterable[Mapping[str, Any]],
        source_state: Mapping[str, Any],
        projection_version: str = "portfolio-projection-v2",
    ) -> list[KnowledgeWriteReceipt]:
        checkpoint = checkpoint_for(portfolio_id, source_state, projection_version=projection_version)
        receipts: list[KnowledgeWriteReceipt] = []
        for projection in projections:
            entity_type = str(projection.get("entity_type") or "holding")
            title = str(projection.get("title") or projection.get("symbol") or "Projection")
            generated_text = str(projection.get("generated_text") or projection.get("body") or "").strip()
            existing_body = projection.get("existing_body")
            body = render_projection_body(generated_text, existing_body=str(existing_body) if existing_body is not None else None)
            metadata = dict(projection.get("metadata") or {})
            metadata.update(
                {
                    "schema_version": 2,
                    "entity_type": entity_type,
                    "title": title,
                    "document_role": "projection",
                    "portfolio_id": portfolio_id,
                    "search_scope": "excluded",
                    "projection_of": "portfolio_transactions",
                    "projection_version": checkpoint.projection_version,
                    "generated_at": checkpoint.generated_at,
                    "source_checkpoint": checkpoint.checkpoint_id,
                    "source_content_hashes": [checkpoint.source_hash],
                    "content_status": "generated",
                }
            )
            key = str(projection.get("stable_key") or title)
            command = KnowledgeWriteCommand(
                operation="regenerate_projection",
                idempotency_key=f"projection:{portfolio_id}:{entity_type}:{key}:{checkpoint.checkpoint_id}",
                document_key=str(metadata.get("document_key")) if metadata.get("document_key") else None,
                entity_type=entity_type,
                producer="portfolio-projection",
                producer_version=projection_version,
                actor="app",
                payload={
                    "metadata": metadata,
                    "body": body,
                    "profile_id": "portfolio_projection",
                    "projection_command": "upsert_note",
                },
            )
            # The first broker implementation intentionally keeps the public
            # operation explicit.  A projection adapter can translate it to a
            # note write without allowing a generic caller to do so.
            translated = KnowledgeWriteCommand(
                **{
                    **command.__dict__,
                    "operation": "upsert_note",
                    "payload": {**command.payload, "profile_id": "portfolio_projection"},
                }
            )
            receipts.append(self.writer.submit(translated))
        return receipts

    def rebuild_from_transaction_store(
        self,
        *,
        store: Any,
        portfolio_id: str,
        projections: Iterable[Mapping[str, Any]],
        projection_version: str = "portfolio-projection-v2",
    ) -> list[KnowledgeWriteReceipt]:
        """Replay the external event source, then publish deterministic views."""
        state = dict(store.replay(portfolio_id))
        checkpoint = store.checkpoint(portfolio_id)
        source_state = {
            "state": state,
            "checkpoint": {
                "sequence": checkpoint.sequence,
                "event_log_hash": checkpoint.event_log_hash,
                "state_hash": checkpoint.state_hash,
            },
        }
        return self.rebuild(
            portfolio_id=portfolio_id,
            projections=projections,
            source_state=source_state,
            projection_version=projection_version,
        )
