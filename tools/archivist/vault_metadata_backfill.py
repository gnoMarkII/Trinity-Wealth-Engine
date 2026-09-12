"""Automated YAML Frontmatter Metadata Backfill Engine for Obsidian Vault V2.

Scans notes lacking standardized frontmatter and non-destructively prepends V2 metadata,
resolving schema debt across legacy ingested files.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from core.logger import get_logger
from tools.archivist.artifact_writer import ArtifactWriter
from tools.archivist.identity_store import DurableIdentityStore
from tools.archivist.metadata import normalize_legacy_metadata, parse_note, validate_note
from tools.archivist.writer import _portableize_wikilinks
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.vault_policy import is_searchable_note

logger = get_logger(__name__)

_DATE_REGEX = re.compile(r"(\d{4}-\d{2}-\d{2})")
_H1_REGEX = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _infer_entity_type_from_path(rel_parts: tuple[str, ...]) -> str:
    """Infers appropriate entity_type based on V2 PARA folder hierarchy."""
    rel_str = "/".join(rel_parts)
    if "Books" in rel_parts:
        return "book_note"
    if "YouTube_Summaries" in rel_parts:
        return "youtube_insight"
    if "News" in rel_parts:
        return "article_note"
    if "Earnings" in rel_parts or "Earnings_Calls" in rel_parts:
        return "earnings_call"
    if "Analysis" in rel_parts:
        return "equity_analysis"
    if "Stocks" in rel_parts:
        return "company"
    if "Macroeconomics" in rel_parts or "Strategies" in rel_parts:
        return "macro_strategy"
    if "Daily_Snapshots" in rel_parts:
        return "macro_snapshot"
    return "concept"


def backfill_vault_metadata(
    vault_root: Optional[Union[str, Path]] = None,
    dry_run: bool = True,
    limit: Optional[int] = None,
) -> dict[str, int]:
    """Backfills incomplete published notes without touching archive/templates."""
    v_root = Path(vault_root).resolve() if vault_root else Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
    if not v_root.exists():
        return {"scanned": 0, "backfilled": 0, "already_has_meta": 0}

    cat = None
    identity_store = DurableIdentityStore(root=v_root)
    try:
        from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
        from tools.archivist.catalog_runtime import resolve_catalog_path
        cat_db = resolve_catalog_path(v_root, require_exists=True)
        if cat_db.exists():
            cat = SqliteNoteCatalogAdapter(db_path=cat_db, vault_root=v_root)
    except Exception:
        pass

    scanned = 0
    backfilled = 0
    already_has_meta = 0
    skipped_malformed = 0

    for md_path in v_root.rglob("*.md"):
        if not is_searchable_note(md_path, vault_root=v_root):
            continue

        scanned += 1
        try:
            content = md_path.read_text(encoding="utf-8")
        except OSError:
            continue

        raw_meta, raw_body, parse_issues = parse_note(content)
        if parse_issues:
            skipped_malformed += 1
            continue
        # A title/entity pair is not enough for R7.  Only a strict V2 profile
        # with no legacy wikilinks is considered already repaired.
        if (
            all(str(raw_meta.get(field) or "").strip() for field in ("note_id", "document_key", "entity_type", "title"))
            and raw_meta.get("schema_version") == 2
            and not re.search(r"\[\[|!\[\[|obsidian://", content)
            and validate_note(raw_meta, mode="strict")[0] is not None
        ):
            already_has_meta += 1
            continue

        # Extract title from H1 or filename
        h1_match = _H1_REGEX.search(raw_body)
        title = h1_match.group(1).strip() if h1_match else md_path.stem.replace("_", " ")

        # Extract or infer date
        date_match = _DATE_REGEX.search(md_path.name) or _DATE_REGEX.search(raw_body[:500])
        date_val = date_match.group(1) if date_match else None

        # Infer entity type
        rel = md_path.relative_to(v_root)
        entity_type = _infer_entity_type_from_path(rel.parts)

        # Build updated metadata
        meta, _ = normalize_legacy_metadata(dict(raw_meta), producer="vault_metadata_backfill")
        meta.setdefault("title", title)
        meta.setdefault("entity_type", entity_type)
        if date_val:
            meta.setdefault("date", date_val)
        existing_tags = meta.get("tags") or []
        if isinstance(existing_tags, str):
            existing_tags = [existing_tags]
        if "legacy-backfilled" not in existing_tags:
            existing_tags.append("legacy-backfilled")
        meta["tags"] = existing_tags
        canonical_type = str(meta.get("entity_type") or "concept").strip().lower()
        if canonical_type in {"stock_hub", "equity_analysis", "quant_snapshot", "earnings_call"} and not meta.get("ticker"):
            meta["legacy_entity_type"] = canonical_type
            meta["entity_type"] = "concept"
        if canonical_type in {"earnings_call"} and not meta.get("period"):
            meta["legacy_entity_type"] = canonical_type
            meta["entity_type"] = "concept"
        if canonical_type in {"youtube_summary"} and not meta.get("video_id"):
            meta["legacy_entity_type"] = canonical_type
            meta["entity_type"] = "concept"
        meta["schema_version"] = 2

        # R5/R6 imported some opaque note IDs under a hashed legacy key while
        # the file frontmatter later received a placeholder key.  The durable
        # allocation is authoritative; reconcile the file to it before the
        # shared writer performs its conflict check.
        if meta.get("note_id"):
            durable_identity = identity_store.get_note_identity_by_note_id(str(meta["note_id"]))
            if durable_identity:
                meta["document_key"] = durable_identity.document_key

        if not dry_run:
            committed = ArtifactWriter(vault_paths=VaultPaths(v_root)).write_note(
                metadata=meta,
                body=_portableize_wikilinks(raw_body, v_root, md_path),
                filename=md_path.stem,
                target_path=md_path,
            )
            if cat:
                try:
                    cat.upsert_note_from_file(committed.primary_file)
                except Exception:
                    pass

        backfilled += 1
        if limit and backfilled >= limit:
            break

    return {
        "scanned": scanned,
        "backfilled": backfilled,
        "already_has_meta": already_has_meta,
        "skipped_malformed": skipped_malformed,
    }
