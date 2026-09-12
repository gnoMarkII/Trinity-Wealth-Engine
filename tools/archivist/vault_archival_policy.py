"""Data Lifecycle Management (DLM) and Archival Policy Engine for Obsidian Vault V2.

Moves aged news and raw daily snapshots (> 90 days) from 30_Knowledge_Base/News
into 40_Archive/News while maintaining SQLite catalog tracking and preventing vault bloat.
"""
from __future__ import annotations

import os
import re
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Union

from core.logger import get_logger

logger = get_logger(__name__)

_DATE_REGEX = re.compile(r"(\d{4}-\d{2}-\d{2})")


def apply_archival_policy(
    vault_root: Optional[Union[str, Path]] = None,
    max_age_days: int = 90,
    dry_run: bool = True,
) -> dict[str, int]:
    """Archives news items older than `max_age_days` to 40_Archive/News."""
    v_root = Path(vault_root).resolve() if vault_root else Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
    if not v_root.exists():
        return {"scanned": 0, "archived": 0, "retained": 0}

    news_dir = v_root / "30_Knowledge_Base" / "News"
    archive_dir = v_root / "40_Archive" / "News"
    if not dry_run:
        from tools.archivist.maintenance_guard import assert_write_allowed
        assert_write_allowed(archive_dir)
    if not news_dir.exists():
        return {"scanned": 0, "archived": 0, "retained": 0}

    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).date()

    cat = None
    try:
        from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
        from tools.archivist.catalog_runtime import resolve_catalog_path
        cat_db = resolve_catalog_path(v_root, require_exists=True)
        if cat_db.exists():
            cat = SqliteNoteCatalogAdapter(db_path=cat_db, vault_root=v_root)
    except Exception:
        pass

    scanned = 0
    archived = 0
    retained = 0

    for item in news_dir.rglob("*.md"):
        if "Inbox" in item.parts:
            continue
        scanned += 1

        # Determine note date from filename or stat
        date_match = _DATE_REGEX.search(item.name)
        if date_match:
            try:
                item_date = datetime.strptime(date_match.group(1), "%Y-%m-%d").date()
            except ValueError:
                item_date = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).date()
        else:
            item_date = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).date()

        if item_date < cutoff_date:
            year_str = str(item_date.year)
            target_folder = archive_dir / year_str
            target_path = target_folder / item.name

            if not dry_run:
                target_folder.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(target_path))
                if cat:
                    try:
                        # Re-index moved file at new path
                        old_rel = item.relative_to(v_root).as_posix()
                        old_entry = cat.get_by_path(old_rel)
                        if old_entry:
                            cat.delete_note(old_entry.note_id)
                        cat.upsert_note_from_file(target_path)
                    except Exception as e:
                        logger.warning("Failed updating catalog for archived %s: %s", item.name, e)

            archived += 1
        else:
            retained += 1

    return {
        "scanned": scanned,
        "archived": archived,
        "retained": retained,
    }
