"""Vault Policy and Searchability Filtering for Obsidian Vault V2.

Defines rules to distinguish inventory scanning from research search,
and enforces security/filename constraints (path traversal prevention, Windows naming rules).
"""
from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Optional, Union

_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}

_INVALID_FILE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Folders completely excluded from normal research/semantic search
_SEARCH_EXCLUDED_PARTS = {
    ".obsidian",
    ".trash",
    ".sync_history",
    ".chroma_index",
    ".system",
    ".evidence_cache",
    "99_Templates",
    "90_Attachments",
    "00_Index",
    "01_Daily_Logs",
    "00_Inbox",
    "Revisions",  # Frozen historical revisions in 40_Archive/Revisions/ must NOT pollute active search
}

_SYSTEM_FILES = {
    "index.md",
    "Portfolio_Holdings.md",
    "Portfolio_Dashboard.md",
    "Watchlist.md",
    "Trading_Journal.md",
}


def sanitize_filename(name: str) -> str:
    """Sanitizes filename for cross-platform and Windows filesystem safety."""
    cleaned = _INVALID_FILE_CHARS.sub("", name).strip()
    # Strip trailing dots or spaces which are invalid on Windows
    cleaned = cleaned.rstrip(". ")
    if not cleaned:
        cleaned = "untitled"
    # Check reserved Windows device names
    base_stem = cleaned.split(".")[0].upper()
    if base_stem in _WINDOWS_RESERVED_NAMES:
        cleaned = f"_{cleaned}"
    return cleaned


def is_searchable(relative_path: Union[str, Path]) -> bool:
    """Determines whether a note is part of the active, searchable knowledge base.

    Excludes system, cache, backup, template, daily logs, and frozen revisions.
    """
    p = Path(relative_path)
    parts = p.parts

    # Exclude system files
    if p.name in _SYSTEM_FILES:
        return False

    # Exclude dot files
    if p.name.startswith("."):
        return False

    # Exclude forbidden directories
    for part in parts:
        if part in _SEARCH_EXCLUDED_PARTS:
            return False
        if part.startswith("."):
            return False
        if part.startswith(".pre_migration_backup"):
            return False

    # Only markdown files are searchable text
    if p.suffix.lower() != ".md":
        return False

    return True


def is_searchable_note(file_path: Union[str, Path], vault_root: Optional[Union[str, Path]] = None) -> bool:
    """Helper that determines if a file path within a vault is searchable."""
    p = Path(file_path)
    if vault_root:
        try:
            rel = p.resolve().relative_to(Path(vault_root).resolve())
            return is_searchable(rel)
        except ValueError:
            pass
    return is_searchable(p)


def is_retired_note(
    vault_root: Union[str, Path],
    *,
    note_id: Optional[str] = None,
    document_key: Optional[str] = None,
) -> bool:
    """Check durable retirement tombstones before a writer materializes a note."""
    if not note_id and not document_key:
        return False
    tombstones = Path(vault_root).resolve() / ".system" / "retired_notes.jsonl"
    if not tombstones.is_file():
        return False
    try:
        latest_status: Optional[str] = None
        for line in tombstones.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            matches = (
                bool(note_id and str(record.get("note_id") or "") == str(note_id))
                or bool(document_key and str(record.get("document_key") or "") == str(document_key))
            )
            if matches:
                # Event-sourced tombstones remain backward compatible: legacy
                # rows without status mean retired, while a later restore
                # event re-opens the identity for an explicit application flow.
                latest_status = str(record.get("status") or "retired").lower()
        if latest_status is not None:
            return latest_status != "restored"
    except (OSError, UnicodeDecodeError, ValueError):
        # A malformed tombstone file must fail closed for writes. Reads and
        # diagnostics can still report the corruption explicitly.
        return True
    return False
