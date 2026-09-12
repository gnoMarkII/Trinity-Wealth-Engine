"""Fail-closed helpers for generated Markdown blocks with human annotations."""
from __future__ import annotations

import hashlib
import re


class ManagedBlockError(ValueError):
    """Managed block markers are missing, nested, or ambiguous."""


def _markers(block_id: str) -> tuple[str, str]:
    clean = str(block_id or "").strip()
    if not clean or not re.fullmatch(r"[A-Za-z0-9_.:-]+", clean):
        raise ManagedBlockError(f"invalid managed block id: {block_id!r}")
    return f"<!-- managed:start {clean} -->", f"<!-- managed:end {clean} -->"


def extract_managed_block(text: str, block_id: str) -> tuple[str, str, str]:
    """Return text before, managed content, and text after the block."""
    start, end = _markers(block_id)
    start_positions = [m.start() for m in re.finditer(re.escape(start), text)]
    end_positions = [m.start() for m in re.finditer(re.escape(end), text)]
    if len(start_positions) != 1 or len(end_positions) != 1:
        raise ManagedBlockError(f"managed block {block_id!r} must have exactly one start and end marker")
    start_pos = start_positions[0]
    end_pos = end_positions[0]
    if end_pos < start_pos:
        raise ManagedBlockError(f"managed block {block_id!r} end precedes start")
    content_start = start_pos + len(start)
    content = text[content_start:end_pos]
    if re.search(r"<!-- managed:(?:start|end) [^>]+ -->", content):
        raise ManagedBlockError(f"nested managed blocks are not supported: {block_id!r}")
    return text[:start_pos], content, text[end_pos + len(end):]


def replace_managed_block(text: str, block_id: str, content: str) -> str:
    before, _old, after = extract_managed_block(text, block_id)
    start, end = _markers(block_id)
    clean_content = str(content).strip("\n")
    return f"{before}{start}\n{clean_content}\n{end}{after}"


def annotation_hash(text: str, block_id: str) -> str:
    before, _managed, after = extract_managed_block(text, block_id)
    return hashlib.sha256((before + after).encode("utf-8")).hexdigest()
