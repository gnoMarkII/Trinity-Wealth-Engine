"""Markdown-specific rendering helpers for portfolio journal entries.

Keeping these helpers in the Markdown adapter layer means domain events stay
plain data while both direct journal writes and staged Unit-of-Work commits
render the same Obsidian wikilinks.
"""
import os
import re
from pathlib import Path

from tools.portfolio.domain.constants import _CASH_SYMBOLS
from tools.archivist.metadata import dump_note, parse_note
from tools.portfolio.adapters.markdown.identity import portfolio_note_identity
from tools.archivist.portable_links import default_vault_root, render_symbol_markdown_link


_TRADE_TITLE_RE = re.compile(
    r"(\*\*\[[\w\s]+\]\*\*\s+)([A-Z][\w.\-]*)([^\]]*\]\*\*)(?!\s*—\s*\[\[)"
)
_SYMBOL_TITLE_RE = re.compile(
    r"(?m)^(?P<prefix>\*\*\[(?:TRADE\s+NOTE\s*-\s*|EDIT\s+|REMOVE\s+))"
    r"(?P<symbol>[A-Z][\w.\-]*)(?P<suffix>\]\*\*)(?P<body>[^\n]*)$"
)


def inject_journal_links(
    content: str,
    *,
    vault_root: str | Path | None = None,
    source_path: str | Path | None = None,
) -> str:
    """Add a portable Markdown link for a traded holding symbol when possible."""
    root = Path(vault_root).resolve() if vault_root else default_vault_root()
    source = Path(source_path) if source_path else root / "index.md"

    def _render(symbol: str) -> str:
        return render_symbol_markdown_link(root, source, symbol, label=symbol)

    def _already_referenced(match: re.Match, tail: str = "") -> bool:
        return bool(re.search(r"—\s*(?:\[\[|\[[^\]]+\]\()", tail))

    def _replace(match: re.Match) -> str:
        symbol = match.group(2)
        if symbol in _CASH_SYMBOLS:
            return match.group(0)
        tail = content[match.end():match.end() + 256]
        if _already_referenced(match, tail):
            return match.group(0)
        return f"{match.group(1)}{symbol}{match.group(3)} — {_render(symbol)}"

    rendered = _TRADE_TITLE_RE.sub(_replace, content)

    # System-generated portfolio events use a symbol inside their title
    # (``**[TRADE NOTE - AAPL]**`` / ``**[EDIT AAPL]**``), whereas the
    # historical journal helper handled ``**[BUY]** AAPL ...``.  Support
    # both forms so staged UoW events and direct journal writes render the
    # same wikilink without changing already-linked entries.
    def _append_symbol_link(match: re.Match) -> str:
        symbol = match.group("symbol")
        body = match.group("body")
        if symbol in _CASH_SYMBOLS or re.search(r"—\s*(?:\[\[|\[[^\]]+\]\()", match.group(0)):
            return match.group(0)
        return (
            f"{match.group('prefix')}{symbol}{match.group('suffix')}"
            f"{body} — {_render(symbol)}"
        )

    return _SYMBOL_TITLE_RE.sub(_append_symbol_link, rendered)


def inject_journal_wikilinks(
    content: str,
    *,
    vault_root: str | Path | None = None,
    source_path: str | Path | None = None,
) -> str:
    """Backward-compatible entry point using the portable link renderer."""
    return inject_journal_links(
        content,
        vault_root=vault_root,
        source_path=source_path,
    )


def ensure_journal_frontmatter(
    existing: str,
    *,
    portfolio_id: str,
    vault_root: str | Path | None = None,
) -> tuple[dict, str]:
    """Return canonical journal metadata and body for a read-modify-write."""
    root = Path(vault_root).resolve() if vault_root else default_vault_root()
    text = existing or ""
    if text.startswith("---"):
        metadata, body, issues = parse_note(text)
        if issues:
            raise ValueError(f"Cannot append to malformed journal frontmatter: {issues}")
    else:
        metadata, body = {}, text.strip()

    metadata = dict(metadata)
    if not metadata.get("note_id") or not metadata.get("document_key"):
        note_id, document_key = portfolio_note_identity(root, portfolio_id, "journal")
        metadata["note_id"] = metadata.get("note_id") or note_id
        metadata["document_key"] = metadata.get("document_key") or document_key

    metadata.update(
        {
            "schema_version": 2,
            "title": metadata.get("title") or f"Trading Journal {portfolio_id}",
            "entity_type": "portfolio_state",
            "document_role": "journal",
            "portfolio_id": portfolio_id,
            "search_scope": "excluded",
        }
    )
    return metadata, body


def serialize_journal(
    existing: str,
    appended_body: str,
    *,
    portfolio_id: str,
    vault_root: str | Path | None = None,
) -> str:
    """Serialize a journal after appending one or more rendered blocks."""
    metadata, body = ensure_journal_frontmatter(
        existing,
        portfolio_id=portfolio_id,
        vault_root=vault_root,
    )
    joined = "\n\n".join(part.strip() for part in (body, appended_body) if part.strip())
    return dump_note(metadata, joined)


__all__ = [
    "ensure_journal_frontmatter",
    "inject_journal_links",
    "inject_journal_wikilinks",
    "serialize_journal",
]
