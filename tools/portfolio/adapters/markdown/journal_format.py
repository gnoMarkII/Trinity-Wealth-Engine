"""Markdown-specific rendering helpers for portfolio journal entries.

Keeping these helpers in the Markdown adapter layer means domain events stay
plain data while both direct journal writes and staged Unit-of-Work commits
render the same Obsidian wikilinks.
"""
import re

from tools.portfolio.domain.constants import _CASH_SYMBOLS


_TRADE_TITLE_RE = re.compile(
    r"(\*\*\[[\w\s]+\]\*\*\s+)([A-Z][\w.\-]*)([^\]]*\]\*\*)(?!\s*—\s*\[\[)"
)
_SYMBOL_TITLE_RE = re.compile(
    r"(?m)^(?P<prefix>\*\*\[(?:TRADE\s+NOTE\s*-\s*|EDIT\s+|REMOVE\s+))"
    r"(?P<symbol>[A-Z][\w.\-]*)(?P<suffix>\]\*\*)(?P<body>[^\n]*)$"
)


def inject_journal_wikilinks(content: str) -> str:
    """Add an Obsidian wikilink for a traded holding symbol when appropriate."""

    def _replace(match: re.Match) -> str:
        symbol = match.group(2)
        if symbol in _CASH_SYMBOLS:
            return match.group(0)
        return f"{match.group(1)}{symbol}{match.group(3)} — [[{symbol}]]"

    rendered = _TRADE_TITLE_RE.sub(_replace, content)

    # System-generated portfolio events use a symbol inside their title
    # (``**[TRADE NOTE - AAPL]**`` / ``**[EDIT AAPL]**``), whereas the
    # historical journal helper handled ``**[BUY]** AAPL ...``.  Support
    # both forms so staged UoW events and direct journal writes render the
    # same wikilink without changing already-linked entries.
    def _append_symbol_link(match: re.Match) -> str:
        symbol = match.group("symbol")
        body = match.group("body")
        if symbol in _CASH_SYMBOLS or f"[[{symbol}]]" in match.group(0):
            return match.group(0)
        return (
            f"{match.group('prefix')}{symbol}{match.group('suffix')}"
            f"{body} — [[{symbol}]]"
        )

    return _SYMBOL_TITLE_RE.sub(_append_symbol_link, rendered)


__all__ = ["inject_journal_wikilinks"]
