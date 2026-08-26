"""Pure NotebookLM source filename rules."""
from __future__ import annotations

import re

_SUFFIX_PATTERN = re.compile(r"_rev\d+_[0-9a-f]{8}_(verified|unverified)$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_source_filename(stem: str) -> tuple[str, str | None, bool]:
    """Parse historical and revisioned NotebookLM source filename formats."""
    date_part, sep, rest = stem.partition("_")
    if sep and _DATE_PATTERN.match(date_part):
        body = rest
        parsed_date = date_part
    else:
        body = stem
        parsed_date = None
    match = _SUFFIX_PATTERN.search(body)
    if match:
        return body[: match.start()], parsed_date, match.group(1) == "verified"
    return body, parsed_date, True


__all__ = ["parse_source_filename"]
