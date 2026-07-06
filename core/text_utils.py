import re

_MOJIBAKE_MARKERS = ("à¸", "à¹", "ðŸ", "â€", "â€”", "âš", "Ã", "Â")


def _repair_mojibake_chunk(text: str) -> str:
    repaired = text
    for _ in range(3):
        if not any(marker in repaired for marker in _MOJIBAKE_MARKERS):
            break
        changed = False
        for encoding in ("cp1252", "latin1"):
            try:
                candidate = repaired.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue
            if candidate != repaired:
                repaired = candidate
                changed = True
                break
        if not changed:
            break
    return repaired


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 decoded-as-Windows mojibake without touching real Thai text."""
    if not any(marker in text for marker in _MOJIBAKE_MARKERS):
        return text
    # If model output already contains valid Thai, split on Thai spans and only
    # repair the non-Thai spans that can still contain formatter/prompt mojibake.
    parts = re.split(r"([\u0e00-\u0e7f]+)", text)
    return "".join(
        part if re.fullmatch(r"[\u0e00-\u0e7f]+", part) else _repair_mojibake_chunk(part)
        for part in parts
    )
