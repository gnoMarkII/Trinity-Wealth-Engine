from html.parser import HTMLParser
import html
import re
from typing import Optional, Sequence, Union

_MOJIBAKE_MARKERS = ("à¸", "à¹", "ðŸ", "â€", "â€”", "âš", "Ã", "Â")


class HTMLTextExtractor(HTMLParser):
    """Safe HTML to plain text extractor using Python standard library.

    Correctly distinguishes between actual HTML tags (<tag>...</tag>)
    and mathematical/financial comparison operators (e.g. P/E < 15, Yield > 4%).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._pieces: list[str] = []
        self._ignore_tags = {"script", "style", "noscript", "svg"}
        self._current_ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag.lower() in self._ignore_tags:
            self._current_ignored_depth += 1
        elif tag.lower() in ("p", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._pieces.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._ignore_tags:
            self._current_ignored_depth = max(0, self._current_ignored_depth - 1)
        elif tag.lower() in ("p", "div", "li", "tr"):
            self._pieces.append(" ")

    def handle_data(self, data: str) -> None:
        if self._current_ignored_depth == 0 and data:
            self._pieces.append(data)

    def get_text(self) -> str:
        raw_text = "".join(self._pieces)
        return re.sub(r"[ \t\r\f\v]+", " ", raw_text).strip()


def strip_html_tags(raw_html: str) -> str:
    """Safely extract plain text from HTML, preserving financial comparison signs like < and >."""
    if not raw_html:
        return ""
    if "<" not in raw_html and ">" not in raw_html:
        return html.unescape(raw_html).strip()
    try:
        extractor = HTMLTextExtractor()
        extractor.feed(raw_html)
        extractor.close()
        text = extractor.get_text()
        return text
    except Exception:
        # Fallback only matching explicit valid HTML tags (starts with a letter)
        # Prevents mathematical/financial comparison signs (< 15, > 4%) from being wiped
        fallback = re.sub(r"</?[a-zA-Z][a-zA-Z0-9:-]*\b[^>]*>", " ", raw_html)
        return html.unescape(re.sub(r"\s+", " ", fallback)).strip()


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


def extract_labeled_value(
    text: Optional[str],
    labels: Union[str, Sequence[str]],
    value_pattern: str = r"([A-Za-z0-9\-_/.]+)",
    *,
    flags: int = re.IGNORECASE,
) -> Optional[str]:
    """Extracts a string token appearing after specified label(s) in unstructured text.

    Supports:
    - Multiple alternative labels in Thai / English (e.g. ['Tax Invoice No.', 'เลขที่ใบกำกับภาษี'])
    - Colons, spaces, whitespace after label
    - Clean strip of whitespace
    """
    if not text:
        return None
    if isinstance(labels, str):
        label_list = [labels]
    else:
        label_list = list(labels)

    escaped = [lbl.strip() if lbl.strip().startswith("(?") else re.escape(lbl.strip()) for lbl in label_list if lbl.strip()]
    if not escaped:
        return None

    label_pat = "|".join(escaped)
    rgx = re.compile(
        rf"(?:{label_pat})(?:(?:\s*\([^)\n]+\))?\s*[:：=]|\s+)\s*{value_pattern}",
        flags,
    )
    m = rgx.search(text)
    if m:
        return m.group(1).strip()
    return None
