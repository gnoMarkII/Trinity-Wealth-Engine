"""Context-aware financial metric and valuation parser.

Pure domain utility for deterministic extraction of valuation multiples (PE, PBV),
yields, spreads, and price targets from unstructured text and table cells.
Prevents fragile regexes from mistaking calendar years (e.g. FY2026) for valuation multiples.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any, Optional, Sequence, Union

# Match units such as x, เท่า, %, bps, THB, USD, บาท
_UNIT_ANCHORED_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*(?:(?:x|bps|thb|usd)(?![a-z0-9])|%|เท่า|บาท)",
    re.IGNORECASE,
)

# Match metric labels followed by numeric values
_LABEL_ANCHORED_RE = re.compile(
    r"(?:p/e|pe|p/bv|pbv|eps|dps|yield|spread|target|เป้าหมาย|ราคาเป้าหมาย|ราคา)\s*(?:is|:|:=|=)?\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Year tokens to ignore when other metrics exist
_YEAR_TOKEN_RE = re.compile(
    r"(?<!\d)(?:FY\s*20\d{2}|FY\s*\d{2}|(?:19|20)\d{2}|25\d{2})(?!\d)",
    re.IGNORECASE,
)

_GENERIC_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?|\.\d+")

_NIL_STRINGS = frozenset({"", "-", "--", "n/a", "na", "null", "none", "nan"})


def extract_valuation_metric(val: Any) -> Optional[float]:
    """Deterministically extracts a valuation multiple or financial metric from text or numbers.

    Rules:
    - Passes existing numeric values (int/float) directly.
    - Strips commas, whitespace, and indicator arrows (▲, ▼).
    - Ignores null-like placeholders ('-', 'N/A', 'null').
    - Anchors to valuation units (x, %, bps, บาท, THB, USD) and labels (PE, PBV, Target).
    - Filters out calendar years (e.g. 'FY2026 Target PE is 15.5x' -> 15.5, not 2026).
    - Falls back to raw number extraction if the only token is numeric.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if not s or s.lower() in _NIL_STRINGS:
        return None

    # Handle downward indicator arrow before sign or number (e.g., "▼1.25" -> "-1.25")
    is_down_indicated = "▼" in s
    clean = s.replace(",", "").replace("▲", "").replace("▼", "").strip()

    # 1. Check unit-anchored pattern (e.g., '15.5x', '2.50%', '32.47 THB')
    unit_match = _UNIT_ANCHORED_RE.search(clean)
    if unit_match:
        try:
            num = float(unit_match.group(1))
            return -abs(num) if is_down_indicated and num > 0 and "-" not in unit_match.group(1) else num
        except ValueError:
            pass

    # 2. Check label-anchored pattern (e.g., 'Target: 45.0', 'PE 15.5')
    label_match = _LABEL_ANCHORED_RE.search(clean)
    if label_match:
        try:
            num = float(label_match.group(1))
            return -abs(num) if is_down_indicated and num > 0 and "-" not in label_match.group(1) else num
        except ValueError:
            pass

    # 3. Strip calendar years (FY2026, 2024, 2567) to prevent year collision
    text_without_years = _YEAR_TOKEN_RE.sub(" ", clean).strip()
    if text_without_years:
        num_match = _GENERIC_NUMBER_RE.search(text_without_years)
        if num_match:
            try:
                num = float(num_match.group(0))
                return -abs(num) if is_down_indicated and num > 0 and "-" not in num_match.group(0) else num
            except ValueError:
                pass

    # 4. Fallback to searching original string if it was a standalone year or plain number
    fallback_match = _GENERIC_NUMBER_RE.search(clean)
    if fallback_match:
        try:
            num = float(fallback_match.group(0))
            return -abs(num) if is_down_indicated and num > 0 and "-" not in fallback_match.group(0) else num
        except ValueError:
            return None

    return None


def extract_financial_decimal(val: Any) -> Optional[Decimal]:
    """Deterministically extracts a Decimal value from text, numbers, or accounting format.

    Rules:
    - Supports accounting parentheses e.g. '(1,500.50)' -> Decimal('-1500.50')
    - Strips commas, percentage signs, and currency symbols
    - Handles indicators '▲', '▼'
    - Ensures result is finite
    """
    if val is None:
        return None
    if isinstance(val, Decimal):
        return val if val.is_finite() else None
    if isinstance(val, (int, float)):
        try:
            d = Decimal(str(val))
            return d if d.is_finite() else None
        except (InvalidOperation, ValueError):
            return None

    s = str(val).strip()
    if not s or s.lower() in _NIL_STRINGS:
        return None

    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1].strip()

    if "▼" in s:
        is_negative = True

    clean = s.replace(",", "").replace("%", "").replace("▲", "").replace("▼", "").replace("฿", "").replace("$", "").strip()
    clean = re.sub(r"(?i)(?:(?<![a-z0-9])(?:thb|usd)(?![a-z0-9])|บาท)", "", clean).strip()

    m = re.search(r"[-+]?\d+(?:\.\d+)?|\.\d+", clean)
    if not m:
        return None

    try:
        d = Decimal(m.group(0))
        if is_negative and d > 0 and "-" not in m.group(0):
            d = -d
        return d if d.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _attempt_json_loads(text: str) -> Optional[Any]:
    """Helper to parse JSON with multi-pass repairs for LLM output quirks."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    # Pass 1: Remove line comments // ...
    candidate = re.sub(r"//.*", "", text)

    # Pass 2: Repair trailing commas before closing braces/brackets
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)

    # Pass 3: Fix missing commas between key-value lines
    candidate = re.sub(r'("(?:[^"\\]|\\.)*")\s*\n\s*(")', r'\1,\n\2', candidate)

    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Pass 4: Fix single quotes replacing double quotes
    if "'" in candidate and '"' not in candidate:
        try:
            return json.loads(candidate.replace("'", '"'))
        except Exception:
            pass

    return None


def extract_json_block(text: Optional[str]) -> Optional[Any]:
    """Extracts parsed JSON object from text containing Markdown fences or raw JSON.

    Robust against:
    - Code fences with or without 'json' tag anywhere in text
    - Prefixes or suffixes (e.g. conversational LLM preambles)
    - Nested balanced curly braces
    - LLM JSON quirks (trailing commas, // comments, single quotes)
    """
    if not text:
        return None
    cleaned = str(text).strip()

    # 1. Look for ```json ... ``` code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        res = _attempt_json_loads(fence_match.group(1).strip())
        if res is not None:
            return res

    # 2. Balanced brace scanner
    start_idx = cleaned.find("{")
    if start_idx != -1:
        brace_count = 0
        in_string = False
        escape = False
        end_idx = -1

        for i in range(start_idx, len(cleaned)):
            c = cleaned[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    brace_count += 1
                elif c == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break

        if end_idx != -1:
            candidate = cleaned[start_idx : end_idx + 1]
            res = _attempt_json_loads(candidate)
            if res is not None:
                return res

    # 3. Outer brace fallback
    if "{" in cleaned and "}" in cleaned:
        s = cleaned.find("{")
        e = cleaned.rfind("}") + 1
        res = _attempt_json_loads(cleaned[s:e])
        if res is not None:
            return res

    return _attempt_json_loads(cleaned)


def extract_labeled_decimal(
    text: Optional[str],
    labels: Union[str, Sequence[str]],
    *,
    default: Optional[Decimal] = None,
    next_line: bool = False,
) -> Optional[Decimal]:
    """Extracts a financial decimal value appearing after specified label(s) in unstructured text.

    Supports:
    - Multiple alternative labels in Thai / English (e.g. ['Total Buy', 'รวมมูลค่าซื้อ', 'จำนวนเงินสุทธิ', 'Net Amount'])
    - Colons, spaces, whitespace after label
    - next_line=True to match values placed on subsequent lines after the label (e.g. BOT as of date\\n 35.25)
    - Uses extract_financial_decimal on the captured numeric token
    """
    if not text:
        return default
    if isinstance(labels, str):
        label_list = [labels]
    else:
        label_list = list(labels)

    escaped = [re.escape(lbl.strip()) for lbl in label_list if lbl.strip()]
    if not escaped:
        return default

    label_pat = "|".join(escaped)
    delim = r"[^\n]*\n\s*" if next_line else r"[^:\d\n]*[:：=]?\s*"
    rgx = re.compile(
        rf"(?:{label_pat}){delim}([+-]?\s*[\d,]+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    m = rgx.search(text)
    if m:
        dec = extract_financial_decimal(m.group(1))
        if dec is not None:
            return dec
    return default


