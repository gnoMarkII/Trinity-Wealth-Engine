from datetime import date, datetime, timezone
import re
from typing import Optional, Sequence, Tuple, Union

THAI_MONTH_MAP = {
    "ม.ค.": 1, "มค": 1, "มกราคม": 1, "jan": 1, "january": 1,
    "ก.พ.": 2, "กพ": 2, "กุมภาพันธ์": 2, "feb": 2, "february": 2,
    "มี.ค.": 3, "มีค": 3, "มีนาคม": 3, "mar": 3, "march": 3,
    "เม.ย.": 4, "เมย": 4, "เมษายน": 4, "apr": 4, "april": 4,
    "พ.ค.": 5, "พค": 5, "พฤษภาคม": 5, "may": 5,
    "มิ.ย.": 6, "มิย": 6, "มิถุนายน": 6, "jun": 6, "june": 6,
    "ก.ค.": 7, "กค": 7, "กรกฎาคม": 7, "jul": 7, "july": 7,
    "ส.ค.": 8, "สค": 8, "สิงหาคม": 8, "aug": 8, "august": 8,
    "ก.ย.": 9, "กย": 9, "กันยายน": 9, "sep": 9, "september": 9,
    "ต.ค.": 10, "ตค": 10, "ตุลาคม": 10, "oct": 10, "october": 10,
    "พ.ย.": 11, "พย": 11, "พฤศจิกายน": 11, "nov": 11, "november": 11,
    "ธ.ค.": 12, "ธค": 12, "ธันวาคม": 12, "dec": 12, "december": 12,
}


def _normalize_year(year: int) -> int:
    """Normalize Buddhist Era (พ.ศ.) or 2-digit years to Gregorian (ค.ศ.)."""
    if year >= 2400:
        return year - 543
    if 40 <= year <= 99:
        # 2-digit Thai Buddhist year (e.g. 67 -> 2567 -> 2024)
        return (2500 + year) - 543
    if 0 <= year <= 39:
        # 2-digit Christian year (e.g. 24 -> 2024)
        return 2000 + year
    return year


def parse_thai_or_iso_date(raw: Union[str, int, float, date, datetime, None]) -> Optional[date]:
    """Parse any Thai or international date format into a standard Gregorian datetime.date.

    Handles:
    - ISO strings: '2026-10-25', '2026-10-25T08:00:00Z', '2026/10/25'
    - Slash/dash formats: '25/10/2569' (BE -> 2026), '25/10/2026', '25-10-2026'
    - Thai textual formats: '25 ต.ค. 2567', '25 ตุลาคม 2567', '25 ต.ค. 67'
    - English textual formats: '25 Oct 2026', 'October 25, 2026'
    - Unix timestamps: 1729843200
    - Pure date/datetime objects
    """
    if raw is None:
        return None

    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw

    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(raw, tz=timezone.utc).date()
        except Exception:
            return None

    clean = str(raw).strip()
    if not clean or clean.lower() in ("none", "null", "n/a", "-"):
        return None

    # 1. Standard ISO format: YYYY-MM-DD
    m_iso = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", clean)
    if m_iso:
        y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        y = _normalize_year(y)
        try:
            return date(y, m, d)
        except ValueError:
            pass

    # 2. Day-first numeric format: DD/MM/YYYY or DD-MM-YYYY
    m_dmy = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})", clean)
    if m_dmy:
        d, m, y = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
        y = _normalize_year(y)
        try:
            return date(y, m, d)
        except ValueError:
            pass

    # 3. Compact numeric format: YYYYMMDD
    m_compact = re.match(r"^(\d{4})(\d{2})(\d{2})$", clean)
    if m_compact:
        y, m, d = int(m_compact.group(1)), int(m_compact.group(2)), int(m_compact.group(3))
        y = _normalize_year(y)
        try:
            return date(y, m, d)
        except ValueError:
            pass

    # 4. Textual format with month name/abbr:
    # e.g. '25 ต.ค. 2567', '25 ต.ค. 67', '25 ตุลาคม 2567', '25 Oct 2026', 'October 25, 2026'
    tokens = re.split(r"[\s,]+", clean)
    if len(tokens) >= 3:
        # Case A: Day Month Year (e.g. 25 ต.ค. 2567)
        if tokens[0].isdigit() and tokens[2].isdigit():
            d = int(tokens[0])
            m_key = tokens[1].lower()
            m = THAI_MONTH_MAP.get(m_key)
            y = _normalize_year(int(tokens[2]))
            if m:
                try:
                    return date(y, m, d)
                except ValueError:
                    pass

        # Case B: Month Day Year (e.g. October 25 2026)
        if tokens[1].isdigit() and tokens[2].isdigit():
            m_key = tokens[0].lower()
            m = THAI_MONTH_MAP.get(m_key)
            d = int(tokens[1])
            y = _normalize_year(int(tokens[2]))
            if m:
                try:
                    return date(y, m, d)
                except ValueError:
                    pass

    return None


def format_iso_date(d: Optional[Union[date, datetime]]) -> str:
    """Format date to YYYY-MM-DD string, returning empty string if None."""
    if d is None:
        return ""
    if isinstance(d, datetime):
        return d.date().isoformat()
    return d.isoformat()


def parse_fiscal_quarter(text: Optional[str]) -> Optional[str]:
    """Extract standard fiscal quarter string 'FYYYYYQX' from diverse expressions.

    Supports:
    - 'Q1 2024', 'Q1-2024', 'Q1/2024', 'Q1_2024', 'Q12024' -> 'FY2024Q1'
    - '2024 Q1', '2024-Q1', '2024/Q1', '2024Q1' -> 'FY2024Q1'
    - 'FY2024Q1', 'FY24Q1' -> 'FY2024Q1'
    - 'Q1 2567' (BE) -> 'FY2024Q1'
    """
    if not text:
        return None
    clean = str(text).strip()

    # 1. FYYYYYQX or FYYYQX
    m_fy = re.search(r"FY(\d{2,4})[\s\-_/]*Q([1-4])", clean, re.IGNORECASE)
    if m_fy:
        y = _normalize_year(int(m_fy.group(1)))
        q = int(m_fy.group(2))
        return f"FY{y}Q{q}"

    # 2. QX YYYY or QX-YYYY or QX/YYYY
    m_qy = re.search(r"Q([1-4])[\s\-_/]*(\d{4})", clean, re.IGNORECASE)
    if m_qy:
        q = int(m_qy.group(1))
        y = _normalize_year(int(m_qy.group(2)))
        return f"FY{y}Q{q}"

    # 3. YYYY QX or YYYY-QX or YYYY/QX
    m_yq = re.search(r"(\d{4})[\s\-_/]*Q([1-4])", clean, re.IGNORECASE)
    if m_yq:
        y = _normalize_year(int(m_yq.group(1)))
        q = int(m_yq.group(2))
        return f"FY{y}Q{q}"

    return None


def parse_quarter_tuple(text: Optional[str]) -> Optional[Tuple[int, int]]:
    """Extract (year, quarter) tuple from text, e.g. (2024, 1)."""
    fq = parse_fiscal_quarter(text)
    if not fq:
        return None
    return int(fq[2:6]), int(fq[7])


_GENERIC_DATE_RE = re.compile(
    r"(?<!\d)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})(?!\d)"
)


def extract_date_by_labels(
    text: Optional[str],
    labels: Union[str, Sequence[str]],
    *,
    pattern: Optional[str] = None,
) -> Optional[date]:
    """Finds and parses a date appearing after specified label(s) in unstructured text.

    Supports:
    - Multiple alternative labels in Thai / English (e.g. ['Effective Date', 'วันที่คำสั่งมีผล'])
    - Colons, spaces, whitespace after label
    - Uses parse_thai_or_iso_date on the captured token
    """
    if not text:
        return None
    if isinstance(labels, str):
        label_list = [labels]
    else:
        label_list = list(labels)

    escaped = [re.escape(lbl.strip()) for lbl in label_list if lbl.strip()]
    if not escaped:
        return None

    label_pat = "|".join(escaped)
    val_pat = pattern or _GENERIC_DATE_RE.pattern

    rgx = re.compile(
        rf"(?:{label_pat})[^:\d\n]*[:：]?\s*({val_pat})",
        re.IGNORECASE,
    )
    m = rgx.search(text)
    if m:
        raw_val = m.group(1).strip()
        return parse_thai_or_iso_date(raw_val)
    return None

