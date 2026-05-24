import re

_THAI_ID_PATTERN = re.compile(r'\b(\d)-?(\d{4})-?(\d{5})-?(\d{2})-?(\d)\b')

_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Credit card with separators: 4-4-4-4 groups (Visa/MC/Discover/etc.)
    (re.compile(r'\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b'), "[REDACTED_CREDIT_CARD]"),
    # Credit card without separators: major BIN prefixes only (Visa/MC/Amex)
    (re.compile(r'\b(?:4\d{15}|5[1-5]\d{14}|2[2-7]\d{14}|3[47]\d{13})\b'), "[REDACTED_CREDIT_CARD]"),
    # Email
    (re.compile(r'\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b', re.IGNORECASE), "[REDACTED_EMAIL]"),
    # Thai mobile: 0[6-9]x format (10 digits), with optional separators
    (re.compile(r'\b0[6-9]\d[-.\s]?\d{3}[-.\s]?\d{4}\b'), "[REDACTED_PHONE]"),
    # International format: +66 or 0066 prefix
    (re.compile(r'(?:\+66|0066)[-.\s]?\d{1,2}[-.\s]?\d{3,4}[-.\s]?\d{4}\b'), "[REDACTED_PHONE]"),
]


def _is_valid_thai_id(digits: str) -> bool:
    """Validate Thai national ID via official mod-11 checksum."""
    if len(digits) != 13 or not digits.isdigit():
        return False
    total = sum(int(d) * (13 - i) for i, d in enumerate(digits[:12]))
    check = (11 - (total % 11)) % 10
    return check == int(digits[12])


def _redact_thai_id(match: re.Match) -> str:
    digits = "".join(match.groups())
    return "[REDACTED_THAI_ID]" if _is_valid_thai_id(digits) else match.group(0)


def anonymize_pii(text: str) -> tuple[str, bool]:
    """Redact PII from text. Returns (cleaned_text, was_pii_found)."""
    result = _THAI_ID_PATTERN.sub(_redact_thai_id, text)
    for pattern, placeholder in _PII_PATTERNS:
        result = pattern.sub(placeholder, result)
    return result, result != text
