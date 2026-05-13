import re

_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Thai national ID: 13 digits in 1-4-5-2-1 format, with or without dashes
    (re.compile(r'\b\d-?\d{4}-?\d{5}-?\d{2}-?\d\b'), "[REDACTED_THAI_ID]"),
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


class PIIMiddleware:
    def anonymize(self, text: str) -> tuple[str, bool]:
        """Redact PII from text. Returns (cleaned_text, was_pii_found)."""
        result = text
        for pattern, placeholder in _PII_PATTERNS:
            result = pattern.sub(placeholder, result)
        return result, result != text
