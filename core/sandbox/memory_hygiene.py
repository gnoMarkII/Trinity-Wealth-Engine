"""Memory Sanitization and Prompt Injection Guard for manager_sandbox/memory/."""
from __future__ import annotations

import re
from dataclasses import dataclass
from core.security import anonymize_pii

_INJECTION_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(system\s+)?rules", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(safety|security|guardrails)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+debug\s+mode|unrestricted|jailbroken)", re.IGNORECASE),
    re.compile(r"override\s+(the\s+)?system\s+prompt", re.IGNORECASE),
)

_HARMFUL_TAGS_PATTERN = re.compile(
    r"<\s*(script|iframe|object|embed|style)[^>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_JAVASCRIPT_URI_PATTERN = re.compile(r"javascript\s*:", re.IGNORECASE)


@dataclass(frozen=True)
class MemorySanitizationResult:
    is_safe: bool
    sanitized_content: str
    violations: tuple[str, ...]


def sanitize_memory_markdown(raw_content: str) -> MemorySanitizationResult:
    """Sanitize and validate markdown content before storing in manager_sandbox/memory/."""
    violations: list[str] = []

    # 1. Prompt Injection Checks
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(raw_content):
            violations.append(f"Prompt injection pattern detected: '{pattern.pattern}'")

    # 2. Harmful HTML / JS Tag Stripping
    sanitized = _HARMFUL_TAGS_PATTERN.sub("", raw_content)
    if sanitized != raw_content:
        violations.append("Removed potentially harmful embedded HTML tags (<script>, <iframe>, etc.)")

    if _JAVASCRIPT_URI_PATTERN.search(sanitized):
        sanitized = _JAVASCRIPT_URI_PATTERN.sub("blocked-uri:", sanitized)
        violations.append("Blocked 'javascript:' URI protocol")

    # 3. PII Anonymization
    sanitized, had_pii = anonymize_pii(sanitized)
    if had_pii:
        violations.append("PII detected and redacted")

    return MemorySanitizationResult(
        is_safe=len([v for v in violations if "Prompt injection" in v]) == 0,
        sanitized_content=sanitized.strip(),
        violations=tuple(violations),
    )

