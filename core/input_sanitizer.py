"""Hostile Input Sanitizer & Untrusted Instruction Defense (Institutional Hedge Fund Grade).

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies per Hexagonal
Architecture AST Domain Layer Isolation rules.

Reference:
Claude Code for Hedge Funds, Chapter 4 (pp. 31-34) & Appendix E (Lab 4):
"Treat documents and web pages as hostile to the instruction hierarchy. An earnings release
can contain text that resembles a request. A copied page can include hidden or visible
instructions. The research task is to extract source facts, not to let source content
redefine project behavior."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple


class HostileInstructionSeverity(str, Enum):
    INFO = "INFO"
    SUSPICIOUS = "SUSPICIOUS"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class SanitizationException:
    """Structured exception emitted when untrusted instruction patterns are detected."""
    exception_type: str                   # e.g. "UNTRUSTED_INSTRUCTION"
    pattern_name: str                     # e.g. "PROMPT_INJECTION", "IMPERATIVE_ACTION_OVERRIDE"
    matched_snippet: str
    severity: HostileInstructionSeverity
    remediation: str = "FLAGGED_FOR_HUMAN_REVIEW"
    owner: str = "security_review"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exception_type": self.exception_type,
            "pattern_name": self.pattern_name,
            "matched_snippet": self.matched_snippet,
            "severity": self.severity.value,
            "remediation": self.remediation,
            "owner": self.owner,
        }


@dataclass(frozen=True)
class SanitizationResult:
    """Result of defensive sanitization scan on raw external text."""
    is_safe: bool
    cleaned_text: str
    exceptions: Tuple[SanitizationException, ...] = ()
    original_char_count: int = 0
    sanitized_char_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "cleaned_text": self.cleaned_text,
            "exceptions": [e.to_dict() for e in self.exceptions],
            "original_char_count": self.original_char_count,
            "sanitized_char_count": self.sanitized_char_count,
        }


# Invisible Unicode characters used in evasion / obfuscation
_INVISIBLE_UNICODE_CHARS = [
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\ufeff",  # Zero-width no-break space / BOM
    "\u202e",  # Right-to-left override
    "\u2060",  # Word joiner
]

_INVISIBLE_UNICODE_REGEX = re.compile(
    "[" + "".join(re.escape(c) for c in _INVISIBLE_UNICODE_CHARS) + "]"
)

# Rule definitions: (Pattern Name, Regex, Severity)
_HOSTILE_RULES: List[Tuple[str, Pattern[str], HostileInstructionSeverity]] = [
    # 1. Direct prompt injection & system instruction hijacking
    (
        "PROMPT_INJECTION_DIRECT",
        re.compile(
            r"(?i)\b(?:ignore\s+(?:all\s+)?(?:project\s+rules|prior\s+instructions|previous\s+instructions|system\s+prompts?)|"
            r"forget\s+(?:all\s+)?rules|system\s+prompt\s+override|override\s+system\s+prompt|"
            r"you\s+are\s+now\s+in\s+developer\s+mode|disregard\s+(?:all\s+)?prior\s+rules)\b"
        ),
        HostileInstructionSeverity.CRITICAL,
    ),
    # 2. Imperative position, trade, or state file alteration
    (
        "IMPERATIVE_STATE_MUTATION",
        re.compile(
            r"(?i)\b(?:change\s+(?:the\s+)?position\s+file|update\s+portfolio\s+weight\s+to|"
            r"override\s+position\s+limits?|execute\s+(?:an?\s+)?unauthorized\s+trade|"
            r"liquidate\s+all\s+positions\s+immediately|sell\s+all\s+assets\s+now|"
            r"buy\s+100%\s+allocation|write\s+directly\s+to\s+state\s+database)\b"
        ),
        HostileInstructionSeverity.CRITICAL,
    ),
    # 3. Delimiter breakouts and instruction envelope spoofing
    (
        "DELIMITER_BREAKOUT",
        re.compile(
            r"(?:</?system>|</?context>|</?instruction>|</?human>|</?assistant>|"
            r"\[SYSTEM_PROMPT\]|\[INSTRUCTION\]|```(?:system|instruction|admin))",
            re.IGNORECASE,
        ),
        HostileInstructionSeverity.CRITICAL,
    ),
    # 4. Role impersonation attacks
    (
        "ROLE_IMPERSONATION",
        re.compile(
            r"(?i)\b(?:i\s+am\s+(?:the\s+)?portfolio\s+manager\s+and\s+i\s+command|"
            r"administrator\s+override\s+code\s*[:=]|as\s+the\s+fund\s+owner\s*,\s*execute)\b"
        ),
        HostileInstructionSeverity.SUSPICIOUS,
    ),
]


def strip_invisible_characters(text: str) -> Tuple[str, bool]:
    """Strips zero-width spaces, RTL overrides, and hidden Unicode formatting.
    
    Returns (cleaned_text, had_hidden_characters).
    """
    cleaned = _INVISIBLE_UNICODE_REGEX.sub("", text)
    return cleaned, cleaned != text


def sanitize_untrusted_input(
    raw_text: str,
    max_snippet_len: int = 120,
) -> SanitizationResult:
    """Scans and defensive-sanitizes raw external source text (filings, news, transcripts).

    Returns a SanitizationResult indicating whether the source text is safe, along
    with any detected UNTRUSTED_INSTRUCTION exceptions.
    """
    if not raw_text:
        return SanitizationResult(
            is_safe=True,
            cleaned_text="",
            exceptions=(),
            original_char_count=0,
            sanitized_char_count=0,
        )

    orig_len = len(raw_text)
    cleaned, had_invisible = strip_invisible_characters(raw_text)
    exceptions: List[SanitizationException] = []

    if had_invisible:
        exceptions.append(
            SanitizationException(
                exception_type="UNTRUSTED_INSTRUCTION",
                pattern_name="INVISIBLE_UNICODE_EVASION",
                matched_snippet="[Hidden zero-width or RTL formatting characters stripped]",
                severity=HostileInstructionSeverity.SUSPICIOUS,
                remediation="STRIPPED_INVISIBLE_FORMATTING",
            )
        )

    for pattern_name, pattern, severity in _HOSTILE_RULES:
        for match in pattern.finditer(cleaned):
            start = max(0, match.start() - 20)
            end = min(len(cleaned), match.end() + 20)
            snippet = cleaned[start:end].strip()
            if len(snippet) > max_snippet_len:
                snippet = snippet[:max_snippet_len] + "..."

            exceptions.append(
                SanitizationException(
                    exception_type="UNTRUSTED_INSTRUCTION",
                    pattern_name=pattern_name,
                    matched_snippet=snippet,
                    severity=severity,
                    remediation="FLAGGED_FOR_HUMAN_REVIEW",
                    owner="security_review",
                )
            )

    is_safe = not any(
        e.severity in (HostileInstructionSeverity.CRITICAL, HostileInstructionSeverity.SUSPICIOUS)
        for e in exceptions
    )

    return SanitizationResult(
        is_safe=is_safe,
        cleaned_text=cleaned,
        exceptions=tuple(exceptions),
        original_char_count=orig_len,
        sanitized_char_count=len(cleaned),
    )

