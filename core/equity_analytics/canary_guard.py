"""Untrusted Data Canary & Prompt Injection Guard.

Architecture Rule (Layer 1 - Core):
Pure domain security primitives. Strictly standard library only.
Implements Chapter 4 & 7 of 'Claude Code for Hedge Funds':
- Classifies external sources (filings, earnings transcripts, news, user uploads) as UNTRUSTED DATA.
- Detects prompt injection markers, delimiter escapes, and system override attempts.
- Enforces canary token integrity: rejects contaminated rows before sending to LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple


class CanaryViolationSeverity(str, Enum):
    INFO = "INFO"
    SUSPICIOUS = "SUSPICIOUS"
    CRITICAL = "CRITICAL"


class UntrustedInputError(ValueError):
    """Raised when an untrusted input contains critical prompt injection or malicious escape sequences."""

    def __init__(self, pattern_matched: str, snippet: str):
        self.pattern_matched = pattern_matched
        self.snippet = snippet
        super().__init__(
            f"Prompt injection pattern detected: '{pattern_matched}' in snippet: '{snippet[:60]}...'. Processing aborted."
        )


@dataclass(frozen=True)
class CanaryCheckResult:
    """Evaluation result of scanning untrusted text."""
    is_safe: bool
    violations: List[Dict[str, str]] = field(default_factory=list)
    sanitized_text: str = ""
    canary_token_intact: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "violations": self.violations,
            "canary_token_intact": self.canary_token_intact,
            "violations_count": len(self.violations),
        }


# High-risk adversarial prompt injection patterns
_CRITICAL_INJECTION_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "SYSTEM_OVERRIDE_DIRECTIVE",
        re.compile(
            r"(?i)\b(ignore\s+(all\s+)?(previous|prior|above)\s+instructions?|disregard\s+(all\s+)?(previous|prior)\s+instructions?|forget\s+(all\s+)?(previous|prior)\s+instructions?)\b"
        ),
    ),
    (
        "ROLE_PLAY_HIJACK",
        re.compile(
            r"(?i)\b(you\s+are\s+now\s+(a\s+|an\s+)?(unrestricted|jailbroken|evil|unfiltered|godmode)|act\s+as\s+(an\s+unrestricted|a\s+rogue))\b"
        ),
    ),
    (
        "DELIMITER_ESCAPE_ATTEMPT",
        re.compile(
            r"(?i)(<\s*\|\s*im_start\s*\|>|<\s*\|\s*im_end\s*\|>|\[\s*INST\s*\]|\[\s*/\s*INST\s*\]|<SYS>|<\s*/\s*SYS\s*>)"
        ),
    ),
    (
        "SECRET_EXFILTRATION_PROBE",
        re.compile(
            r"(?i)\b(reveal\s+your\s+system\s+prompt|print\s+(the\s+)?initial\s+prompt|dump\s+(all\s+)?system\s+instructions|show\s+your\s+secret\s+key)\b"
        ),
    ),
    (
        "ARBITRARY_TOOL_CALL_TRIGGER",
        re.compile(
            r"(?i)\b(execute\s+(the\s+)?following\s+tool|call\s+tool:\s*`?[a-z0-9_]+`?|invoke_subagent\b)"
        ),
    ),
]


def scan_untrusted_text(
    raw_text: str,
    canary_token: Optional[str] = None,
    raise_on_critical: bool = False,
) -> CanaryCheckResult:
    """Scans untrusted external text (filings, transcripts, news) for prompt injections.

    Rules:
    - If canary_token is provided, ensures the canary token is not modified or spoofed.
    - Matches critical injection signatures.
    - If raise_on_critical is True, immediately raises UntrustedInputError upon match.
    - Returns CanaryCheckResult with list of violations and sanitized text.
    """
    if not raw_text:
        return CanaryCheckResult(is_safe=True, sanitized_text="", canary_token_intact=True)

    violations: List[Dict[str, str]] = []

    # 1. Canary Token Check
    canary_intact = True
    if canary_token is not None:
        if canary_token not in raw_text:
            canary_intact = False
            violations.append({
                "rule": "CANARY_TOKEN_MISSING_OR_CORRUPTED",
                "severity": CanaryViolationSeverity.CRITICAL.value,
                "snippet": f"Expected canary token '{canary_token}' was absent or altered.",
            })

    # 2. Pattern Matching
    for rule_name, pattern in _CRITICAL_INJECTION_PATTERNS:
        match = pattern.search(raw_text)
        if match:
            matched_str = match.group(0)
            snippet = raw_text[max(0, match.start() - 20): min(len(raw_text), match.end() + 20)]
            if raise_on_critical:
                raise UntrustedInputError(rule_name, snippet)
            violations.append({
                "rule": rule_name,
                "severity": CanaryViolationSeverity.CRITICAL.value,
                "snippet": snippet.strip(),
                "matched_text": matched_str,
            })

    # 3. Sanitization (defangs delimiters and tags)
    sanitized = raw_text
    sanitized = re.sub(r"(<\s*\|\s*im_start\s*\|>|<\s*\|\s*im_end\s*\|>)", "[DEFANGED_DELIMITER]", sanitized)
    sanitized = re.sub(r"(<SYS>|</SYS>|\[INST\]|\[/INST\])", "[DEFANGED_TAG]", sanitized, flags=re.IGNORECASE)

    is_safe = (len(violations) == 0 and canary_intact)

    return CanaryCheckResult(
        is_safe=is_safe,
        violations=violations,
        sanitized_text=sanitized,
        canary_token_intact=canary_intact,
    )

