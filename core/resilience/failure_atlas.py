"""Institutional Failure Atlas & Smallest Bounded Repair Engine.

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies (no FastAPI, Pydantic,
SQLite, or network I/O) per Hexagonal Architecture AST Domain Layer Isolation rules.

Reference:
Claude Code for Hedge Funds: Build a Reproducible As-Of Research State
Chapter I (Appendix I / Chapter 11): Institutional Failure Atlas
"The fastest safe repair begins by identifying what kind of proof failed...
Repair the smallest failing unit."
"""
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Dict, Optional


class FailureClass(str, Enum):
    """The 9 institutional failure classifications from Claude Code for Hedge Funds."""
    ACCESS = "ACCESS"           # API key missing/invalid, network connection refused, rate limit (HTTP 429)
    SOURCE = "SOURCE"           # Filing mismatch, unverified document locator, missing accession
    STATE = "STATE"             # Protected hash mismatch, retroactively modified historical prices
    SCHEMA = "SCHEMA"           # Missing required clock/cutoff, missing key fields, invalid JSON
    ARITHMETIC = "ARITHMETIC"   # Driver sum mismatch, zero variance in factor beta denominator, negative share price
    INTEGRITY = "INTEGRITY"     # Prompt injection detected, canary token corrupted or stripped
    PERMISSION = "PERMISSION"   # Unauthorized output path, attempting write outside workspace/sandbox
    SCHEDULING = "SCHEDULING"   # Late run drift (>120 min), stale market close data
    REVIEW = "REVIEW"           # Unsigned packet, blank reviewer identity, attempting trade without human PM signoff


@dataclass(frozen=True)
class FailureIncident:
    """A triage-classified operational or analytical failure with its smallest repair boundary."""
    failure_class: FailureClass
    symptom: str
    likely_cause: str
    smallest_repair: str
    escalate_when: str
    affected_artifact: str = ""
    is_blocking: bool = False
    error_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_class": self.failure_class.value,
            "symptom": self.symptom,
            "likely_cause": self.likely_cause,
            "smallest_repair": self.smallest_repair,
            "escalate_when": self.escalate_when,
            "affected_artifact": self.affected_artifact,
            "is_blocking": self.is_blocking,
            "error_type": self.error_type,
        }


def triage_failure(
    error: Exception | str,
    context: Optional[Dict[str, Any]] = None,
) -> FailureIncident:
    """Triages an exception or error message into one of 9 institutional failure classes.
    
    Provides the exact, smallest bounded repair action to prevent cascading system failure.
    """
    ctx = context or {}
    err_msg = str(error)
    err_type = type(error).__name__ if isinstance(error, Exception) else "RuntimeError"
    err_lower = err_msg.lower()

    # 1. INTEGRITY (Canary, prompt injection, security breaches)
    if any(k in err_lower for k in ["canary", "prompt injection", "untrusted instruction", "system prompt override"]):
        return FailureIncident(
            failure_class=FailureClass.INTEGRITY,
            symptom=f"Integrity guard tripped: {err_msg}",
            likely_cause="Malicious prompt injection or corrupted canary token detected in untrusted inputs.",
            smallest_repair="Discard untrusted payload, quarantine agent input, and defang markdown/HTML tags.",
            escalate_when="Attacks target production orchestrator or persist across multiple data providers.",
            affected_artifact=ctx.get("artifact", "untrusted_input"),
            is_blocking=True,
            error_type=err_type,
        )

    # 2. STATE (Tampered files, checksum mismatch, hash verification failure)
    if any(k in err_lower for k in ["state integrity", "sha256 mismatch", "tampered", "frozen state", "checksum mismatch"]):
        return FailureIncident(
            failure_class=FailureClass.STATE,
            symptom=f"State hash integrity mismatch: {err_msg}",
            likely_cause="Historical frozen input file was retroactively modified or deleted.",
            smallest_repair="Restore historical state from immutable backup or recreate checkpoint manifest.",
            escalate_when="Mismatches span multiple historical state snapshots or indicate disk corruption.",
            affected_artifact=ctx.get("artifact", "state_manifest"),
            is_blocking=True,
            error_type=err_type,
        )

    # 3. ACCESS (Network, API key, rate limits)
    if any(k in err_lower for k in ["rate limit", "429", "unauthorized", "401", "forbidden", "connection refused", "timeout", "api key"]):
        return FailureIncident(
            failure_class=FailureClass.ACCESS,
            symptom=f"External service access failed: {err_msg}",
            likely_cause="Network disconnection, expired API key, or provider rate limiting.",
            smallest_repair="Rotate API key or retry request with exponential backoff; do not alter analytical models.",
            escalate_when="Outage persists past SLA window (>15 minutes).",
            affected_artifact=ctx.get("artifact", "network_adapter"),
            is_blocking=False,
            error_type=err_type,
        )

    # 4. SOURCE (Filing mismatch, citation laundering, unverified URL)
    if any(k in err_lower for k in ["citation mismatch", "anti-citation", "unverified source", "missing accession", "10-k", "10-q"]):
        return FailureIncident(
            failure_class=FailureClass.SOURCE,
            symptom=f"Source attribution failure: {err_msg}",
            likely_cause="Claim numbers do not tie out against primary filing disclosure facts.",
            smallest_repair="Demote claim evidence tier to UNVERIFIED and prompt analyst to verify accession.",
            escalate_when="Factual mismatch occurs on core earnings or valuation drivers for active positions.",
            affected_artifact=ctx.get("artifact", "thesis_claim"),
            is_blocking=False,
            error_type=err_type,
        )

    # 5. ARITHMETIC (Driver sums, zero variance, negative prices, division by zero)
    if any(k in err_lower for k in ["driver sum", "zero variance", "division by zero", "negative price", "unbalanced bridge", "ddof"]):
        return FailureIncident(
            failure_class=FailureClass.ARITHMETIC,
            symptom=f"Mathematical invariant breach: {err_msg}",
            likely_cause="Variance drivers do not sum to total delta, or benchmark returns have zero variance.",
            smallest_repair="Isolate calculation error to single asset card; rebalance variance drivers without halting portfolio run.",
            escalate_when="Arithmetic errors compromise portfolio-level NAV or gross exposure bounds.",
            affected_artifact=ctx.get("artifact", "math_engine"),
            is_blocking=False,
            error_type=err_type,
        )

    # 6. PERMISSION (Unauthorized filesystem access, write outside workspace)
    if any(k in err_lower for k in ["permission denied", "outside workspace", "unauthorized path", "read-only"]):
        return FailureIncident(
            failure_class=FailureClass.PERMISSION,
            symptom=f"Filesystem permission restriction: {err_msg}",
            likely_cause="Attempting to write outside approved workspace sandbox or touching protected root files.",
            smallest_repair="Restrict file write to approved artifact or scratch directory.",
            escalate_when="Permission failures block core database or audit log persistence.",
            affected_artifact=ctx.get("artifact", "filesystem"),
            is_blocking=True,
            error_type=err_type,
        )

    # 7. SCHEDULING (Late-run drift, stale market data)
    if any(k in err_lower for k in ["drift", "late run", "stale data", "as-of clock"]):
        return FailureIncident(
            failure_class=FailureClass.SCHEDULING,
            symptom=f"Clock or scheduling drift: {err_msg}",
            likely_cause="Morning packet generated >120 minutes past market close or target cutoff.",
            smallest_repair="Display late-run warning banner in Morning Review; allow PM to review with explicit timestamp.",
            escalate_when="Drift exceeds 24 hours.",
            affected_artifact=ctx.get("artifact", "scheduler"),
            is_blocking=False,
            error_type=err_type,
        )

    # 8. REVIEW (Unsigned packet, trade execution without human sign-off)
    if any(k in err_lower for k in ["signoff", "unsigned", "review required", "human pm", "missing reviewer"]):
        return FailureIncident(
            failure_class=FailureClass.REVIEW,
            symptom=f"PM review governance check: {err_msg}",
            likely_cause="Attempting portfolio execution or trade transition without human sign-off.",
            smallest_repair="Route packet to Human PM Review queue; require explicit sign-off button click.",
            escalate_when="Unauthorized trades attempt automated execution.",
            affected_artifact=ctx.get("artifact", "pm_review"),
            is_blocking=True,
            error_type=err_type,
        )

    # 9. SCHEMA (Default/fallback if missing fields or schema violation)
    return FailureIncident(
        failure_class=FailureClass.SCHEMA,
        symptom=f"Schema or structural anomaly: {err_msg}",
        likely_cause="Unexpected dictionary key, missing field, or data contract mismatch.",
        smallest_repair="Backfill default values for optional schema fields; avoid rebuilding unaffected quadrants.",
        escalate_when="Missing required primary keys or structural entity IDs.",
        affected_artifact=ctx.get("artifact", "schema"),
        is_blocking=False,
        error_type=err_type,
    )

