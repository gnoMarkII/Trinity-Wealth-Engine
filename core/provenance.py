"""Point-in-Time Temporal Provenance & 5-Clock Hierarchy (Chapter 1 & Appendix A).

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies per Hexagonal
Architecture AST Domain Layer Isolation rules.

Reference:
Claude Code for Hedge Funds, Chapter 1 (pp. 7-15) & Appendix A:
"The 5 Clocks: event_time, available_time, observed_time, revised_time, compute_time.
Any observation where available_time > cutoff is strictly excluded from As-Of point-in-time
calculation to guarantee zero look-ahead bias."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlparse


def parse_utc_timestamp(ts: str) -> datetime:
    """Parses ISO-8601 timestamp string into timezone-aware UTC datetime."""
    clean = ts.strip()
    if clean.endswith("Z"):
        clean = clean[:-1] + "+00:00"
    elif len(clean) == 10 and clean[4] == "-" and clean[7] == "-":
        clean += "T00:00:00+00:00"
    try:
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception as e:
        raise ValueError(f"Invalid ISO-8601 timestamp '{ts}': {e}") from e


def normalize_utc_iso(ts: str) -> str:
    """Normalizes an ISO timestamp to canonical UTC 'YYYY-MM-DDTHH:MM:SSZ' format."""
    dt = parse_utc_timestamp(ts)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def is_available_as_of(available_time: str, cutoff_time: str) -> bool:
    """Verifies observation satisfies point-in-time cutoff constraint without look-ahead bias."""
    try:
        avail_dt = parse_utc_timestamp(available_time)
        cutoff_dt = parse_utc_timestamp(cutoff_time)
        return avail_dt <= cutoff_dt
    except Exception:
        return available_time <= cutoff_time


@dataclass(frozen=True)
class TemporalClocks:
    """The 5 canonical clocks governing point-in-time financial state."""
    event_time: str                         # Wall-clock moment economic event occurred (e.g. Q3 close)
    available_time: str                     # Moment public filing/disclosure became accessible
    observed_time: str                      # Moment internal agent/system fetched the data
    compute_time: str                       # Moment analytical calculation was executed
    revised_time: Optional[str] = None      # Moment retroactive restatement was announced, if any

    def is_valid_as_of(self, cutoff_time: str) -> bool:
        """Point-in-time invariant: available_time must be <= cutoff_time."""
        return is_available_as_of(self.available_time, cutoff_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_time": self.event_time,
            "available_time": self.available_time,
            "observed_time": self.observed_time,
            "compute_time": self.compute_time,
            "revised_time": self.revised_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalClocks":
        return cls(
            event_time=data["event_time"],
            available_time=data["available_time"],
            observed_time=data["observed_time"],
            compute_time=data["compute_time"],
            revised_time=data.get("revised_time"),
        )


def get_source_independence_key(url: str, publisher: str, source_type: str) -> str:
    """
    Standardize source grouping logic to evaluate consensus and independence.
    """
    if source_type == "official":
        return f"official_{publisher.lower()}"
    
    if url:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            if domain:
                return f"domain_{domain}"
        except Exception:
            pass
            
    return f"publisher_{publisher.lower()}"
