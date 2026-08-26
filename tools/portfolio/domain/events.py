"""System and domain events for portfolio journal and lifecycle mutations."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SystemJournalEvent:
    """Represents a system-generated qualitative event to be logged alongside ledger changes."""
    event_type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_entry(
        cls,
        event_type: str,
        message: str,
        date_str: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SystemJournalEvent":
        """Build an event using the journal's stable, human-readable timestamp.

        Journal markdown historically stores timestamps as ``YYYY-MM-DD
        HH:MM:SS``.  Keeping that normalization at the domain-event boundary
        lets a repository stage the event atomically while preserving the
        existing reader/parser contract.  Explicit malformed values are left
        untouched so callers retain the legacy error/visibility behaviour.
        """
        timestamp = _normalize_journal_timestamp(date_str)
        return cls(
            event_type=event_type,
            message=message,
            timestamp=timestamp,
            metadata=dict(metadata or {}),
        )


def _normalize_journal_timestamp(date_str: Optional[str]) -> str:
    """Normalize an optional API date into the journal timestamp format."""
    if date_str:
        value = str(date_str).strip()
        try:
            if len(value) == 10:
                # Historical JournalVaultAdapter convention for date-only
                # entries is midday, avoiding an accidental timezone shift.
                return f"{value} 12:00:00"
            else:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                # Preserve the caller's wall-clock components; the journal
                # format has no timezone field and callers already choose the
                # intended date/time before crossing this boundary.
                parsed = parsed.replace(tzinfo=None)
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            return value
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
