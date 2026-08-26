"""Domain State Machine & Idempotency Key computation for Earnings Call Context."""
from enum import Enum
import hashlib


class EarningsCallRunStatus(str, Enum):
    NEW = "new"
    SUMMARIZED = "summarized"
    NOTE_WRITTEN = "note_written"
    KANBAN_PENDING = "kanban_pending"
    COMPLETED = "completed"
    FAILED = "failed"


class EarningsCallKanbanStatus(str, Enum):
    NONE = "none"
    PENDING = "pending"
    CREATED = "created"
    EXISTING = "existing"
    FAILED = "failed"


def compute_source_key(
    canonical_ticker: str,
    canonical_period: str,
    transcript: str,
    prompt_version: str = "v1",
) -> tuple[str, str]:
    """Computes a deterministic source_key and transcript_hash from normalized inputs."""
    transcript_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    raw_key = f"{canonical_ticker}:{canonical_period}:{transcript_hash}:{prompt_version}"
    source_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    return source_key, transcript_hash
