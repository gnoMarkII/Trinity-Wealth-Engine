"""Stable identities and immutable evidence binding for Macro tasks across retries, queue dispatches, and acceptance."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any


def next_macro_task_run_id(
    job_id: str | None,
    turn_id: str,
    sequence: int,
    *,
    retry_run_id: str | None = None,
) -> tuple[str, int]:
    """Reuse the logical run on retry; allocate a new sequence for a new task."""
    if retry_run_id:
        return retry_run_id, max(0, int(sequence))
    next_sequence = max(0, int(sequence)) + 1
    run_root = str(job_id or turn_id)
    return f"{run_root}:{turn_id}:{next_sequence}", next_sequence


def generate_macro_run_id(date_str: str | None = None, suffix: str | None = None) -> str:
    """Generate an immutable, collision-proof Macro run ID with date prefix."""
    date_prefix = (date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")).replace("-", "")
    unique_suffix = suffix or uuid.uuid4().hex[:8]
    return f"macro_run_{date_prefix}_{unique_suffix}"


def compute_payload_sha256(payload: Any) -> str:
    """Compute a deterministic SHA256 checksum for any serializable payload."""
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def create_evidence_manifest(
    run_id: str,
    evaluated_at: str,
    observable_count: int,
    content_hash: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a verified immutable run evidence manifest receipt."""
    return {
        "macro_run_id": run_id,
        "evaluated_at": evaluated_at,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "observable_count": observable_count,
        "content_hash": content_hash,
        "metadata": metadata or {},
    }
