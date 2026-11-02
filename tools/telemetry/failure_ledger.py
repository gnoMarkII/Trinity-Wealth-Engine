"""Failure Telemetry and Probe Event Ledger.

Architecture Layer: tools/telemetry/
Dependencies: Standard Library only. Complies with AST dependency rules.

Records and analyzes validation probe events across Macro and Equity harnesses:
- Tracks probe retries, failures, and resolution status.
- Aggregates recurring patterns to guide prompt refinements and active guardrails.
- Uses configurable file paths via env var for test isolation and safe execution.
"""
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TELEMETRY_DIR = Path(os.getenv("TELEMETRY_DIR", "data/telemetry"))
DEFAULT_TELEMETRY_FILE = Path(os.getenv("TELEMETRY_EVENTS_PATH", DEFAULT_TELEMETRY_DIR / "probe_events.jsonl"))


def get_telemetry_file_path(override_path: Optional[Path | str] = None) -> Path:
    """Resolve active telemetry events file path."""
    if override_path:
        return Path(override_path)
    env_path = os.getenv("TELEMETRY_EVENTS_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_TELEMETRY_FILE


def record_probe_event(
    domain: str,
    event_type: str,
    issues: List[str],
    resolved: bool,
    retries_count: int = 1,
    candidate_summary: Optional[Dict[str, Any]] = None,
    context_meta: Optional[Dict[str, Any]] = None,
    output_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Record a structured validation/probe event to the JSONL ledger.

    Args:
        domain: "macro" or "equity"
        event_type: "probe_retry", "probe_resolved", or "probe_failed"
        issues: list of warning IDs or verification error strings
        resolved: True if retry resolved the issues, False otherwise
        retries_count: number of retry attempts made
        candidate_summary: summary of candidate metrics/allocations
        context_meta: metadata (e.g. agent name, ticker, observable count)
        output_path: optional target path (defaults to env or data/telemetry)

    Returns:
        The recorded event payload.
    """
    target = get_telemetry_file_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    event_payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domain": str(domain).lower().strip(),
        "event_type": str(event_type).strip(),
        "issues": list(issues),
        "issues_count": len(issues),
        "resolved": bool(resolved),
        "retries_count": int(retries_count),
        "candidate_summary": candidate_summary or {},
        "context_meta": context_meta or {},
    }

    try:
        line = json.dumps(event_payload, ensure_ascii=False)
        with open(target, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning(f"Failed to record probe event to {target}: {exc}")

    return event_payload


def read_probe_events(
    domain: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: Optional[int] = None,
    input_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """Read probe events from ledger with optional filtering."""
    target = get_telemetry_file_path(input_path)
    if not target.is_file():
        return []

    events: List[Dict[str, Any]] = []
    try:
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    if domain and payload.get("domain") != domain.lower().strip():
                        continue
                    if event_type and payload.get("event_type") != event_type.strip():
                        continue
                    events.append(payload)
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.warning(f"Failed to read probe events from {target}: {exc}")
        return []

    if limit and limit > 0:
        return events[-limit:]
    return events


def aggregate_probe_statistics(
    events: Optional[List[Dict[str, Any]]] = None,
    input_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Aggregate Pareto statistics across recorded probe events."""
    if events is None:
        events = read_probe_events(input_path=input_path)

    total_events = len(events)
    if total_events == 0:
        return {
            "total_events": 0,
            "resolved_rate": 0.0,
            "issue_counts": {},
            "top_recurring_issues": [],
        }

    resolved_count = sum(1 for e in events if e.get("resolved") is True)
    issue_freq: Dict[str, int] = {}

    for e in events:
        for issue in e.get("issues", []):
            issue_freq[issue] = issue_freq.get(issue, 0) + 1

    sorted_issues = sorted(issue_freq.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "total_events": total_events,
        "resolved_rate": round(resolved_count / total_events, 4) if total_events else 0.0,
        "issue_counts": issue_freq,
        "top_recurring_issues": sorted_issues,
    }

