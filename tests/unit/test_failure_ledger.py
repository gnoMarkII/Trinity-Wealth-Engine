"""Unit tests for tools/telemetry/failure_ledger.py"""
from pathlib import Path
from tools.telemetry.failure_ledger import (
    record_probe_event,
    read_probe_events,
    aggregate_probe_statistics,
)


def test_record_and_read_probe_events(tmp_path: Path):
    log_file = tmp_path / "probe_events.jsonl"

    e1 = record_probe_event(
        domain="macro",
        event_type="probe_retry",
        issues=["SUPPORTING_DATA_MISMATCH", "SINGLE_SOURCE_PENALTY"],
        resolved=True,
        retries_count=1,
        context_meta={"agent": "allocator"},
        output_path=log_file,
    )
    assert e1["domain"] == "macro"
    assert e1["resolved"] is True
    assert e1["issues_count"] == 2

    e2 = record_probe_event(
        domain="equity",
        event_type="probe_failed",
        issues=["dcf_contradiction"],
        resolved=False,
        retries_count=1,
        context_meta={"ticker": "AAPL"},
        output_path=log_file,
    )

    all_events = read_probe_events(input_path=log_file)
    assert len(all_events) == 2

    macro_events = read_probe_events(domain="macro", input_path=log_file)
    assert len(macro_events) == 1
    assert macro_events[0]["domain"] == "macro"

    stats = aggregate_probe_statistics(input_path=log_file)
    assert stats["total_events"] == 2
    assert stats["resolved_rate"] == 0.5
    assert stats["issue_counts"]["SUPPORTING_DATA_MISMATCH"] == 1
    assert stats["issue_counts"]["dcf_contradiction"] == 1

