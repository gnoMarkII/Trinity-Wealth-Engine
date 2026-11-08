"""Unit tests for Institutional Failure Atlas & Smallest Bounded Repair Engine.

Hexagonal Architecture Rule: Pure domain testing using standard library fixtures.
"""
import pytest

from core.resilience.failure_atlas import (
    FailureClass,
    FailureIncident,
    triage_failure,
)


def test_triage_integrity_failure():
    """Prompt injection or canary corruption maps to INTEGRITY with blocking status."""
    err = ValueError("Prompt injection detected: untrusted system instruction token")
    incident = triage_failure(err, context={"artifact": "analyst_notes.txt"})
    assert incident.failure_class == FailureClass.INTEGRITY
    assert incident.is_blocking is True
    assert "quarantine agent input" in incident.smallest_repair
    assert incident.affected_artifact == "analyst_notes.txt"


def test_triage_state_failure():
    """State hash mismatch maps to STATE with blocking status."""
    err = RuntimeError("State integrity SHA256 mismatch on prices_history.csv")
    incident = triage_failure(err)
    assert incident.failure_class == FailureClass.STATE
    assert incident.is_blocking is True
    assert "Restore historical state" in incident.smallest_repair


def test_triage_access_failure():
    """Rate limits or timeouts map to ACCESS and non-blocking retry."""
    err = ConnectionError("HTTP 429 Too Many Requests: Rate limit exceeded")
    incident = triage_failure(err)
    assert incident.failure_class == FailureClass.ACCESS
    assert incident.is_blocking is False
    assert "Rotate API key or retry" in incident.smallest_repair


def test_triage_source_failure():
    """Citation mismatch maps to SOURCE with tier demotion suggestion."""
    err = ValueError("Anti-citation laundering violation: Claimed margin does not match SEC 10-Q")
    incident = triage_failure(err)
    assert incident.failure_class == FailureClass.SOURCE
    assert incident.is_blocking is False
    assert "Demote claim evidence tier" in incident.smallest_repair


def test_triage_arithmetic_failure():
    """Driver sum invariant breach maps to ARITHMETIC."""
    err = ValueError("Unbalanced bridge: Driver sum does not equal revenue revision")
    incident = triage_failure(err)
    assert incident.failure_class == FailureClass.ARITHMETIC
    assert incident.is_blocking is False
    assert "Isolate calculation error" in incident.smallest_repair


def test_triage_permission_failure():
    """Filesystem permission restriction maps to PERMISSION and blocking status."""
    err = PermissionError("Attempted file write outside workspace directory")
    incident = triage_failure(err)
    assert incident.failure_class == FailureClass.PERMISSION
    assert incident.is_blocking is True
    assert "Restrict file write" in incident.smallest_repair


def test_triage_scheduling_failure():
    """Late run clock drift maps to SCHEDULING with warning banner repair."""
    err = "As-of clock drift exceeds 120 minutes: late run"
    incident = triage_failure(err)
    assert incident.failure_class == FailureClass.SCHEDULING
    assert incident.is_blocking is False
    assert "late-run warning banner" in incident.smallest_repair


def test_triage_review_failure():
    """Unsigned packet or missing PM signoff maps to REVIEW with blocking status."""
    err = "Review required: cannot execute trades with unsigned packet"
    incident = triage_failure(err)
    assert incident.failure_class == FailureClass.REVIEW
    assert incident.is_blocking is True
    assert "Route packet to Human PM Review queue" in incident.smallest_repair


def test_triage_schema_fallback():
    """Unknown data structure anomaly defaults to SCHEMA."""
    err = KeyError("Missing expected field 'portfolio_id'")
    incident = triage_failure(err)
    assert incident.failure_class == FailureClass.SCHEMA
    assert incident.is_blocking is False
    assert "Backfill default values" in incident.smallest_repair


def test_incident_dict_serialization():
    """FailureIncident serializes to dict for API and UI rendering."""
    incident = triage_failure("Test error")
    d = incident.to_dict()
    assert "failure_class" in d
    assert "smallest_repair" in d
    assert "is_blocking" in d

