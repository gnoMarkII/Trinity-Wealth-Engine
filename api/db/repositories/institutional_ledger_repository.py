"""DAO functions for append-only institutional ledgers.

Functions receive a connection owned by an adapter or UnitOfWork and never
commit or roll back it.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _row_dict(row: sqlite3.Row | tuple, cursor: sqlite3.Cursor) -> dict[str, Any]:
    if isinstance(row, sqlite3.Row):
        return dict(row)
    return dict(zip((col[0] for col in cursor.description or ()), row))


def append_routine_event(
    conn: sqlite3.Connection,
    *,
    execution_id: str,
    event_type: str,
    payload: Mapping[str, Any],
    portfolio_id: str | None = None,
) -> dict[str, Any]:
    execution_id = str(execution_id).strip()
    event_type = str(event_type).strip()
    if not execution_id or not event_type:
        raise ValueError("execution_id and event_type are required")
    conn.row_factory = sqlite3.Row
    record = dict(payload)
    payload_json = _canonical(record)
    payload_sha = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    event_id = str(record.get("event_id") or record.get("routine_event_id") or f"ROUTEVT-{uuid4().hex}")
    existing = conn.execute(
        "SELECT * FROM routine_execution_events WHERE event_id=?", (event_id,)
    ).fetchone()
    if existing is not None:
        conn.row_factory = sqlite3.Row
        if (
            existing["execution_id"] != execution_id
            or existing["event_type"] != event_type
            or existing["payload_sha256"] != payload_sha
        ):
            raise ValueError("routine event ID was reused with different content")
        item = dict(existing)
        item["payload"] = json.loads(item.pop("payload_json"))
        return item
    row = conn.execute(
        "SELECT COALESCE(MAX(event_sequence), 0) FROM routine_execution_events WHERE execution_id=?",
        (execution_id,),
    ).fetchone()
    sequence = int(row[0]) + 1
    event = {
        "event_id": event_id,
        "execution_id": execution_id,
        "event_sequence": sequence,
        "portfolio_id": portfolio_id,
        "event_type": event_type,
        "scheduled_at_utc": record.get("scheduled_utc"),
        "started_at_utc": record.get("actual_start_utc"),
        "completed_at_utc": record.get("completed_utc"),
        "payload_json": payload_json,
        "payload_sha256": payload_sha,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    conn.execute(
        """
        INSERT INTO routine_execution_events
        (event_id, execution_id, event_sequence, portfolio_id, event_type,
         scheduled_at_utc, started_at_utc, completed_at_utc, payload_json,
         payload_sha256, created_at_utc)
        VALUES (:event_id, :execution_id, :event_sequence, :portfolio_id,
                :event_type, :scheduled_at_utc, :started_at_utc, :completed_at_utc,
                :payload_json, :payload_sha256, :created_at_utc)
        """,
        event,
    )
    event["payload"] = json.loads(payload_json)
    return event


def list_routine_events(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM routine_execution_events ORDER BY created_at_utc, execution_id, event_sequence"
    ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        result.append(item)
    return result


def append_catalyst_revision(
    conn: sqlite3.Connection,
    catalyst: Mapping[str, Any],
    *,
    portfolio_id: str | None = None,
) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    payload = dict(catalyst)
    catalyst_id = str(payload.get("catalyst_id") or "").strip()
    entity_id = str(payload.get("entity_id") or "").strip()
    if not catalyst_id or not entity_id:
        raise ValueError("catalyst_id and entity_id are required")
    payload_json = _canonical(payload)
    payload_sha = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    prior = conn.execute(
        "SELECT revision, payload_sha256 FROM catalyst_revisions WHERE catalyst_id=? ORDER BY revision DESC LIMIT 1",
        (catalyst_id,),
    ).fetchone()
    if prior and prior[1] == payload_sha:
        row = conn.execute(
            "SELECT * FROM catalyst_revisions WHERE catalyst_id=? AND revision=?",
            (catalyst_id, int(prior[0])),
        ).fetchone()
        return _catalyst_row(row)
    revision = int(prior[0]) + 1 if prior else 1
    recorded = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    conn.execute(
        """
        INSERT INTO catalyst_revisions
        (catalyst_id, revision, portfolio_id, entity_id, linked_claim_id, event_name,
         expected_date, observed_date, source_id, source_revision, prior_revision,
         payload_json, payload_sha256, recorded_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            catalyst_id, revision, portfolio_id, entity_id,
            payload.get("linked_claim_id"), payload.get("event_name", ""),
            payload.get("expected_date"), payload.get("observed_date"),
            payload.get("source_id"), payload.get("source_revision"),
            revision - 1 if revision > 1 else None, payload_json, payload_sha, recorded,
        ),
    )
    row = conn.execute(
        "SELECT * FROM catalyst_revisions WHERE catalyst_id=? AND revision=?",
        (catalyst_id, revision),
    ).fetchone()
    return _catalyst_row(row)


def _catalyst_row(row: sqlite3.Row | tuple | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row) if isinstance(row, sqlite3.Row) else None
    if item is None:
        raise TypeError("catalyst row requires sqlite3.Row")
    item["payload"] = json.loads(item.pop("payload_json"))
    return item


def list_latest_catalysts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT c.* FROM catalyst_revisions c
        JOIN (
            SELECT catalyst_id, MAX(revision) AS revision
            FROM catalyst_revisions GROUP BY catalyst_id
        ) latest ON latest.catalyst_id=c.catalyst_id AND latest.revision=c.revision
        ORDER BY c.catalyst_id
        """
    ).fetchall()
    return [_catalyst_row(row) for row in rows]


def insert_source_capture(conn: sqlite3.Connection, capture: Mapping[str, Any]) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    source = dict(capture)
    required = ("source_id", "revision", "document_type", "source_locator", "artifact_ref", "content_sha256", "captured_at_utc")
    missing = [name for name in required if not str(source.get(name) or "").strip()]
    if missing:
        raise ValueError(f"source capture missing required fields: {', '.join(missing)}")
    metadata_json = _canonical(source.get("metadata") or {})
    conn.execute(
        """
        INSERT INTO source_captures
        (source_id, revision, issuer_id, ticker, document_type, fiscal_period,
         source_locator, artifact_ref, content_sha256, captured_at_utc,
         available_at_utc, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_id, revision) DO NOTHING
        """,
        (
            source["source_id"], source["revision"], source.get("issuer_id"),
            source.get("ticker"), source["document_type"], source.get("fiscal_period"),
            source["source_locator"], source["artifact_ref"], source["content_sha256"],
            source["captured_at_utc"], source.get("available_at_utc"), metadata_json,
        ),
    )
    row = conn.execute(
        "SELECT * FROM source_captures WHERE source_id=? AND revision=?",
        (source["source_id"], source["revision"]),
    ).fetchone()
    item = dict(row)
    expected = {
        "issuer_id": source.get("issuer_id"),
        "ticker": source.get("ticker"),
        "document_type": source["document_type"],
        "fiscal_period": source.get("fiscal_period"),
        "source_locator": source["source_locator"],
        "artifact_ref": source["artifact_ref"],
        "content_sha256": source["content_sha256"],
        "captured_at_utc": source["captured_at_utc"],
        "available_at_utc": source.get("available_at_utc"),
        "metadata_json": metadata_json,
    }
    if any(item[name] != value for name, value in expected.items()):
        raise ValueError("source_id/revision already exists with different content hash or metadata")
    item["metadata"] = json.loads(item.pop("metadata_json"))
    return item


def list_source_captures(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM source_captures ORDER BY captured_at_utc, source_id, revision"
    ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json"))
        result.append(item)
    return result


def insert_catalyst_assessment(conn: sqlite3.Connection, assessment: Mapping[str, Any]) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    row_data = dict(assessment)
    key = str(row_data.get("idempotency_key") or "").strip()
    stable_payload = {k: v for k, v in row_data.items() if k not in {"assessment_id", "recorded_at_utc"}}
    payload_sha = _sha(stable_payload)
    if not key:
        raise ValueError("idempotency_key is required")
    row_data.setdefault("assessment_id", f"ASSESS-{uuid4().hex}")
    row_data.setdefault("recorded_at_utc", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    conn.execute(
        """
        INSERT INTO catalyst_assessments
        (assessment_id, idempotency_key, portfolio_id, catalyst_id, exception_fingerprint,
         source_id, source_revision, reviewer_subject_id, reviewer_session_id,
         assessment_text, payload_json, payload_sha256, recorded_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(idempotency_key) DO NOTHING
        """,
        (
            row_data["assessment_id"], key, row_data.get("portfolio_id"),
            row_data["catalyst_id"], row_data["exception_fingerprint"],
            row_data.get("source_id"), row_data.get("source_revision"),
            row_data["reviewer_subject_id"], row_data.get("reviewer_session_id"),
            row_data["assessment_text"], _canonical(row_data), payload_sha, row_data["recorded_at_utc"],
        ),
    )
    row = conn.execute(
        "SELECT * FROM catalyst_assessments WHERE idempotency_key=?", (key,)
    ).fetchone()
    item = dict(row)
    if item["payload_sha256"] != payload_sha:
        raise ValueError("idempotency key was reused with a different assessment")
    return item


def list_catalyst_assessments(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    result = []
    for row in conn.execute(
        "SELECT * FROM catalyst_assessments ORDER BY recorded_at_utc, assessment_id"
    ).fetchall():
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        result.append(item)
    return result


def insert_estimate_record(conn: sqlite3.Connection, record: Mapping[str, Any]) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    payload = dict(record)
    estimate_id = str(payload.get("estimate_id") or "").strip()
    if not estimate_id:
        raise ValueError("estimate_id is required")
    payload_json = _canonical(payload)
    payload_sha = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    conn.execute(
        """
        INSERT INTO estimate_records
        (estimate_id, prior_record_id, ticker, fiscal_period, metric_name,
         estimate_type, value_decimal_text, unit, source_id, source_revision,
         source_as_of_utc, available_at_utc, approved_by_subject_id, payload_json,
         payload_sha256, recorded_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(estimate_id) DO NOTHING
        """,
        (
            estimate_id, payload.get("prior_record_id"), str(payload["ticker"]).upper(),
            payload["period"], payload["metric_name"], payload["estimate_type"],
            str(payload["value"]), payload.get("unit", ""), payload.get("source_id"),
            payload.get("source_revision"), payload.get("as_of_utc"),
            payload.get("available_at_utc") or payload.get("as_of_utc") or "",
            payload.get("approved_by_subject_id") or (
                payload.get("owner") if payload.get("approval_status") == "APPROVED" else None
            ),
            payload_json, payload_sha,
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        ),
    )
    row = conn.execute(
        "SELECT * FROM estimate_records WHERE estimate_id=?", (estimate_id,)
    ).fetchone()
    item = dict(row)
    if item["payload_sha256"] != payload_sha:
        raise ValueError(f"estimate ID {estimate_id} already exists with different payload")
    item["payload"] = json.loads(item.pop("payload_json"))
    return item


def list_estimate_records(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM estimate_records ORDER BY available_at_utc, estimate_id"
    ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        result.append(item)
    return result


def insert_import_receipt(
    conn: sqlite3.Connection,
    *,
    source_digest: str,
    record_family: str,
    legacy_record_id: str,
    payload_sha256: str,
    target_record_id: str,
    import_status: str = "IMPORTED",
) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    if import_status not in ("DRY_RUN", "IMPORTED", "SKIPPED_IDENTICAL"):
        raise ValueError("invalid import_status")
    receipt_id = f"IMPORT-{uuid4().hex}"
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    conn.execute(
        """
        INSERT INTO state_import_receipts
        (receipt_id, source_digest, record_family, legacy_record_id,
         payload_sha256, target_record_id, import_status, imported_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_digest, record_family, legacy_record_id) DO NOTHING
        """,
        (
            receipt_id, source_digest, record_family, legacy_record_id,
            payload_sha256, target_record_id, import_status, now,
        ),
    )
    row = conn.execute(
        """
        SELECT * FROM state_import_receipts
        WHERE source_digest=? AND record_family=? AND legacy_record_id=?
        """,
        (source_digest, record_family, legacy_record_id),
    ).fetchone()
    item = dict(row)
    if item["payload_sha256"] != payload_sha256 or item["target_record_id"] != target_record_id:
        raise ValueError("legacy import receipt conflicts with an existing payload")
    return item


def list_import_receipts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(
        "SELECT * FROM state_import_receipts ORDER BY imported_at_utc, receipt_id"
    ).fetchall()]
