"""SQLite DAO for Thesis Claims, Exceptions and PM Review Sign-offs.

Architecture Rule (Layer 3 - Tools & Adapters):
Pure Data Access Object (DAO) receiving a managed connection from UnitOfWork.
Strictly ZERO calls to conn.commit() or conn.rollback() per AST DAO isolation rules.
"""
from __future__ import annotations

from decimal import Decimal
import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.thesis.evidence_taxonomy import EvidenceTier
from core.thesis.models import (
    CatalystEvent,
    FalsifierRule,
    ThesisClaimRecord,
    ThesisExceptionRow,
)


def insert_thesis_claim(conn: sqlite3.Connection, claim: ThesisClaimRecord) -> None:
    """Inserts a new immutable thesis claim record into SQLite."""
    falsifiers_json = json.dumps([f.to_dict() for f in claim.falsifiers], ensure_ascii=False)
    catalysts_json = json.dumps([c.to_dict() for c in claim.linked_catalysts], ensure_ascii=False)
    now = time.time()
    conn.execute(
        """
        INSERT INTO thesis_claims (
            claim_id, ticker, version, prior_claim_id, as_of_utc,
            claim_statement, falsifiers_json, catalysts_json, status,
            evidence_tier, author_role, checksum_sha256, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            claim.claim_id,
            claim.ticker,
            claim.version,
            claim.prior_claim_id,
            claim.as_of_utc,
            claim.claim_statement,
            falsifiers_json,
            catalysts_json,
            claim.status,
            claim.evidence_tier.value,
            claim.author_role,
            claim.checksum_sha256,
            now,
        ),
    )


def _row_to_claim(row: sqlite3.Row | tuple) -> ThesisClaimRecord:
    falsifiers_raw = json.loads(row["falsifiers_json"]) if isinstance(row, sqlite3.Row) else json.loads(row[6])
    catalysts_raw = json.loads(row["catalysts_json"]) if isinstance(row, sqlite3.Row) else json.loads(row[7])

    falsifiers = tuple(FalsifierRule.from_dict(f) for f in falsifiers_raw)
    catalysts = tuple(CatalystEvent.from_dict(c) for c in catalysts_raw)

    if isinstance(row, sqlite3.Row):
        return ThesisClaimRecord(
            claim_id=row["claim_id"],
            ticker=row["ticker"],
            version=row["version"],
            prior_claim_id=row["prior_claim_id"],
            as_of_utc=row["as_of_utc"],
            claim_statement=row["claim_statement"],
            falsifiers=falsifiers,
            linked_catalysts=catalysts,
            status=row["status"],
            evidence_tier=EvidenceTier(row["evidence_tier"]),
            author_role=row["author_role"],
            checksum_sha256=row["checksum_sha256"],
        )
    return ThesisClaimRecord(
        claim_id=row[0],
        ticker=row[1],
        version=row[2],
        prior_claim_id=row[3],
        as_of_utc=row[4],
        claim_statement=row[5],
        falsifiers=falsifiers,
        linked_catalysts=catalysts,
        status=row[8],
        evidence_tier=EvidenceTier(row[9]),
        author_role=row[10],
        checksum_sha256=row[11],
    )


def get_active_claims(conn: sqlite3.Connection, ticker: Optional[str] = None) -> List[ThesisClaimRecord]:
    """Retrieves all currently ACTIVE thesis claims, optionally filtered by ticker."""
    conn.row_factory = sqlite3.Row
    if ticker:
        cursor = conn.execute(
            """
            SELECT * FROM thesis_claims
            WHERE ticker = ? AND status = 'ACTIVE'
            ORDER BY version DESC
            """,
            (ticker.strip().upper(),),
        )
    else:
        cursor = conn.execute(
            """
            SELECT * FROM thesis_claims
            WHERE status = 'ACTIVE'
            ORDER BY ticker ASC, version DESC
            """
        )
    return [_row_to_claim(r) for r in cursor.fetchall()]


def get_claim_by_id(conn: sqlite3.Connection, claim_id: str) -> Optional[ThesisClaimRecord]:
    """Fetches a specific claim record by its unique ID."""
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT * FROM thesis_claims WHERE claim_id = ?",
        (claim_id,),
    )
    row = cursor.fetchone()
    return _row_to_claim(row) if row else None


def get_claims_as_of(conn: sqlite3.Connection, ticker: str, as_of_utc: str) -> List[ThesisClaimRecord]:
    """Time-travel query: retrieves the active claims in effect at the specified historical timestamp."""
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        """
        SELECT * FROM thesis_claims
        WHERE ticker = ? AND as_of_utc <= ?
        ORDER BY version DESC
        """,
        (ticker.strip().upper(), as_of_utc),
    )
    rows = cursor.fetchall()
    # Return latest version that was effective at or before as_of_utc
    if not rows:
        return []
    latest_version = rows[0]["version"]
    return [_row_to_claim(r) for r in rows if r["version"] == latest_version]


def insert_exception(conn: sqlite3.Connection, exc: ThesisExceptionRow) -> None:
    """Inserts a new thesis exception detected by the watchdog or updates unacknowledged exception."""
    conn.execute(
        """
        INSERT INTO thesis_exceptions (
            exception_id, claim_id, ticker, detected_at_utc,
            exception_type, metric_name, expected_value, observed_value,
            source_citation, human_action, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(exception_id) DO UPDATE SET
            observed_value=excluded.observed_value,
            expected_value=excluded.expected_value,
            detected_at_utc=excluded.detected_at_utc
        """,
        (
            exc.exception_id,
            exc.claim_id,
            exc.ticker,
            exc.detected_at_utc,
            exc.exception_type,
            exc.metric_name,
            exc.expected_value,
            exc.observed_value,
            exc.source_citation,
            exc.human_action,
            exc.status,
        ),
    )


def list_pending_exceptions(conn: sqlite3.Connection, ticker: Optional[str] = None, limit: int = 50) -> List[ThesisExceptionRow]:
    """Retrieves pending exceptions requiring PM review."""
    conn.row_factory = sqlite3.Row
    if ticker:
        cursor = conn.execute(
            """
            SELECT * FROM thesis_exceptions
            WHERE ticker = ? AND status = 'PENDING'
            ORDER BY detected_at_utc DESC
            LIMIT ?
            """,
            (ticker.strip().upper(), limit),
        )
    else:
        cursor = conn.execute(
            """
            SELECT * FROM thesis_exceptions
            WHERE status = 'PENDING'
            ORDER BY detected_at_utc DESC
            LIMIT ?
            """,
            (limit,),
        )
    results: List[ThesisExceptionRow] = []
    for row in cursor.fetchall():
        results.append(
            ThesisExceptionRow(
                exception_id=row["exception_id"],
                claim_id=row["claim_id"],
                ticker=row["ticker"],
                detected_at_utc=row["detected_at_utc"],
                exception_type=row["exception_type"],
                metric_name=row["metric_name"],
                expected_value=row["expected_value"],
                observed_value=row["observed_value"],
                source_citation=row["source_citation"],
                human_action=row["human_action"],
                status=row["status"],
            )
        )
    return results


def acknowledge_exception(conn: sqlite3.Connection, exception_id: str) -> None:
    """Updates exception status to ACKNOWLEDGED after human PM review."""
    conn.execute(
        "UPDATE thesis_exceptions SET status = 'ACKNOWLEDGED' WHERE exception_id = ?",
        (exception_id,),
    )



def save_packet_snapshot(
    conn: sqlite3.Connection,
    *,
    packet_id: str,
    portfolio_id: str,
    review_date: str,
    cutoff_utc: str,
    input_fingerprint_sha256: str,
    policy_version: str,
    packet_sha256: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist or read back an immutable packet revision."""
    conn.row_factory = sqlite3.Row
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    existing = conn.execute(
        """
        SELECT * FROM pm_review_packets
        WHERE portfolio_id=? AND review_date=? AND input_fingerprint_sha256=? AND policy_version=?
        """,
        (portfolio_id, review_date, input_fingerprint_sha256, policy_version),
    ).fetchone()
    if existing:
        if existing["packet_sha256"] != packet_sha256 or existing["payload_json"] != payload_json:
            raise ValueError("packet fingerprint already exists with different frozen content")
        item = dict(existing)
        item["payload"] = json.loads(item.pop("payload_json"))
        return item

    revision_row = conn.execute(
        "SELECT COALESCE(MAX(revision),0) FROM pm_review_packets WHERE portfolio_id=? AND review_date=?",
        (portfolio_id, review_date),
    ).fetchone()
    revision = int(revision_row[0]) + 1
    conn.execute(
        """
        INSERT INTO pm_review_packets
        (packet_id, portfolio_id, review_date, revision, cutoff_utc,
         input_fingerprint_sha256, policy_version, packet_sha256, payload_json, created_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            packet_id, portfolio_id, review_date, revision, cutoff_utc,
            input_fingerprint_sha256, policy_version, packet_sha256, payload_json,
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        ),
    )
    row = conn.execute(
        "SELECT * FROM pm_review_packets WHERE packet_id=?", (packet_id,)
    ).fetchone()
    item = dict(row)
    item["payload"] = json.loads(item.pop("payload_json"))
    return item


def get_packet_snapshot(
    conn: sqlite3.Connection,
    portfolio_id: str,
    packet_sha256: str,
) -> Optional[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT * FROM pm_review_packets
        WHERE portfolio_id=? AND packet_sha256=?
        ORDER BY revision DESC LIMIT 1
        """,
        (portfolio_id, packet_sha256),
    ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["payload"] = json.loads(item.pop("payload_json"))
    return item


def _decode_signoff(row: sqlite3.Row | tuple) -> Dict[str, Any]:
    keys = row.keys() if hasattr(row, "keys") else ()
    item = dict(row) if hasattr(row, "keys") else {}
    for sql_name, output_name in (
        ("verified_sample_ids_json", "verified_sample_ids"),
        ("acknowledged_exception_ids_json", "acknowledged_exception_ids"),
    ):
        raw = item.get(sql_name) if sql_name in keys else None
        if raw is None:
            item[output_name] = None
        else:
            try:
                value = json.loads(raw)
                item[output_name] = value if isinstance(value, list) else None
            except (TypeError, ValueError):
                item[output_name] = None
    return item


def insert_signoff(
    conn: sqlite3.Connection,
    signoff_id: str,
    review_date: str,
    signed_off_by: str,
    signed_off_at_utc: str,
    acknowledged_exceptions_count: int,
    notes: Optional[str],
    packet_sha256: str,
    portfolio_id: str = "default",
    verified_sample_ids: Optional[Sequence[str]] = None,
    acknowledged_exception_ids: Optional[Sequence[str]] = None,
    packet_id: Optional[str] = None,
    request_digest: Optional[str] = None,
    principal_subject_id: Optional[str] = None,
    principal_session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Append an immutable sign-off and return the canonical committed row."""
    conn.row_factory = sqlite3.Row
    samples_json = (
        json.dumps(list(verified_sample_ids), ensure_ascii=False, separators=(",", ":"))
        if verified_sample_ids is not None else None
    )
    acknowledgements = list(acknowledged_exception_ids) if acknowledged_exception_ids is not None else None
    if acknowledgements is not None and len(acknowledgements) != len(set(acknowledgements)):
        raise ValueError("duplicate exception acknowledgements are not allowed")
    if acknowledgements is not None and int(acknowledged_exceptions_count) != len(acknowledgements):
        raise ValueError("acknowledged exception count must match persisted acknowledgement IDs")
    ack_json = (
        json.dumps(acknowledgements, ensure_ascii=False, separators=(",", ":"))
        if acknowledgements is not None else None
    )
    if packet_id is not None:
        packet = conn.execute(
            """
            SELECT packet_sha256, portfolio_id, review_date
            FROM pm_review_packets WHERE packet_id=?
            """,
            (packet_id,),
        ).fetchone()
        if (
            packet is None
            or packet["packet_sha256"] != packet_sha256
            or packet["portfolio_id"] != portfolio_id
            or packet["review_date"] != review_date
        ):
            raise ValueError("sign-off must reference an existing packet in the same portfolio/date/hash scope")

    conn.execute(
        """
        INSERT INTO pm_review_signoffs (
            signoff_id, portfolio_id, review_date, signed_off_by, signed_off_at_utc,
            acknowledged_exceptions_count, notes, packet_sha256, verified_sample_ids_json,
            packet_id, acknowledged_exception_ids_json, request_digest,
            principal_subject_id, principal_session_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(portfolio_id, packet_sha256) DO NOTHING
        """,
        (
            signoff_id, portfolio_id, review_date, signed_off_by, signed_off_at_utc,
            acknowledged_exceptions_count, notes, packet_sha256, samples_json,
            packet_id, ack_json, request_digest, principal_subject_id, principal_session_id,
        ),
    )
    row = conn.execute(
        """
        SELECT * FROM pm_review_signoffs
        WHERE portfolio_id=? AND packet_sha256=?
        ORDER BY signed_off_at_utc, rowid LIMIT 1
        """,
        (portfolio_id, packet_sha256),
    ).fetchone()
    if row is None:
        raise RuntimeError("sign-off insert did not return its canonical row")
    return _decode_signoff(row)


def get_signoff(
    conn: sqlite3.Connection,
    review_date: str,
    portfolio_id: str = "default",
    packet_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch sign-off evidence scoped to portfolio and optional frozen packet hash."""
    conn.row_factory = sqlite3.Row
    if packet_sha256:
        row = conn.execute(
            """
            SELECT * FROM pm_review_signoffs
            WHERE portfolio_id=? AND packet_sha256=?
            ORDER BY signed_off_at_utc DESC, rowid DESC LIMIT 1
            """,
            (portfolio_id, packet_sha256),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT * FROM pm_review_signoffs
            WHERE portfolio_id=? AND review_date=?
            ORDER BY signed_off_at_utc DESC, rowid DESC LIMIT 1
            """,
            (portfolio_id, review_date),
        ).fetchone()
    return _decode_signoff(row) if row else None


def list_signoffs(
    conn: sqlite3.Connection,
    limit: int = 30,
    portfolio_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List historical approvals, preserving NULL for evidence absent in legacy rows."""
    conn.row_factory = sqlite3.Row
    if portfolio_id:
        rows = conn.execute(
            """
            SELECT * FROM pm_review_signoffs WHERE portfolio_id=?
            ORDER BY review_date DESC, signed_off_at_utc DESC, rowid DESC LIMIT ?
            """,
            (portfolio_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM pm_review_signoffs
            ORDER BY review_date DESC, signed_off_at_utc DESC, rowid DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_decode_signoff(row) for row in rows]
