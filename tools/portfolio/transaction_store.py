"""External transactional source for portfolio state.

Markdown remains the human-readable projection.  This store is deliberately
outside the Obsidian tree and records an append-only event stream plus
replayable state snapshots.  It gives sync/import services a durable recovery
source without making a Markdown file the transaction log.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from tools.archivist.runtime_layout import runtime_root_for
from tools.archivist.vault_paths import VaultPaths


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class PortfolioCheckpoint:
    portfolio_id: str
    sequence: int
    event_log_hash: str
    state_hash: str
    created_at: str


class PortfolioTransactionStore:
    """Append/replay portfolio transactions in external runtime storage."""

    def __init__(
        self,
        *,
        vault_paths: Optional[VaultPaths] = None,
        runtime_root: Optional[str | Path] = None,
        runtime_base: Optional[str | Path] = None,
    ) -> None:
        self.vault_paths = vault_paths or VaultPaths()
        if runtime_root is not None and runtime_base is not None:
            raise ValueError("provide either runtime_root (isolated override) or runtime_base (canonical base), not both")
        runtime = (
            Path(runtime_root).resolve()
            if runtime_root is not None
            else runtime_root_for(self.vault_paths.root, runtime_base, create=True)
        )
        if runtime.is_relative_to(self.vault_paths.root):
            raise ValueError(f"portfolio transaction runtime must be outside vault: {runtime}")
        self.runtime_root = runtime
        self.db_path = runtime / "portfolio" / "transactions.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS portfolio_events (
                    event_id TEXT PRIMARY KEY,
                    portfolio_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    UNIQUE(portfolio_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS idx_portfolio_events_stream
                    ON portfolio_events(portfolio_id, sequence);
                CREATE TABLE IF NOT EXISTS portfolio_command_receipts (
                    portfolio_id TEXT NOT NULL,
                    use_case TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    command_id TEXT NOT NULL,
                    checkpoint_sequence INTEGER NOT NULL,
                    checkpoint_state_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(portfolio_id, use_case, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS idx_portfolio_receipts_cmd
                    ON portfolio_command_receipts(portfolio_id, command_id);
                CREATE TABLE IF NOT EXISTS portfolio_cash_flow_events (
                    event_id TEXT PRIMARY KEY,
                    portfolio_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_digest TEXT NOT NULL,
                    action TEXT NOT NULL CHECK (action IN ('deposit','withdraw')),
                    amount_native_decimal TEXT NOT NULL,
                    currency TEXT NOT NULL CHECK (currency IN ('THB','USD')),
                    fx_rate_to_thb_decimal TEXT,
                    fx_source_id TEXT,
                    amount_thb_decimal TEXT,
                    occurred_at_utc TEXT NOT NULL,
                    recorded_at_utc TEXT NOT NULL,
                    source_state_sequence INTEGER NOT NULL,
                    source_state_sha256 TEXT NOT NULL,
                    flow_kind TEXT NOT NULL,
                    flow_coverage_status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE(portfolio_id, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS idx_cash_flow_portfolio_sequence
                    ON portfolio_cash_flow_events(portfolio_id, source_state_sequence);
                CREATE TRIGGER IF NOT EXISTS trg_cash_flow_events_immutable_update
                    BEFORE UPDATE ON portfolio_cash_flow_events
                    BEGIN SELECT RAISE(ABORT, 'cash flow events are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS trg_cash_flow_events_immutable_delete
                    BEFORE DELETE ON portfolio_cash_flow_events
                    BEGIN SELECT RAISE(ABORT, 'cash flow events are append-only'); END;
                """
            )

    def append_event(
        self,
        portfolio_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        event_id: Optional[str] = None,
        expected_sequence: Optional[int] = None,
        command_receipt: Optional[Mapping[str, Any]] = None,
        related_events: Optional[list[Mapping[str, Any]]] = None,
    ) -> int:
        portfolio_id = str(portfolio_id or "").strip()
        event_type = str(event_type or "").strip()
        if not portfolio_id or not event_type:
            raise ValueError("portfolio_id and event_type are required")
        if not isinstance(payload, Mapping):
            raise TypeError("portfolio transaction payload must be a mapping")
        event_id = str(event_id or f"evt_{uuid.uuid4().hex}")
        payload_json = _canonical(dict(payload))
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT sequence FROM portfolio_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if existing is not None:
                conn.commit()
                return int(existing["sequence"])
            row = conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM portfolio_events WHERE portfolio_id=?",
                (portfolio_id,),
            ).fetchone()
            current = int(row["sequence"] if row else 0)
            if expected_sequence is not None and current != int(expected_sequence):
                conn.rollback()
                raise RuntimeError(
                    f"portfolio transaction conflict for {portfolio_id}: expected sequence {expected_sequence}, actual {current}"
                )
            sequence = current + 1
            conn.execute(
                "INSERT INTO portfolio_events(event_id, portfolio_id, sequence, event_type, payload_json, occurred_at) VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, portfolio_id, sequence, event_type, payload_json, now),
            )
            for related in related_events or []:
                related_type = str(related.get("event_type") or "").strip()
                related_payload = dict(related.get("payload") or {})
                if not related_type or not related_payload:
                    raise ValueError("related portfolio events require event_type and payload")
                if related_type == "cash_flow_recorded":
                    idempotency_key = str(related_payload.get("idempotency_key") or "").strip()
                    if not idempotency_key:
                        raise ValueError("cash flow event requires idempotency_key")
                    prior_flow = conn.execute(
                        "SELECT request_digest FROM portfolio_cash_flow_events WHERE portfolio_id=? AND idempotency_key=?",
                        (portfolio_id, idempotency_key),
                    ).fetchone()
                    if prior_flow:
                        raise RuntimeError("cash flow idempotency key was already committed")
                source_state_sequence = sequence
                related_payload["source_state_sequence"] = source_state_sequence
                related_payload["source_state_sha256"] = hashlib.sha256(
                    _canonical(dict(payload)).encode("utf-8")
                ).hexdigest()
                related_payload.setdefault("portfolio_id", portfolio_id)
                related_payload.setdefault("recorded_at_utc", now)
                related_id = str(related_payload.get("event_id") or f"evt_{uuid.uuid4().hex}")
                related_json = _canonical(related_payload)
                sequence += 1
                conn.execute(
                    "INSERT INTO portfolio_events(event_id, portfolio_id, sequence, event_type, payload_json, occurred_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (related_id, portfolio_id, sequence, related_type, related_json, now),
                )
                if related_type == "cash_flow_recorded":
                    conn.execute(
                        """
                        INSERT INTO portfolio_cash_flow_events
                        (event_id, portfolio_id, idempotency_key, request_digest, action,
                         amount_native_decimal, currency, fx_rate_to_thb_decimal, fx_source_id,
                         amount_thb_decimal, occurred_at_utc, recorded_at_utc,
                         source_state_sequence, source_state_sha256, flow_kind,
                         flow_coverage_status, payload_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            related_id, portfolio_id, idempotency_key,
                            str(related_payload["request_digest"]), str(related_payload["action"]),
                            str(related_payload["amount_native_decimal"]), str(related_payload["currency"]),
                            related_payload.get("fx_rate_to_thb_decimal"), related_payload.get("fx_source_id"),
                            related_payload.get("amount_thb_decimal"), str(related_payload["occurred_at_utc"]),
                            str(related_payload["recorded_at_utc"]), source_state_sequence,
                            str(related_payload["source_state_sha256"]), str(related_payload["flow_kind"]),
                            str(related_payload["flow_coverage_status"]), related_json,
                        ),
                    )
            if command_receipt and command_receipt.get("idempotency_key"):
                conn.execute(
                    """
                    INSERT INTO portfolio_command_receipts
                    (portfolio_id, use_case, idempotency_key, request_hash, command_id,
                     checkpoint_sequence, checkpoint_state_hash, result_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(portfolio_id),
                        str(command_receipt.get("use_case", "apply_allocation")),
                        str(command_receipt.get("idempotency_key", "")),
                        str(command_receipt.get("request_hash", "")),
                        str(command_receipt.get("command_id", "")),
                        sequence,
                        str(command_receipt.get("checkpoint_state_hash", "")),
                        _canonical(command_receipt.get("result", {})),
                        now,
                    ),
                )
            conn.commit()
        return sequence

    def append_state(
        self,
        portfolio_id: str,
        state: Mapping[str, Any],
        *,
        ledger_rows: Optional[list[Mapping[str, Any]]] = None,
        event_id: Optional[str] = None,
        expected_sequence: Optional[int] = None,
        command_receipt: Optional[Mapping[str, Any]] = None,
        related_events: Optional[list[Mapping[str, Any]]] = None,
    ) -> int:
        return self.append_event(
            portfolio_id,
            "state_snapshot",
            {
                "state": dict(state),
                "ledger_rows": [dict(row) for row in (ledger_rows or [])],
            },
            event_id=event_id,
            expected_sequence=expected_sequence,
            command_receipt=command_receipt,
            related_events=related_events,
        )

    def cash_flow_coverage_between(
        self,
        portfolio_id: str,
        *,
        after_sequence: int,
        through_sequence: int,
    ) -> bool:
        """True only when every state snapshot in the interval has a durable event explanation."""
        events = self.events(portfolio_id, through_sequence=through_sequence)
        explained = {
            int(event["payload"]["source_state_sequence"])
            for event in events
            if event["event_type"] != "state_snapshot"
            and isinstance(event.get("payload"), dict)
            and str(event["payload"].get("source_state_sequence", "")).isdigit()
        }
        for event in events:
            sequence = int(event["sequence"])
            if sequence <= int(after_sequence) or sequence > int(through_sequence):
                continue
            if event["event_type"] != "state_snapshot":
                continue
            # The final snapshot is the performance capture itself; it has no preceding user cash flow.
            if sequence == int(through_sequence):
                continue
            if sequence not in explained:
                return False
        return True

    def get_cash_flow_event(self, portfolio_id: str, idempotency_key: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM portfolio_cash_flow_events WHERE portfolio_id=? AND idempotency_key=?",
                (str(portfolio_id), str(idempotency_key)),
            ).fetchone()
            if not row:
                return None
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            return item

    def list_cash_flow_events(
        self,
        portfolio_id: str,
        *,
        after_sequence: Optional[int] = None,
        through_sequence: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        clauses = ["portfolio_id=?"]
        params: list[Any] = [str(portfolio_id)]
        if after_sequence is not None:
            clauses.append("source_state_sequence > ?")
            params.append(int(after_sequence))
        if through_sequence is not None:
            clauses.append("source_state_sequence <= ?")
            params.append(int(through_sequence))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM portfolio_cash_flow_events WHERE "
                + " AND ".join(clauses)
                + " ORDER BY source_state_sequence, event_id",
                params,
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["payload"] = json.loads(item.pop("payload_json"))
                result.append(item)
            return result

    def get_command_receipt(
        self,
        portfolio_id: str,
        use_case: str,
        idempotency_key: str,
    ) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT portfolio_id, use_case, idempotency_key, request_hash,
                       command_id, checkpoint_sequence, checkpoint_state_hash,
                       result_json, created_at
                FROM portfolio_command_receipts
                WHERE portfolio_id=? AND use_case=? AND idempotency_key=?
                """,
                (str(portfolio_id), str(use_case), str(idempotency_key)),
            ).fetchone()
            if not row:
                return None
            return {
                "portfolio_id": str(row["portfolio_id"]),
                "use_case": str(row["use_case"]),
                "idempotency_key": str(row["idempotency_key"]),
                "request_hash": str(row["request_hash"]),
                "command_id": str(row["command_id"]),
                "checkpoint_sequence": int(row["checkpoint_sequence"]),
                "checkpoint_state_hash": str(row["checkpoint_state_hash"]),
                "result": json.loads(str(row["result_json"])),
                "created_at": str(row["created_at"]),
            }

    def get_command_receipt_by_command_id(
        self,
        portfolio_id: str,
        command_id: str,
    ) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT portfolio_id, use_case, idempotency_key, request_hash,
                       command_id, checkpoint_sequence, checkpoint_state_hash,
                       result_json, created_at
                FROM portfolio_command_receipts
                WHERE portfolio_id=? AND command_id=?
                ORDER BY checkpoint_sequence DESC LIMIT 1
                """,
                (str(portfolio_id), str(command_id)),
            ).fetchone()
            if not row:
                return None
            return {
                "portfolio_id": str(row["portfolio_id"]),
                "use_case": str(row["use_case"]),
                "idempotency_key": str(row["idempotency_key"]),
                "request_hash": str(row["request_hash"]),
                "command_id": str(row["command_id"]),
                "checkpoint_sequence": int(row["checkpoint_sequence"]),
                "checkpoint_state_hash": str(row["checkpoint_state_hash"]),
                "result": json.loads(str(row["result_json"])),
                "created_at": str(row["created_at"]),
            }

    def events(self, portfolio_id: str, *, through_sequence: Optional[int] = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM portfolio_events WHERE portfolio_id=?"
        params: list[Any] = [str(portfolio_id)]
        if through_sequence is not None:
            query += " AND sequence <= ?"
            params.append(int(through_sequence))
        query += " ORDER BY sequence ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "event_id": str(row["event_id"]),
                "portfolio_id": str(row["portfolio_id"]),
                "sequence": int(row["sequence"]),
                "event_type": str(row["event_type"]),
                "payload": json.loads(str(row["payload_json"])),
                "occurred_at": str(row["occurred_at"]),
            }
            for row in rows
        ]

    def replay(self, portfolio_id: str, *, through_sequence: Optional[int] = None) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for event in self.events(portfolio_id, through_sequence=through_sequence):
            payload = event["payload"]
            if event["event_type"] == "state_snapshot":
                state = dict(payload.get("state") or {})
            elif event["event_type"] in {"state_patch", "transaction"}:
                state.update(dict(payload.get("state_patch") or payload.get("state") or {}))
            elif event["event_type"] == "delete_field":
                state.pop(str(payload.get("field") or ""), None)
        return state

    def replay_ledger(self, portfolio_id: str, *, through_sequence: Optional[int] = None) -> list[dict[str, Any]]:
        """Replay the transactional trade ledger kept beside portfolio state."""
        rows: list[dict[str, Any]] = []
        for event in self.events(portfolio_id, through_sequence=through_sequence):
            payload = event["payload"]
            if event["event_type"] == "state_snapshot" and "ledger_rows" in payload:
                rows = [dict(row) for row in (payload.get("ledger_rows") or [])]
            elif event["event_type"] == "ledger_replace":
                rows = [dict(row) for row in (payload.get("rows") or [])]
            elif event["event_type"] == "ledger_append" and payload.get("row") is not None:
                rows.append(dict(payload["row"]))
        return rows

    def checkpoint(self, portfolio_id: str) -> PortfolioCheckpoint:
        events = self.events(portfolio_id)
        state = self.replay(portfolio_id)
        event_log_hash = hashlib.sha256(_canonical(events).encode("utf-8")).hexdigest()
        state_hash = hashlib.sha256(_canonical(state).encode("utf-8")).hexdigest()
        return PortfolioCheckpoint(
            portfolio_id=str(portfolio_id),
            sequence=int(events[-1]["sequence"]) if events else 0,
            event_log_hash=event_log_hash,
            state_hash=state_hash,
            created_at=_utc_now(),
        )
