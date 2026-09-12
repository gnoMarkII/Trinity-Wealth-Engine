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
        )

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
