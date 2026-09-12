"""Durable, idempotent broker for application-level Vault write commands.

The broker's queue and receipts live outside the Obsidian tree.  A command is
accepted into SQLite before execution, claimed with a renewable lease and a
monotonic fencing token, then committed through one injected executor.  A
restarted process can safely reclaim expired work because the canonical
``ArtifactWriter`` is itself content-addressed and optimistic-concurrency
aware.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

from application.knowledge.errors import WriteUnavailableError
from application.knowledge.write_models import KnowledgeWriteCommand, KnowledgeWriteReceipt
from application.knowledge.write_ports import KnowledgeWritePort
from tools.archivist.artifact_writer import StaleWriteConflictError
from tools.archivist.maintenance_guard import MaintenanceLeaseConflictError
from tools.archivist.schema_registry import SchemaRegistry, load_default_registry
from tools.archivist.write_adapter import AdapterCommit, ArtifactWriterKnowledgeAdapter, WriteAdapterError
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.runtime_layout import runtime_root_for


class KnowledgeWriteExecutor(Protocol):
    def commit(self, command: KnowledgeWriteCommand, *, fencing_token: int = 0) -> AdapterCommit:
        ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _after(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=max(0.0, seconds))).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS broker_commands (
    command_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    command_fingerprint TEXT NOT NULL,
    command_json TEXT NOT NULL,
    operation TEXT NOT NULL,
    producer TEXT NOT NULL,
    document_key TEXT,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    available_at TEXT,
    lease_owner TEXT,
    lease_until TEXT,
    fencing_token INTEGER NOT NULL DEFAULT 0,
    accepted_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    committed_at TEXT,
    result_json TEXT,
    error_code TEXT,
    error_message TEXT,
    retryable INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_broker_commands_ready
    ON broker_commands(status, available_at, accepted_at);
CREATE INDEX IF NOT EXISTS idx_broker_commands_lease
    ON broker_commands(lease_until);
CREATE TABLE IF NOT EXISTS broker_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    command_id TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    fencing_token INTEGER,
    detail_json TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_broker_events_command
    ON broker_events(command_id, event_id);
"""


class KnowledgeWriteBroker(KnowledgeWritePort):
    """SQLite-backed broker implementing the application write port."""

    def __init__(
        self,
        *,
        vault_paths: Optional[VaultPaths] = None,
        executor: Optional[KnowledgeWriteExecutor] = None,
        registry: Optional[SchemaRegistry] = None,
        runtime_root: Optional[str | Path] = None,
        runtime_base: Optional[str | Path] = None,
        broker_id: Optional[str] = None,
        lease_seconds: int = 120,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        self.vault_paths = vault_paths or VaultPaths()
        self.registry = registry or load_default_registry()
        self.broker_id = broker_id or f"broker_{uuid.uuid4().hex[:12]}"
        self.lease_seconds = max(1, int(lease_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        if runtime_root is not None and runtime_base is not None:
            raise ValueError("provide either runtime_root (isolated override) or runtime_base (canonical base), not both")
        runtime = (
            Path(runtime_root).resolve()
            if runtime_root is not None
            else runtime_root_for(self.vault_paths.root, runtime_base, create=True)
        )
        if runtime.is_relative_to(self.vault_paths.root):
            raise ValueError(f"write broker runtime must be outside vault: {runtime}")
        runtime.mkdir(parents=True, exist_ok=True)
        self.runtime_root = runtime
        self.db_path = runtime / "broker" / "knowledge_write.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.executor = executor or ArtifactWriterKnowledgeAdapter(
            vault_paths=self.vault_paths,
            registry=self.registry,
            fencing_guard=self._assert_fencing_token,
        )
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def submit(self, command: KnowledgeWriteCommand) -> KnowledgeWriteReceipt:
        if not isinstance(command, KnowledgeWriteCommand):
            raise TypeError("KnowledgeWriteBroker.submit requires KnowledgeWriteCommand")
        now = _utc_now()
        command_json = _json(command.to_dict())
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM broker_commands WHERE idempotency_key = ?",
                (command.idempotency_key,),
            ).fetchone()
            if existing is not None:
                conn.commit()
                if existing["command_fingerprint"] != command.command_fingerprint:
                    return self._conflict_receipt(
                        command,
                        code="idempotency_key_reused",
                        message="idempotency_key already belongs to a different command fingerprint",
                    )
                existing_receipt = self._row_to_receipt(existing)
                if existing_receipt is not None:
                    if existing_receipt.status == "committed":
                        return replace(
                            existing_receipt,
                            status="duplicate_reused",
                            warnings=tuple(existing_receipt.warnings) + ("idempotent_replay",),
                        )
                    return existing_receipt
                return self._row_to_receipt(existing, fallback_command=command)  # type: ignore[return-value]
            conn.execute(
                """
                INSERT INTO broker_commands (
                    command_id, idempotency_key, command_fingerprint, command_json,
                    operation, producer, document_key, status, accepted_at, updated_at,
                    available_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'accepted', ?, ?, ?)
                """,
                (
                    command.command_id,
                    command.idempotency_key,
                    command.command_fingerprint,
                    command_json,
                    command.operation,
                    command.producer,
                    command.document_key,
                    now,
                    now,
                    now,
                ),
            )
            self._event(conn, command.command_id, None, "accepted", None, {"producer": command.producer})
            conn.commit()

        # Synchronous execution provides a simple local transport while the
        # persisted row still makes a process crash recoverable.
        self.drain(limit=1, command_id=command.command_id)
        receipt = self.get_receipt(command_id=command.command_id)
        if receipt is None:
            raise WriteUnavailableError(f"write receipt disappeared for {command.command_id}")
        return receipt

    def get_receipt(
        self,
        *,
        command_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Optional[KnowledgeWriteReceipt]:
        if not command_id and not idempotency_key:
            raise ValueError("command_id or idempotency_key is required")
        with self._connect() as conn:
            if command_id:
                row = conn.execute("SELECT * FROM broker_commands WHERE command_id = ?", (command_id,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM broker_commands WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
        return self._row_to_receipt(row) if row is not None else None

    def drain(self, *, limit: int = 100, command_id: Optional[str] = None) -> int:
        processed = 0
        for _ in range(max(0, int(limit))):
            claimed = self._claim(command_id=command_id)
            if claimed is None:
                break
            row, token = claimed
            self._execute_claim(row, token)
            processed += 1
            if command_id:
                break
        return processed

    def recover_expired(self, *, limit: int = 100) -> int:
        """Re-run accepted, retryable, and expired leases after a restart."""
        return self.drain(limit=limit)

    def pending_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM broker_commands WHERE status IN ('accepted', 'retry_wait', 'leased', 'committing')"
            ).fetchone()
        return int(row["count"] if row else 0)

    def health(self) -> dict[str, Any]:
        """Return operator-safe queue health without exposing command payloads."""
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            counts = {
                str(row["status"]): int(row["count"])
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM broker_commands GROUP BY status"
                ).fetchall()
            }
            oldest = conn.execute(
                """
                SELECT accepted_at FROM broker_commands
                WHERE status IN ('accepted', 'retry_wait', 'leased', 'committing')
                ORDER BY accepted_at ASC LIMIT 1
                """
            ).fetchone()
        oldest_age = None
        if oldest and oldest["accepted_at"]:
            try:
                accepted = datetime.fromisoformat(str(oldest["accepted_at"]).replace("Z", "+00:00"))
                oldest_age = max(0.0, (now - accepted).total_seconds())
            except ValueError:
                oldest_age = None
        return {
            "broker_id": self.broker_id,
            "db_path": str(self.db_path),
            "queue_depth": sum(counts.get(status, 0) for status in ("accepted", "retry_wait", "leased", "committing")),
            "oldest_pending_age_seconds": oldest_age,
            "status_counts": counts,
            "conflicts": counts.get("conflict", 0),
            "dead_letters": counts.get("dead_letter", 0),
        }

    def retry(self, command_id: str) -> Optional[KnowledgeWriteReceipt]:
        """Operator retry for retryable/dead-letter work with an audit event."""
        command_id = str(command_id or "").strip()
        if not command_id:
            raise ValueError("command_id is required")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status, fencing_token FROM broker_commands WHERE command_id=?",
                (command_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            status = str(row["status"])
            if status not in {"retry_wait", "dead_letter"}:
                conn.commit()
                return self.get_receipt(command_id=command_id)
            conn.execute(
                """
                UPDATE broker_commands
                SET status='accepted', available_at=?, retryable=0, error_code=NULL,
                    error_message=NULL, updated_at=?, lease_owner=NULL, lease_until=NULL
                WHERE command_id=? AND status=?
                """,
                (now, now, command_id, status),
            )
            self._event(conn, command_id, status, "accepted", int(row["fencing_token"] or 0), {"operator_retry": True})
            conn.commit()
        self.drain(limit=1, command_id=command_id)
        return self.get_receipt(command_id=command_id)

    def _assert_fencing_token(self, command_id: str, token: int) -> None:
        """Fail closed when a lease has expired or another worker fenced it."""
        if not command_id:
            return
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status, lease_owner, lease_until, fencing_token FROM broker_commands WHERE command_id=?",
                (command_id,),
            ).fetchone()
        if row is None or str(row["lease_owner"] or "") != self.broker_id or int(row["fencing_token"] or 0) != int(token):
            raise StaleWriteConflictError(
                f"stale broker fencing token for {command_id}: token={token}, owner={self.broker_id}"
            )
        lease_until = str(row["lease_until"] or "")
        try:
            expired = datetime.fromisoformat(lease_until.replace("Z", "+00:00")) <= datetime.now(timezone.utc)
        except ValueError:
            expired = True
        if str(row["status"]) not in {"leased", "committing"} or expired:
            raise StaleWriteConflictError(f"broker lease is no longer valid for {command_id}")

    def _claim(self, *, command_id: Optional[str]) -> Optional[tuple[sqlite3.Row, int]]:
        now = _utc_now()
        lease_until = _after(self.lease_seconds)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            query = """
                SELECT * FROM broker_commands
                WHERE ((
                    status IN ('accepted', 'retry_wait')
                    AND (available_at IS NULL OR available_at <= ?)
                ) OR (
                    status IN ('leased', 'committing')
                    AND lease_until IS NOT NULL AND lease_until <= ?
                ))
            """
            params: list[Any] = [now, now]
            if command_id:
                query += " AND command_id = ?"
                params.append(command_id)
            query += " ORDER BY accepted_at ASC LIMIT 1"
            row = conn.execute(query, params).fetchone()
            if row is None:
                conn.commit()
                return None
            token = int(row["fencing_token"] or 0) + 1
            previous = str(row["status"])
            conn.execute(
                """
                UPDATE broker_commands
                SET status='leased', attempts=attempts+1, lease_owner=?, lease_until=?,
                    fencing_token=?, updated_at=?
                WHERE command_id=? AND status=? AND fencing_token=?
                """,
                (self.broker_id, lease_until, token, now, row["command_id"], previous, token - 1),
            )
            if conn.execute("SELECT changes()").fetchone()[0] != 1:
                conn.rollback()
                return None
            self._event(conn, row["command_id"], previous, "leased", token, {"owner": self.broker_id})
            conn.commit()
            updated = conn.execute("SELECT * FROM broker_commands WHERE command_id = ?", (row["command_id"],)).fetchone()
            return (updated or row), token

    def _execute_claim(self, row: sqlite3.Row, token: int) -> None:
        command = KnowledgeWriteCommand.from_dict(json.loads(str(row["command_json"])))
        now = _utc_now()
        if not self._transition(row["command_id"], token, "committing", None, None):
            return
        try:
            result = self.executor.commit(command, fencing_token=token)
        except StaleWriteConflictError as exc:
            self._finish(
                row["command_id"], token, "conflict", error_code="stale_write", error_message=str(exc), retryable=False
            )
        except WriteAdapterError as exc:
            self._finish(
                row["command_id"], token, "rejected", error_code="adapter_rejected", error_message=str(exc), retryable=False
            )
        except MaintenanceLeaseConflictError as exc:
            self._finish(
                row["command_id"], token, "retry_wait", error_code="maintenance_lease", error_message=str(exc),
                retryable=True, available_at=_after(self.retry_backoff_seconds),
            )
        except (ValueError, TypeError, KeyError) as exc:
            self._finish(
                row["command_id"], token, "rejected", error_code="invalid_payload", error_message=str(exc), retryable=False
            )
        except Exception as exc:  # noqa: BLE001 - durable retry boundary
            attempts = int(row["attempts"] or 0)
            if attempts >= self.max_attempts:
                self._finish(
                    row["command_id"], token, "dead_letter", error_code="max_attempts", error_message=str(exc), retryable=False
                )
            else:
                self._finish(
                    row["command_id"], token, "retry_wait", error_code="transient_error", error_message=str(exc), retryable=True,
                    available_at=_after(self.retry_backoff_seconds * max(1, attempts)),
                )
        else:
            receipt = KnowledgeWriteReceipt(
                command_id=command.command_id,
                idempotency_key=command.idempotency_key,
                status="committed",
                operation=command.operation,
                producer=command.producer,
                document_key=command.document_key,
                note_id=result.note_id,
                revision_id=result.revision_id,
                relative_path=result.relative_path,
                content_hash=result.content_hash,
                artifact_set_hash=result.artifact_set_hash,
                committed_at=now,
                retryable=False,
                warnings=result.warnings,
                registry_digest=self.registry.digest(),
                broker_fencing_token=token,
            )
            self._finish(row["command_id"], token, "committed", result_json=_json(receipt.to_dict()), receipt=receipt)
    def _transition(
        self,
        command_id: str,
        token: int,
        status: str,
        error_code: Optional[str],
        error_message: Optional[str],
    ) -> bool:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status FROM broker_commands WHERE command_id=? AND fencing_token=? AND lease_owner=?", (command_id, token, self.broker_id)).fetchone()
            if row is None:
                conn.rollback()
                return False
            previous = str(row["status"])
            conn.execute(
                "UPDATE broker_commands SET status=?, updated_at=?, error_code=?, error_message=? WHERE command_id=? AND fencing_token=? AND lease_owner=?",
                (status, _utc_now(), error_code, error_message, command_id, token, self.broker_id),
            )
            self._event(conn, command_id, previous, status, token, {"owner": self.broker_id})
            conn.commit()
            return True

    def _finish(
        self,
        command_id: str,
        token: int,
        status: str,
        *,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        retryable: bool = False,
        available_at: Optional[str] = None,
        result_json: Optional[str] = None,
        receipt: Optional[KnowledgeWriteReceipt] = None,
    ) -> bool:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status FROM broker_commands WHERE command_id=? AND fencing_token=? AND lease_owner=?", (command_id, token, self.broker_id)).fetchone()
            if row is None:
                conn.rollback()
                return False
            previous = str(row["status"])
            conn.execute(
                """
                UPDATE broker_commands
                SET status=?, updated_at=?, committed_at=?, result_json=?, error_code=?, error_message=?,
                    retryable=?, available_at=?, lease_owner=NULL, lease_until=NULL
                WHERE command_id=? AND fencing_token=? AND lease_owner=?
                """,
                (
                    status,
                    _utc_now(),
                    _utc_now() if status == "committed" else None,
                    result_json or (_json(receipt.to_dict()) if receipt else None),
                    error_code,
                    error_message,
                    1 if retryable else 0,
                    available_at,
                    command_id,
                    token,
                    self.broker_id,
                ),
            )
            self._event(conn, command_id, previous, status, token, {"error_code": error_code, "retryable": retryable})
            conn.commit()
            return True

    @staticmethod
    def _event(
        conn: sqlite3.Connection,
        command_id: str,
        from_status: Optional[str],
        to_status: str,
        token: Optional[int],
        detail: Optional[dict[str, Any]],
    ) -> None:
        conn.execute(
            "INSERT INTO broker_events(command_id, from_status, to_status, fencing_token, detail_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (command_id, from_status, to_status, token, _json(detail or {}), _utc_now()),
        )

    def _conflict_receipt(self, command: KnowledgeWriteCommand, *, code: str, message: str) -> KnowledgeWriteReceipt:
        return KnowledgeWriteReceipt(
            command_id=command.command_id,
            idempotency_key=command.idempotency_key,
            status="conflict",
            operation=command.operation,
            producer=command.producer,
            document_key=command.document_key,
            conflict_code=code,
            error_code=code,
            error_message=message,
            retryable=False,
            registry_digest=self.registry.digest(),
        )

    @staticmethod
    def _row_to_receipt(
        row: sqlite3.Row,
        *,
        fallback_command: Optional[KnowledgeWriteCommand] = None,
    ) -> Optional[KnowledgeWriteReceipt]:
        result_json = row["result_json"]
        if result_json:
            try:
                payload = json.loads(str(result_json))
                if isinstance(payload, dict):
                    return KnowledgeWriteReceipt(
                        command_id=str(payload.get("command_id") or row["command_id"]),
                        idempotency_key=str(payload.get("idempotency_key") or row["idempotency_key"]),
                        status=str(payload.get("status") or row["status"]),
                        operation=str(payload.get("operation") or row["operation"]),
                        producer=str(payload.get("producer") or row["producer"]),
                        document_key=payload.get("document_key"),
                        note_id=payload.get("note_id"),
                        revision_id=payload.get("revision_id"),
                        relative_path=payload.get("relative_path"),
                        content_hash=payload.get("content_hash"),
                        artifact_set_hash=payload.get("artifact_set_hash"),
                        committed_at=payload.get("committed_at"),
                        conflict_code=payload.get("conflict_code"),
                        retryable=bool(payload.get("retryable")),
                        warnings=tuple(payload.get("warnings") or ()),
                        error_code=payload.get("error_code"),
                        error_message=payload.get("error_message"),
                        registry_digest=payload.get("registry_digest"),
                        broker_fencing_token=payload.get("broker_fencing_token"),
                    )
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        command = fallback_command
        if command is None:
            try:
                command = KnowledgeWriteCommand.from_dict(json.loads(str(row["command_json"])))
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
        return KnowledgeWriteReceipt(
            command_id=command.command_id,
            idempotency_key=command.idempotency_key,
            status=str(row["status"]),
            operation=command.operation,
            producer=command.producer,
            document_key=command.document_key,
            conflict_code=str(row["error_code"]) if row["status"] == "conflict" and row["error_code"] else None,
            retryable=bool(row["retryable"]),
            error_code=row["error_code"],
            error_message=row["error_message"],
        )
