"""Pure command and receipt models for multi-app Vault writes."""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Optional


WriteOperation = Literal[
    "upsert_note",
    "publish_capture",
    "append_journal_entry",
    "retire_note",
    "restore_note",
    "register_attachment",
    "regenerate_projection",
]

WriteStatus = Literal[
    "accepted",
    "validated",
    "leased",
    "committing",
    "committed",
    "duplicate_reused",
    "conflict",
    "rejected",
    "retry_wait",
    "dead_letter",
]

ALLOWED_OPERATIONS = frozenset(
    {
        "upsert_note",
        "publish_capture",
        "append_journal_entry",
        "retire_note",
        "restore_note",
        "register_attachment",
        "regenerate_projection",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_required(value: Any, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field} is required")
    return result


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class KnowledgeWriteCommand:
    """A path-independent, idempotent request to mutate canonical storage."""

    operation: WriteOperation
    payload: Mapping[str, Any]
    idempotency_key: str
    document_key: Optional[str] = None
    entity_type: Optional[str] = None
    command_id: str = field(default_factory=lambda: f"cmd_{uuid.uuid4().hex}")
    protocol_version: int = 1
    producer: str = "unknown"
    producer_version: str = "unknown"
    expected_revision_id: Optional[str] = None
    expected_content_hash: Optional[str] = None
    causation_id: Optional[str] = None
    correlation_id: Optional[str] = None
    submitted_at: str = field(default_factory=_utc_now)
    actor: str = "application"

    def __post_init__(self) -> None:
        operation = _clean_required(self.operation, "operation")
        if operation not in ALLOWED_OPERATIONS:
            raise ValueError(f"Unsupported Vault write operation: {operation}")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "idempotency_key", _clean_required(self.idempotency_key, "idempotency_key"))
        object.__setattr__(self, "command_id", _clean_required(self.command_id, "command_id"))
        object.__setattr__(self, "producer", _clean_required(self.producer, "producer"))
        object.__setattr__(self, "actor", _clean_required(self.actor, "actor"))
        if self.protocol_version != 1:
            raise ValueError(f"Unsupported write protocol version: {self.protocol_version}")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        if len(_canonical_json(self.payload).encode("utf-8")) > 5 * 1024 * 1024:
            raise ValueError("write payload exceeds the 5 MiB command limit")
        if self.document_key is not None:
            object.__setattr__(self, "document_key", str(self.document_key).strip() or None)
        if self.entity_type is not None:
            object.__setattr__(self, "entity_type", str(self.entity_type).strip().lower() or None)

    @property
    def payload_hash(self) -> str:
        return hashlib.sha256(_canonical_json(dict(self.payload)).encode("utf-8")).hexdigest()

    @property
    def command_fingerprint(self) -> str:
        value = {
            "protocol_version": self.protocol_version,
            "operation": self.operation,
            "payload_hash": self.payload_hash,
            "document_key": self.document_key,
            "entity_type": self.entity_type,
            "idempotency_key": self.idempotency_key,
        }
        return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "command_id": self.command_id,
            "idempotency_key": self.idempotency_key,
            "operation": self.operation,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "document_key": self.document_key,
            "entity_type": self.entity_type,
            "payload": dict(self.payload),
            "payload_hash": self.payload_hash,
            "expected_revision_id": self.expected_revision_id,
            "expected_content_hash": self.expected_content_hash,
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "submitted_at": self.submitted_at,
            "actor": self.actor,
            "command_fingerprint": self.command_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KnowledgeWriteCommand":
        if not isinstance(value, Mapping):
            raise TypeError("write command must be an object")
        allowed = {
            "operation",
            "payload",
            "idempotency_key",
            "document_key",
            "entity_type",
            "command_id",
            "protocol_version",
            "producer",
            "producer_version",
            "expected_revision_id",
            "expected_content_hash",
            "causation_id",
            "correlation_id",
            "submitted_at",
            "actor",
        }
        kwargs = {key: value[key] for key in allowed if key in value}
        return cls(**kwargs)


@dataclass(frozen=True)
class KnowledgeWriteReceipt:
    """Immutable result envelope returned by a write adapter or broker."""

    command_id: str
    idempotency_key: str
    status: WriteStatus
    operation: str
    producer: str
    document_key: Optional[str] = None
    note_id: Optional[str] = None
    revision_id: Optional[str] = None
    relative_path: Optional[str] = None
    content_hash: Optional[str] = None
    artifact_set_hash: Optional[str] = None
    committed_at: Optional[str] = None
    conflict_code: Optional[str] = None
    retryable: bool = False
    warnings: tuple[str, ...] = ()
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    registry_digest: Optional[str] = None
    broker_fencing_token: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.command_id or not self.idempotency_key:
            raise ValueError("receipt requires command_id and idempotency_key")
        if self.status not in {
            "accepted",
            "validated",
            "leased",
            "committing",
            "committed",
            "duplicate_reused",
            "conflict",
            "rejected",
            "retry_wait",
            "dead_letter",
        }:
            raise ValueError(f"Unknown receipt status: {self.status}")

    @property
    def is_success(self) -> bool:
        return self.status in {"committed", "duplicate_reused"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "idempotency_key": self.idempotency_key,
            "status": self.status,
            "operation": self.operation,
            "producer": self.producer,
            "document_key": self.document_key,
            "note_id": self.note_id,
            "revision_id": self.revision_id,
            "relative_path": self.relative_path,
            "content_hash": self.content_hash,
            "artifact_set_hash": self.artifact_set_hash,
            "committed_at": self.committed_at,
            "conflict_code": self.conflict_code,
            "retryable": self.retryable,
            "warnings": list(self.warnings),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "registry_digest": self.registry_digest,
            "broker_fencing_token": self.broker_fencing_token,
        }
