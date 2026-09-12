"""Versioned transport schemas for multi-app knowledge writes."""
from __future__ import annotations

from typing import Any, Literal, Optional

import json

from pydantic import BaseModel, Field, field_validator


WriteOperationLiteral = Literal[
    "upsert_note",
    "publish_capture",
    "append_journal_entry",
    "retire_note",
    "restore_note",
    "register_attachment",
    "regenerate_projection",
]


class KnowledgeWriteRequest(BaseModel):
    protocol_version: int = 1
    command_id: Optional[str] = None
    idempotency_key: str = Field(min_length=1, max_length=512)
    operation: WriteOperationLiteral
    producer: str = Field(min_length=1, max_length=200)
    producer_version: str = Field(default="unknown", max_length=100)
    document_key: Optional[str] = Field(default=None, max_length=512)
    entity_type: Optional[str] = Field(default=None, max_length=100)
    payload: dict[str, Any]
    expected_revision_id: Optional[str] = None
    expected_content_hash: Optional[str] = None
    causation_id: Optional[str] = None
    correlation_id: Optional[str] = None
    actor: str = Field(default="app", max_length=100)

    @field_validator("payload")
    @classmethod
    def validate_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "target_path" in value:
            raise ValueError("client-supplied target_path is not accepted by the transport")
        size = len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
        if size > 5 * 1024 * 1024:
            raise ValueError("write payload exceeds the 5 MiB command limit")
        return value

    @field_validator("actor")
    @classmethod
    def validate_actor(cls, value: str) -> str:
        if str(value).strip().lower() in {"migration", "reconciliation", "vault-maintenance"}:
            raise ValueError("privileged actors are not accepted from the app transport")
        return value


class KnowledgeWriteResponse(BaseModel):
    model_config = {"extra": "allow"}

    command_id: str
    idempotency_key: str
    status: str
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
    warnings: list[str] = Field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    registry_digest: Optional[str] = None
    broker_fencing_token: Optional[int] = None
