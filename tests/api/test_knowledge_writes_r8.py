from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas.knowledge_writes import KnowledgeWriteRequest


def test_transport_rejects_path_injection_and_privileged_actor() -> None:
    with pytest.raises(ValidationError):
        KnowledgeWriteRequest(
            idempotency_key="path",
            operation="upsert_note",
            producer="client",
            payload={"target_path": "../../outside.md"},
        )
    with pytest.raises(ValidationError):
        KnowledgeWriteRequest(
            idempotency_key="actor",
            operation="upsert_note",
            producer="client",
            actor="migration",
            payload={},
        )


def test_transport_rejects_oversized_payload() -> None:
    with pytest.raises(ValidationError):
        KnowledgeWriteRequest(
            idempotency_key="large",
            operation="upsert_note",
            producer="client",
            payload={"body": "x" * (5 * 1024 * 1024)},
        )
