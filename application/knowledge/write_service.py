"""Application service for validating and submitting Vault write commands."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from application.knowledge.errors import WriteCommandValidationError
from application.knowledge.write_models import KnowledgeWriteCommand, KnowledgeWriteReceipt
from application.knowledge.write_ports import KnowledgeWritePort


class KnowledgeWriteService:
    """Keeps inbound clients independent from the concrete Vault writer."""

    def __init__(self, writer: KnowledgeWritePort) -> None:
        self._writer = writer

    def submit(self, command: KnowledgeWriteCommand) -> KnowledgeWriteReceipt:
        if not isinstance(command, KnowledgeWriteCommand):
            raise WriteCommandValidationError("submit requires KnowledgeWriteCommand")
        return self._writer.submit(command)

    def submit_dict(self, value: Mapping[str, Any]) -> KnowledgeWriteReceipt:
        try:
            command = KnowledgeWriteCommand.from_dict(value)
        except (TypeError, ValueError) as exc:
            raise WriteCommandValidationError(str(exc)) from exc
        return self.submit(command)

    def get_receipt(
        self,
        *,
        command_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Optional[KnowledgeWriteReceipt]:
        if not command_id and not idempotency_key:
            raise ValueError("command_id or idempotency_key is required")
        return self._writer.get_receipt(command_id=command_id, idempotency_key=idempotency_key)
