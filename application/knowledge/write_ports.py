"""Application write ports; implementations belong to infrastructure."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from application.knowledge.write_models import KnowledgeWriteCommand, KnowledgeWriteReceipt


@runtime_checkable
class KnowledgeWritePort(Protocol):
    """Submit path-independent write commands and retrieve immutable receipts."""

    def submit(self, command: KnowledgeWriteCommand) -> KnowledgeWriteReceipt:
        ...

    def get_receipt(
        self,
        *,
        command_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Optional[KnowledgeWriteReceipt]:
        ...


@runtime_checkable
class KnowledgeWriteStatusPort(Protocol):
    """Read-only status port kept separate from write submission."""

    def get_receipt(
        self,
        *,
        command_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Optional[KnowledgeWriteReceipt]:
        ...
