"""Application-facing schema and policy port definitions.

These protocols intentionally contain no filesystem or Vault implementation
imports.  API/application code can validate a command through a registry
adapter without learning where the registry is stored.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, runtime_checkable


@runtime_checkable
class SchemaRegistryPort(Protocol):
    def digest(self) -> str:
        ...

    def policy_digest(self) -> str:
        ...

    def validate_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        profile_id: Optional[str] = None,
        require_schema_version: bool = True,
        allow_identity_allocation: bool = False,
    ) -> tuple[bool, list[dict[str, str]]]:
        ...

    def is_index_eligible(
        self,
        metadata: Mapping[str, Any],
        *,
        profile_id: Optional[str] = None,
        vector: bool = True,
    ) -> bool:
        ...
