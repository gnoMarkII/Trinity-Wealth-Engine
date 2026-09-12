"""Pure names for portable multi-app storage policy decisions."""
from __future__ import annotations

from typing import Final

PRIMARY_RETRIEVAL_NAMESPACE: Final[str] = "primary"
SENSITIVITY_NAMESPACES: Final[frozenset[str]] = frozenset(
    {"public", "internal", "confidential", "restricted"}
)
VECTOR_EXCLUDED_LIFECYCLE: Final[frozenset[str]] = frozenset(
    {"draft", "superseded", "retired", "generated"}
)


def is_sensitive_namespace(namespace: str) -> bool:
    return str(namespace or "").strip().lower() in SENSITIVITY_NAMESPACES
