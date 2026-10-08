"""Typed application errors for Investor Essence with stable error codes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class InvestorEssenceError(Exception):
    """Base domain and application error."""
    def __init__(self, message: str, code: str = "INVESTOR_ESSENCE_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ResourceNotFoundError(InvestorEssenceError):
    """Resource not found (HTTP 404)."""
    def __init__(self, resource_type: str, resource_id: str = "") -> None:
        if resource_id:
            msg = f"Resource '{resource_type}' with id '{resource_id}' was not found."
            self.resource_type = resource_type
            self.resource_id = resource_id
        else:
            msg = resource_type
            self.resource_type = "resource"
            self.resource_id = resource_type
        super().__init__(msg, code="RESOURCE_NOT_FOUND")


class RevisionConflictError(InvestorEssenceError):
    """Optimistic lock / revision mismatch (HTTP 409)."""
    def __init__(
        self,
        resource_id: str,
        expected_revision: int = 0,
        actual_revision: int = 0,
        latest_revision: Optional[int] = None,
    ) -> None:
        actual = latest_revision if latest_revision is not None else actual_revision
        if expected_revision or actual:
            msg = f"Revision conflict on resource '{resource_id}': expected {expected_revision}, but found {actual}."
        else:
            msg = resource_id
        super().__init__(msg, code="REVISION_CONFLICT")
        self.resource_id = resource_id
        self.expected_revision = expected_revision
        self.actual_revision = actual


class CurrentRefConflictError(InvestorEssenceError):
    """Current pointer CAS failed due to concurrent promotion (HTTP 409)."""
    def __init__(self, scope: str, expected_ref: Optional[str] = None, actual_ref: Optional[str] = None) -> None:
        if expected_ref or actual_ref:
            msg = f"Current reference conflict in scope '{scope}': expected '{expected_ref}', but found '{actual_ref}'."
        else:
            msg = scope
        super().__init__(msg, code="CURRENT_REF_CONFLICT")
        self.scope = scope
        self.expected_ref = expected_ref
        self.actual_ref = actual_ref


class PortfolioConflictError(InvestorEssenceError):
    """Portfolio checkpoint changed during planning / apply (HTTP 409)."""
    def __init__(self, portfolio_id: str, expected_seq: int, actual_seq: int) -> None:
        super().__init__(
            f"Portfolio '{portfolio_id}' checkpoint changed: expected sequence {expected_seq}, but found {actual_seq}.",
            code="PORTFOLIO_CONFLICT",
        )
        self.portfolio_id = portfolio_id
        self.expected_seq = expected_seq
        self.actual_seq = actual_seq


class IdempotencyConflictError(InvestorEssenceError):
    """Same idempotency key with different request payload (HTTP 409)."""
    def __init__(self, key: str) -> None:
        super().__init__(
            f"Idempotency conflict: key '{key}' was already used with a different request payload.",
            code="IDEMPOTENCY_CONFLICT",
        )
        self.key = key


class ValidationFailedError(InvestorEssenceError):
    """Domain validation failed (HTTP 422)."""
    def __init__(self, message: str, issues: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__(message, code="VALIDATION_FAILED")
        self.issues = issues or []


class AxisIncompleteError(InvestorEssenceError):
    """Investment Axis missing mandatory sections or confirmed numbers (HTTP 422)."""
    def __init__(self, issues: List[str]) -> None:
        super().__init__(
            "Investment Axis is incomplete:\n" + "\n".join(issues),
            code="AXIS_INCOMPLETE",
        )
        self.issues = issues


class WorkerUnavailableError(InvestorEssenceError):
    """Workflow worker is disabled or unreachable (HTTP 503)."""
    def __init__(self, message: str = "Workflow worker is unavailable.") -> None:
        super().__init__(message, code="WORKER_UNAVAILABLE")


class ProviderUnavailableError(InvestorEssenceError):
    """External LLM provider is unavailable (HTTP 503)."""
    def __init__(self, message: str = "AI generation provider is unavailable.") -> None:
        super().__init__(message, code="PROVIDER_UNAVAILABLE")


class KnowledgeUnavailableError(InvestorEssenceError):
    """Knowledge broker or vault storage is unreachable (HTTP 503)."""
    def __init__(self, message: str = "Knowledge broker or vault is unavailable.") -> None:
        super().__init__(message, code="KNOWLEDGE_UNAVAILABLE")
