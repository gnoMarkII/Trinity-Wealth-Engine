"""Outbound port interfaces for Investor Essence (Hexagonal Architecture).

Driven adapters implement these interfaces; application services depend only on these protocols.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from core.investor_essence.interview import EssenceSession
from core.investor_essence.models import (
    ArtifactRef,
    BucketPlanDraft,
    EssenceSummaryDraft,
    FinancialContextSnapshot,
    InvestmentAxisDraft,
)
from application.investor_essence.dto import (
    AllocationApplyReceipt,
    AllocationPreview,
    ApplyAllocationCommand,
    AxisProposal,
    BucketPlanProposal,
    EssenceSummaryProposal,
    EvidenceSnapshot,
    OperationView,
    PortfolioPlanningSnapshot,
    PreviewAllocationCommand,
    QuestionProposal,
)


class SessionRepositoryPort(Protocol):
    """Persistence port for adaptive interview sessions and summaries."""
    def get(self, session_id: str) -> Optional[EssenceSession]: ...
    def get_current(self, scope: str = "workspace") -> Optional[EssenceSession]: ...
    def save(self, session: EssenceSession) -> None: ...
    def get_summary(self, session_id: str) -> Optional[EssenceSummaryDraft]: ...
    def save_summary(self, summary: EssenceSummaryDraft) -> None: ...


class PlanningRepositoryPort(Protocol):
    """Persistence port for financial context, drafts, and confirmed pointers."""
    def save_context_snapshot(self, snapshot: FinancialContextSnapshot) -> None: ...
    def get_context_snapshot(self, portfolio_id: str) -> Optional[FinancialContextSnapshot]: ...
    def save_axis_draft(self, draft: InvestmentAxisDraft) -> None: ...
    def get_axis_draft(self, draft_id: str) -> Optional[InvestmentAxisDraft]: ...
    def save_bucket_draft(self, draft: BucketPlanDraft) -> None: ...
    def get_bucket_draft(self, draft_id: str) -> Optional[BucketPlanDraft]: ...
    def get_confirmed_pointer(self, scope: str, kind: str) -> Optional[str]: ...
    def set_confirmed_pointer(
        self, scope: str, kind: str, artifact_ref: str, expected_ref: Optional[str]
    ) -> bool: ...


class OperationRepositoryPort(Protocol):
    """Persistence port for asynchronous durable operations and worker leases."""
    def enqueue(
        self,
        operation_id: str,
        stage: str,
        resource_id: str,
        resource_revision: int,
        input_hash: str,
        prompt_version: str,
        frozen_input: Dict[str, Any],
    ) -> None: ...
    def get(self, operation_id: str) -> Optional[OperationView]: ...
    def claim_lease(self, worker_id: str, lease_seconds: int) -> Optional[Dict[str, Any]]: ...
    def complete_with_fence(self, operation_id: str, fence: int, result: Dict[str, Any]) -> bool: ...
    def fail_with_fence(
        self, operation_id: str, fence: int, error_code: str, error_message: str, retryable: bool
    ) -> bool: ...


class IntentRepositoryPort(Protocol):
    """Persistence port for confirmation & apply intents and idempotent command receipts."""
    def save_intent(
        self,
        intent_id: str,
        scope: str,
        kind: str,
        idempotency_key: str,
        request_hash: str,
        payload: Dict[str, Any],
    ) -> None: ...
    def get_intent_by_idempotency(
        self, scope: str, kind: str, idempotency_key: str
    ) -> Optional[Dict[str, Any]]: ...
    def update_intent_receipt(self, intent_id: str, receipt: Dict[str, Any], status: str) -> None: ...
    def save_command_receipt(
        self, scope: str, use_case: str, idempotency_key: str, request_hash: str, result: Dict[str, Any]
    ) -> None: ...
    def get_command_receipt(
        self, scope: str, use_case: str, idempotency_key: str
    ) -> Optional[Dict[str, Any]]: ...


class InvestorRuntimeUow(Protocol):
    """Unit of Work interface wrapping session, planning, operation, and intent transactions."""
    sessions: SessionRepositoryPort
    planning: PlanningRepositoryPort
    operations: OperationRepositoryPort
    intents: IntentRepositoryPort

    def __enter__(self) -> "InvestorRuntimeUow": ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...


class InvestorRuntimeUowFactory(Protocol):
    """Factory to create a new isolated Unit of Work session."""
    def open(self) -> InvestorRuntimeUow: ...


class InterviewGeneratorPort(Protocol):
    """Port to LLM adapter generating adaptive interview questions."""
    def generate_next_question(
        self, evidence: EvidenceSnapshot, prompt_version: str
    ) -> QuestionProposal: ...
    def generate_clarification(
        self, evidence: EvidenceSnapshot, unresolved_topic: str, prompt_version: str
    ) -> QuestionProposal: ...


class EssenceGeneratorPort(Protocol):
    """Port to LLM adapter generating essence bulleted summaries."""
    def generate_summary(
        self, evidence: EvidenceSnapshot, prompt_version: str
    ) -> EssenceSummaryProposal: ...


class AxisGeneratorPort(Protocol):
    """Port to LLM adapter generating 8-section investment axis drafts."""
    def generate_axis(
        self,
        accepted_claims: List[Dict[str, Any]],
        financial_context: Dict[str, Any],
        prompt_version: str,
    ) -> AxisProposal: ...


class BucketGeneratorPort(Protocol):
    """Port to LLM adapter proposing target purpose buckets from axis policy."""
    def generate_buckets(
        self, confirmed_axis: Dict[str, Any], prompt_version: str
    ) -> BucketPlanProposal: ...


class ConfirmedKnowledgeReaderPort(Protocol):
    """Port for exact reading and verification of confirmed Vault notes."""
    def read_exact(self, ref: ArtifactRef) -> Optional[Dict[str, Any]]: ...
    def read_confirmed_versions(self, scope: str, entity_type: str) -> List[Dict[str, Any]]: ...


class KnowledgeWritePort(Protocol):
    """Port to existing Knowledge Write Broker for publishing confirmed notes."""
    def submit(self, command: Dict[str, Any]) -> Dict[str, Any]: ...
    def get_receipt(self, command_id: str) -> Optional[Dict[str, Any]]: ...


class PortfolioPlanningPort(Protocol):
    """Bridge port to Portfolio subsystem for snapshot, preview, apply, and repair."""
    def snapshot(self, portfolio_id: str) -> PortfolioPlanningSnapshot: ...
    def preview(self, command: PreviewAllocationCommand) -> AllocationPreview: ...
    def apply(self, command: ApplyAllocationCommand) -> AllocationApplyReceipt: ...
    def get_apply_receipt(self, portfolio_id: str, command_id: str) -> Optional[AllocationApplyReceipt]: ...
    def repair_projection(self, portfolio_id: str, command_id: str) -> AllocationApplyReceipt: ...


class ClockPort(Protocol):
    """Port for current time in UTC."""
    def now_utc(self) -> str: ...
    def now_epoch(self) -> float: ...


class IdGeneratorPort(Protocol):
    """Port for generating unique IDs (UUIDs)."""
    def new_id(self) -> str: ...
