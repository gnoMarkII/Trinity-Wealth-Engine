"""Pure Domain models and value objects for Investor Essence.

This module is strictly isolated in the Domain Layer (Hexagonal Architecture):
- Pure Python dataclasses, Enums, and Decimal
- NO I/O, NO SQLite, NO frameworks (FastAPI/LangChain/Pydantic)
- NO system clock or random UUID generation (supplied externally via ports/commands)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

SCHEMA_VERSION: int = 1
PERCENT_TOLERANCE: Decimal = Decimal("0.01")
HUNDRED_PERCENT: Decimal = Decimal("100.00")


class EvidenceType(str, Enum):
    ACTUAL_EXPERIENCE = "actual_experience"
    HYPOTHETICAL = "hypothetical"
    SELF_REPORT = "self_report"


class AnswerKind(str, Enum):
    CHOICE = "choice"
    FREE_TEXT = "free_text"
    UNSURE = "unsure"
    SKIPPED = "skipped"


class SourceKind(str, Enum):
    USER_STATED = "user_stated"
    AI_INFERRED = "ai_inferred"
    USER_EDITED = "user_edited"


class FitRating(str, Enum):
    EXACT = "exact"
    PARTIAL = "partial"
    REJECTED = "rejected"


class NumericPolicyOrigin(str, Enum):
    USER_ANSWER = "user_answer"
    USER_INPUT = "user_input"
    AI_PROPOSAL = "ai_proposal"


class AllocationBasis(str, Enum):
    PURPOSE = "purpose"
    ASSET_CLASS = "asset_class"


class SessionStatus(str, Enum):
    INTERVIEWING = "interviewing"
    SUMMARY_PENDING = "summary_pending"
    REVIEW = "review"
    CONFIRMED = "confirmed"


class BucketPlanStatus(str, Enum):
    DRAFT = "draft"
    APPLYING = "applying"
    APPLIED = "applied"


@dataclass(frozen=True)
class ArtifactRef:
    """Canonical pointer to a published Vault note artifact."""
    document_key: str
    note_id: str
    revision_id: str
    content_hash: str
    artifact_set_hash: str


@dataclass(frozen=True)
class ContentOption:
    """One of the 4 content options in an adaptive interview question."""
    option_id: str
    option_key: str
    text: str


@dataclass(frozen=True)
class GeneratedQuestion:
    """An adaptive interview question created from prior Q&A context."""
    question_id: str
    sequence_no: int
    text: str
    options: List[ContentOption]
    evidence_type: EvidenceType
    coverage_topics: List[str]
    is_clarification: bool = False
    created_at_iso: str = ""

    def __post_init__(self) -> None:
        if not self.is_clarification and not (1 <= self.sequence_no <= 10):
            raise ValueError(f"Base interview sequence must be between 1 and 10, got {self.sequence_no}")
        if len(self.options) != 4:
            raise ValueError(f"Question must have exactly 4 content options, got {len(self.options)}")
        option_keys = {opt.option_key for opt in self.options}
        if len(option_keys) != 4:
            raise ValueError("Option keys within question must be distinct")

    def get_option(self, option_id: str) -> Optional[ContentOption]:
        for opt in self.options:
            if opt.option_id == option_id:
                return opt
        return None


@dataclass(frozen=True)
class Answer:
    """User response to a generated question."""
    answer_id: str
    question_id: str
    answer_kind: AnswerKind
    option_id: Optional[str] = None
    free_text: Optional[str] = None
    evidence_type: EvidenceType = EvidenceType.SELF_REPORT
    created_at_iso: str = ""


@dataclass(frozen=True)
class EvidenceRef:
    """Evidence pointer tying a claim directly back to an answered question."""
    answer_id: str
    question_id: str
    revision: int
    quote: str
    evidence_type: EvidenceType


@dataclass(frozen=True)
class EssenceClaim:
    """Single bulleted summary claim extracted from interview Q&A."""
    claim_id: str
    text: str
    source_kind: SourceKind
    evidence_refs: List[EvidenceRef]
    counter_evidence_refs: List[EvidenceRef] = field(default_factory=list)
    fit_rating: Optional[FitRating] = None
    edited_text: Optional[str] = None
    text_revision: int = 1
    is_accepted: bool = False

    @property
    def effective_text(self) -> str:
        return self.edited_text if self.edited_text is not None else self.text


@dataclass(frozen=True)
class CoverageItem:
    """Coverage audit for a key investor pillar."""
    topic: str
    status: str  # covered | needs_review | not_ready
    supporting_answer_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class EssenceSummaryDraft:
    """Bulleted essence synthesis pending user review."""
    summary_id: str
    session_id: str
    statement: str
    claims: List[EssenceClaim]
    unresolved_topics: List[str]
    coverage_report: List[CoverageItem]
    context_hash: str
    revision: int = 1
    created_at_iso: str = ""


@dataclass(frozen=True)
class PerClaimConfirmationSnapshot:
    """Auditable state of each claim at the exact moment of confirmation."""
    claim_id: str
    final_text: str
    text_revision: int
    source_kind: SourceKind
    fit_rating: FitRating


@dataclass(frozen=True)
class EssenceConfirmationSnapshot:
    """Immutable snapshot of confirmed essence claims ready for Vault publication."""
    confirmation_id: str
    session_id: str
    accepted_claims: List[EssenceClaim]
    unresolved_topics: List[str]
    per_claim_snapshot: List[PerClaimConfirmationSnapshot]
    evidence_snapshot_hash: str
    confirmed_at_iso: str


@dataclass(frozen=True)
class NumericPolicyField:
    """Quantitative constraint or target in an investment policy."""
    field_id: str
    value: Optional[Decimal]
    unit: str
    calculation_basis: str
    origin: NumericPolicyOrigin
    source_refs: List[str] = field(default_factory=list)
    assumptions: str = ""
    confirmed_text_revision: int = 1
    is_confirmed: bool = False


@dataclass(frozen=True)
class FinancialReadinessIssue:
    """Audited issue in financial context preventing numeric policy confirmation."""
    code: str
    severity: str  # blocking | warning
    message: str
    field_path: str


@dataclass(frozen=True)
class FinancialContextSnapshot:
    """Reported financial profile facts and portfolio readiness context."""
    snapshot_id: str
    portfolio_id: str
    horizon_years: Optional[Decimal] = None
    target_use_amount: Optional[Decimal] = None
    target_use_range: Optional[str] = None
    target_use_timeline: Optional[str] = None
    emergency_reserves_amount: Optional[Decimal] = None
    emergency_reserves_months: Optional[Decimal] = None
    obligations_monthly: Optional[Decimal] = None
    obligations_description: Optional[str] = None
    withdrawal_frequency: Optional[str] = None
    withdrawal_amount: Optional[Decimal] = None
    experience_description: Optional[str] = None
    unknown_fields: List[str] = field(default_factory=list)
    as_of: str = ""
    source: str = "user_reported"
    portfolio_checkpoint_refs: Dict[str, Any] = field(default_factory=dict)
    readiness_issues: List[FinancialReadinessIssue] = field(default_factory=list)
    is_ready_for_numeric_policy: bool = False


@dataclass(frozen=True)
class AllocationPlanRow:
    """Single row in an asset allocation policy."""
    allocation_id: str
    category_name: str
    target_percent: Decimal
    role_description: str


@dataclass(frozen=True)
class InvestmentAxisDraft:
    """Draft 8-section investment policy for a specific portfolio."""
    draft_id: str
    portfolio_id: str
    essence_ref: ArtifactRef
    context_ref: str
    basic_policy: str
    risk_limits: Dict[str, NumericPolicyField]
    invest_targets: List[str]
    exclude_targets: List[str]
    primary_methods: List[str]
    secondary_methods: List[str]
    investment_horizon: str
    allocation_basis: AllocationBasis
    allocation_rows: List[AllocationPlanRow]
    rebalance_frequency: str
    role_models: List[str]
    non_actions: List[str]
    numeric_fields: Dict[str, NumericPolicyField]
    assumptions: List[str] = field(default_factory=list)
    clarifications: List[str] = field(default_factory=list)
    revision: int = 1
    created_at_iso: str = ""


@dataclass(frozen=True)
class ConfirmedInvestmentAxis:
    """Audited, confirmed 8-section investment policy."""
    confirmation_id: str
    portfolio_id: str
    accepted_essence_ref: ArtifactRef
    context_ref: str
    complete_sections: Dict[str, Any]
    allocation_basis: AllocationBasis
    allocation_rows: List[AllocationPlanRow]
    non_actions: List[str]
    confirmed_at_iso: str
    content_hash: str


@dataclass(frozen=True)
class PurposeBucketDraft:
    """Target bucket defined by purpose or money role."""
    bucket_id: str
    name: str
    role: str
    color: str
    target_percent: Decimal
    source_value_ids: List[str] = field(default_factory=list)
    source_axis_allocation_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AllocationMappingCell:
    """Matrix weight tying an axis category to a target purpose bucket."""
    axis_allocation_id: str
    bucket_id: str
    portfolio_weight_percent: Decimal


@dataclass(frozen=True)
class BucketRemappingItem:
    """Remapping from previous bucket ID to new bucket ID (or None for unassigned)."""
    old_bucket_id: str
    target_bucket_id: Optional[str]
    affected_holding_count: int = 0


@dataclass(frozen=True)
class BucketPlanDraft:
    """Draft target bucket plan compiled from a confirmed axis policy."""
    draft_id: str
    portfolio_id: str
    essence_ref: ArtifactRef
    axis_ref: ArtifactRef
    context_ref: str
    portfolio_checkpoint: Dict[str, Any]
    purpose_buckets: List[PurposeBucketDraft]
    allocation_basis: AllocationBasis
    mapping_weights: List[AllocationMappingCell]
    constraints: List[str]
    remapping: List[BucketRemappingItem]
    status: BucketPlanStatus = BucketPlanStatus.DRAFT
    revision: int = 1
    created_at_iso: str = ""


@dataclass(frozen=True)
class AllocationPlanOrigin:
    """Provenance tracking recorded on Portfolio operational state."""
    applied_draft_id: str
    command_id: str
    essence_ref: ArtifactRef
    axis_ref: ArtifactRef
    context_ref: str
    allocation_basis: AllocationBasis
    applied_mapping: List[Dict[str, Any]]
    applied_at_iso: str
    targets_hash: str
    is_modified_since_apply: bool = False
