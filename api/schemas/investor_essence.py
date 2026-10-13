"""Pydantic transport schemas for Investor Essence HTTP API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class ValidationIssueSchema(BaseModel):
    code: str
    severity: str  # blocking | warning | info
    field_path: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    issue: Optional[str] = None
    level: Optional[str] = None

    @model_validator(mode="after")
    def populate_compat_aliases(self) -> "ValidationIssueSchema":
        if not self.issue:
            object.__setattr__(self, "issue", self.message)
        if not self.level:
            object.__setattr__(self, "level", self.severity)
        return self


class SourceRefSchema(BaseModel):
    kind: str  # question | session | user_input | holding | goal | external
    ref_id: str
    revision: Optional[int] = None
    quote: Optional[str] = None
    as_of: Optional[str] = None


class ContentOptionSchema(BaseModel):
    option_id: str
    option_key: str
    text: str


class GeneratedQuestionSchema(BaseModel):
    question_id: str
    sequence_no: int
    text: str
    options: List[ContentOptionSchema]
    evidence_type: str
    coverage_topics: List[str]
    is_clarification: bool = False


class AnswerDetailSchema(BaseModel):
    answer_id: str
    answer_kind: str
    option_id: Optional[str] = None
    free_text: Optional[str] = None
    evidence_type: str


class QAItemSchema(BaseModel):
    question_id: str
    sequence_no: int
    text: str
    options: List[ContentOptionSchema]
    evidence_type: str
    coverage_topics: List[str]
    is_clarification: bool
    answer: Optional[AnswerDetailSchema] = None


class SessionResponse(BaseModel):
    session_id: str
    status: str
    revision: int
    active_branch_id: str
    questions_count: int
    answers_count: int
    is_complete: bool
    current_question: Optional[GeneratedQuestionSchema] = None
    qa_history: List[QAItemSchema] = Field(default_factory=list)


class StartSessionRequest(BaseModel):
    scope: str = Field(default="workspace")
    prompt_version: str = Field(default="1.0")
    portfolio_id: Optional[str] = None
    clear_downstream: bool = Field(default=True)


class RecordAnswerRequest(BaseModel):
    answer_kind: str = Field(default="choice")  # choice | free_text | unsure | skipped
    option_id: Optional[str] = None
    free_text: Optional[str] = None
    expected_revision: Optional[int] = None


class AdvanceQuestionRequest(BaseModel):
    expected_revision: Optional[int] = None
    prompt_version: str = Field(default="1.0")


class ClarificationRequest(BaseModel):
    unresolved_topic: str
    expected_revision: Optional[int] = None
    prompt_version: str = Field(default="1.0")


class InterviewConfigResponse(BaseModel):
    interview_version: str = "1.0"
    prompt_version: str = "1.0"
    questions_count: int = 10
    options_per_question: int = 4
    topics: List[str] = Field(
        default_factory=lambda: [
            "life_goals",
            "priorities",
            "horizon_liquidity",
            "risk_experience",
            "portfolio_habits",
            "constraints_beliefs",
        ]
    )


class EvidenceRefSchema(BaseModel):
    answer_id: str
    question_id: str
    revision: int
    quote: str
    evidence_type: str


class ClaimSchema(BaseModel):
    claim_id: str
    text: str
    effective_text: str
    source_kind: str
    fit_rating: Optional[str] = None
    edited_text: Optional[str] = None
    text_revision: int
    is_accepted: bool
    evidence_refs: List[EvidenceRefSchema] = Field(default_factory=list)


class CoverageItemSchema(BaseModel):
    topic: str
    status: str
    supporting_answer_ids: List[str] = Field(default_factory=list)


class SummaryResponse(BaseModel):
    summary_id: str
    session_id: str
    statement: str
    claims: List[ClaimSchema]
    unresolved_topics: List[str]
    coverage_report: List[CoverageItemSchema]
    revision: int


class SummarizeRequest(BaseModel):
    expected_revision: Optional[int] = None
    prompt_version: str = Field(default="1.0")


class ReviewClaimRequest(BaseModel):
    fit_rating: Optional[str] = None  # exact | partial | rejected
    edited_text: Optional[str] = None
    is_excluded: Optional[bool] = None
    expected_revision: Optional[int] = None


class ConfirmEssenceRequest(BaseModel):
    accepted_claim_ids: Optional[List[str]] = None
    expected_summary_revision: Optional[int] = None
    idempotency_key: Optional[str] = None


class ConfirmationResponse(BaseModel):
    confirmation_id: str
    status: str
    accepted_claims_count: int
    artifact_ref: Optional[Dict[str, str]] = None
    message: str = ""


class FinancialContextResponse(BaseModel):
    snapshot_id: str
    portfolio_id: str
    horizon_years: Optional[str] = None
    target_use_amount: Optional[str] = None
    target_use_range: Optional[str] = None
    target_use_timeline: Optional[str] = None
    emergency_reserves_amount: Optional[str] = None
    emergency_reserves_months: Optional[str] = None
    obligations_monthly: Optional[str] = None
    obligations_description: Optional[str] = None
    withdrawal_frequency: Optional[str] = None
    withdrawal_amount: Optional[str] = None
    experience_description: Optional[str] = None
    monthly_investable_contribution: Optional[str] = None
    essential_monthly_expense: Optional[str] = None
    reserves_outside_portfolio: Optional[str] = None
    facts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    unknown_fields: List[str] = Field(default_factory=list)
    as_of: str = ""
    source: str = "user_reported"
    readiness_issues: List[ValidationIssueSchema] = Field(default_factory=list)
    is_ready_for_numeric_policy: bool = False


class UpdateFinancialContextRequest(BaseModel):
    horizon_years: Optional[str] = None
    target_use_amount: Optional[str] = None
    target_use_range: Optional[str] = None
    target_use_timeline: Optional[str] = None
    emergency_reserves_amount: Optional[str] = None
    emergency_reserves_months: Optional[str] = None
    obligations_monthly: Optional[str] = None
    obligations_description: Optional[str] = None
    withdrawal_frequency: Optional[str] = None
    withdrawal_amount: Optional[str] = None
    experience_description: Optional[str] = None
    monthly_investable_contribution: Optional[str] = None
    essential_monthly_expense: Optional[str] = None
    reserves_outside_portfolio: Optional[str] = None
    unknown_fields: Optional[List[str]] = None


class CreateAxisDraftRequest(BaseModel):
    portfolio_id: str
    prompt_version: str = Field(default="1.0")


class AxisDraftResponse(BaseModel):
    draft_id: str
    portfolio_id: str
    essence_ref: Dict[str, Any]
    context_ref: str
    basic_policy: str
    risk_limits: Dict[str, Any]
    invest_targets: List[str]
    exclude_targets: List[str]
    primary_methods: List[str]
    secondary_methods: List[str]
    investment_horizon: str
    allocation_basis: str
    allocation_rows: List[Dict[str, Any]]
    rebalance_frequency: str
    role_models: List[str]
    non_actions: List[str]
    assumptions: List[str] = Field(default_factory=list)
    clarifications: List[str] = Field(default_factory=list)
    revision: int = 1
    completeness_issues: List[str] = Field(default_factory=list)
    is_complete: bool = False


class UpdateAxisDraftRequest(BaseModel):
    basic_policy: Optional[str] = None
    risk_limits: Optional[Dict[str, Any]] = None
    invest_targets: Optional[List[str]] = None
    exclude_targets: Optional[List[str]] = None
    primary_methods: Optional[List[str]] = None
    secondary_methods: Optional[List[str]] = None
    investment_horizon: Optional[str] = None
    allocation_rows: Optional[List[Dict[str, Any]]] = None
    rebalance_frequency: Optional[str] = None
    role_models: Optional[List[str]] = None
    non_actions: Optional[List[str]] = None
    expected_revision: Optional[int] = None


class ConfirmAxisRequest(BaseModel):
    idempotency_key: Optional[str] = None
    expected_draft_revision: Optional[int] = None


class CreateBucketPlanRequest(BaseModel):
    portfolio_id: str
    prompt_version: str = Field(default="1.0")


class PurposeBucketSchema(BaseModel):
    bucket_id: str
    name: str
    role: str
    color: str
    target_percent: str
    source_value_ids: List[str] = Field(default_factory=list)
    source_axis_allocation_ids: List[str] = Field(default_factory=list)
    linked_goal_name: Optional[str] = None
    eligible_assets: List[str] = Field(default_factory=list)
    cash_inflow_routing_pct: Optional[str] = None
    action_rules: List[str] = Field(default_factory=list)
    target_amount: Optional[str] = None
    minimum_amount: Optional[str] = None
    need_by_date: Optional[str] = None
    inflow_route_mode: str = Field(default="fixed_ratio")
    inflow_priority: int = Field(default=1)
    when_full_redirect_bucket_id: Optional[str] = None
    goal_id: Optional[str] = None
    schema_version: str = Field(default="2.0")


class MappingWeightSchema(BaseModel):
    axis_allocation_id: str
    bucket_id: str
    portfolio_weight_percent: str


class BucketRemappingSchema(BaseModel):
    old_bucket_id: str
    target_bucket_id: Optional[str] = None
    affected_holding_count: int = 0


class BucketPlanDraftResponse(BaseModel):
    draft_id: str
    portfolio_id: str
    essence_ref: Dict[str, Any]
    axis_ref: Dict[str, Any]
    context_ref: str
    portfolio_checkpoint: Dict[str, Any]
    purpose_buckets: List[PurposeBucketSchema]
    allocation_basis: str
    mapping_weights: List[MappingWeightSchema]
    constraints: List[str]
    remapping: List[BucketRemappingSchema]
    status: str
    revision: int
    validation_issues: List[str] = Field(default_factory=list)
    is_valid: bool = False


class UpdateBucketPlanRequest(BaseModel):
    purpose_buckets: Optional[List[Dict[str, Any]]] = None
    mapping_weights: Optional[List[Dict[str, Any]]] = None
    remapping: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[List[str]] = None
    expected_revision: Optional[int] = None


class AllocationPreviewResponse(BaseModel):
    checkpoint_sequence: int
    checkpoint_state_hash: str
    validated_targets: List[Dict[str, Any]]
    affected_holdings: List[Dict[str, Any]]
    before_allocation: Dict[str, Any]
    after_allocation: Dict[str, Any]
    issues: List[str]
    payload_hash: str
    preview_id: str = ""
    draft_id: str = ""
    draft_revision: int = 1
    source_refs: Dict[str, Any] = Field(default_factory=dict)


class ApplyBucketPlanRequest(BaseModel):
    idempotency_key: Optional[str] = None
    expected_draft_revision: Optional[int] = None
    expected_checkpoint_sequence: Optional[int] = None
    expected_checkpoint_state_hash: Optional[str] = None
    preview_id: Optional[str] = None
    payload_hash: Optional[str] = None


class AllocationApplyReceiptResponse(BaseModel):
    command_id: str
    portfolio_id: str
    request_hash: str
    applied_sequence: int
    applied_state_hash: str
    applied_at_iso: str
    canonical_status: str
    projection_status: str
    warnings: List[str] = Field(default_factory=list)


