"""Pure Domain layer for Investor Essence and Investment Axis.

This package contains:
- Domain aggregates: EssenceSession
- Value objects: Question, Answer, Claim, Confirmation, FinancialContext, Axis, BucketPlan
- Domain validation and invariant rules
"""
from core.investor_essence.models import (
    HUNDRED_PERCENT,
    PERCENT_TOLERANCE,
    SCHEMA_VERSION,
    AllocationBasis,
    AllocationMappingCell,
    AllocationPlanOrigin,
    AllocationPlanRow,
    Answer,
    AnswerKind,
    ArtifactRef,
    BucketPlanDraft,
    BucketPlanStatus,
    BucketRemappingItem,
    ConfirmedInvestmentAxis,
    ContentOption,
    CoverageItem,
    EssenceClaim,
    EssenceConfirmationSnapshot,
    EssenceSummaryDraft,
    EvidenceRef,
    EvidenceType,
    FinancialContextSnapshot,
    FinancialReadinessIssue,
    FitRating,
    GeneratedQuestion,
    InvestmentAxisDraft,
    NumericPolicyField,
    NumericPolicyOrigin,
    PerClaimConfirmationSnapshot,
    PurposeBucketDraft,
    SessionStatus,
    SourceKind,
)
from core.investor_essence.interview import EssenceSession
from core.investor_essence.claims import (
    build_confirmation_snapshot,
    edit_claim_text,
    exclude_claim,
    rate_claim_fit,
)
from core.investor_essence.financial_context import (
    evaluate_financial_readiness,
    extract_years_from_text,
    update_readiness,
)
from core.investor_essence.bucket_policy import (
    BucketPolicy,
    GoalFundingLink,
    InflowRoutingMode,
    purpose_bucket_to_policy,
)
from core.investor_essence.funding import (
    calculate_baseline_contribution,
    calculate_bucket_gap,
    calculate_emergency_reserve_requirement,
    calculate_funding_progress,
    calculate_inflow_routing,
)
from core.investor_essence.bucket_validation import (
    reconcile_allocation_matrix,
    reconcile_bucket_remapping,
    validate_bucket_plan_completeness,
)
from core.investor_essence.validation import (
    to_decimal_2dp,
    validate_allocation_matrix,
    validate_percent_sum,
)

__all__ = [
    "HUNDRED_PERCENT",
    "PERCENT_TOLERANCE",
    "SCHEMA_VERSION",
    "AllocationBasis",
    "AllocationMappingCell",
    "AllocationPlanOrigin",
    "AllocationPlanRow",
    "Answer",
    "AnswerKind",
    "ArtifactRef",
    "BucketPlanDraft",
    "BucketPlanStatus",
    "BucketRemappingItem",
    "ConfirmedInvestmentAxis",
    "ContentOption",
    "CoverageItem",
    "EssenceClaim",
    "EssenceConfirmationSnapshot",
    "EssenceSession",
    "EssenceSummaryDraft",
    "EvidenceRef",
    "EvidenceType",
    "FinancialContextSnapshot",
    "FinancialReadinessIssue",
    "FitRating",
    "GeneratedQuestion",
    "InvestmentAxisDraft",
    "NumericPolicyField",
    "NumericPolicyOrigin",
    "PerClaimConfirmationSnapshot",
    "PurposeBucketDraft",
    "SessionStatus",
    "SourceKind",
    "build_confirmation_snapshot",
    "build_confirmed_axis",
    "edit_claim_text",
    "evaluate_financial_readiness",
    "extract_years_from_text",
    "exclude_claim",
    "rate_claim_fit",
    "reconcile_allocation_matrix",
    "reconcile_bucket_remapping",
    "to_decimal_2dp",
    "update_readiness",
    "validate_allocation_matrix",
    "validate_axis_completeness",
    "validate_bucket_plan_completeness",
    "validate_percent_sum",
]
