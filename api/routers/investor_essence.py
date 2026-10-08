"""FastAPI router for Investor Essence and Investment Axis endpoints."""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.auth import require_session
from api.dependencies import (
    get_bucket_planning_service,
    get_claim_review_service,
    get_confirmation_service,
    get_financial_context_service,
    get_interview_service,
    get_investment_axis_service,
)
from api.schemas.investor_essence import (
    AdvanceQuestionRequest,
    AllocationApplyReceiptResponse,
    AllocationPreviewResponse,
    ApplyBucketPlanRequest,
    AxisDraftResponse,
    BucketPlanDraftResponse,
    ConfirmAxisRequest,
    ConfirmEssenceRequest,
    ConfirmationResponse,
    CreateAxisDraftRequest,
    CreateBucketPlanRequest,
    FinancialContextResponse,
    InterviewConfigResponse,
    RecordAnswerRequest,
    ReviewClaimRequest,
    SessionResponse,
    StartSessionRequest,
    SummarizeRequest,
    SummaryResponse,
    UpdateAxisDraftRequest,
    UpdateBucketPlanRequest,
    UpdateFinancialContextRequest,
)
from application.investor_essence import (
    AxisDraftView,
    AxisIncompleteError,
    BucketPlanningService,
    ClaimReviewService,
    ConfirmationService,
    CurrentRefConflictError,
    FinancialContextService,
    FinancialContextView,
    IdempotencyConflictError,
    InterviewService,
    InvestmentAxisService,
    InvestorEssenceError,
    KnowledgeUnavailableError,
    PortfolioConflictError,
    ProviderUnavailableError,
    ResourceNotFoundError,
    RevisionConflictError,
    ValidationFailedError,
    WorkerUnavailableError,
)
from core.investor_essence.models import AnswerKind, FitRating

router = APIRouter(
    prefix="/api/investor",
    tags=["Investor Essence & Axis"],
    dependencies=[Depends(require_session)],
)


def _handle_error(exc: Exception) -> None:
    if isinstance(exc, ResourceNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.message)
    if isinstance(
        exc,
        (
            RevisionConflictError,
            CurrentRefConflictError,
            IdempotencyConflictError,
            PortfolioConflictError,
        ),
    ):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=exc.message)
    if isinstance(exc, (ValidationFailedError, AxisIncompleteError)):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.message)
    if isinstance(exc, (WorkerUnavailableError, ProviderUnavailableError, KnowledgeUnavailableError)):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.message)
    if isinstance(exc, InvestorEssenceError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message)
    raise exc


# -----------------------------------------------------------------------------
# Milestone 1: Investor Essence Endpoints
# -----------------------------------------------------------------------------

@router.get("/essence/interview-config", response_model=InterviewConfigResponse)
def get_interview_config() -> InterviewConfigResponse:
    """Returns interview metadata and 10-question / 4-option protocol parameters."""
    return InterviewConfigResponse()


@router.post("/essence/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def start_session(
    request: StartSessionRequest,
    service: InterviewService = Depends(get_interview_service),
) -> SessionResponse:
    """Starts a new adaptive interview session and generates Question 1."""
    try:
        view = service.start_session(
            scope=request.scope,
            prompt_version=request.prompt_version,
        )
        return SessionResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.get("/essence/sessions/current", response_model=SessionResponse)
def get_current_session(
    scope: str = Query(default="workspace"),
    service: InterviewService = Depends(get_interview_service),
) -> SessionResponse:
    """Resumes the most recent session for the current scope."""
    view = service.get_current_session(scope=scope)
    if not view:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No current interview session found for scope '{scope}'",
        )
    return SessionResponse.model_validate(view, from_attributes=True)


@router.get("/essence/sessions/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: str,
    service: InterviewService = Depends(get_interview_service),
) -> SessionResponse:
    """Fetches session state, answered questions, and current active question."""
    try:
        view = service.get_session(session_id)
        return SessionResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.put("/essence/sessions/{session_id}/answers/{question_id}", response_model=SessionResponse)
def record_answer(
    session_id: str,
    question_id: str,
    request: RecordAnswerRequest,
    service: InterviewService = Depends(get_interview_service),
) -> SessionResponse:
    """Records an answer to a question in the active branch."""
    try:
        answer_kind = AnswerKind(request.answer_kind)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid answer_kind: {request.answer_kind}",
        )

    try:
        view = service.record_answer(
            session_id=session_id,
            question_id=question_id,
            answer_kind=answer_kind,
            option_id=request.option_id,
            free_text=request.free_text,
            expected_revision=request.expected_revision,
        )
        return SessionResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.post("/essence/sessions/{session_id}/next-question", response_model=SessionResponse)
def advance_next_question(
    session_id: str,
    request: Optional[AdvanceQuestionRequest] = None,
    service: InterviewService = Depends(get_interview_service),
) -> SessionResponse:
    """Generates and advances to the next adaptive question (Q2..Q10)."""
    expected_rev = request.expected_revision if request else None
    prompt_ver = request.prompt_version if request else "1.0"
    try:
        view = service.advance_next_question(
            session_id=session_id,
            expected_revision=expected_rev,
            prompt_version=prompt_ver,
        )
        return SessionResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.post("/essence/sessions/{session_id}/summarize", response_model=SummaryResponse)
def generate_summary(
    session_id: str,
    request: Optional[SummarizeRequest] = None,
    service: ClaimReviewService = Depends(get_claim_review_service),
) -> SummaryResponse:
    """Synthesizes the completed 10-question interview into a bulleted essence draft."""
    expected_rev = request.expected_revision if request else None
    prompt_ver = request.prompt_version if request else "1.0"
    try:
        view = service.generate_summary(
            session_id=session_id,
            expected_revision=expected_rev,
            prompt_version=prompt_ver,
        )
        return SummaryResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.get("/essence/sessions/{session_id}/summary", response_model=SummaryResponse)
def get_summary(
    session_id: str,
    service: ClaimReviewService = Depends(get_claim_review_service),
) -> SummaryResponse:
    """Retrieves the current draft summary for a session."""
    view = service.get_summary(session_id)
    if not view:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Summary draft for session '{session_id}' not found",
        )
    return SummaryResponse.model_validate(view, from_attributes=True)


@router.put("/essence/sessions/{session_id}/summary/claims/{claim_id}", response_model=SummaryResponse)
def review_claim(
    session_id: str,
    claim_id: str,
    request: ReviewClaimRequest,
    service: ClaimReviewService = Depends(get_claim_review_service),
) -> SummaryResponse:
    """Rates, edits, or excludes a claim in the draft summary."""
    try:
        if request.is_excluded:
            view = service.exclude_claim(
                session_id=session_id,
                claim_id=claim_id,
                expected_revision=request.expected_revision,
            )
        elif request.edited_text:
            view = service.edit_claim(
                session_id=session_id,
                claim_id=claim_id,
                new_text=request.edited_text,
                expected_revision=request.expected_revision,
            )
        elif request.fit_rating:
            try:
                fit = FitRating(request.fit_rating)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid fit_rating: {request.fit_rating}",
                )
            view = service.rate_claim(
                session_id=session_id,
                claim_id=claim_id,
                fit_rating=fit,
                expected_revision=request.expected_revision,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Must provide one of 'fit_rating', 'edited_text', or 'is_excluded'",
            )
        return SummaryResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.post("/essence/sessions/{session_id}/confirm", response_model=ConfirmationResponse)
def confirm_essence(
    session_id: str,
    request: ConfirmEssenceRequest,
    service: ConfirmationService = Depends(get_confirmation_service),
) -> ConfirmationResponse:
    """Confirms accepted essence claims and sets the workspace confirmed pointer."""
    try:
        view = service.confirm_essence(
            session_id=session_id,
            accepted_claim_ids=request.accepted_claim_ids,
            expected_summary_revision=request.expected_summary_revision,
            idempotency_key=request.idempotency_key or "",
        )
        return ConfirmationResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.get("/essence/current", response_model=Dict[str, Any])
def get_current_essence(
    service: ConfirmationService = Depends(get_confirmation_service),
) -> Dict[str, Any]:
    """Retrieves the current confirmed investor essence pointer."""
    confirmed = service.get_current_confirmed_essence()
    if not confirmed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No confirmed investor essence found for this workspace",
        )
    return confirmed


# -----------------------------------------------------------------------------
# Milestone 2: Financial Context & Investment Axis Endpoints
# -----------------------------------------------------------------------------

@router.get("/financial-context", response_model=FinancialContextResponse)
def get_financial_context(
    portfolio_id: str = Query(...),
    service: FinancialContextService = Depends(get_financial_context_service),
) -> FinancialContextResponse:
    """Retrieves the reported financial context facts and readiness audit for a portfolio."""
    try:
        view = service.get_financial_context(portfolio_id=portfolio_id)
        return FinancialContextResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.put("/financial-context", response_model=FinancialContextResponse)
def update_financial_context(
    portfolio_id: str = Query(...),
    request: UpdateFinancialContextRequest = ...,
    service: FinancialContextService = Depends(get_financial_context_service),
) -> FinancialContextResponse:
    """Updates reported financial context facts and recalculates readiness."""
    try:
        updates = request.model_dump(exclude_unset=True)
        view = service.update_financial_context(portfolio_id=portfolio_id, updates=updates)
        return FinancialContextResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.post("/investment-axis/drafts", response_model=AxisDraftResponse, status_code=status.HTTP_201_CREATED)
def create_axis_draft(
    request: CreateAxisDraftRequest,
    service: InvestmentAxisService = Depends(get_investment_axis_service),
) -> AxisDraftResponse:
    """Generates an 8-section investment policy draft using Command 2 and confirmed essence."""
    try:
        view = service.create_axis_draft(
            portfolio_id=request.portfolio_id,
            prompt_version=request.prompt_version,
        )
        return AxisDraftResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.get("/investment-axis/drafts/{draft_id}", response_model=AxisDraftResponse)
def get_axis_draft(
    draft_id: str,
    service: InvestmentAxisService = Depends(get_investment_axis_service),
) -> AxisDraftResponse:
    """Retrieves an existing investment axis draft with completeness audit."""
    view = service.get_axis_draft(draft_id=draft_id)
    if not view:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Investment axis draft '{draft_id}' not found",
        )
    return AxisDraftResponse.model_validate(view, from_attributes=True)


@router.put("/investment-axis/drafts/{draft_id}", response_model=AxisDraftResponse)
def update_axis_draft(
    draft_id: str,
    request: UpdateAxisDraftRequest,
    service: InvestmentAxisService = Depends(get_investment_axis_service),
) -> AxisDraftResponse:
    """Updates sections, non-actions, or numeric values of an investment axis draft."""
    try:
        updates = request.model_dump(exclude_unset=True)
        expected_rev = updates.pop("expected_revision", None)
        view = service.update_axis_draft(
            draft_id=draft_id,
            updates=updates,
            expected_revision=expected_rev,
        )
        return AxisDraftResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.post("/investment-axis/drafts/{draft_id}/confirm", response_model=ConfirmationResponse)
def confirm_investment_axis(
    draft_id: str,
    request: Optional[ConfirmAxisRequest] = None,
    service: InvestmentAxisService = Depends(get_investment_axis_service),
) -> ConfirmationResponse:
    """Validates 8 sections, non-actions >= 3, and numeric constraints, then sets confirmed pointer."""
    idempotency_key = request.idempotency_key if request and request.idempotency_key else ""
    try:
        view = service.confirm_axis(draft_id=draft_id, idempotency_key=idempotency_key)
        return ConfirmationResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.get("/investment-axis/current", response_model=Dict[str, Any])
def get_current_axis(
    portfolio_id: str = Query(...),
    service: InvestmentAxisService = Depends(get_investment_axis_service),
) -> Dict[str, Any]:
    """Retrieves the current confirmed investment axis pointer for a portfolio."""
    confirmed = service.get_current_confirmed_axis(portfolio_id=portfolio_id)
    if not confirmed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No confirmed investment axis found for portfolio '{portfolio_id}'",
        )
    return confirmed


# -----------------------------------------------------------------------------
# Milestone 3: Purpose Bucket Plans & Apply Endpoints
# -----------------------------------------------------------------------------

@router.post("/bucket-plans", response_model=BucketPlanDraftResponse, status_code=status.HTTP_201_CREATED)
def create_bucket_plan(
    request: CreateBucketPlanRequest,
    service: BucketPlanningService = Depends(get_bucket_planning_service),
) -> BucketPlanDraftResponse:
    """Generates 3-5 purpose buckets and multi-dimensional mapping from confirmed axis."""
    try:
        view = service.create_bucket_plan_draft(
            portfolio_id=request.portfolio_id,
            prompt_version=request.prompt_version,
        )
        return BucketPlanDraftResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.get("/bucket-plans/{draft_id}", response_model=BucketPlanDraftResponse)
def get_bucket_plan(
    draft_id: str,
    service: BucketPlanningService = Depends(get_bucket_planning_service),
) -> BucketPlanDraftResponse:
    """Retrieves an existing bucket plan draft with matrix mapping and validation."""
    view = service.get_bucket_plan_draft(draft_id=draft_id)
    if not view:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bucket plan draft '{draft_id}' not found",
        )
    return BucketPlanDraftResponse.model_validate(view, from_attributes=True)


@router.put("/bucket-plans/{draft_id}", response_model=BucketPlanDraftResponse)
def update_bucket_plan(
    draft_id: str,
    request: UpdateBucketPlanRequest,
    service: BucketPlanningService = Depends(get_bucket_planning_service),
) -> BucketPlanDraftResponse:
    """Updates purpose buckets, mapping weights, or holding remappings."""
    try:
        updates = request.model_dump(exclude_unset=True)
        expected_rev = updates.pop("expected_revision", None)
        view = service.update_bucket_plan_draft(
            draft_id=draft_id,
            updates=updates,
            expected_revision=expected_rev,
        )
        return BucketPlanDraftResponse.model_validate(view, from_attributes=True)
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.get("/bucket-plans/{draft_id}/preview", response_model=AllocationPreviewResponse)
def preview_bucket_plan(
    draft_id: str,
    service: BucketPlanningService = Depends(get_bucket_planning_service),
) -> AllocationPreviewResponse:
    """Previews before/after allocation targets and affected holdings."""
    try:
        preview = service.preview_bucket_plan(draft_id=draft_id)
        return AllocationPreviewResponse(
            checkpoint_sequence=preview.checkpoint_sequence,
            checkpoint_state_hash=preview.checkpoint_state_hash,
            validated_targets=preview.validated_targets,
            affected_holdings=preview.affected_holdings,
            before_allocation=preview.before_allocation,
            after_allocation=preview.after_allocation,
            issues=preview.issues,
            payload_hash=preview.payload_hash,
        )
    except Exception as exc:
        _handle_error(exc)
        raise exc


@router.post("/bucket-plans/{draft_id}/apply", response_model=AllocationApplyReceiptResponse)
def apply_bucket_plan(
    draft_id: str,
    request: Optional[ApplyBucketPlanRequest] = None,
    service: BucketPlanningService = Depends(get_bucket_planning_service),
) -> AllocationApplyReceiptResponse:
    """Atomically commits new targets and remaps holdings in a single Portfolio transaction."""
    idempotency_key = request.idempotency_key if request and request.idempotency_key else ""
    try:
        receipt = service.apply_bucket_plan(draft_id=draft_id, idempotency_key=idempotency_key)
        return AllocationApplyReceiptResponse(
            command_id=receipt.command_id,
            portfolio_id=receipt.portfolio_id,
            request_hash=receipt.request_hash,
            applied_sequence=receipt.applied_sequence,
            applied_state_hash=receipt.applied_state_hash,
            applied_at_iso=receipt.applied_at_iso,
            canonical_status=receipt.canonical_status,
            projection_status=receipt.projection_status,
            warnings=receipt.warnings,
        )
    except Exception as exc:
        _handle_error(exc)
        raise exc
