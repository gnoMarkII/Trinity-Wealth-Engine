"""Inbound HTTP adapter for Equity Position Review & Decision Framework."""
import logging
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_equity_query_service, get_position_review_service
from api.schemas.position_review import (
    HoldingSummaryDTO,
    PortfolioAwareTriggerDTO,
    PositionReviewResponseDTO,
    PriceMoveDecompositionDTO,
    ThenVsNowSnapshotDTO,
)
from application.equity.position_review_service import PositionReviewApplicationService
from application.equity.query_service import EquityResearchQueryService
from application.equity.validation import validate_ticker

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{ticker}/position-review", response_model=PositionReviewResponseDTO)
def get_position_review(
    ticker: str,
    market: str = "US",
    portfolio_id: str = "default",
    service: PositionReviewApplicationService = Depends(get_position_review_service),
    equity_query_service: EquityResearchQueryService = Depends(get_equity_query_service),
) -> PositionReviewResponseDTO:
    """Retrieves comprehensive single-stock Position Review analysis."""
    try:
        clean_ticker = validate_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        report = service.run_position_review(ticker=clean_ticker, market=market, portfolio_id=portfolio_id)
    except TypeError:
        report = service.run_position_review(ticker=clean_ticker, market=market)
    except Exception as exc:
        log.exception("Position review execution failed for %s: %s", clean_ticker, exc)
        raise HTTPException(status_code=500, detail=f"Failed to execute position review: {exc}") from exc

    # Map domain decompositions
    decomps_dto = [
        PriceMoveDecompositionDTO(
            horizon_label=d.horizon_label,
            start_date=d.start_date,
            end_date=d.end_date,
            start_price=d.start_price,
            end_price=d.end_price,
            total_return_pct=d.total_return_pct,
            start_eps=d.start_eps,
            end_eps=d.end_eps,
            eps_growth_pct=d.eps_growth_pct,
            start_pe=d.start_pe,
            end_pe=d.end_pe,
            pe_expansion_pct=d.pe_expansion_pct,
            primary_driver=d.primary_driver,
            diagnosis_tier=d.diagnosis_tier,
            diagnosis_summary=d.diagnosis_summary,
        )
        for d in report.decompositions
    ]

    # Map domain ThenVsNow
    then_vs_now_dto = None
    if report.then_vs_now:
        tn = report.then_vs_now
        then_vs_now_dto = ThenVsNowSnapshotDTO(
            is_held=tn.is_held,
            period_label=tn.period_label,
            then_date=tn.then_date,
            now_date=tn.now_date,
            then_price=tn.then_price,
            now_price=tn.now_price,
            then_pe=tn.then_pe,
            now_pe=tn.now_pe,
            then_growth_pct=tn.then_growth_pct,
            now_growth_pct=tn.now_growth_pct,
            what_was_true_then=tn.what_was_true_then,
            what_is_true_now=tn.what_is_true_now,
            thesis_reflection_question=tn.thesis_reflection_question,
        )

    # Map domain Triggers
    triggers_dto = [
        PortfolioAwareTriggerDTO(
            trigger_id=t.trigger_id,
            title=t.title,
            condition_text=t.condition_text,
            category=t.category,
            case_impact=t.case_impact,
            impact_description=t.impact_description,
            is_currently_triggered=t.is_currently_triggered,
            metric_current_value=t.metric_current_value,
            metric_threshold_value=t.metric_threshold_value,
        )
        for t in report.triggers
    ]

    # Map holding summary
    hs = report.holding_summary or {}
    holding_dto = HoldingSummaryDTO(
        is_held=hs.get("is_held", False),
        units=float(hs.get("units") or 0.0),
        avg_cost=hs.get("avg_cost"),
        unrealized_pnl_pct=hs.get("unrealized_pnl_pct"),
        current_weight_pct=float(hs.get("current_weight_pct") or 0.0),
        bucket_id=hs.get("bucket_id"),
        entry_date=hs.get("entry_date"),
        added_direction=hs.get("added_direction"),
    )

    curr_p = decomps_dto[0].end_price if decomps_dto else 100.0

    company_name = getattr(report, "company_name", None) or clean_ticker
    action_stance = getattr(report, "action_stance", None) or "HOLD_WAIT"
    action_stance_reason = getattr(report, "action_stance_reason", None)

    try:
        sidecar = equity_query_service.get_detail(clean_ticker)
        if sidecar:
            if sidecar.get("company_name"):
                company_name = sidecar["company_name"]
            quant = sidecar.get("quant_signals") or {}
            scorecard = quant.get("deterministic_scorecard") or {}
            if scorecard.get("action_stance") or quant.get("action_stance"):
                action_stance = scorecard.get("action_stance") or quant.get("action_stance")
            if scorecard.get("action_stance_reason") or quant.get("action_stance_reason"):
                action_stance_reason = scorecard.get("action_stance_reason") or quant.get("action_stance_reason")
    except Exception as e:
        log.debug("Failed to read equity sidecar for %s in position review: %s", clean_ticker, e)

    return PositionReviewResponseDTO(
        ticker=report.ticker,
        company_name=company_name,
        current_price=curr_p,
        as_of_date=report.as_of_date,
        action_stance=action_stance,
        action_stance_reason=action_stance_reason,
        holding_summary=holding_dto,
        decompositions=decomps_dto,
        then_vs_now=then_vs_now_dto,
        triggers=triggers_dto,
        honest_gaps=report.honest_gaps,
    )
