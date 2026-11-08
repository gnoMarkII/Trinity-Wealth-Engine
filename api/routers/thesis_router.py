"""REST Router for Structured Theses, Falsifiers & As-Of Research Queries."""
from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.auth import require_session
from api.dependencies import get_thesis_service, get_thesis_synthesis_service
from application.thesis.thesis_service import ThesisApplicationService
from application.thesis.thesis_synthesis_service import ThesisSynthesisApplicationService
from core.equity_analytics.earnings_bridge import DriverCategory, EarningsBridgeEngine, EarningsDriver
from core.security import validate_ticker
from core.thesis.evidence_taxonomy import EvidenceTier
from core.thesis.models import CatalystEvent, FalsifierRule
from api.routers.thesis.router_catalyst import router as catalyst_router

router = APIRouter(prefix="/api/thesis", tags=["thesis"], dependencies=[Depends(require_session)])
router.include_router(catalyst_router)


class FalsifierRuleDTO(BaseModel):
    falsifier_id: str
    metric_name: str
    operator: str
    threshold_value: str
    unit: str = "USD"
    period_basis: str = "QUARTERLY"
    invalidation_impact: str = "BREAKS_BUY_CASE"
    narrative_explanation: str = ""


class CatalystEventDTO(BaseModel):
    catalyst_id: str
    event_name: str
    expected_date_str: str
    confidence_window_days: int = 7
    source_locator: str = ""
    status: str = "PENDING"
    observed_date_str: Optional[str] = None


class ThesisClaimDTO(BaseModel):
    claim_id: str
    ticker: str
    version: int
    prior_claim_id: Optional[str] = None
    as_of_utc: str
    claim_statement: str
    falsifiers: List[FalsifierRuleDTO]
    linked_catalysts: List[CatalystEventDTO]
    status: str
    evidence_tier: str
    author_role: str
    checksum_sha256: str


class CreateThesisRequestDTO(BaseModel):
    claim_statement: str = Field(..., min_length=5, description="Core thesis statement")
    falsifiers: List[FalsifierRuleDTO] = Field(default_factory=list)
    catalysts: List[CatalystEventDTO] = Field(default_factory=list)
    evidence_tier: str = Field(default="DOCUMENTED")
    author_role: str = Field(default="HUMAN_PM")
    prior_claim_id: Optional[str] = Field(default=None, description="Set for revisions")
    as_of_utc: Optional[str] = None


class WatchdogScanResponseDTO(BaseModel):
    exceptions_detected: int
    exceptions: List[Dict[str, Any]]


class SynthesizeThesisRequestDTO(BaseModel):
    dry_run: bool = Field(default=False)
    custom_stop_loss: Optional[float] = None
    custom_growth_floor: Optional[float] = None
    custom_margin_floor: Optional[float] = None
    author_role: str = Field(default="HERMES_ADVISOR")


class BatchSynthesizeRequestDTO(BaseModel):
    portfolio_id: str = Field(default="default")
    dry_run: bool = Field(default=False)


class EarningsBridgeDriverDTO(BaseModel):
    name: str
    category: str
    impact_value: str
    unit: str = "USD"
    narrative: str = ""


class EarningsBridgeRequestDTO(BaseModel):
    ticker: str
    period: str
    as_of_date: Optional[str] = None
    revenue_reported: str
    revenue_consensus: str
    eps_reported: str
    eps_consensus: str
    revenue_unit: str = "USD_M"
    eps_unit: str = "USD"
    ebit_margin_reported: Optional[str] = None
    ebit_margin_consensus: Optional[str] = None
    revenue_drivers: Optional[List[EarningsBridgeDriverDTO]] = None


@router.get("/{ticker}", response_model=Optional[ThesisClaimDTO])
def get_active_thesis(
    ticker: str,
    service: ThesisApplicationService = Depends(get_thesis_service),
) -> Optional[ThesisClaimDTO]:
    """Retrieves the current active thesis claim for a ticker."""
    try:
        clean = validate_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    claim = service.get_active_thesis(clean)
    if not claim:
        return None
    return ThesisClaimDTO(**claim.to_dict())


@router.get("/{ticker}/as-of", response_model=Optional[ThesisClaimDTO])
def get_thesis_as_of(
    ticker: str,
    as_of_utc: str = Query(..., description="Historical ISO-8601 UTC timestamp"),
    service: ThesisApplicationService = Depends(get_thesis_service),
) -> Optional[ThesisClaimDTO]:
    """Time-travel query: returns the active thesis in effect at a historical point in time."""
    try:
        clean = validate_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    claim = service.get_thesis_as_of(ticker=clean, as_of_utc=as_of_utc)
    if not claim:
        return None
    return ThesisClaimDTO(**claim.to_dict())


@router.post("/evaluate", response_model=WatchdogScanResponseDTO)
def evaluate_watchdog_scan(
    ticker: Optional[str] = Query(default=None),
    service: ThesisApplicationService = Depends(get_thesis_service),
) -> WatchdogScanResponseDTO:
    """Manually triggers watchdog evaluation for active theses."""
    try:
        exceptions = service.run_watchdog_scan(ticker=ticker)
        return WatchdogScanResponseDTO(
            exceptions_detected=len(exceptions),
            exceptions=[e.to_dict() for e in exceptions],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Watchdog evaluation failed: {exc}",
        )


@router.post("/synthesize/{ticker}")
def synthesize_thesis_for_ticker(
    ticker: str,
    req: SynthesizeThesisRequestDTO = Body(default_factory=SynthesizeThesisRequestDTO),
    service: ThesisSynthesisApplicationService = Depends(get_thesis_synthesis_service),
) -> Dict[str, Any]:
    """Synthesizes structured thesis and 3-pillar falsifiers for a ticker."""
    try:
        return service.synthesize_thesis_for_ticker(
            ticker=ticker,
            dry_run=req.dry_run,
            custom_stop_loss=req.custom_stop_loss,
            custom_growth_floor=req.custom_growth_floor,
            custom_margin_floor=req.custom_margin_floor,
            author_role=req.author_role,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to synthesize thesis for {ticker}: {exc}",
        )


@router.post("/batch-synthesize")
def batch_synthesize_portfolio_theses(
    req: BatchSynthesizeRequestDTO = Body(default_factory=BatchSynthesizeRequestDTO),
    service: ThesisSynthesisApplicationService = Depends(get_thesis_synthesis_service),
) -> Dict[str, Any]:
    """Batch-synthesizes theses for all uncovered assets in the portfolio."""
    try:
        return service.batch_synthesize_portfolio_theses(
            portfolio_id=req.portfolio_id,
            dry_run=req.dry_run,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch thesis synthesis failed: {exc}",
        )


@router.post("/earnings-bridge")
def generate_earnings_bridge(
    req: EarningsBridgeRequestDTO = Body(...),
) -> Dict[str, Any]:
    """Calculates deterministic reported-vs-expected driver bridge with math invariant validation."""
    try:
        now_date = req.as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        drivers = []
        if req.revenue_drivers:
            for d in req.revenue_drivers:
                cat = DriverCategory(d.category.upper()) if d.category.upper() in DriverCategory.__members__ else DriverCategory.ONE_OFF_OTHER
                drivers.append(
                    EarningsDriver(
                        name=d.name,
                        category=cat,
                        impact_value=Decimal(str(d.impact_value)),
                        unit=d.unit,
                        narrative=d.narrative,
                    )
                )

        bridge = EarningsBridgeEngine.build_earnings_bridge(
            ticker=req.ticker,
            period=req.period,
            as_of_date=now_date,
            revenue_reported=Decimal(str(req.revenue_reported)),
            revenue_consensus=Decimal(str(req.revenue_consensus)),
            eps_reported=Decimal(str(req.eps_reported)),
            eps_consensus=Decimal(str(req.eps_consensus)),
            revenue_unit=req.revenue_unit,
            eps_unit=req.eps_unit,
            ebit_margin_reported=Decimal(str(req.ebit_margin_reported)) if req.ebit_margin_reported is not None else None,
            ebit_margin_consensus=Decimal(str(req.ebit_margin_consensus)) if req.ebit_margin_consensus is not None else None,
            revenue_drivers=drivers,
        )
        return bridge.to_dict()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to generate earnings bridge: {exc}",
        )


@router.post("/{ticker}", response_model=ThesisClaimDTO)
def record_or_revise_thesis(
    ticker: str,
    req: CreateThesisRequestDTO,
    service: ThesisApplicationService = Depends(get_thesis_service),
) -> ThesisClaimDTO:
    """Records a new thesis or appends a revision to the append-only ledger."""
    try:
        falsifiers = [
            FalsifierRule(
                falsifier_id=f.falsifier_id,
                metric_name=f.metric_name,
                operator=f.operator,
                threshold_value=Decimal(f.threshold_value),
                unit=f.unit,
                period_basis=f.period_basis,
                invalidation_impact=f.invalidation_impact,
                narrative_explanation=f.narrative_explanation,
            )
            for f in req.falsifiers
        ]
        catalysts = [
            CatalystEvent(
                catalyst_id=c.catalyst_id,
                event_name=c.event_name,
                expected_date_str=c.expected_date_str,
                confidence_window_days=c.confidence_window_days,
                source_locator=c.source_locator,
                status=c.status,
                observed_date_str=c.observed_date_str,
            )
            for c in req.catalysts
        ]
        tier = EvidenceTier(req.evidence_tier.upper()) if req.evidence_tier.upper() in EvidenceTier.__members__ else EvidenceTier.DOCUMENTED

        if req.prior_claim_id:
            record = service.revise_thesis(
                ticker=ticker,
                claim_statement=req.claim_statement,
                falsifiers=falsifiers,
                catalysts=catalysts,
                evidence_tier=tier,
                author_role=req.author_role,
                prior_claim_id=req.prior_claim_id,
                as_of_utc=req.as_of_utc,
            )
        else:
            record = service.record_new_thesis(
                ticker=ticker,
                claim_statement=req.claim_statement,
                falsifiers=falsifiers,
                catalysts=catalysts,
                evidence_tier=tier,
                author_role=req.author_role,
                as_of_utc=req.as_of_utc,
            )
        return ThesisClaimDTO(**record.to_dict())
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to record thesis: {exc}",
        )

