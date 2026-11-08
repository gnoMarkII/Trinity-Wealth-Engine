"""FastAPI Router for Portfolio Advisory Desk.

Architecture Layer: api/routers/portfolio/
Handles:
- GET /advisory: Deterministic Tier 1 calculations, 11-point guardrails, and cached narrative (< 50ms)
- POST /advisory/synthesize: Tier 2 AI Playbook synthesis with Post-LLM consistency validation
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, status

from api.auth import require_session
from api.dependencies import get_portfolio_advisory_service, get_pm_review_service
from application.manager.pm_review_service import PMReviewApplicationService
from api.schemas.portfolio import FactorExposureSnapshotDTO
from api.routers.portfolio.common import handle_portfolio_exceptions
from application.portfolio.advisory_service import PortfolioAdvisoryService
from schemas.portfolio_advisory_schemas import (
    PortfolioAdvisoryNarrativeDTO,
    PortfolioAdvisoryResponseDTO,
    SynthesizeAdvisoryPlaybookRequest,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["Portfolio Advisory"])


@router.get(
    "/api/portfolio/advisory",
    response_model=PortfolioAdvisoryResponseDTO,
    summary="Get Tier 1 Portfolio Advisory Report and Cached Playbook",
    dependencies=[Depends(require_session)],
)
async def get_portfolio_advisory_endpoint(
    refresh_prices: bool = Query(default=False, description="Whether to trigger an immediate live price fetch before evaluation"),
    portfolio_id: str = Query(default="default", description="Portfolio identifier to evaluate"),
    advisory_service: PortfolioAdvisoryService = Depends(get_portfolio_advisory_service),
) -> PortfolioAdvisoryResponseDTO:
    """Returns deterministic Tier 1 cross-scale advice and cached AI playbook in < 50ms."""
    with handle_portfolio_exceptions():
        if portfolio_id == "default":
            return advisory_service.get_portfolio_advisory(refresh_prices=refresh_prices)
        return advisory_service.get_portfolio_advisory(refresh_prices=refresh_prices, portfolio_id=portfolio_id)


@router.post(
    "/api/portfolio/advisory/synthesize",
    response_model=PortfolioAdvisoryNarrativeDTO,
    status_code=status.HTTP_200_OK,
    summary="Synthesize AI Strategic Playbook",
    dependencies=[Depends(require_session)],
)
async def synthesize_advisory_playbook_endpoint(
    request: Optional[SynthesizeAdvisoryPlaybookRequest] = None,
    portfolio_id: str = Query(default="default", description="Portfolio identifier to synthesize"),
    advisory_service: PortfolioAdvisoryService = Depends(get_portfolio_advisory_service),
) -> PortfolioAdvisoryNarrativeDTO:
    """Executes single-pass Tier 2 strategic playbook synthesis and validates consistency against numbers."""
    with handle_portfolio_exceptions():
        force_refresh = request.force_refresh if request else False
        target_pid = (request.portfolio_id if request and request.portfolio_id != "default" else portfolio_id)
        if target_pid == "default":
            return advisory_service.synthesize_playbook(force_refresh=force_refresh)
        return advisory_service.synthesize_playbook(force_refresh=force_refresh, portfolio_id=target_pid)


@router.get(
    "/api/portfolio/factor-exposure",
    response_model=FactorExposureSnapshotDTO,
    summary="Get Chapter 8 Factor Exposure & Missing Price Invariant Snapshot",
    dependencies=[Depends(require_session)],
)
async def get_factor_exposure_endpoint(
    portfolio_id: str = Query(default="default", description="Portfolio identifier to evaluate"),
    service: PMReviewApplicationService = Depends(get_pm_review_service),
) -> FactorExposureSnapshotDTO:
    """Returns Chapter 8 Gross/Net Exposure, Long/Short breakdown, and 1-Factor Beta."""
    packet = service.build_morning_packet(portfolio_id=portfolio_id)
    p_delta = packet.portfolio_delta
    missing_symbols = [
        q["symbol"] for q in packet.quarantined_holdings
        if q.get("guardrail_code") in ("MISSING_PRICE_EXCEPTION", "MISSING_AS_OF_PRICE")
    ]
    return FactorExposureSnapshotDTO(
        gross_market_value=float(p_delta.get("gross_market_value", 0.0)),
        net_market_value=float(p_delta.get("net_market_value", 0.0)),
        long_market_value=float(p_delta.get("long_market_value", 0.0)),
        short_market_value=float(p_delta.get("short_market_value", 0.0)),
        long_exposure_pct=float(p_delta.get("long_exposure_pct", 100.0)),
        short_exposure_pct=float(p_delta.get("short_exposure_pct", 0.0)),
        net_exposure_pct=float(p_delta.get("net_exposure_pct", 100.0)),
        eligible_positions_count=int(p_delta.get("eligible_positions_count", 0)),
        portfolio_beta=p_delta.get("portfolio_beta"),
        beta_benchmark=p_delta.get("beta_benchmark", "SPY"),
        factor_observations_count=p_delta.get("factor_observations_count"),
        status="REVIEW_REQUIRED",
        missing_price_symbols=missing_symbols,
    )
