"""API Router for 3D Smart Opportunity Finder.

Architecture Layer: api/routers/portfolio/ (Inbound Transport / HTTP Only)
Endpoint: GET /api/portfolio/watchlist/opportunities
Adheres strictly to AST Rule S2: Pure HTTP delegation, no direct filesystem or DB I/O.
"""
from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, Depends, Query

from api.dependencies import get_portfolio_opportunity_service
from api.routers.portfolio.common import handle_portfolio_exceptions
from application.portfolio.opportunity_service import PortfolioOpportunityService
from schemas.opportunity_schemas import PortfolioOpportunitiesResponseDTO

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/portfolio/watchlist", tags=["Portfolio Watchlist Opportunities"])


@router.get(
    "/opportunities",
    response_model=PortfolioOpportunitiesResponseDTO,
    summary="Get 3D Smart Opportunity Finder candidates for portfolio watchlist",
)
def get_portfolio_opportunities(
    portfolio_id: str = Query("default", description="Portfolio ID to evaluate opportunities for"),
    force_refresh: bool = Query(False, description="Bypass cache and force re-evaluation"),
    service: PortfolioOpportunityService = Depends(get_portfolio_opportunity_service),
) -> PortfolioOpportunitiesResponseDTO:
    """Evaluate investable candidates across Macro Alignment, Investor Axis, and Portfolio Capacity."""
    with handle_portfolio_exceptions():
        return service.get_opportunities(
            portfolio_id=portfolio_id,
            force_refresh=force_refresh,
        )

