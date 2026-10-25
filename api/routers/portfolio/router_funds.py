"""FastAPI Router for Thai Mutual Funds and Look-Through Constituents.

Architecture Layer: api/routers/portfolio/
Handles:
- GET /api/portfolio/funds/{symbol}/constituents: Single fund constituent look-through
- GET /api/portfolio/{portfolio_id}/look-through: Portfolio-wide look-through overlap analysis
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import require_session
from api.dependencies import get_fund_look_through_service
from api.routers.portfolio.common import handle_portfolio_exceptions
from application.portfolio.fund_look_through_service import FundLookThroughService
from schemas.portfolio_advisory_schemas import (
    FundConstituentsDTO,
    PortfolioLookThroughReportDTO,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["Portfolio Funds"])


@router.get(
    "/api/portfolio/funds/{symbol}/constituents",
    response_model=FundConstituentsDTO,
    summary="Get single fund top holdings, sector breakdown, and asset allocation",
    dependencies=[Depends(require_session)],
)
async def get_fund_constituents_endpoint(
    symbol: str,
    look_through_service: FundLookThroughService = Depends(get_fund_look_through_service),
) -> FundConstituentsDTO:
    """Returns look-through constituent holdings and sector distributions for a Thai mutual fund or PVD symbol."""
    with handle_portfolio_exceptions():
        dto = look_through_service.get_fund_constituents_dto(symbol=symbol)
        if not dto:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Constituents data not found for fund '{symbol}'",
            )
        return dto


@router.get(
    "/api/portfolio/{portfolio_id}/look-through",
    response_model=PortfolioLookThroughReportDTO,
    summary="Get portfolio-wide look-through overlap and true economic exposure",
    dependencies=[Depends(require_session)],
)
async def get_portfolio_look_through_endpoint(
    portfolio_id: str,
    look_through_service: FundLookThroughService = Depends(get_fund_look_through_service),
) -> PortfolioLookThroughReportDTO:
    """Calculates cross-fund equity overlaps, true sector allocations, and hidden concentration caps for a portfolio."""
    with handle_portfolio_exceptions():
        return look_through_service.get_portfolio_look_through_dto(portfolio_id=portfolio_id)

