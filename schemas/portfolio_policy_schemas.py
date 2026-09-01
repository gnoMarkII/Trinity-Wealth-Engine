"""Schemas for Scoped Portfolio Policy & Execution Constraints (Phase 5 & v3.1).

Defines user/strategy-level policy constraints: allocation targets, cash buffers,
sector/country concentration limits, ADTV liquidity limits, and risk budgets.
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class PortfolioPolicy(BaseModel):
    """Scoped Portfolio Policy configuration (v3.1)"""
    policy_id: str = Field(..., description="Unique policy identifier")
    portfolio_id: str = Field(..., description="Target portfolio identifier")
    version: str = "1.0"
    owner: str = "user"
    scope: Literal["portfolio_wide", "bucket", "asset"] = "portfolio_wide"
    target_ticker: Optional[str] = Field(None, description="Specific ticker if scope == 'asset'")
    target_bucket: Optional[str] = Field(None, description="Specific asset bucket if scope == 'bucket'")
    target_weight_pct: float = Field(..., ge=0.0, le=100.0, description="Target allocation weight (%)")
    max_weight_pct: float = Field(..., ge=0.0, le=100.0, description="Maximum allocation ceiling (%)")
    max_sector_exposure_pct: float = Field(default=25.0, ge=0.0, le=100.0, description="Max exposure to a single sector (%)")
    max_country_exposure_pct: float = Field(default=60.0, ge=0.0, le=100.0, description="Max exposure to a single country (%)")
    max_currency_exposure_pct: float = Field(default=60.0, ge=0.0, le=100.0, description="Max exposure to a single foreign currency (%)")
    risk_budget_volatility_cap_pct: float = Field(default=18.0, ge=0.0, description="Portfolio annualized volatility cap (%)")
    max_correlation_limit: float = Field(default=0.70, ge=-1.0, le=1.0, description="Max pairwise correlation limit with major holdings")
    minimum_cash_buffer_pct: float = Field(default=5.0, ge=0.0, le=100.0, description="Required unencumbered cash buffer (%)")
    minimum_margin_of_safety_pct: float = Field(default=15.0, ge=0.0, description="Minimum required DCF upside (%)")
    max_adtv_participation_rate: float = Field(
        default=0.02,
        ge=0.001,
        le=0.20,
        description="Max allowed order size as a fraction of 20D ADTV (e.g. 0.02 = 2%)"
    )
