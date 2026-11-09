"""Pydantic Transport DTOs for 3D Smart Opportunity Finder.

Architecture Layer: schemas/ (Transport / API Contracts)
Provides serialization for /api/portfolio/watchlist/opportunities endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class OpportunityCandidateDTO(BaseModel):
    """Investable candidate metadata and current exposure."""
    security_id: str = Field(description="Unique security identity, e.g. 'US:MSFT' or 'TH:CPALL'")
    issuer_id: str = Field(description="Issuer root ticker symbol")
    symbol: str = Field(description="Stock symbol / ticker")
    market: str = Field(description="Market classification: US, TH, or GLOBAL")
    currency: str = Field(description="Quotation currency (USD, THB)")
    asset_type: str = Field(description="Asset classification (EQUITY, REIT, etc.)")
    company_name: str = Field(description="Legal entity or display name")
    sector_name: str = Field(description="Mapped GICS English sector name")
    sector_name_th: str = Field(description="Sector name in Thai")
    source_pools: List[str] = Field(default_factory=list, description="Candidate origin pools: POOL_A_FUND_LOOK_THROUGH, POOL_B_EQUITY_SIDECAR")
    source_refs: List[str] = Field(default_factory=list, description="Provenance references, e.g. ['fund:SCBS&P500E', 'sidecar:MSFT']")
    already_directly_held: bool = Field(default=False, description="True if stock already held as a direct position")
    already_watchlisted: bool = Field(default=False, description="True if stock already present in current portfolio's watchlist")
    current_direct_value_thb: float = Field(default=0.0, description="Direct market value in THB")
    current_indirect_value_thb: float = Field(default=0.0, description="Underlying mutual fund exposure in THB")
    current_total_exposure_thb: float = Field(default=0.0, description="Consolidated direct + indirect exposure in THB")
    current_total_weight_pct: float = Field(default=0.0, description="Consolidated portfolio weight percentage")


class CandidateEvaluationResultDTO(BaseModel):
    """Consolidated 3D evaluation result for a candidate."""
    candidate: OpportunityCandidateDTO = Field(description="Candidate details")
    policy_fit: str = Field(description="Investor principles fit: pass, fail, needs_review")
    policy_reasons: List[str] = Field(default_factory=list, description="Provenance rationale for policy evaluation")
    suggested_bucket_id: Optional[str] = Field(default=None, description="Suggested allocation bucket ID from applied policy")
    applied_bucket_name: Optional[str] = Field(default=None, description="Applied allocation bucket display name")

    evidence_readiness: str = Field(description="Equity research readiness: sufficient, partial, stale, unavailable")
    composite_score: Optional[float] = Field(default=None, description="Micro quant composite score (0-100)")
    valuation_upside_pct: Optional[float] = Field(default=None, description="Estimated fundamental upside percentage")
    target_price: Optional[float] = Field(default=None, description="Fair value / target price in local currency")
    target_price_basis: Optional[str] = Field(default=None, description="Methodology basis of target price")
    moat_rating: Optional[str] = Field(default=None, description="Economic moat rating (Wide, Narrow, None)")
    evidence_reasons: List[str] = Field(default_factory=list, description="Evidence status rationale")

    macro_alignment: str = Field(description="Macro sector alignment: tailwind, neutral, headwind, unavailable")
    macro_stage: Optional[str] = Field(default=None, description="Sector rotation quadrant (Leading, Improving, Weakening, Lagging)")
    macro_reasons: List[str] = Field(default_factory=list, description="Macro alignment rationale")

    capacity_status: str = Field(description="Portfolio room status: available, constrained, unknown, not_applicable")
    remaining_capacity_thb: Optional[float] = Field(default=None, description="Remaining deployable capacity in THB")
    issuer_room_thb: Optional[float] = Field(default=None, description="Remaining single-stock room in THB before Cap")
    sector_room_thb: Optional[float] = Field(default=None, description="Remaining sector room in THB before Cap")
    capacity_reasons: List[str] = Field(default_factory=list, description="Capacity calculation rationale")

    research_priority: int = Field(default=3, description="1: High Conviction, 2: Secondary, 3: Watch/Monitor, 4: Review Needed")
    priority_reasons: List[str] = Field(default_factory=list, description="Priority determination rationale")
    is_actionable: bool = Field(default=False, description="True if passes policy, has room, and ready for study")
    liquidity_status: Optional[str] = Field(default=None, description="ADTV liquidity check status")
    adtv_thb: Optional[float] = Field(default=None, description="Dated 20-day ADTV converted to THB")
    liquidity_room_thb: Optional[float] = Field(default=None, description="5% ADTV liquidity participation cap in THB")


class PortfolioOpportunitiesResponseDTO(BaseModel):
    """Full API Response for Portfolio Opportunity Finder."""
    portfolio_id: str = Field(description="Evaluated portfolio ID")
    evaluation_id: str = Field(description="Unique evaluation snapshot ID")
    evaluated_at: str = Field(description="ISO timestamp of evaluation")
    readiness: str = Field(description="Report readiness: ready, degraded, not_ready")
    total_candidates_scanned: int = Field(default=0, description="Total candidates identified across Pool A + B")
    passed_candidates_count: int = Field(default=0, description="Count of candidates passing investor principles")
    candidates: List[CandidateEvaluationResultDTO] = Field(default_factory=list, description="Ranked candidate evaluation results")
    warnings: List[str] = Field(default_factory=list, description="Operational warnings or degraded data notices")
    missing_requirements: List[str] = Field(default_factory=list, description="Missing prerequisites (e.g., unconfirmed axis)")
    snapshot_meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata and parameters used during evaluation")
