"""Opportunity Finder Domain Models.

Architecture Layer: core/opportunity_finder/ (Pure Domain)
Value objects and dataclasses for 3-Dimensional Opportunity Evaluation:
1. Macro / Sector Alignment
2. Investor Principles / Axis Fit
3. Portfolio Look-Through / Capacity
Strictly adheres to Pure Domain: no frameworks, Pydantic, HTTP, or DB imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

PolicyFitStatus = Literal["pass", "fail", "needs_review"]
EvidenceReadiness = Literal["sufficient", "partial", "stale", "unavailable"]
MacroAlignment = Literal["tailwind", "neutral", "headwind", "unavailable"]
CapacityStatus = Literal["available", "constrained", "unknown", "not_applicable"]
ReportReadiness = Literal["ready", "degraded", "not_ready"]
CandidatePool = Literal["POOL_A_FUND_LOOK_THROUGH", "POOL_B_EQUITY_SIDECAR"]


@dataclass
class OpportunityCandidate:
    """Investable security candidate merged from Pool A (fund constituents) and Pool B (sidecars)."""
    security_id: str                          # e.g. "US:MSFT", "TH:CPALL"
    issuer_id: str                            # e.g. "MSFT", "CPALL"
    symbol: str                               # e.g. "MSFT", "CPALL"
    market: Literal["US", "TH", "GLOBAL"]
    currency: str                             # "USD", "THB"
    asset_type: Literal["EQUITY", "REIT", "FUND", "ETF", "GOLD", "BOND", "CASH", "UNKNOWN"]
    company_name: str
    sector_name: str                          # GICS English
    sector_name_th: str
    source_pools: List[CandidatePool] = field(default_factory=list)
    source_refs: List[str] = field(default_factory=list)   # e.g. ["fund:SCBS&P500E", "sidecar:AAPL"]
    already_directly_held: bool = False
    already_watchlisted: bool = False
    current_direct_value_thb: Decimal = Decimal("0")
    current_indirect_value_thb: Decimal = Decimal("0")
    current_total_exposure_thb: Decimal = Decimal("0")
    current_total_weight_pct: Decimal = Decimal("0")


@dataclass
class CandidateEvaluationResult:
    """Deterministic 3-Dimensional evaluation result for a candidate."""
    candidate: OpportunityCandidate
    # Dimension 1: Investor Principles / Axis
    policy_fit: PolicyFitStatus
    policy_reasons: List[str] = field(default_factory=list)
    suggested_bucket_id: Optional[str] = None
    applied_bucket_name: Optional[str] = None

    # Dimension 2: Equity Evidence Readiness
    evidence_readiness: EvidenceReadiness = "unavailable"
    composite_score: Optional[float] = None
    valuation_upside_pct: Optional[Decimal] = None
    target_price: Optional[Decimal] = None
    target_price_basis: Optional[str] = None
    moat_rating: Optional[str] = None
    evidence_reasons: List[str] = field(default_factory=list)

    # Dimension 3: Macro / Sector Alignment
    macro_alignment: MacroAlignment = "neutral"
    macro_stage: Optional[str] = None         # "Leading", "Improving", "Weakening", "Lagging"
    macro_reasons: List[str] = field(default_factory=list)

    # Portfolio Capacity & Exposure Guard
    capacity_status: CapacityStatus = "unknown"
    remaining_capacity_thb: Optional[Decimal] = None
    issuer_room_thb: Optional[Decimal] = None
    sector_room_thb: Optional[Decimal] = None
    capacity_reasons: List[str] = field(default_factory=list)

    # Prioritization & Actionability
    research_priority: int = 3                 # 1: High Conviction, 2: Secondary, 3: Watch/Monitor, 4: Review Needed
    priority_reasons: List[str] = field(default_factory=list)
    is_actionable: bool = False                # True if pass, tailwind/neutral, and capacity available


@dataclass
class OpportunityReport:
    """Consolidated Opportunity Finder evaluation report."""
    portfolio_id: str
    evaluation_id: str
    evaluated_at: str
    readiness: ReportReadiness
    candidates: List[CandidateEvaluationResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    total_candidates_scanned: int = 0
    passed_candidates_count: int = 0
    snapshot_meta: Dict[str, Any] = field(default_factory=dict)
