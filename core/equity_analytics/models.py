"""Pure domain models for Position Review & Decision Framework.

Architecture Rule (Layer 1 - Core):
Pure Python primitives and dataclasses only — strictly zero external dependencies (no FastAPI, Pydantic,
SQLite, or yfinance) per Hexagonal Architecture AST Domain Layer Isolation rules.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PriceMoveDecomposition:
    """Decomposes historical price change into fundamental EPS growth vs P/E multiple expansion."""
    horizon_label: str                  # e.g. "1Y", "3Y", "since_entry"
    start_date: str                     # YYYY-MM-DD
    end_date: str                       # YYYY-MM-DD
    start_price: float
    end_price: float
    total_return_pct: float             # ((end_price - start_price) / start_price) * 100
    start_eps: float
    end_eps: float
    eps_growth_pct: float               # ((end_eps - start_eps) / abs(start_eps)) * 100
    start_pe: float
    end_pe: float
    pe_expansion_pct: float             # ((end_pe - start_pe) / start_pe) * 100
    primary_driver: str                 # "earnings_growth" | "multiple_expansion" | "twin_drivers" | "twin_headwinds"
    diagnosis_tier: str                 # "FUNDAMENTAL_DRIVEN" | "SPECULATIVE_EXPANSION" | "VALUATION_COMPRESSION_CUSHION" | "TWIN_DETERIORATION"
    diagnosis_summary: str


@dataclass(frozen=True)
class ThenVsNowSnapshot:
    """Contrasts market regime and company fundamentals at entry period versus today (Anti-Anchoring)."""
    is_held: bool                       # True if user holds position in portfolio
    period_label: str                   # e.g. "Around Entry (Nov 2023)" or "1 Year Ago (Oct 2025)"
    then_date: str
    now_date: str
    then_price: float
    now_price: float
    what_was_true_then: str             # Descriptive market state when position was bought
    what_is_true_now: str               # Current business and expectation reality
    thesis_reflection_question: str     # Reflection observation for investor to confirm or correct
    then_pe: Optional[float] = None
    now_pe: Optional[float] = None
    then_growth_pct: Optional[float] = None
    now_growth_pct: Optional[float] = None


@dataclass(frozen=True)
class PortfolioAwareTrigger:
    """An actionable, checkable condition linked directly to a decision case impact."""
    trigger_id: str
    title: str
    condition_text: str
    category: str                       # "event_catalyst" | "fundamental_floor" | "valuation_ceiling" | "portfolio_concentration" | "anti_averaging_down"
    case_impact: str                    # "SETTLES_BUY_VS_SELL" | "BREAKS_BUY_CASE" | "STRENGTHENS_SELL_CASE" | "FORCES_PORTFOLIO_TRIM" | "BLOCKS_DIP_BUYING"
    impact_description: str
    is_currently_triggered: bool
    metric_current_value: Optional[float] = None
    metric_threshold_value: Optional[float] = None


@dataclass
class PositionReviewReport:
    """Comprehensive single-stock position review report."""
    ticker: str
    as_of_date: str
    holding_summary: Optional[Dict[str, Any]] = None  # units, avg_cost, pnl_pct, weight_pct, added_direction
    decompositions: List[PriceMoveDecomposition] = field(default_factory=list)
    then_vs_now: Optional[ThenVsNowSnapshot] = None
    triggers: List[PortfolioAwareTrigger] = field(default_factory=list)
    honest_gaps: List[str] = field(default_factory=list)
    company_name: Optional[str] = None
    action_stance: Optional[str] = None
    action_stance_reason: Optional[str] = None
