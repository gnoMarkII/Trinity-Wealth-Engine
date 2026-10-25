"""Pure Domain Models and Enums for 2-Tier Hybrid Portfolio Advisory Engine.

Architecture Layer: core/portfolio_advisory/ (Pure Domain)
No third-party framework or database imports allowed per Hexagonal Architecture rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class AdvisoryDataMode(str, Enum):
    """Graceful degradation modes based on upstream data freshness and availability."""
    FULL_CROSS_SCALE = "FULL_CROSS_SCALE"  # Macro + Micro Quant + Portfolio all available
    PORTFOLIO_PLUS_MICRO = "PORTFOLIO_PLUS_MICRO"  # Macro missing/stale, Micro + Portfolio available
    PORTFOLIO_BUCKET_ONLY = "PORTFOLIO_BUCKET_ONLY"  # Micro missing/stale, Portfolio balance only


class AssetEvaluationTrack(str, Enum):
    """3-Track Asset Classification for financial advice."""
    TRACK_1_DIRECT_EQUITIES = "TRACK_1_DIRECT_EQUITIES"  # Stocks, Equities, REITs (Evaluated via Macro x Micro)
    TRACK_2_MACRO_LIQUID = "TRACK_2_MACRO_LIQUID"  # Gold, Mutual Funds, ETFs, Bonds (Evaluated via Macro & Drift)
    TRACK_3_LOCKED_VEHICLE = "TRACK_3_LOCKED_VEHICLE"  # Tax vehicles: PVD, RMF, SSF, ThaiESG & Cash


class AdvisoryActionCode(str, Enum):
    """Deterministic Action Codes ordered by risk priority."""
    # 1. Urgent Risk Reduction
    URGENT_REDUCE = "URGENT_REDUCE"  # Stop loss breached or Stage 4 markdown
    TRIM_OVERWEIGHT = "TRIM_OVERWEIGHT"  # Above target or dynamic conviction cap
    LET_WINNER_RUN = "LET_WINNER_RUN"  # Conviction >= 8.0, profit > 0, cap raised to 20%

    # 2. Capital Deployment (Separated into Immediate vs Conditional Buy-Zone)
    ACCUMULATE_NOW = "ACCUMULATE_NOW"  # Price currently in buy zone -> Immediate Execution Budget
    BREAKOUT_BUY = "BREAKOUT_BUY"  # Technical breakout confirmed -> Immediate Execution Budget
    ACCUMULATE_ON_DIP = "ACCUMULATE_ON_DIP"  # Price above buy zone -> Conditional Limit Order Budget

    # 3. Expansion & Hold
    INITIATE_WATCHLIST = "INITIATE_WATCHLIST"  # High conviction watchlist item beating hurdle +1.5
    HOLD_WITHIN_BAND = "HOLD_WITHIN_BAND"  # Position drift within +/-2.5% or < 10,000 THB
    HOLD_LOCKED_VEHICLE = "HOLD_LOCKED_VEHICLE"  # Tax-locked retirement vehicle / cash reserves

    # 4. Data Anomaly & Price Guard
    QUARANTINED_DATA_ANOMALY = "QUARANTINED_DATA_ANOMALY"  # Price scale, currency mismatch, or invalid split
    WAIT_PRICE_REFRESH = "WAIT_PRICE_REFRESH"  # Price older than 3 business days


@dataclass
class DataAnomalyWarning:
    """Guardrail warning detailing data or temporal anomalies detected before engine execution."""
    guardrail_code: str
    ticker: str
    message: str
    severity: Literal["critical", "warning", "info"] = "warning"
    remedy_hint: str = ""


@dataclass
class QuarantinedHoldingItem:
    """Holding isolated by safety guardrails to protect capital from data anomalies."""
    symbol: str
    reason: str
    guardrail_code: str
    current_value_thb: Decimal
    details: str = ""


@dataclass
class LotSizingBreakdown:
    """Market-aware order lot sizing adhering to real exchange rules."""
    market: Literal["TH", "US", "GLOBAL"]
    board_lot_shares: Optional[int] = None  # SET 100-share board lot multiplier
    odd_lot_shares: Optional[int] = None  # SET odd lot shares (1..99)
    whole_shares: Optional[int] = None  # US exchange whole shares
    fractional_shares: Optional[Decimal] = None  # US 4-decimal place fractional shares (Dime/WealthX)
    amount_thb: Decimal = Decimal("0")
    amount_native: Decimal = Decimal("0")
    estimated_execution_price_native: Decimal = Decimal("0")
    native_currency: Literal["THB", "USD"] = "THB"


@dataclass
class DualCurrencyCashPlan:
    """Two-basket cash ledger and execution budget preserving dual currency solvency."""
    cash_thb_available: Decimal
    cash_usd_available: Decimal
    usd_thb_fx_rate: Decimal
    immediate_deploy_thb: Decimal = Decimal("0")
    immediate_deploy_usd: Decimal = Decimal("0")
    conditional_limit_order_thb: Decimal = Decimal("0")
    conditional_limit_order_usd: Decimal = Decimal("0")
    thb_surplus_after_immediate: Decimal = Decimal("0")
    usd_surplus_after_immediate: Decimal = Decimal("0")
    fx_conversion_needed_thb: Decimal = Decimal("0")  # THB to convert to USD if USD cash insufficient


@dataclass
class HoldingAdvisoryItem:
    """Deterministic per-holding advice synthesized across Macro x Micro x Portfolio."""
    symbol: str
    asset_name: str
    asset_type: str
    market: Literal["TH", "US"]
    native_currency: Literal["THB", "USD"]
    evaluation_track: AssetEvaluationTrack
    bucket_id: Optional[str]
    bucket_name: Optional[str]
    current_units: Decimal
    current_price_native: Decimal
    current_value_thb: Decimal
    current_weight_pct: Decimal
    target_weight_pct: Decimal
    weight_drift_pct: Decimal
    action_code: AdvisoryActionCode
    action_rationale: str
    lot_sizing: Optional[LotSizingBreakdown] = None
    is_immediate_execution: bool = False
    conditional_buy_zone_min: Optional[Decimal] = None
    conditional_buy_zone_max: Optional[Decimal] = None
    core_conviction_score: Optional[Decimal] = None
    business_conviction_score: Optional[Decimal] = None
    price_stage: Optional[str] = None
    live_upside_12m_pct: Optional[Decimal] = None
    macro_alignment: Literal["tailwind", "neutral", "headwind", "unavailable"] = "unavailable"
    sector_quadrant: Optional[str] = None
    breached_falsifiers: list[str] = field(default_factory=list)
    data_warnings: list[str] = field(default_factory=list)


@dataclass
class PortfolioRiskRadar:
    """Cross-portfolio risk and concentration radar metrics."""
    weighted_beta: Optional[Decimal] = None
    sector_crowding: dict[str, Decimal] = field(default_factory=dict)  # Sectors > 30% NAV
    fx_exposure_pct: dict[str, Decimal] = field(default_factory=dict)  # e.g. {"THB": 65.0, "USD": 35.0}
    purpose_bucket_drift: dict[str, Decimal] = field(default_factory=dict)
    unassigned_holdings_value_thb: Decimal = Decimal("0")
    unassigned_holdings_nav_pct: Decimal = Decimal("0")
    unassigned_drift_guard_active: bool = False  # True if unassigned holdings > 5% NAV
    hhi: Optional[Decimal] = None  # Herfindahl-Hirschman Index (0.0000 .. 1.0000)
    effective_holdings_count: Optional[Decimal] = None  # 1 / HHI
    top_1_weight_pct: Decimal = Decimal("0")
    top_3_weight_pct: Decimal = Decimal("0")
    top_5_weight_pct: Decimal = Decimal("0")
    concentration_risk_level: Literal["LOW", "MODERATE", "HIGH"] = "LOW"


@dataclass
class PortfolioAdvisoryQuantReport:
    """Complete Tier 1 Deterministic Portfolio Advisory Quant Report (< 50ms)."""
    evaluated_at: str
    data_mode: AdvisoryDataMode
    portfolio_state_hash: str
    total_nav_thb: Decimal
    cash_plan: DualCurrencyCashPlan
    risk_radar: PortfolioRiskRadar
    holdings_advisory: list[HoldingAdvisoryItem] = field(default_factory=list)
    quarantined_holdings: list[QuarantinedHoldingItem] = field(default_factory=list)
    anomaly_warnings: list[DataAnomalyWarning] = field(default_factory=list)
    macro_regime_summary: Optional[dict[str, Any]] = None
    cached_narrative: Optional[dict[str, Any]] = None
    look_through_summary: Optional[Any] = None


@dataclass
class PortfolioAdvisoryNarrative:
    """Tier 2 Structured Strategic Playbook Synthesized by LLM."""
    synthesized_at: str
    portfolio_state_hash: str
    macro_executive_summary: str
    immediate_action_playbook: list[str]
    conditional_dip_plan: list[str]
    risk_mitigation_and_hedging: list[str]
    whats_working: list[str] = field(default_factory=list)
    what_concerns_me: list[str] = field(default_factory=list)
    strategic_discussion_points: list[str] = field(default_factory=list)
    consistency_verified: bool = True
