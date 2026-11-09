"""Pydantic Schemas for Portfolio Advisory API DTOs and Vault Persistence.

Architecture Layer: schemas/
DTOs and serialization schemas matching core/portfolio_advisory/ domain dataclasses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class LotSizingDTO(BaseModel):
    market: Literal["TH", "US", "GLOBAL"]
    board_lot_shares: Optional[int] = None
    odd_lot_shares: Optional[int] = None
    whole_shares: Optional[int] = None
    fractional_shares: Optional[float] = None
    amount_thb: float
    amount_native: float
    estimated_execution_price_native: float
    native_currency: Literal["THB", "USD"] = "THB"


class DualCurrencyCashPlanDTO(BaseModel):
    cash_thb_available: float
    cash_usd_available: float
    usd_thb_fx_rate: float
    immediate_deploy_thb: float = 0.0
    immediate_deploy_usd: float = 0.0
    conditional_limit_order_thb: float = 0.0
    conditional_limit_order_usd: float = 0.0
    thb_surplus_after_immediate: float = 0.0
    usd_surplus_after_immediate: float = 0.0
    fx_conversion_needed_thb: float = 0.0


class DataAnomalyWarningDTO(BaseModel):
    guardrail_code: str
    ticker: str
    message: str
    severity: Literal["critical", "warning", "info"] = "warning"
    remedy_hint: str = ""


class QuarantinedHoldingDTO(BaseModel):
    symbol: str
    reason: str
    guardrail_code: str
    current_value_thb: float
    details: str = ""


class HoldingAdvisoryDTO(BaseModel):
    symbol: str
    asset_name: str
    asset_type: str
    market: Literal["TH", "US"]
    native_currency: Literal["THB", "USD"]
    evaluation_track: str
    bucket_id: Optional[str] = None
    bucket_name: Optional[str] = None
    current_units: float
    current_price_native: float
    current_value_thb: float
    current_weight_pct: float
    target_weight_pct: float
    weight_drift_pct: float
    action_code: str
    action_rationale: str
    lot_sizing: Optional[LotSizingDTO] = None
    is_immediate_execution: bool = False
    conditional_buy_zone_min: Optional[float] = None
    conditional_buy_zone_max: Optional[float] = None
    core_conviction_score: Optional[float] = None
    business_conviction_score: Optional[float] = None
    price_stage: Optional[str] = None
    live_upside_12m_pct: Optional[float] = None
    macro_alignment: Literal["tailwind", "neutral", "headwind", "unavailable"] = "unavailable"
    sector_quadrant: Optional[str] = None
    breached_falsifiers: list[str] = Field(default_factory=list)
    data_warnings: list[str] = Field(default_factory=list)
    risk_contribution_pct: Optional[float] = None
    estimated_friction_bps: Optional[float] = None
    volatility_pct: Optional[float] = None
    volatility_source: Literal["observed", "estimated_default", "unavailable"] = "unavailable"
    is_volatility_estimated: bool = False


class PortfolioRiskRadarDTO(BaseModel):
    weighted_beta: Optional[float] = None
    sector_crowding: dict[str, float] = Field(default_factory=dict)
    fx_exposure_pct: dict[str, float] = Field(default_factory=dict)
    purpose_bucket_drift: dict[str, float] = Field(default_factory=dict)
    unassigned_holdings_value_thb: float = 0.0
    unassigned_holdings_nav_pct: float = 0.0
    unassigned_drift_guard_active: bool = False
    hhi: Optional[float] = None
    effective_holdings_count: Optional[float] = None
    top_1_weight_pct: float = 0.0
    top_3_weight_pct: float = 0.0
    top_5_weight_pct: float = 0.0
    concentration_risk_level: Literal["LOW", "MODERATE", "HIGH"] = "LOW"
    bucket_active_share_pct: Optional[float] = None
    bucket_active_share_status: Literal["AVAILABLE", "UNAVAILABLE"] = "AVAILABLE"
    risk_contributions_pct: dict[str, float] = Field(default_factory=dict)
    inverse_vol_weights_pct: dict[str, float] = Field(default_factory=dict)
    total_portfolio_volatility_pct: Optional[float] = None
    risk_contribution_status: Literal["AVAILABLE", "UNAVAILABLE_ZERO_VARIANCE", "NO_ACTIVE_ASSETS"] = "AVAILABLE"
    volatility_estimated_symbols: list[str] = Field(default_factory=list)
    drawdown_pct: Optional[float] = None
    drawdown_posture_tier: Literal["NORMAL", "TIER_1_CAUTION", "TIER_2_THROTTLE", "TIER_3_DEFENSIVE"] = "NORMAL"
    drawdown_status: Literal["AVAILABLE", "UNAVAILABLE_UNKNOWN_PEAK"] = "UNAVAILABLE_UNKNOWN_PEAK"
    drawdown_hurdle_adjustment: float = 0.0
    drawdown_action_restrictions: list[str] = Field(default_factory=list)


class PortfolioAdvisoryNarrativeDTO(BaseModel):
    synthesized_at: str = Field(default="")
    portfolio_state_hash: str = Field(default="")
    synthesis_input_hash: str = Field(default="")
    macro_executive_summary: str = Field(default="")
    immediate_action_playbook: list[str] = Field(default_factory=list)
    conditional_dip_plan: list[str] = Field(default_factory=list)
    risk_mitigation_and_hedging: list[str] = Field(default_factory=list)
    whats_working: list[str] = Field(default_factory=list)
    what_concerns_me: list[str] = Field(default_factory=list)
    strategic_discussion_points: list[str] = Field(default_factory=list)
    consistency_verified: bool = True
    generated_by: str = "portfolio_synthesizer"


class FundHoldingItemDTO(BaseModel):
    name: str
    percent: float
    ticker: Optional[str] = None
    asset_type: str = "EQUITY"


class FundSectorItemDTO(BaseModel):
    sector_name_th: str
    sector_name_en: str
    percent: float


class FundConstituentsDTO(BaseModel):
    fund_id: str
    symbol: str
    as_of_date: str
    master_fund_name: Optional[str] = None
    master_fund_ticker: Optional[str] = None
    top_holdings: list[FundHoldingItemDTO] = Field(default_factory=list)
    sector_breakdown: list[FundSectorItemDTO] = Field(default_factory=list)
    asset_allocation: dict[str, float] = Field(default_factory=dict)


class LookThroughHoldingExposureDTO(BaseModel):
    ticker: str
    company_name: str
    market: str
    direct_value_thb: float
    direct_weight_pct: float
    indirect_value_thb: float
    indirect_weight_pct: float
    total_value_thb: float
    total_weight_pct: float
    source_funds: list[tuple[str, float]] = Field(default_factory=list)
    is_cap_breached: bool = False
    is_hidden_concentration: bool = False


class TrueSectorExposureDTO(BaseModel):
    sector_name: str
    sector_name_th: str
    total_value_thb: float
    weight_pct: float
    direct_stocks_pct: float
    funds_pct: float
    is_crowded: bool = False


class PortfolioLookThroughReportDTO(BaseModel):
    portfolio_id: str
    evaluated_at: str
    total_nav_thb: float
    holdings_exposure: list[LookThroughHoldingExposureDTO] = Field(default_factory=list)
    top_overlapped_stocks: list[LookThroughHoldingExposureDTO] = Field(default_factory=list)
    sector_exposure: list[TrueSectorExposureDTO] = Field(default_factory=list)
    hidden_concentration_warnings: list[str] = Field(default_factory=list)
    cap_breached_count: int = 0


class PortfolioAdvisoryResponseDTO(BaseModel):
    evaluated_at: str
    data_mode: str
    portfolio_state_hash: str
    synthesis_input_hash: str = ""
    total_nav_thb: float
    cash_plan: DualCurrencyCashPlanDTO
    risk_radar: PortfolioRiskRadarDTO
    holdings_advisory: list[HoldingAdvisoryDTO] = Field(default_factory=list)
    quarantined_holdings: list[QuarantinedHoldingDTO] = Field(default_factory=list)
    anomaly_warnings: list[DataAnomalyWarningDTO] = Field(default_factory=list)
    macro_regime_summary: Optional[dict[str, Any]] = None
    cached_narrative: Optional[PortfolioAdvisoryNarrativeDTO] = None
    look_through_summary: Optional[PortfolioLookThroughReportDTO] = None


class SynthesizeAdvisoryPlaybookRequest(BaseModel):
    force_refresh: bool = False
    portfolio_id: str = "default"


class AdvisorySidecarRecord(BaseModel):
    """Obsidian Vault record schema for Portfolio Advisory Sidecars."""
    evaluated_at: str
    portfolio_state_hash: str
    synthesis_input_hash: str = ""
    data_mode: str
    total_nav_thb: float
    cash_plan: DualCurrencyCashPlanDTO
    risk_radar: PortfolioRiskRadarDTO
    holdings_advisory: list[HoldingAdvisoryDTO] = Field(default_factory=list)
    quarantined_holdings: list[QuarantinedHoldingDTO] = Field(default_factory=list)
    anomaly_warnings: list[DataAnomalyWarningDTO] = Field(default_factory=list)
    narrative: Optional[PortfolioAdvisoryNarrativeDTO] = None
    look_through_summary: Optional[PortfolioLookThroughReportDTO] = None
