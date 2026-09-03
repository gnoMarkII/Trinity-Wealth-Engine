"""Schemas สำหรับ Equity Intel Pipeline (Micro Quant Agent) — มิเรอร์โครงสร้างของ schemas/macro_schemas.py

แยกชั้นชัดเจน: QuantSignals / DeterministicScorecard (deterministic ล้วน, ห้าม LLM แตะ) vs EquitySentimentContext/EquityNarrativeOutput
(LLM-derived — categorical/text เท่านั้น ไม่มี numeric field ที่ดูเหมือน deterministic)
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

DataStatus = Literal["available", "partial", "unavailable", "not_applicable"]

PriceSource = Literal[
    "ohlcv_close",
    "verified_live_quote",
    "intraday_snapshot",
    "stale_eod",
    "unavailable",
]


class DCFScenario(BaseModel):
    target_price: Optional[float] = None
    upside_pct: Optional[float] = None
    margin_of_safety_pct: Optional[float] = None


class DCFResult(BaseModel):
    wacc_pct: Optional[float] = None
    cost_of_equity_pct: Optional[float] = None
    cost_of_debt_pct: Optional[float] = None
    risk_free_rate_pct: Optional[float] = None
    erp_pct: Optional[float] = None
    observable_refs: list[str] = Field(default_factory=list)
    scenarios: dict[str, DCFScenario] = Field(default_factory=dict)
    valuation_verdict: Literal["undervalued", "fairly_valued", "overvalued", "unavailable"] = "unavailable"
    is_actionable: bool = True
    actionability_reason: Optional[str] = None
    invalidation_reasons: list[str] = Field(default_factory=list)


class MarginMetricItem(BaseModel):
    value_pct: Optional[float] = None
    period_end: Optional[str] = None
    period_type: Optional[str] = None  # "TTM", "quarterly", "annual", "guidance_forward"
    definition: str = ""
    source_provenance: Optional[str] = None


class ExplicitFCFProjection(BaseModel):
    year_index: int  # 1 to 5
    projected_revenue: float
    projected_ebit: float
    projected_ebit_margin_pct: float
    projected_nopat: float
    projected_reinvestment_currency: float  # Absolute net reinvestment (CapEx - Deprec + Delta NWC) in currency units
    projected_fcf: float
    discount_factor: float
    pv_fcf: float


class AtomicMarketSnapshot(BaseModel):
    analysis_price: Optional[float] = None
    analysis_price_as_of: Optional[str] = None
    price_source: PriceSource = "ohlcv_close"
    latest_ohlcv_close: Optional[float] = None
    latest_ohlcv_date: Optional[str] = None
    shares_outstanding: Optional[float] = None
    market_cap: Optional[float] = None
    raw_analysis_price: Optional[str] = None
    raw_analysis_price_str: Optional[str] = None
    market_cap_str: Optional[str] = None
    market_cap_cents: Optional[int] = None
    is_provisional: bool = False
    volume_confirmation: Literal["confirmed", "provisional", "unavailable"] = "confirmed"
    price_sync_status: Literal["synced", "quote_ohlcv_mismatch", "stale", "unavailable"] = "synced"
    freshness_status: Literal["fresh", "stale", "session_synced", "out_of_session", "unavailable"] = "fresh"
    market_session_status: Literal["pre_market", "open", "after_hours", "closed", "unavailable"] = "closed"
    data_freshness_status: Literal["fresh", "stale", "stale_one_session", "stale_multiple_sessions", "unavailable", "unknown"] = "fresh"
    expected_latest_session_date: Optional[str] = None
    actual_latest_session_date: Optional[str] = None
    missing_trading_sessions: int = 0
    retrieved_at: str


class ReverseDCFResult(BaseModel):
    explicit_forecast_5y: list[ExplicitFCFProjection] = Field(default_factory=list)
    sum_pv_5y_fcf: Optional[float] = None
    terminal_value_undiscounted: Optional[float] = None
    terminal_value_pv: Optional[float] = None
    enterprise_value: Optional[float] = None
    net_cash_debt: Optional[float] = None
    equity_value: Optional[float] = None
    intrinsic_value_today: Optional[float] = None
    target_price_12m: Optional[float] = None
    upside_12m_pct: Optional[float] = None
    market_implied_growth_pct: Optional[float] = None
    market_implied_margin_pct: Optional[float] = None
    solver_status: Literal["converged", "bounded_extreme", "no_solution", "not_applicable"] = "converged"
    fixed_parameters: dict[str, Any] = Field(default_factory=dict)
    valuation_horizon_months: int = 12
    status: DataStatus = "available"
    is_eligible: bool = True
    exclusion_reason: Optional[str] = None
    valuation_verdict: Literal["overvalued", "fairly_valued", "undervalued", "unavailable"] = "unavailable"
    is_actionable: bool = True
    actionability_reason: Optional[str] = None
    raw_wacc_pct: Optional[float] = None
    effective_wacc_pct: Optional[float] = None
    wacc_adjustment_reason: Optional[str] = None
    reported_ebit_margin_pct: Optional[float] = None
    ebit_margin_fiscal_period: Optional[str] = None
    ebit_margin_period_type: Optional[Literal["annual", "quarterly", "ttm", "unknown"]] = None
    ebit_margin_source_tier: Optional[Literal["filing_authoritative", "primary_best_effort", "fallback", "unknown"]] = None
    target_price_exit_multiple_12m: Optional[float] = None
    exit_multiple_used: Optional[float] = None
    intrinsic_value_exit_multiple: Optional[float] = None
    consensus_target_price: Optional[float] = None
    consensus_target_high: Optional[float] = None
    consensus_target_low: Optional[float] = None
    analyst_count: Optional[int] = None
    base_revenue: Optional[float] = None
    base_revenue_period_type: Optional[Literal["annual", "quarterly", "ttm", "unknown"]] = None


class SmartMoneyFlags(BaseModel):
    insider_signal: Literal["buying", "selling", "neutral"] = "neutral"
    insider_buy_count_90d: int = 0
    insider_sell_count_90d: int = 0
    institutional_ownership_pct: Optional[float] = None
    insider_ownership_pct: Optional[float] = None
    short_interest_pct: Optional[float] = None
    short_squeeze_risk: bool = False
    overall_smart_money_flag: Literal["bullish_signal", "bearish_signal", "neutral"] = "neutral"


class PiotroskiFScoreBreakdown(BaseModel):
    # Profitability (0-4)
    roa_positive: Optional[bool] = False
    cfo_positive: Optional[bool] = False
    delta_roa_positive: Optional[bool] = False
    accrual_quality: Optional[bool] = False
    
    # Leverage, Liquidity and Source of Funds (0-3)
    delta_leverage_improved: Optional[bool] = False
    delta_liquidity_improved: Optional[bool] = False
    no_share_dilution: Optional[bool] = False
    
    # Operating Efficiency (0-2)
    delta_gross_margin_improved: Optional[bool] = False
    delta_asset_turnover_improved: Optional[bool] = False
    
    f_score: Optional[int] = None
    status: DataStatus = "available"
    is_eligible: bool = True
    exclusion_reason: Optional[str] = None

    @property
    def profitability_points(self) -> int:
        return sum(1 for v in (self.roa_positive, self.cfo_positive, self.delta_roa_positive, self.accrual_quality) if v is True)

    @property
    def leverage_liquidity_points(self) -> int:
        return sum(1 for v in (self.delta_leverage_improved, self.delta_liquidity_improved, self.no_share_dilution) if v is True)

    @property
    def operating_efficiency_points(self) -> int:
        return sum(1 for v in (self.delta_gross_margin_improved, self.delta_asset_turnover_improved) if v is True)


class TacticalSetup(BaseModel):
    price_stage: Literal["STAGE_1_BASE", "STAGE_2_MARKUP", "STAGE_3_DISTRIBUTION", "STAGE_4_MARKDOWN", "UNKNOWN"] = "UNKNOWN"
    current_price: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    atr_14: Optional[float] = None
    key_support_level: Optional[float] = None
    key_resistance_level: Optional[float] = None
    buy_zone_min: Optional[float] = None
    buy_zone_max: Optional[float] = None
    invalidation_stop_loss: Optional[float] = None
    tactical_target_price: Optional[float] = None
    tactical_risk_reward_ratio: Optional[float] = None
    current_rr_ratio: Optional[float] = None
    buy_zone_rr_min: Optional[float] = None
    buy_zone_rr_max: Optional[float] = None
    is_in_buy_zone: Optional[bool] = None
    pullback_entry_status: Literal["below_stop", "in_buy_zone", "between_zone_and_target", "at_or_above_target", "unavailable"] = "unavailable"
    breakout_trigger_price: Optional[float] = None
    breakout_target_price: Optional[float] = None
    breakout_stop_loss: Optional[float] = None
    breakout_planned_rr: Optional[float] = None
    breakout_current_rr: Optional[float] = None
    max_breakout_chase_price: Optional[float] = None
    breakout_entry_status: Literal["pre_trigger", "eligible", "chased", "expired"] = "pre_trigger"
    breakout_entry_eligible: bool = False
    breakout_volume_ratio: Optional[float] = None
    breakout_volume_baseline: Optional[float] = None
    breakout_volume_confirmed: Optional[bool] = None
    horizon_timeframe: str = "1-3M"
    status: DataStatus = "available"


class InsiderConviction(BaseModel):
    open_market_p_count_90d: int = 0
    open_market_p_value_usd: float = 0.0
    open_market_p_value_cents: int = 0
    open_market_p_value_usd_str: str = "0.00"
    open_market_s_count_90d: int = 0
    open_market_s_value_usd: float = 0.0
    open_market_s_value_cents: int = 0
    open_market_s_value_usd_str: str = "0.00"
    rule_10b5_1_s_count_90d: int = 0
    rule_10b5_1_s_value_usd: float = 0.0
    rule_10b5_1_s_value_cents: int = 0
    rule_10b5_1_s_value_usd_str: str = "0.00"
    unflagged_s_count_90d: int = 0
    unflagged_s_value_usd: float = 0.0
    unflagged_s_value_cents: int = 0
    unflagged_s_value_usd_str: str = "0.00"
    tax_withholding_count_90d: int = 0
    tax_withholding_value_usd: float = 0.0
    tax_withholding_value_cents: int = 0
    tax_withholding_value_usd_str: str = "0.00"
    exercise_count_90d: int = 0
    c_suite_p_count: int = 0
    filing_count_90d: int = 0
    transaction_lot_count_90d: int = 0
    rule_10b5_1_filing_count_90d: int = 0
    rule_10b5_1_lot_count_90d: int = 0
    quarantined_filing_count_90d: int = 0
    quarantined_lot_count_90d: int = 0
    insider_buy_range_min: Optional[float] = None
    insider_buy_range_max: Optional[float] = None
    signal_confidence: Literal["high", "moderate", "neutral", "unavailable"] = "neutral"
    amendment_unresolved: bool = False
    status: Literal["bullish_cluster", "moderate_buying", "neutral_no_signal", "selling_activity", "requires_review", "unavailable", "not_applicable"] = "neutral_no_signal"
    data_status: DataStatus = "available"


class VerifiedGuidanceClaim(BaseModel):
    """Explicitly verified numerical or controlled-vocabulary guidance statement bound to quote."""
    metric_name: str  # e.g., "revenue_growth_pct", "operating_margin_trajectory"
    numeric_value: Optional[float] = None
    controlled_vocabulary_value: Optional[str] = None  # "expanding", "stable", "contracting"
    unit: str = "%"  # "%", "bps", "currency", "ratio"
    denominator: Optional[str] = None  # e.g. "YoY", "vs FY24", "Full Year"
    fiscal_period: str  # e.g. "2025-Q2" or "FY2025"
    quote_text: str
    quote_hash: str
    char_start: int
    char_end: int
    source_ref: str  # e.g. "vault:///30_Knowledge_Base/Earnings_Calls/NVDA/2025-Q2_NVDA_Earnings_Call.md#offset_120_245"


class EarningsGuidanceContext(BaseModel):
    latest_fiscal_period: Optional[str] = None
    fiscal_quarter: Optional[str] = None
    call_date: Optional[str] = None
    revenue_growth_guidance_pct: Optional[float] = None
    revenue_guidance_yoy_pct: Optional[float] = None
    operating_margin_trajectory: Optional[Literal["expanding", "stable", "contracting", "unspecified"]] = None
    margin_guidance_direction: Optional[Literal["expanding", "stable", "contracting", "unspecified"]] = None
    management_tone: Optional[Literal["bullish", "neutral", "cautious"]] = "neutral"
    guidance_summary: str = ""
    verified_claims: list[VerifiedGuidanceClaim] = Field(default_factory=list)
    key_executive_quotes: list[str] = Field(default_factory=list)
    guidance_quotes: list[str] = Field(default_factory=list)
    source_note_path: Optional[str] = None
    status: DataStatus = "available"


class DeterministicScorecard(BaseModel):
    core_conviction_score: float = 5.0  # 1.0 - 10.0 (Aggregate / Business conviction)
    business_conviction_score: Optional[float] = None  # 1.0 - 10.0 (Fundamental + Expectations/Guidance only)
    investment_conviction_score: Optional[float] = None  # 1.0 - 10.0 (Requires actionable valuation, else None)
    execution_readiness_score: float = 5.0  # 1.0 - 10.0 (Technical + ADTV + Insider only)
    action_stance: Literal["ACCUMULATE_NOW", "ACCUMULATE_ON_DIP", "BREAKOUT_BUY", "HOLD_WAIT", "REDUCE", "INSUFFICIENT_DATA"] = "HOLD_WAIT"
    action_stance_reason: Optional[str] = None
    stance_mode: Literal["active", "conditional", "wait", "reduce"] = "wait"
    setup_readiness_score: Optional[float] = None
    execution_score_breakdown: Optional[dict[str, Any]] = None
    fundamental_quality_score: Optional[float] = None  # 0-100
    guidance_expectation_score: Optional[float] = None  # 0-100 (Aggregate of analyst + guidance)
    analyst_expectations_score: Optional[float] = None  # 0-100 (from EPS revisions/consensus)
    management_guidance_score: Optional[float] = None  # 0-100 (from verified guidance claims)
    valuation_margin_score: Optional[float] = None  # 0-100
    coverage_pct: float = 100.0
    usable_coverage_pct: float = 100.0
    verified_coverage_pct: float = 100.0
    applicable_pillars_count: int = 4
    methodology_version: str = "2.0.0"
    reweighting_metadata: Optional[dict[str, Any]] = None
    data_quality_flags: list[str] = Field(default_factory=list)


class QuantSignalsFailureResult(BaseModel):
    """Structured Error DTO returned when quant pipeline cannot proceed."""
    run_status: Literal["error"] = "error"
    ticker: str
    market: Literal["TH", "US"]
    error_code: str
    error_message: str
    data_quality_flags: list[str] = Field(default_factory=list)
    evaluated_at: str


class ThesisFalsifier(BaseModel):
    falsifier_id: str
    metric_name: str
    condition: str
    threshold_value: Optional[float] = None
    source_basis: Literal["forecast_driver_delta", "guidance_quote", "statement_baseline"]
    source_ref: Optional[str] = None
    source_quote: Optional[str] = None
    narrative_explanation: str

    @model_validator(mode="after")
    def validate_source_ref_when_threshold_present(self) -> "ThesisFalsifier":
        if self.threshold_value is not None and not self.source_ref:
            raise ValueError(f"Falsifier {self.falsifier_id} has threshold_value={self.threshold_value} but source_ref is missing.")
        return self


class QuantSignals(BaseModel):
    """ผลลัพธ์จาก compute_equity_quant_signals — deterministic 100% ไม่มี field ไหนมาจาก LLM"""
    ticker: str
    market: Literal["TH", "US"]
    company_name: Optional[str] = Field(default=None, description="ชื่อบริษัทเต็ม (yfinance shortName) — ใช้ปรับปรุงคุณภาพ Vault search และหัวรายงาน")
    value_score: Optional[float] = Field(default=None, description="0-100, None ถ้าข้อมูลไม่พอ (เช่น P/E ติดลบ)")
    quality_score: Optional[float] = None
    momentum_score: Optional[float] = Field(
        default=None,
        description="วัดความแรงของโมเมนตัมขาขึ้นเชิงเทคนิค ไม่ใช่คำแนะนำซื้อ — ดู tools/market/quant_scoring.py",
    )
    beta: Optional[float] = None
    volatility_pct: Optional[float] = None
    mdd_pct: Optional[float] = None
    upside_pct: Optional[float] = None
    downside_pct: Optional[float] = None
    raw_analysis_price: Optional[float] = None
    raw_analysis_price_str: Optional[str] = None
    # Growth
    revenue_growth_yoy_pct: Optional[float] = None
    net_income_growth_yoy_pct: Optional[float] = None
    growth_score: Optional[float] = None
    # Dividend
    dividend_yield_pct: Optional[float] = None
    payout_ratio_pct: Optional[float] = None
    dividend_score: Optional[float] = None
    # Solvency — Risk Gate แยกจาก Composite Score (ดู compute_solvency_score docstring)
    de_ratio_pct: Optional[float] = Field(default=None, description="debtToEquity จาก yfinance ตรงๆ (150.0 = D/E 1.5x)")
    current_ratio: Optional[float] = None
    solvency_score: Optional[float] = None
    # Cash Flow & Capital Quality
    fcf_yield_pct: Optional[float] = None
    fcf_margin_pct: Optional[float] = None
    fcf_cagr_3y: Optional[float] = None
    interest_coverage: Optional[float] = None
    net_debt_ebitda: Optional[float] = None
    roic_pct: Optional[float] = None
    ocf_to_net_income: Optional[float] = None
    fcf_quality_score: Optional[float] = None
    debt_quality_score: Optional[float] = None
    # Trading Liquidity
    adtv_local_currency: Optional[float] = Field(
        default=None,
        description="มูลค่าซื้อขายเฉลี่ยต่อวัน สกุลท้องถิ่น (THB สำหรับ TH, USD สำหรับ US) — ไม่ใช่ USD-normalized ห้ามเทียบข้าม market ตรงๆ",
    )
    # Composite
    composite_score: Optional[float] = Field(
        default=None,
        description="Weighted: Value25/Quality25/Growth25/Momentum15/Dividend10 (re-normalize ถ้าขาดมิติ) — Solvency ไม่รวม (risk gate แยก)",
    )
    # Peer/Sector Relative Valuation — Contextual เท่านั้น ไม่รวมใน composite_score
    peer_sector: Optional[str] = None
    peer_count: Optional[int] = None
    pe_vs_peer_avg_pct: Optional[float] = Field(default=None, description="บวก=แพงกว่า peer เฉลี่ย, ลบ=ถูกกว่า")
    peer_relative_score: Optional[float] = None
    # Historical Price Context — Percentile/Z-score ของ 'ราคา' ไม่ใช่ Valuation Multiple (yfinance
    # ไม่มี point-in-time fundamentals พอคำนวณ P/E percentile ย้อนหลังได้) — Contextual เท่านั้น
    price_percentile_5y: Optional[float] = Field(default=None, description="0-100, เปอร์เซ็นไทล์ของราคาปัจจุบันเทียบช่วง 5 ปี (ราคา ไม่ใช่ P/E)")
    price_zscore_5y: Optional[float] = None
    # Earnings Momentum & Revisions — Contextual เท่านั้น ไม่รวมใน composite_score
    eps_revision_net_30d: Optional[int] = Field(default=None, description="จำนวนนักวิเคราะห์ที่ปรับกำไรขึ้น ลบด้วยปรับลง ใน 30 วันล่าสุด (ปีบัญชีปัจจุบัน)")
    eps_estimate_change_30d_pct: Optional[float] = None
    earnings_momentum_score: Optional[float] = None
    # DCF Engine & Valuation
    dcf_result: Optional[DCFResult] = None
    # Smart Money Signals
    smart_money_flags: Optional[SmartMoneyFlags] = None
    evaluated_at: str = Field(description="ISO format string (ไม่ใช่ datetime object)")
    data_quality_flags: list[str] = Field(
        default_factory=list,
        description="เช่น 'insufficient_trading_history:beta', 'negative_earnings:pe_undefined'",
    )
    # Institutional Additive Fields
    piotroski_breakdown: Optional[PiotroskiFScoreBreakdown] = None
    reverse_dcf_result: Optional[ReverseDCFResult] = None
    tactical_setup: Optional[TacticalSetup] = None
    insider_conviction: Optional[InsiderConviction] = None
    earnings_guidance_context: Optional[EarningsGuidanceContext] = None
    deterministic_scorecard: Optional[DeterministicScorecard] = None
    thesis_falsifiers: list[ThesisFalsifier] = Field(default_factory=list)
    dcf_discrepancy_warning: Optional[str] = None
    evidence_snapshot: Optional['AnalysisEvidenceSnapshot'] = None
    atomic_market_snapshot: Optional[AtomicMarketSnapshot] = None
    metric_basis: dict[str, str] = Field(default_factory=dict)
    gaap_operating_margin: Optional[MarginMetricItem] = None
    non_gaap_operating_margin: Optional[MarginMetricItem] = None
    historical_gaap_operating_margin: Optional[MarginMetricItem] = None
    provider_ebit_margin: Optional[MarginMetricItem] = None
    valuation_margin_source_used: str = "Standardized TTM GAAP Operating Margin"


class EvidenceItemMetadata(BaseModel):
    """Item-Level Temporal Provenance & Source Metadata (v3.1)"""
    source_as_of: Optional[str] = Field(None, description="Timestamp of data state at source (ISO 8601)")
    retrieved_at: str = Field(..., description="Timestamp when engine fetched data (ISO 8601)")
    fiscal_period_end: Optional[str] = Field(None, description="Period end date e.g. 2024-09-30")
    reported_at: Optional[str] = Field(None, description="Official filing disclosure timestamp")
    exchange_timezone: str = "America/New_York"
    currency: str = "USD"
    unit: str = "units"  # e.g., "USD", "THB", "shares", "ratio", "percent"
    source_uri: str = Field(..., description="URI/file path e.g. yfinance:///AAPL/info or vault:///Earnings_Calls/...")
    payload_hash: str = Field(..., description="SHA256 checksum of raw payload")
    provider_tier: Literal["primary_best_effort", "filing_authoritative", "vault_internal", "fallback"] = "primary_best_effort"
    status: DataStatus = "available"
    stale_reason: Optional[str] = None


class CorporateActionsEvidence(BaseModel):
    """Corporate Actions with explicit separation between Chart Price Basis and Valuation Price Basis"""
    metadata: EvidenceItemMetadata
    chart_price_basis: Literal["split_and_dividend_adjusted", "split_only", "unadjusted"] = "split_and_dividend_adjusted"
    valuation_price_basis: Literal["unadjusted_close", "split_adjusted_only", "split_and_dividend_adjusted"] = "unadjusted_close"
    splits_history: list[dict[str, Any]] = Field(default_factory=list)
    dividends_history: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceManifestItem(BaseModel):
    """Manifest item pointing to raw evidence in Content-Addressed Store (CAS) or Vault"""
    item_id: str = Field(..., description="e.g. financials_income, macro_risk_free, earnings_transcript")
    metadata: EvidenceItemMetadata
    storage_ref: Optional[str] = Field(None, description="Relative path in CAS e.g. .evidence_cache/{sha256}.json")
    query_slice: Optional[dict[str, Any]] = Field(None, description="e.g. {'history_window': '5Y', 'interval': '1d'}")


class SnapshotMetadata(BaseModel):
    """Auditable Snapshot Metadata with Versioning, Checksum, and Execution Run ID"""
    analysis_run_id: str = Field(..., description="Unique Run ID e.g. run_AAPL_20260828_123456_a1b2")
    schema_version: str = "3.1"
    as_of_date: str = Field(..., description="Market close as-of date (YYYY-MM-DD)")
    generated_at: str = Field(..., description="ISO 8601 UTC timestamp")
    snapshot_sha256: str = Field(..., description="SHA256 checksum of canonical normalized manifest")
    data_quality_flags: list[str] = Field(default_factory=list)
    coverage_pct: float = 100.0


class AnalysisEvidenceSnapshot(BaseModel):
    """Immutable Analysis Evidence Snapshot (Manifest Pattern v3.1)"""
    metadata: SnapshotMetadata
    manifest_items: dict[str, EvidenceManifestItem] = Field(default_factory=dict)
    corporate_actions: Optional[CorporateActionsEvidence] = None
    derived_features: dict[str, Any] = Field(default_factory=dict)


class TechnicalOHLCVEvidence(BaseModel):
    ticker: str
    market: str = "US"
    bars: list[dict[str, Any]] = Field(default_factory=list)
    interval: str = "1d"
    count: int = 0


class MarketSnapshotEvidence(BaseModel):
    ticker: str
    market: str = "US"
    analysis_price: float
    analysis_price_as_of: str
    price_source: str = "ohlcv_close"
    shares_outstanding: Optional[float] = None
    market_cap: Optional[float] = None
    retrieved_at: str


class MacroValuationEvidence(BaseModel):
    risk_free_rate_pct: float
    erp_pct: float
    crp_pct: float = 0.0
    observable_refs: list[str] = Field(default_factory=list)
    source_uri: str = "macro:///registry"


class ReverseDCFInputEvidence(BaseModel):
    base_revenue: float
    base_ebit_margin_pct: float
    shares_outstanding: float
    total_cash: float = 0.0
    total_debt: float = 0.0
    beta: float = 1.0
    raw_wacc_pct: Optional[float] = None
    effective_wacc_pct: Optional[float] = None
    risk_free_rate_pct: float = 4.25
    erp_pct: float = 5.50
    terminal_growth_pct: float = 2.50
    reinvestment_rate_pct: float = 10.0


class OwnershipEvidence(BaseModel):
    heldPercentInstitutions: Optional[float] = None
    heldPercentInsiders: Optional[float] = None
    shortPercentOfFloat: Optional[float] = None
    sharesShort: Optional[float] = None
    shortRatio: Optional[float] = None
    insider_transactions: Optional[dict[str, Any]] = None


class AnalystTargetEvidence(BaseModel):
    targetMeanPrice: Optional[float] = None
    targetHighPrice: Optional[float] = None
    targetLowPrice: Optional[float] = None
    upside_pct: Optional[float] = None
    downside_pct: Optional[float] = None
    numberOfAnalystOpinions: Optional[int] = None


class EquitySentimentContext(BaseModel):
    """มิเรอร์ NarrativeContext (schemas/macro_schemas.py) — categorical ไม่ใช่ raw float เพื่อไม่ให้
    ดูเหมือนตัวเลข deterministic ทั้งที่เป็นการประเมินเชิงคุณภาพของ LLM"""
    evaluated_at: str
    market_sentiment: Literal["bullish", "neutral", "bearish"]
    key_themes: list[str] = Field(default_factory=list)
    tail_risks: list[str] = Field(default_factory=list)
    sources_summary: str
    report_references: list[dict[str, Any]] = Field(default_factory=list)


def build_unavailable_sentiment_context(evaluated_at: Optional[str] = None) -> EquitySentimentContext:
    from datetime import datetime, timezone
    now_iso = evaluated_at or datetime.now(timezone.utc).isoformat()
    return EquitySentimentContext(
        evaluated_at=now_iso,
        market_sentiment="neutral",
        key_themes=["Sentiment analysis unavailable (LLM offline or pending)"],
        tail_risks=[],
        sources_summary="Automated fallback: LLM narrative unavailable at run time.",
        report_references=[],
    )


class EquityNarrativeOutput(BaseModel):
    """Schema แคบที่ equity_synthesizer ผูกกับ LLM ตัวสุดท้าย — ไม่มี numeric field ให้แตะเลย"""
    narrative_analysis: str
    base_case_summary: str


class MicroQuantOutput(BaseModel):
    """ผลลัพธ์สุดท้ายของ equity_intel pipeline — numeric fields มาจาก Python เท่านั้น,
    text fields มาจาก LLM (EquityNarrativeOutput) เท่านั้น ไม่มีการปนกัน"""
    ticker: str
    market: Literal["TH", "US"]
    analysis_date: str
    quant_signals: QuantSignals
    sentiment_context: EquitySentimentContext
    narrative_analysis: str
    base_case_summary: str
    narrative_status: Literal["available", "unavailable", "pending"] = "available"
    error_code: Optional[str] = None
    numeric_revision_id: Optional[str] = None
    evidence_snapshot_hash: Optional[str] = None
    parent_numeric_revision_id: Optional[str] = None
    generated_by: str = "equity_intel"
    # Institutional Additive Fields
    piotroski_breakdown: Optional[PiotroskiFScoreBreakdown] = None
    reverse_dcf_result: Optional[ReverseDCFResult] = None
    tactical_setup: Optional[TacticalSetup] = None
    insider_conviction: Optional[InsiderConviction] = None
    earnings_guidance_context: Optional[EarningsGuidanceContext] = None
    deterministic_scorecard: Optional[DeterministicScorecard] = None
    thesis_falsifiers: list[ThesisFalsifier] = Field(default_factory=list)
    evidence_snapshot: Optional[AnalysisEvidenceSnapshot] = None

