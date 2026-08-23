"""Equity Intel, Technicals, Targets, and Filings API Schemas."""
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

class EquitySummaryDTO(BaseModel):
    ticker: str
    market: Literal["TH", "US"]
    company_name: Optional[str] = None
    analysis_date: str
    evaluated_at: str
    market_sentiment: Literal["bullish", "neutral", "bearish"]
    composite_score: Optional[float] = None
    data_quality_flags: list[str] = []
    source_file: str
    sidecar_file: str


class EquitySentimentContextDTO(BaseModel):
    evaluated_at: str
    market_sentiment: Literal["bullish", "neutral", "bearish"]
    key_themes: list[str] = []
    tail_risks: list[str] = []
    sources_summary: str
    report_references: list[dict[str, Any]] = []


class EquityDetailDTO(EquitySummaryDTO):
    quant_signals: dict[str, Any]
    sentiment_context: EquitySentimentContextDTO
    narrative_analysis: str
    base_case_summary: str
    generated_by: str = "equity_intel"


class EquityNewsItemDTO(BaseModel):
    title: str
    source: str
    link: str
    published_at: Optional[str] = None
    age_hours: int
    freshness_reason: str
    is_stale: bool
    sources_count: int = 1


class EquityNewsDTO(BaseModel):
    ticker: str
    market: Literal["TH", "US"]
    last_updated: Optional[str] = None
    news_date: Optional[str] = None
    items: list[EquityNewsItemDTO] = []


class EquityNoteItemDTO(BaseModel):
    title: str
    folder: str
    relative_path: str
    obsidian_uri: str
    snippet: str
    modified_at: str
    matched_by: str


class EquityNotesDTO(BaseModel):
    ticker: str
    total_count: int
    items: list[EquityNoteItemDTO] = []


class EquityNoteContentDTO(BaseModel):
    title: str
    relative_path: str
    content: str
    modified_at: Optional[str] = None





class OHLCVCandleDTO(BaseModel):
    timestamp: int          # Unix epoch in milliseconds (e.g. 1700000000000)
    open: float
    high: float
    low: float
    close: float
    volume: float


class PivotLevelsDTO(BaseModel):
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float
    s4: float


class CorporateActionEventDTO(BaseModel):
    event_type: Literal["earnings", "ex_dividend", "split"]
    timestamp: int                                          # Unix epoch ms mapped to trading candle session
    date_str: str                                           # "YYYY-MM-DD" official reported calendar date
    label: str                                              # "E", "XD", or "S"
    color: Literal["green", "red", "blue", "purple"]        # green=Beat, red=Miss, blue=In line/XD, purple=Split
    tooltip: str                                            # Non-inferential, currency-aware descriptive text
    mapping_method: Literal["reported_date", "next_session", "period_enclosing", "unknown"] = "reported_date"
    eps_actual: Optional[float] = None
    eps_estimate: Optional[float] = None
    dividend_amount: Optional[float] = None
    split_numerator: Optional[float] = None
    split_denominator: Optional[float] = None
    split_formatted: Optional[str] = None                   # e.g. "4-for-1 forward split"


class CorporateActionsMetadataDTO(BaseModel):
    status: Literal["available", "partial", "unavailable"] = "available"
    as_of: Optional[str] = None                             # Oldest timestamp among available sources
    earnings_status: Literal["ok", "failed", "empty"] = "ok"
    earnings_as_of: Optional[str] = None
    dividends_status: Literal["ok", "failed", "empty"] = "ok"
    dividends_as_of: Optional[str] = None
    splits_status: Literal["ok", "failed", "empty"] = "ok"
    splits_as_of: Optional[str] = None
    missing_sources: list[str] = Field(default_factory=list) # e.g. ["dividends"]
    data_provenance: str = "Yahoo Finance (yfinance)"


class IndicatorBurnInPolicyDTO(BaseModel):
    algorithm_version: str = "ema_v1.0_sma_seed"
    seed_method: str = "sma_initial_period"
    convergence_tolerance_pct: float = 0.01
    required_burn_in_bars: int = 200
    burn_in_bars_remaining: int = 0
    first_reliable_timestamp: Optional[int] = None
    first_reliable_index: Optional[int] = None


class IndicatorWarmupDetailDTO(BaseModel):
    status: Literal["full", "partial", "unavailable"] = "full"
    required_bars: int = 200
    actual_warmup_bars: int = 0
    burn_in_bars_remaining: int = 0
    first_reliable_timestamp: Optional[int] = None
    first_reliable_index: Optional[int] = None
    burn_in_policy: Optional[IndicatorBurnInPolicyDTO] = None


class OHLCVResponseDTO(BaseModel):
    ticker: str
    market: Literal["TH", "US"]
    currency: Literal["USD", "THB"]
    price_basis: str = "provider_proportional_adj_close_ratio"
    provider_name: str = "yfinance"
    provider_tier: Literal["best_effort", "institutional_licensed"] = "best_effort"
    feed_latency_model: Literal["realtime", "delayed_15m", "eod"] = "delayed_15m"
    current_price: Optional[float] = None
    price_change: Optional[float] = None
    price_change_pct: Optional[float] = None
    price_as_of: Optional[str] = None
    candles: list[OHLCVCandleDTO] = Field(default_factory=list)
    pivot_levels: Optional[PivotLevelsDTO] = None
    pivot_period: Optional[str] = None
    pivot_as_of: Optional[str] = None
    requested_range: str = "6mo"
    interval: str = "1d"
    allowed_ranges: list[str] = Field(default_factory=list)
    effective_capabilities: dict[str, list[str]] = Field(default_factory=dict)
    capability_reasons: dict[str, str] = Field(default_factory=dict)
    display_start_timestamp: Optional[int] = None
    available_warmup_bars: int = 0
    required_warmup_bars: int = 250
    warmup_status: Literal["full", "partial", "unavailable", "sufficient", "insufficient", "not_applicable", "unknown"] = "unknown"
    indicator_warmup: dict[str, IndicatorWarmupDetailDTO] = Field(default_factory=dict)
    events: list[CorporateActionEventDTO] = Field(default_factory=list)
    events_metadata: Optional[CorporateActionsMetadataDTO] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    week52_coverage_calendar_days: int = 0


class CorporateActionFactorDTO(BaseModel):
    event_type: Literal["split", "special_dividend", "spinoff"]
    effective_date: str
    ratio: Optional[float] = None
    amount: Optional[float] = None


class DCFScenarioLevelDTO(BaseModel):
    scenario_name: str                                          # "base", "bull", "bear"
    label: str                                                  # "DCF Base", "DCF Bull", "DCF Bear"
    target_price: float
    upside_pct: Optional[float] = None
    margin_of_safety_pct: Optional[float] = None
    color: Literal["emerald", "green", "rose", "zinc"] = "emerald"


class ValuationTargetsDTO(BaseModel):
    evaluation_id: str                                          # Immutable UUID from canonical ledger
    ticker: str
    market: Literal["TH", "US"]
    currency: Literal["USD", "THB"]
    chart_price_basis: str = "provider_proportional_adj_close_ratio"
    valuation_price_basis: str = "split_adjusted_only"
    comparability_status: Literal["comparable", "not_comparable", "unknown"] = "comparable"
    comparability_reasons: list[str] = Field(default_factory=list)
    corporate_action_factors: list[CorporateActionFactorDTO] = Field(default_factory=list)
    current_price_at_eval: Optional[float] = None
    evaluated_at: str                                           # ISO Date from Canonical Ledger
    as_of_label: str = ""                                       # "as of YYYY-MM-DD"
    model_version: str = "dcf_v1.0"
    valuation_verdict: Literal["undervalued", "fairly_valued", "overvalued", "unknown"] = "unknown"
    wacc_pct: Optional[float] = None
    macro_observable_refs: list[str] = Field(default_factory=list)
    data_quality_flags: list[str] = Field(default_factory=list)
    status: Literal["available", "unavailable", "stale"] = "available"
    scenario_order_valid: bool = True
    scenarios: list[DCFScenarioLevelDTO] = Field(default_factory=list)


class InsiderTransactionDTO(BaseModel):
    transaction_id: str
    transaction_date: str                                       # YYYY-MM-DD
    transaction_code: str                                       # P, S, A, M, F, G
    shares: float
    price_per_share: float
    acquired_or_disposed: Literal["A", "D"]
    shares_owned_following: Optional[float] = None
    ownership_nature: Optional[str] = None                      # D (Direct) or I (Indirect)
    is_derivative: bool = False
    normalized_weight: float = 1.0


class InsiderFilingDTO(BaseModel):
    accession_number: str
    issuer_cik: str
    ticker: str
    filing_url: str
    filed_at: str                                               # YYYY-MM-DD
    timestamp: int                                              # Milliseconds unix timestamp mapped to candle
    reporting_owner_cik: Optional[str] = None
    reporting_owner_name: Optional[str] = None
    is_director: bool = False
    is_officer: bool = False
    is_ten_percent_owner: bool = False
    officer_title: Optional[str] = None
    is_amendment: bool = False
    amends_accession_number: Optional[str] = None
    is_cluster_buy: bool = False
    transactions: list[InsiderTransactionDTO] = Field(default_factory=list)


class InsiderFilingsResponseDTO(BaseModel):
    ticker: str
    market: Literal["TH", "US"]
    requested_range: str = "1y"
    interval: str = "1d"
    net_shares_30d: float = 0.0
    net_shares_90d: float = 0.0
    net_shares_180d: float = 0.0
    cluster_buy_count: int = 0
    total_filings_count: int = 0
    filings: list[InsiderFilingDTO] = Field(default_factory=list)


class EarningsHistoryEntryDTO(BaseModel):
    date_str: str                                               # "YYYY-MM-DD" reported date
    eps_actual: Optional[float] = None                          # Reported EPS
    eps_estimate: Optional[float] = None                        # Estimated EPS


class AnalystContextDTO(BaseModel):
    ticker: str
    provider_symbol: str
    market: Literal["US", "TH"]
    currency: Literal["USD", "THB"]
    exchange_tz: str
    target_mean: Optional[float] = None
    target_high: Optional[float] = None
    target_low: Optional[float] = None
    num_analysts: Optional[int] = None
    next_earnings_date: Optional[str] = None                    # ISO date "YYYY-MM-DD" or None
    days_to_earnings: Optional[int] = None                      # Computed at response time in exchange_tz
    earnings_history: list[EarningsHistoryEntryDTO] = Field(default_factory=list)
    source_as_of: Optional[str] = None                          # ISO datetime
    data_status: Literal["ok", "partial", "stale", "unavailable"] = "ok"
    provider_tier: Literal["best_effort"] = "best_effort"
    synced_at: str                                              # ISO UTC datetime string


