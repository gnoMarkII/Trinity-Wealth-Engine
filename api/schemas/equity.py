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





from tools.market.ohlcv.domain.models import (
    OHLCVCandleDTO,
    PivotLevelsDTO,
    CorporateActionEventDTO,
    CorporateActionsMetadataDTO,
    IndicatorBurnInPolicyDTO,
    IndicatorWarmupDetailDTO,
    OHLCVResponseDTO,
)


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


class EarningsCallSummarizeRequest(BaseModel):
    period: str = Field(..., min_length=2, max_length=20, description="Earnings period, e.g. 'Q4 2024'")
    transcript: str = Field(..., min_length=20, max_length=120000, description="Raw earnings call transcript text")


class EarningsCallRunResponse(BaseModel):
    run_id: str
    ticker: str
    period: str
    status: str
    kanban_status: str
    highlights: Optional[str] = None
    vault_path: Optional[str] = None
    kanban_card_id: Optional[str] = None
    reused_existing_run: bool = False
    is_idempotent_replay: bool = False
    last_error_code: Optional[str] = None
    created_at: float
    updated_at: float


class EarningsCallSummarizeResponse(BaseModel):
    run_id: str
    ticker: str
    period: str
    status: str
    kanban_status: str
    highlights: Optional[str] = None
    vault_path: Optional[str] = None
    kanban_card_id: Optional[str] = None
    reused_existing_run: bool = False
    is_idempotent_replay: bool = False


class EarningsCallNoteItem(BaseModel):
    title: str
    ticker: str
    period: str
    vault_path: str
    highlights: str
    date: str
    last_updated: str
    has_full_transcript: bool = True


class EarningsCallListResponse(BaseModel):
    ticker: str
    total_count: int
    items: list[EarningsCallNoteItem]


