"""Domain Models and DTOs for OHLCV Market Data and Technical Indicators."""
from typing import Literal, Optional
from pydantic import BaseModel, Field


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
