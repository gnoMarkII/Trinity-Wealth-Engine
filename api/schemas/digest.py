"""API Schemas for Daily and Weekly Market Intelligence Digest."""
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field


class DigestPortfolioBadgeDTO(BaseModel):
    symbol: str
    company_name: Optional[str] = None
    impact_type: Literal["direct_holding", "fund_lookthrough", "watchlist"]
    weight_pct: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    source_fund_codes: list[str] = Field(default_factory=list)
    target_entry_price: Optional[float] = None
    sector: Optional[str] = None


class DigestNewsCandidateDTO(BaseModel):
    candidate_id: str
    title: str
    summary: str = ""
    source: str
    links: list[str] = Field(default_factory=list)
    sources_count: int = 1
    published_at: Optional[str] = None
    age_hours: int = 0
    freshness_tier: str = "fresh_today"
    freshness_label: str = ""
    is_stale: bool = False
    matched_tickers: list[str] = Field(default_factory=list)
    portfolio_badges: list[DigestPortfolioBadgeDTO] = Field(default_factory=list)


class DigestYouTubeCandidateDTO(BaseModel):
    video_id: str
    title: str
    channel: str
    link: str
    published_at: Optional[str] = None
    age_hours: int = 0
    freshness_tier: str = "fresh_today"
    freshness_label: str = ""
    thumbnail: str = ""
    is_fetched: bool = False
    cached_summary_snippet: Optional[str] = None
    cached_tickers: list[str] = Field(default_factory=list)


class DigestCandidatesResponseDTO(BaseModel):
    last_fetched_at: Optional[str] = None
    news_candidates: list[DigestNewsCandidateDTO] = []
    youtube_candidates: list[DigestYouTubeCandidateDTO] = []
    stale_dropped_count: int = 0


class DailyDigestSynthesizeRequestDTO(BaseModel):
    selected_video_ids: list[str] = Field(default_factory=list)
    excluded_news_ids: list[str] = Field(default_factory=list)
    custom_youtube_urls: list[str] = Field(default_factory=list)
    custom_news_items: list[str] = Field(default_factory=list)


class DailyDigestItemDTO(BaseModel):
    source_candidate_id: str
    thai_headline: str
    what_happened: str
    why_it_matters: str
    thesis_signal: Literal["thesis_breaker", "buy_on_dip", "thesis_reinforcer", "noise"]
    impact_type: Literal["direct_holding", "fund_lookthrough", "watchlist", "second_order", "macro_general"]
    affected_tickers: list[str] = Field(default_factory=list)
    affected_sectors: list[str] = Field(default_factory=list)
    macro_impact_score: int = 5
    asset_impact_score: int = 5
    time_horizon: Literal["short_term", "structural"] = "short_term"
    youtube_corroboration: Optional[str] = None
    source: str = "Daily Digest"
    link: str = ""
    links: list[str] = Field(default_factory=list)
    published_at: Optional[str] = None
    age_hours: int = 0
    freshness_tier: str = "ultra_fresh"
    freshness_label: str = ""
    portfolio_badges: list[DigestPortfolioBadgeDTO] = Field(default_factory=list)
    is_read: bool = False
    is_new: bool = False


class DailyYouTubeHighlightDTO(BaseModel):
    video_id: str
    thai_title: str
    channel: str
    key_takeaways: list[str] = Field(default_factory=list)
    mentioned_tickers: list[str] = Field(default_factory=list)
    alignment_with_news: Literal["confirmed", "contrarian", "independent"] = "independent"
    alignment_note: str = ""


class DailyDigestPayloadDTO(BaseModel):
    digest_id: str
    date: str
    period: str
    generated_at: str
    market_mood: Literal["risk_on", "risk_off", "mixed"]
    market_mood_summary: str
    transmission_chain: str
    top_portfolio_focus: str
    selected_video_ids: list[str] = Field(default_factory=list)
    items: list[DailyDigestItemDTO] = []
    youtube_highlights: list[DailyYouTubeHighlightDTO] = []


class DailyDigestResponseDTO(BaseModel):
    has_digest: bool
    digest: Optional[DailyDigestPayloadDTO] = None
    snapshot_age_hours: int = 0
    is_snapshot_stale: bool = False
    fresh_pending_count: int = 0
    available_dates: list[str] = Field(default_factory=list)


class WeeklyPortfolioMatrixRowDTO(BaseModel):
    ticker: str
    thesis_status: Literal["strong_hold", "watch_closely", "accumulate_zone", "review_thesis"]
    thesis_status_label: str
    weekly_summary: str


class WeeklyDigestPayloadDTO(BaseModel):
    digest_id: str
    from_date: str
    to_date: str
    generated_at: str
    executive_themes: list[str] = Field(default_factory=list)
    portfolio_health_summary: str
    portfolio_matrix: list[WeeklyPortfolioMatrixRowDTO] = Field(default_factory=list)
    consensus_narratives: list[str] = Field(default_factory=list)
    contrarian_and_pitches: list[str] = Field(default_factory=list)
    week_ahead_checklist: list[str] = Field(default_factory=list)
    items: list[DailyDigestItemDTO] = Field(default_factory=list)
    youtube_highlights: list[DailyYouTubeHighlightDTO] = Field(default_factory=list)


class WeeklyDigestResponseDTO(BaseModel):
    has_digest: bool
    digest: Optional[WeeklyDigestPayloadDTO] = None
    available_dates: list[str] = Field(default_factory=list)


class WeeklyDigestSynthesizeRequestDTO(BaseModel):
    days: int = 7
    selected_video_ids: list[str] = Field(default_factory=list)
    excluded_news_ids: list[str] = Field(default_factory=list)
    custom_youtube_urls: list[str] = Field(default_factory=list)
    custom_news_items: list[str] = Field(default_factory=list)


class DigestReadStateRequestDTO(BaseModel):
    item_ids: list[str] = Field(default_factory=list)
    mark_as_read: bool = True


class DigestReadStateResponseDTO(BaseModel):
    ok: bool = True
    read_item_ids: list[str] = Field(default_factory=list)
