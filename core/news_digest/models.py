"""Domain models and LLM structured output schemas for News and YouTube Digest."""
from typing import Literal, Optional
from pydantic import BaseModel, Field

FreshnessTier = Literal["ultra_fresh", "fresh_today", "weekly_window", "stale"]
ThesisSignal = Literal["thesis_breaker", "buy_on_dip", "thesis_reinforcer", "noise"]
PortfolioImpactType = Literal[
    "direct_holding",
    "fund_lookthrough",
    "watchlist",
    "second_order",
    "macro_general",
]
MarketMood = Literal["risk_on", "risk_off", "mixed"]


class PortfolioHoldingContext(BaseModel):
    """Context of investor holding or watchlist item matching news/video."""
    symbol: str
    company_name: Optional[str] = None
    impact_type: Literal["direct_holding", "fund_lookthrough", "watchlist"]
    weight_pct: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    source_fund_codes: list[str] = Field(default_factory=list)
    target_entry_price: Optional[float] = None
    sector: Optional[str] = None


class PreLLMNewsCandidate(BaseModel):
    """Cleaned, deduplicated, and freshness-checked news candidate ready for triage."""
    candidate_id: str
    title: str
    summary: str = ""
    source: str
    links: list[str] = Field(default_factory=list)
    sources_count: int = 1
    published_at: Optional[str] = None
    age_hours: int = 0
    freshness_tier: FreshnessTier = "fresh_today"
    freshness_label: str = ""
    is_stale: bool = False
    matched_tickers: list[str] = Field(default_factory=list)
    portfolio_badges: list[PortfolioHoldingContext] = Field(default_factory=list)


class PreLLMYouTubeCandidate(BaseModel):
    """Cleaned YouTube candidate ready for investor curation before LLM call."""
    video_id: str
    title: str
    channel: str
    link: str
    published_at: Optional[str] = None
    age_hours: int = 0
    freshness_tier: FreshnessTier = "fresh_today"
    freshness_label: str = ""
    thumbnail: str = ""
    is_fetched: bool = False
    cached_summary_snippet: Optional[str] = None
    cached_tickers: list[str] = Field(default_factory=list)


# --- Schemas for One-Shot Mega-Batch LLM Structured Output ---

class OneShotDigestItemLLM(BaseModel):
    source_candidate_id: str = Field(description="ID of source candidate or video")
    thai_headline: str = Field(description="Concise Thai headline (1 line max)")
    what_happened: str = Field(description="Fact summary of what occurred in 1 concise sentence")
    why_it_matters: str = Field(description="Impact on portfolio or investment decision in 1 sentence")
    thesis_signal: ThesisSignal = Field(description="Thesis impact: thesis_breaker, buy_on_dip, thesis_reinforcer, or noise")
    impact_type: PortfolioImpactType = Field(description="Level of impact to portfolio")
    affected_tickers: list[str] = Field(default_factory=list, description="Direct or 2nd-order affected tickers")
    affected_sectors: list[str] = Field(default_factory=list)
    macro_impact_score: int = Field(default=5, ge=1, le=10)
    asset_impact_score: int = Field(default=5, ge=1, le=10)
    time_horizon: Literal["short_term", "structural"] = "short_term"
    youtube_corroboration: Optional[str] = Field(default=None, description="Relation to selected YouTube clips")


class OneShotYouTubeHighlightLLM(BaseModel):
    video_id: str
    thai_title: str
    channel: str
    key_takeaways: list[str] = Field(default_factory=list, description="Top takeaways (max 3)")
    mentioned_tickers: list[str] = Field(default_factory=list)
    alignment_with_news: Literal["confirmed", "contrarian", "independent"] = "independent"
    alignment_note: str = Field(default="", description="How this aligns with or contradicts daily news")


class OneShotDailyLLMOutput(BaseModel):
    market_mood: MarketMood
    market_mood_summary: str = Field(description="Daily market mood summary in 1-2 sentences")
    transmission_chain: str = Field(description="Macro cause-and-effect chain (e.g. CPI ➔ Yields ➔ Tech)")
    top_portfolio_focus: str = Field(description="Single most important portfolio focus today")
    items: list[OneShotDigestItemLLM] = Field(default_factory=list)
    youtube_highlights: list[OneShotYouTubeHighlightLLM] = Field(default_factory=list)


class WeeklyPortfolioMatrixRowLLM(BaseModel):
    ticker: str
    thesis_status: Literal["strong_hold", "watch_closely", "accumulate_zone", "review_thesis"]
    thesis_status_label: str
    weekly_summary: str = Field(description="Key weekly developments for this ticker")


class OneShotWeeklyLLMOutput(BaseModel):
    executive_themes: list[str] = Field(default_factory=list, description="3-5 macro themes driving market")
    portfolio_health_summary: str
    portfolio_matrix: list[WeeklyPortfolioMatrixRowLLM] = Field(default_factory=list)
    consensus_narratives: list[str] = Field(default_factory=list, description="Where news and YouTube gurus agree")
    contrarian_and_pitches: list[str] = Field(default_factory=list, description="Contrarian views or stock pitches")
    week_ahead_checklist: list[str] = Field(default_factory=list, description="Watchlist and action checklist for next week")
    items: list[OneShotDigestItemLLM] = Field(default_factory=list, description="Top curated weekly news developments")
    youtube_highlights: list[OneShotYouTubeHighlightLLM] = Field(default_factory=list)
