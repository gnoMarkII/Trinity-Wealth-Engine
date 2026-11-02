"""Ports for News and YouTube Digest use cases."""
from typing import Any, Dict, List, Optional, Protocol, Tuple

from core.news_digest.models import (
    OneShotDailyLLMOutput,
    OneShotWeeklyLLMOutput,
    PortfolioHoldingContext,
    PreLLMNewsCandidate,
    PreLLMYouTubeCandidate,
)


class DigestPortfolioReaderPort(Protocol):
    """Port for retrieving portfolio holdings, look-through constituents, and watchlist."""
    def get_portfolio_context(self, portfolio_id: str = "default") -> Tuple[List[PortfolioHoldingContext], Dict[str, PortfolioHoldingContext], str]:
        """Returns (holding_contexts, holdings_map, formatted_portfolio_summary_str)."""
        ...


class DigestCandidateFetcherPort(Protocol):
    """Port for fetching fresh news candidates and YouTube RSS candidates."""
    def fetch_candidates(
        self,
        portfolio_tickers: List[str],
        lookback_days: int = 7,
        max_news_age_hours: int = 36,
    ) -> Tuple[List[PreLLMNewsCandidate], List[PreLLMYouTubeCandidate], int]:
        """Returns (news_candidates, youtube_candidates, stale_dropped_count)."""
        ...


class DigestTranscriptPort(Protocol):
    """Port for loading cached YouTube insights or extracting transcripts on-demand."""
    def resolve_youtube_document(self, video_id: str, custom_url: Optional[str] = None) -> Dict[str, Any]:
        """Returns dict with video_id, title, channel, summary_snippet/content."""
        ...


class DigestLLMPort(Protocol):
    """Port for executing One-Shot Structured LLM synthesis."""
    def invoke_daily(
        self,
        candidates: List[PreLLMNewsCandidate],
        youtube_docs: List[Dict[str, Any]],
        portfolio_summary: str,
    ) -> OneShotDailyLLMOutput:
        ...

    def invoke_weekly(
        self,
        daily_digests: List[Dict[str, Any]],
        recent_news: List[PreLLMNewsCandidate],
        youtube_docs: List[Dict[str, Any]],
        calendar_events: List[Dict[str, Any]],
        portfolio_summary: str,
    ) -> OneShotWeeklyLLMOutput:
        ...


class DigestStoragePort(Protocol):
    """Port for reading and persisting digest state and ticker index."""
    def load_store(self) -> Dict[str, Any]:
        ...

    def save_candidates(
        self,
        news_candidates: List[Dict[str, Any]],
        youtube_candidates: List[Dict[str, Any]],
        stale_dropped_count: int,
        cache_bucket: str = "daily",
    ) -> None:
        ...

    def save_daily(self, daily_payload: Dict[str, Any]) -> None:
        ...

    def save_weekly(self, weekly_payload: Dict[str, Any]) -> None:
        ...

    def mark_read(self, item_ids: List[str]) -> List[str]:
        ...


class DigestCalendarPort(Protocol):
    """Port for fetching upcoming earnings and corporate actions."""
    def get_upcoming_events(self, portfolio_id: str = "default", days_ahead: int = 14) -> List[Dict[str, Any]]:
        """Returns list of upcoming earnings and corporate action event dicts."""
        ...
