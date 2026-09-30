"""RSS News Discovery & Candidate Aggregator Adapter.

Strict Invariants:
1. RSS is a "news discovery and candidate aggregator", NOT a primary source or real-time sub-second feed.
2. Every item preserves original publication timestamp (`published_at`) and publisher credit (`publisher`).
3. Explicit HTTP 429 rate-limiting resilience: returns status="rate_limited" gracefully instead of throwing exceptions.
4. Offline fixture support for zero-network testing.
"""
from datetime import datetime
import html
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.models import NewsCandidate, NewsDiscoverySnapshot
from tools.market.terminal_v2.ports.driven_ports import TickerNewsPort

logger = logging.getLogger(__name__)

NEWS_TTL_SECONDS = 900.0         # 15 minutes
NEWS_MAX_STALE_SECONDS = 7200.0  # 2 hours

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search?q=({symbol})+stock&hl=en-US&gl=US&ceid=US:en"


class RssNewsDiscoveryAdapter(TickerNewsPort):
    """Adapter discovering per-symbol news candidates via RSS search feeds."""

    def __init__(
        self,
        cache: Optional[ThreadSafeTTLCache] = None,
        fixture_path: Optional[Path] = None,
    ):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=NEWS_TTL_SECONDS)
        self._fixture_path = fixture_path

    def _fetch_rss_text(self, symbol: str) -> Tuple[str, str]:
        """Fetch RSS XML text. Returns (xml_text, status_str)."""
        if self._fixture_path and self._fixture_path.exists():
            return self._fixture_path.read_text(encoding="utf-8"), "ok"

        url = GOOGLE_NEWS_RSS_URL.format(symbol=symbol)
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
            if resp.status_code == 429:
                logger.warning("Google News RSS returned 429 (Rate Limited) for %s", symbol)
                return "", "rate_limited"
            resp.raise_for_status()
            return resp.text, "ok"
        except requests.exceptions.RequestException as exc:
            logger.debug("Google News RSS fetch failed for %s: %s", symbol, exc)
            return "", "feed_unavailable"

    def _parse_candidates(self, xml_text: str, symbol: str) -> List[NewsCandidate]:
        if not xml_text:
            return []

        candidates: List[NewsCandidate] = []
        try:
            root = ET.fromstring(xml_text)
        except Exception as exc:
            logger.debug("Failed to parse RSS XML for %s: %s", symbol, exc)
            return []

        now_epoch = time.time()
        for item in root.findall(".//item"):
            title = html.unescape(item.findtext("title", "")).strip()
            link = item.findtext("link", "").strip()
            pub_date = item.findtext("pubDate", "").strip()
            source = item.findtext("source", "Google News").strip()

            if title:
                candidates.append(
                    NewsCandidate(
                        headline=title,
                        publisher=source,
                        source_type="google_news_rss",
                        article_url=link,
                        published_at=pub_date,
                        discovered_at=now_epoch,
                        symbol=symbol,
                        is_stale=False,
                    )
                )

        return candidates

    def get_news_candidates(self, symbol: str, limit: int = 15) -> NewsDiscoverySnapshot:
        sym = symbol.strip().upper()
        capped_limit = min(max(limit, 1), 50)
        cache_key = f"news:discovery:{sym}:{capped_limit}"

        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._load_news_discovery(sym, capped_limit),
            ttl_seconds=NEWS_TTL_SECONDS,
            max_stale_seconds=NEWS_MAX_STALE_SECONDS,
        )

    def _load_news_discovery(self, symbol: str, limit: int) -> NewsDiscoverySnapshot:
        xml_text, status = self._fetch_rss_text(symbol)
        all_candidates = self._parse_candidates(xml_text, symbol)

        capped_items = tuple(all_candidates[:limit])
        as_of = datetime.utcnow().strftime("%Y-%m-%d")

        return NewsDiscoverySnapshot(
            query_symbol=symbol,
            items=capped_items,
            status=status,
            source="Google News RSS Discovery",
            as_of_date=as_of,
            fetched_at=time.time(),
        )
