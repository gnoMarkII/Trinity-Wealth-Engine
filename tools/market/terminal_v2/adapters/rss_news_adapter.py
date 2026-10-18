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
from typing import Dict, List, Optional, Set, Tuple
import urllib.parse
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

    def _fetch_url(self, url: str, symbol: str) -> Tuple[str, str]:
        """Fetch RSS XML text from a given URL. Returns (xml_text, status_str)."""
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
            if resp.status_code == 429:
                logger.warning("RSS feed returned 429 (Rate Limited) for %s from %s", symbol, url)
                return "", "rate_limited"
            resp.raise_for_status()
            return resp.text, "ok"
        except requests.exceptions.RequestException as exc:
            logger.debug("RSS feed fetch failed for %s from %s: %s", symbol, url, exc)
            return "", "feed_unavailable"

    def _fetch_rss_text(self, symbol: str) -> Tuple[str, str]:
        """Fetch primary Google RSS XML text. Returns (xml_text, status_str)."""
        if self._fixture_path and self._fixture_path.exists():
            return self._fixture_path.read_text(encoding="utf-8"), "ok"

        url = GOOGLE_NEWS_RSS_URL.format(symbol=symbol)
        return self._fetch_url(url, symbol)

    def _parse_candidates(
        self,
        xml_text: str,
        symbol: str,
        default_source: str = "Google News",
        source_type: str = "google_news_rss",
    ) -> List[NewsCandidate]:
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
            source = item.findtext("source", default_source).strip() or default_source

            if title:
                candidates.append(
                    NewsCandidate(
                        headline=title,
                        publisher=source,
                        source_type=source_type,
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

    def _build_feed_urls(self, symbol: str) -> List[Tuple[str, str]]:
        clean_sym = symbol.removesuffix(".BK").strip().upper()
        is_th = symbol.endswith(".BK")
        feeds: List[Tuple[str, str]] = []
        if is_th:
            q_th = urllib.parse.quote(f"({clean_sym}) หุ้น")
            feeds.append(("google_news_th", f"https://news.google.com/rss/search?q={q_th}&hl=th&gl=TH&ceid=TH:th"))
            q_en = urllib.parse.quote(f"({clean_sym}) Thailand stock")
            feeds.append(("google_news_en", f"https://news.google.com/rss/search?q={q_en}&hl=en-US&gl=US&ceid=US:en"))
        else:
            feeds.append(("google_news_us", f"https://news.google.com/rss/search?q=({clean_sym})+stock&hl=en-US&gl=US&ceid=US:en"))
            feeds.append(("yahoo_rss", f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={clean_sym}&region=US&lang=en-US"))
        return feeds

    def _load_news_discovery(self, symbol: str, limit: int) -> NewsDiscoverySnapshot:
        if self._fixture_path and self._fixture_path.exists():
            xml_text = self._fixture_path.read_text(encoding="utf-8")
            all_candidates = self._parse_candidates(xml_text, symbol)
            return NewsDiscoverySnapshot(
                query_symbol=symbol,
                items=tuple(all_candidates[:limit]),
                status="ok",
                source="Fixture RSS Discovery",
                as_of_date=datetime.utcnow().strftime("%Y-%m-%d"),
                fetched_at=time.time(),
            )

        is_th = symbol.endswith(".BK")

        all_candidates: List[NewsCandidate] = []
        seen_urls: Set[str] = set()
        seen_headlines: Set[str] = set()
        last_status = "ok"

        def _add_candidates(candidates: List[NewsCandidate]) -> None:
            for c in candidates:
                url_key = c.article_url.strip() if c.article_url else ""
                head_key = c.headline.strip().lower()
                if (url_key and url_key in seen_urls) or (head_key and head_key in seen_headlines):
                    continue
                if url_key:
                    seen_urls.add(url_key)
                if head_key:
                    seen_headlines.add(head_key)
                all_candidates.append(c)

        feeds = self._build_feed_urls(symbol)
        if is_th:
            for feed_type, url in feeds:
                default_src = "Google News TH" if feed_type == "google_news_th" else "Google News EN"
                xml_text, status = self._fetch_url(url, symbol)
                if status == "ok":
                    _add_candidates(self._parse_candidates(xml_text, symbol, default_source=default_src))
                elif last_status == "ok":
                    last_status = status
        else:
            for feed_type, url in feeds:
                if feed_type == "google_news_us":
                    xml_us, status_us = self._fetch_url(url, symbol)
                    if status_us == "ok":
                        _add_candidates(self._parse_candidates(xml_us, symbol, default_source="Google News US"))
                    else:
                        last_status = status_us
                elif feed_type == "yahoo_rss" and (len(all_candidates) < 5 or last_status != "ok"):
                    xml_yf, status_yf = self._fetch_url(url, symbol)
                    if status_yf == "ok":
                        _add_candidates(
                            self._parse_candidates(
                                xml_yf,
                                symbol,
                                default_source="Yahoo Finance RSS",
                                source_type="yahoo_finance_rss",
                            )
                        )
                        last_status = "ok"

        if all_candidates:
            last_status = "ok"

        capped_items = tuple(all_candidates[:limit])
        as_of = datetime.utcnow().strftime("%Y-%m-%d")

        return NewsDiscoverySnapshot(
            query_symbol=symbol,
            items=capped_items,
            status=last_status,
            source="Multi-Source RSS Discovery",
            as_of_date=as_of,
            fetched_at=time.time(),
        )
