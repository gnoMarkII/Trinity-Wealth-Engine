"""Gold Traders Association (Thailand) Keyless Adapter.

Scrapes official retail gold prices (bar and ornament 96.5%) from classic.goldtraders.or.th.
Strict Rule: No silent fallback to GC=F (gold futures) or foreign bullion fixes.
Cache TTL: 120 seconds (2 minutes) to prevent ASP.NET session bans.
"""
import logging
import re
from typing import Optional, Tuple
from bs4 import BeautifulSoup
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import (
    GoldPriceDetail,
    ThaiRetailGoldQuote,
)
from tools.market.terminal_v2.ports.driven_ports import GoldPricePort

logger = logging.getLogger(__name__)

GTA_URL = "https://classic.goldtraders.or.th/"
GOLD_TTL_SECONDS = 120.0
TIMEOUT_SECONDS = 15.0

SPAN_IDS = {
    "bar_sell": "DetailPlace_uc_goldprices1_lblBLSell",
    "bar_buy": "DetailPlace_uc_goldprices1_lblBLBuy",
    "ornament_sell": "DetailPlace_uc_goldprices1_lblOMSell",
    "ornament_buy": "DetailPlace_uc_goldprices1_lblOMBuy",
    "announced": "DetailPlace_uc_goldprices1_lblAsTime",
}


def _extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    cleaned = text.replace(",", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _parse_announcement(text: Optional[str]) -> Tuple[str, Optional[int]]:
    if not text:
        return "", None
    revision_match = re.search(r"ครั้งที่\s*(\d+)", text)
    revision = int(revision_match.group(1)) if revision_match else None
    return text.strip(), revision


class GoldTradersAdapter(GoldPricePort):
    """Native Python adapter for Gold Traders Association retail prices."""

    def __init__(self, cache: Optional[ThreadSafeTTLCache] = None):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=GOLD_TTL_SECONDS)

    def get_retail_gold_quote(self) -> ThaiRetailGoldQuote:
        """Fetch official retail gold prices from Gold Traders Association.

        If GTA is down or parsing fails, returns stale cache or raises DataUnavailableError.
        Never substitutes with GC=F.
        """
        cache_key = "goldtraders:retail:quote"

        def _loader() -> ThaiRetailGoldQuote:
            try:
                resp = requests.get(GTA_URL, headers=BROWSER_HEADERS, timeout=TIMEOUT_SECONDS)
                resp.raise_for_status()
                html = resp.text
            except Exception as exc:
                raise ProviderError(
                    f"Failed to fetch page from Gold Traders Association: {exc}",
                    source="Gold Traders Association",
                ) from exc

            soup = BeautifulSoup(html, "html.parser")

            def _get_val(span_id: str) -> Optional[float]:
                el = soup.find(id=span_id)
                if not el:
                    # Regex fallback in case DOM structure is slightly altered
                    pattern = rf'<span[^>]*id=["\']?{re.escape(span_id)}["\']?[^>]*>(.*?)</span>'
                    match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
                    if match:
                        raw = re.sub(r"<[^>]*>", "", match.group(1))
                        return _extract_number(raw)
                    return None
                return _extract_number(el.get_text())

            bar_buy = _get_val(SPAN_IDS["bar_buy"])
            bar_sell = _get_val(SPAN_IDS["bar_sell"])
            ornament_buy = _get_val(SPAN_IDS["ornament_buy"])
            ornament_sell = _get_val(SPAN_IDS["ornament_sell"])

            time_el = soup.find(id=SPAN_IDS["announced"])
            raw_time = time_el.get_text() if time_el else ""
            if not raw_time:
                match = re.search(rf'<span[^>]*id=["\']?{re.escape(SPAN_IDS["announced"])}["\']?[^>]*>(.*?)</span>', html, re.DOTALL | re.IGNORECASE)
                if match:
                    raw_time = re.sub(r"<[^>]*>", "", match.group(1)).strip()

            announced_at, revision = _parse_announcement(raw_time)

            if bar_buy is None or bar_sell is None or ornament_buy is None or ornament_sell is None:
                raise ProviderError(
                    "Gold Traders Association page structure changed: missing price labels "
                    f"(bar_buy={bar_buy}, bar_sell={bar_sell}, ornament_buy={ornament_buy}, ornament_sell={ornament_sell})",
                    source="Gold Traders Association",
                )

            return ThaiRetailGoldQuote(
                source="Gold Traders Association",
                unit="baht-weight (15.244 g, 96.5%)",
                bar=GoldPriceDetail(buy=bar_buy, sell=bar_sell),
                ornament=GoldPriceDetail(buy=ornament_buy, sell=ornament_sell),
                announced_at=announced_at,
                revision=revision,
                is_stale=False,
            )

        try:
            return self._cache.get_or_set(
                cache_key,
                _loader,
                ttl_seconds=GOLD_TTL_SECONDS,
            )
        except Exception as exc:
            raise DataUnavailableError(
                f"Thai retail gold price is temporarily unavailable from Gold Traders Association: {exc}",
                capability="retail_gold_price",
                source="Gold Traders Association",
            ) from exc
