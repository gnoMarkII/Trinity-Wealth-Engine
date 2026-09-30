"""Polymarket Gamma API Keyless Adapter for Prediction Markets.

Fetches active prediction markets and market-implied outcome odds from Polymarket Gamma API.
Positional parsing and validation of JSON-stringified outcome and outcomePrices arrays.
Strict Rule: Implied prices are market odds for specific contracts, not official forecasts.
"""
import json
import logging
import time
from typing import Any, List, Optional, Tuple
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import ProviderError
from tools.market.terminal_v2.domain.models import (
    PredictionMarketItem,
    PredictionOutcome,
)
from tools.market.terminal_v2.ports.driven_ports import PredictionMarketPort

logger = logging.getLogger(__name__)

GAMMA_MARKETS_URL = (
    "https://gamma-api.polymarket.com/markets?closed=false&limit={limit}&order=volume&ascending=false"
)
POLYMARKET_TTL_SECONDS = 240.0        # 4 minutes
POLYMARKET_MAX_STALE_SECONDS = 600.0  # 10 minutes ceiling


def _parse_string_array(raw_val: Any) -> List[str]:
    """Parse JSON string array or list."""
    if isinstance(raw_val, list):
        return [str(x) for x in raw_val]
    if isinstance(raw_val, str):
        try:
            parsed = json.loads(raw_val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (json.JSONDecodeError, ValueError):
            return []
    return []


class PolymarketAdapter(PredictionMarketPort):
    """Adapter reading prediction market data from Polymarket Gamma API."""

    def __init__(self, cache: Optional[ThreadSafeTTLCache] = None):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=POLYMARKET_TTL_SECONDS)

    def _fetch_markets(self, limit: int) -> Tuple[PredictionMarketItem, ...]:
        url = GAMMA_MARKETS_URL.format(limit=limit)
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise ProviderError(f"Failed to fetch Polymarket markets: {exc}", source="Polymarket") from exc

        if not isinstance(data, list):
            raise ProviderError("Unexpected response shape from Polymarket Gamma API (expected array)", source="Polymarket")

        items: List[PredictionMarketItem] = []
        now_epoch = time.time()

        for m in data:
            q = (m.get("question") or "").strip()
            market_id = str(m.get("id") or m.get("slug") or "")
            if not q or not market_id:
                continue

            labels = _parse_string_array(m.get("outcomes"))
            raw_prices = _parse_string_array(m.get("outcomePrices"))

            outcomes: List[PredictionOutcome] = []
            for idx, label in enumerate(labels):
                price = 0.0
                if idx < len(raw_prices):
                    try:
                        p = float(raw_prices[idx])
                        if 0.0 <= p <= 1.0:
                            price = p
                    except (ValueError, TypeError):
                        price = 0.0
                outcomes.append(PredictionOutcome(label=label, price=price))

            vol_24h = None
            if m.get("volume24hr") is not None:
                try:
                    vol_24h = float(m["volume24hr"])
                except (ValueError, TypeError):
                    vol_24h = None

            slug = m.get("slug") or market_id
            source_url = f"https://polymarket.com/market/{slug}"

            items.append(
                PredictionMarketItem(
                    market_id=market_id,
                    question=q,
                    outcomes=tuple(outcomes),
                    volume_24h_usd=vol_24h,
                    end_date=m.get("endDate"),
                    source_url=source_url,
                    fetched_at=now_epoch,
                    source="Polymarket",
                )
            )

        return tuple(items[:limit])

    def get_prediction_markets(self, limit: int = 12) -> Tuple[PredictionMarketItem, ...]:
        capped = min(max(limit, 1), 50)
        cache_key = f"polymarket:markets:{capped}"
        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._fetch_markets(capped),
            ttl_seconds=POLYMARKET_TTL_SECONDS,
            max_stale_seconds=POLYMARKET_MAX_STALE_SECONDS,
        )
