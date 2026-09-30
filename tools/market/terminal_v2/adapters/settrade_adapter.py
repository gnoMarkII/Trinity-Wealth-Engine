"""Settrade Keyless Adapter for Thai Equity Market.

Fetches 4-investor-type daily flow, valuation multiples, and market breadth directly
from SET's open API without API keys.
Strict Rule: No silent fallback to ^SET.BK or volume indices.
Cache TTL: 60 seconds (anti-ban protection).
"""
import logging
from typing import Optional
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import (
    InvestorTypeRow,
    MarketBreadth,
    MarketValuation,
    ThaiFundFlowSnapshot,
)
from tools.market.terminal_v2.ports.driven_ports import ThaiMarketPort

logger = logging.getLogger(__name__)

BASE_URL = "https://api.settrade.com"
SETTRADE_TTL_SECONDS = 60.0
TIMEOUT_SECONDS = 10.0


class SettradeAdapter(ThaiMarketPort):
    """Native Python adapter for Settrade open market-data endpoints."""

    def __init__(self, cache: Optional[ThreadSafeTTLCache] = None):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=SETTRADE_TTL_SECONDS)

    def get_investor_type_flow(self, market: str = "SET") -> ThaiFundFlowSnapshot:
        """Fetch 4-investor-type flow from Settrade.

        If Settrade is unreachable, returns stale cache or raises DataUnavailableError.
        Never substitutes with ^SET.BK.
        """
        venue = market.upper()
        if venue not in ("SET", "MAI"):
            venue = "SET"

        cache_key = f"settrade:flow:{venue}"

        def _loader() -> ThaiFundFlowSnapshot:
            url = f"{BASE_URL}/api/market/{venue}/investortype"
            try:
                resp = requests.get(url, headers=BROWSER_HEADERS, timeout=TIMEOUT_SECONDS)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                raise ProviderError(
                    f"Failed to fetch investor type flow from Settrade: {exc}",
                    source="Settrade",
                ) from exc

            as_of = data.get("asof_date", "")
            total_value = float(data.get("total_value", 0.0) or 0.0)

            investors_raw = data.get("investors", [])
            investors_list = []
            for row in investors_raw:
                buy_val = float(row.get("buy_value") or 0.0)
                sell_val = float(row.get("sell_value") or 0.0)
                net_val = (
                    float(row["net_value"])
                    if row.get("net_value") is not None
                    else (buy_val - sell_val)
                )
                investors_list.append(
                    InvestorTypeRow(
                        investor_type=str(row.get("type", "")),
                        name_en=str(row.get("type_name_en") or row.get("type", "")),
                        buy_value=buy_val,
                        sell_value=sell_val,
                        net_value=net_val,
                    )
                )

            return ThaiFundFlowSnapshot(
                market=venue,
                as_of=as_of,
                total_value=total_value,
                investors=tuple(investors_list),
                source="Settrade",
                is_stale=False,
            )

        try:
            return self._cache.get_or_set(
                cache_key,
                _loader,
                ttl_seconds=SETTRADE_TTL_SECONDS,
            )
        except Exception as exc:
            raise DataUnavailableError(
                f"Investor type flow data is temporarily unavailable from Settrade: {exc}",
                capability="investor_type_flow",
                source="Settrade",
            ) from exc

    def get_market_statistics(self, market: str = "SET") -> MarketValuation:
        """Fetch venue aggregate valuation metrics (P/E, P/BV, yield)."""
        venue = market.upper()
        if venue not in ("SET", "MAI"):
            venue = "SET"

        cache_key = f"settrade:stats:{venue}"

        def _loader() -> MarketValuation:
            url = f"{BASE_URL}/api/market/{venue}/statistics"
            try:
                resp = requests.get(url, headers=BROWSER_HEADERS, timeout=TIMEOUT_SECONDS)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                raise ProviderError(
                    f"Failed to fetch market statistics from Settrade: {exc}",
                    source="Settrade",
                ) from exc

            return MarketValuation(
                market=venue,
                as_of=data.get("asof_date", ""),
                market_cap=float(data["market_cap"]) if data.get("market_cap") is not None else None,
                pe_ratio=float(data["pe_ratio"]) if data.get("pe_ratio") is not None else None,
                pbv_ratio=float(data["pbv_ratio"]) if data.get("pbv_ratio") is not None else None,
                dividend_yield=float(data["dividend_yield"]) if data.get("dividend_yield") is not None else None,
                turnover_ratio=float(data["turnover_ratio"]) if data.get("turnover_ratio") is not None else None,
                source="Settrade",
                is_stale=False,
            )

        try:
            return self._cache.get_or_set(
                cache_key,
                _loader,
                ttl_seconds=SETTRADE_TTL_SECONDS,
            )
        except Exception as exc:
            raise DataUnavailableError(
                f"Market statistics data is temporarily unavailable from Settrade: {exc}",
                capability="market_statistics",
                source="Settrade",
            ) from exc

    def get_market_breadth(self, market: str = "SET") -> MarketBreadth:
        """Fetch market breadth (gainers, losers, unchanged)."""
        venue = market.upper()
        if venue not in ("SET", "MAI"):
            venue = "SET"

        cache_key = f"settrade:breadth:{venue}"

        def _loader() -> MarketBreadth:
            url = f"{BASE_URL}/api/market/{venue}/info"
            try:
                resp = requests.get(url, headers=BROWSER_HEADERS, timeout=TIMEOUT_SECONDS)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                raise ProviderError(
                    f"Failed to fetch market breadth from Settrade: {exc}",
                    source="Settrade",
                ) from exc

            return MarketBreadth(
                market=venue,
                as_of=data.get("datetime", ""),
                gainers=int(data.get("gainer_amount", 0)),
                losers=int(data.get("loser_amount", 0)),
                unchanged=int(data.get("unchange_amount", 0)),
                source="Settrade",
                is_stale=False,
            )

        try:
            return self._cache.get_or_set(
                cache_key,
                _loader,
                ttl_seconds=SETTRADE_TTL_SECONDS,
            )
        except Exception as exc:
            raise DataUnavailableError(
                f"Market breadth data is temporarily unavailable from Settrade: {exc}",
                capability="market_breadth",
                source="Settrade",
            ) from exc
