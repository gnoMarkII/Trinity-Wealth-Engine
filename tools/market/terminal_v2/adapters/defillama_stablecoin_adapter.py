"""DeFiLlama Keyless Public Adapter for Global Stablecoin Supply.

Fetches total USD-pegged circulating supply, 7d/30d changes, and top stablecoins breakdown.
Strict Rule: Best-effort keyless gateway; cached with ThreadSafeTTLCache.
"""
from datetime import datetime, timezone
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import (
    StablecoinItem,
    StablecoinSupplySnapshot,
)
from tools.market.terminal_v2.ports.driven_ports import StablecoinSupplyPort

logger = logging.getLogger(__name__)

STABLECOINS_URL = "https://stablecoins.llama.fi/stablecoins?includePrices=true"
STABLECOINS_TTL_SECONDS = 3600.0          # 1 hour
STABLECOINS_MAX_STALE_SECONDS = 86400.0   # 24 hours ceiling
REQUEST_TIMEOUT_SECONDS = 12.0


class DefiLlamaStablecoinsAdapter(StablecoinSupplyPort):
    """Adapter reading global stablecoin supply and growth metrics from DeFiLlama."""

    def __init__(
        self,
        cache: Optional[ThreadSafeTTLCache] = None,
        enabled: Optional[bool] = None,
    ):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=STABLECOINS_TTL_SECONDS)
        if enabled is not None:
            self._enabled = enabled
        else:
            env_val = os.getenv("ENABLE_DEFILLAMA_STABLECOINS", "true").strip().lower()
            self._enabled = env_val not in ("0", "false", "no", "off")

    def _fetch_snapshot(self) -> StablecoinSupplySnapshot:
        if not self._enabled:
            raise DataUnavailableError(
                "DeFiLlama stablecoin supply capability is disabled by feature flag",
                capability="stablecoin-supply",
                source="DeFiLlama",
            )

        headers = {**BROWSER_HEADERS, "Accept": "application/json"}
        try:
            resp = requests.get(STABLECOINS_URL, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise ProviderError(
                f"Failed to fetch stablecoin data from DeFiLlama: {exc}",
                source="DeFiLlama",
            ) from exc

        pegged_assets: List[Dict[str, Any]] = data.get("peggedAssets", [])
        if not pegged_assets or not isinstance(pegged_assets, list):
            raise ProviderError("DeFiLlama returned empty or invalid peggedAssets list", source="DeFiLlama")

        # Filter USD-pegged stablecoins
        usd_assets = [p for p in pegged_assets if p.get("pegType") == "peggedUSD"]
        if not usd_assets:
            usd_assets = pegged_assets  # Fallback to all if pegType not populated

        total_circulating = 0.0
        total_prev_week = 0.0
        total_prev_month = 0.0

        items_raw: List[Tuple[float, Dict[str, Any]]] = []

        for p in usd_assets:
            circ_dict = p.get("circulating") or {}
            circ_val = circ_dict.get("peggedUSD")
            if circ_val is not None:
                try:
                    c_f = float(circ_val)
                    if c_f > 0:
                        total_circulating += c_f
                        items_raw.append((c_f, p))
                except (ValueError, TypeError):
                    pass

            pw_dict = p.get("circulatingPrevWeek") or {}
            pw_val = pw_dict.get("peggedUSD")
            if pw_val is not None:
                try:
                    total_prev_week += float(pw_val)
                except (ValueError, TypeError):
                    pass

            pm_dict = p.get("circulatingPrevMonth") or {}
            pm_val = pm_dict.get("peggedUSD")
            if pm_val is not None:
                try:
                    total_prev_month += float(pm_val)
                except (ValueError, TypeError):
                    pass

        change_7d_pct: Optional[float] = None
        if total_prev_week > 0:
            change_7d_pct = round(((total_circulating - total_prev_week) / total_prev_week) * 100.0, 3)

        change_30d_pct: Optional[float] = None
        if total_prev_month > 0:
            change_30d_pct = round(((total_circulating - total_prev_month) / total_prev_month) * 100.0, 3)

        # Sort top stablecoins by circulating supply
        items_raw.sort(key=lambda x: x[0], reverse=True)
        top_items: List[StablecoinItem] = []
        for c_f, p in items_raw[:10]:
            sym = p.get("symbol", "").strip() or "UNKNOWN"
            name = p.get("name", "").strip() or sym
            price_val = p.get("price")
            p_usd = float(price_val) if price_val is not None else 1.0
            share_pct = round((c_f / total_circulating * 100.0), 2) if total_circulating > 0 else None
            top_items.append(
                StablecoinItem(
                    symbol=sym,
                    name=name,
                    circulating_usd=round(c_f, 2),
                    market_share_pct=share_pct,
                    price_usd=p_usd,
                )
            )

        as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return StablecoinSupplySnapshot(
            total_circulating_usd=round(total_circulating, 2),
            change_7d_pct=change_7d_pct,
            change_30d_pct=change_30d_pct,
            top_stablecoins=tuple(top_items),
            as_of_date=as_of,
            is_partial=False,
            completeness_notes="Top 10 USD stablecoins parsed from DeFiLlama keyless API.",
            fetched_at=time.time(),
            source="DeFiLlama",
            unit="USD",
        )

    def get_stablecoin_supply(self) -> StablecoinSupplySnapshot:
        """Fetch total USD stablecoin supply, 7d/30d changes, and top assets."""
        cache_key = "defillama:stablecoin:supply"
        return self._cache.get_or_set(
            key=cache_key,
            loader=self._fetch_snapshot,
            ttl_seconds=STABLECOINS_TTL_SECONDS,
            max_stale_seconds=STABLECOINS_MAX_STALE_SECONDS,
        )
