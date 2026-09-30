"""Hyperliquid REST API Adapter for Crypto & Builder DEX Perpetual Futures.

Queries public info endpoints on api.hyperliquid.xyz for live perpetual mark prices,
open interest, and funding rates.
Strict Rule: Explicitly flags assets as synthetic_crypto_perp.
Never presents HIP-3 perps (xyz:TSLA) as NASDAQ/NYSE cash equities.
Cache TTL: 4.0 seconds (prevents polling spam).
"""
import logging
from typing import Dict, Optional, Tuple
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import LivePerpsQuote
from tools.market.terminal_v2.ports.driven_ports import PerpsQuotePort

logger = logging.getLogger(__name__)

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
HL_TTL_SECONDS = 4.0
TIMEOUT_SECONDS = 8.0


class HyperliquidAdapter(PerpsQuotePort):
    """Native Python adapter for Hyperliquid Info REST API."""

    def __init__(self, cache: Optional[ThreadSafeTTLCache] = None):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=HL_TTL_SECONDS)

    def get_all_mids(self) -> Dict[str, float]:
        """Fetch all mid prices across all perpetual pairs on Hyperliquid."""
        cache_key = "hyperliquid:all_mids"

        def _loader() -> Dict[str, float]:
            try:
                resp = requests.post(
                    HL_INFO_URL,
                    json={"type": "allMids"},
                    headers=BROWSER_HEADERS,
                    timeout=TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                raise ProviderError(
                    f"Failed to fetch allMids from Hyperliquid: {exc}",
                    source="Hyperliquid",
                ) from exc

            if not isinstance(data, dict):
                raise ProviderError(
                    f"Hyperliquid allMids returned unexpected type: {type(data)}",
                    source="Hyperliquid",
                )

            mids: Dict[str, float] = {}
            for sym, px_str in data.items():
                try:
                    mids[sym] = float(px_str)
                except (ValueError, TypeError):
                    continue
            return mids

        try:
            return self._cache.get_or_set(
                cache_key,
                _loader,
                ttl_seconds=HL_TTL_SECONDS,
            )
        except Exception as exc:
            raise DataUnavailableError(
                f"Hyperliquid mids data is temporarily unavailable: {exc}",
                capability="perps_quote",
                source="Hyperliquid",
            ) from exc

    def get_perps_quote(self, symbol: str) -> LivePerpsQuote:
        """Fetch live mark price and contract stats for a perpetual symbol.

        Supports standard coins (e.g. 'BTC', 'ETH') and HIP-3 builder DEX perps
        (e.g. 'xyz:TSLA', 'xyz:NVDA', 'km:US500').
        """
        raw_symbol = symbol.strip()
        # Canonical symbol formatting
        dex_ns = raw_symbol.split(":")[0] if ":" in raw_symbol else ""
        cache_key = f"hyperliquid:perps:{raw_symbol}"

        def _loader() -> LivePerpsQuote:
            # Query metaAndAssetCtxs to get detailed context
            try:
                resp = requests.post(
                    HL_INFO_URL,
                    json={"type": "metaAndAssetCtxs"},
                    headers=BROWSER_HEADERS,
                    timeout=TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                # Fallback to allMids if metaAndAssetCtxs fails
                logger.warning("metaAndAssetCtxs failed, attempting allMids fallback: %s", exc)
                all_mids = self.get_all_mids()
                if raw_symbol in all_mids:
                    return LivePerpsQuote(
                        symbol=raw_symbol,
                        mark_price=all_mids[raw_symbol],
                        dex_namespace=dex_ns,
                        asset_class="synthetic_crypto_perp",
                        contract_type="perpetual_future",
                        source="Hyperliquid",
                        is_stale=False,
                    )
                raise ProviderError(
                    f"Symbol '{raw_symbol}' not found on Hyperliquid: {exc}",
                    source="Hyperliquid",
                ) from exc

            if not isinstance(data, list) or len(data) < 2:
                raise ProviderError(
                    "Hyperliquid metaAndAssetCtxs returned malformed structure",
                    source="Hyperliquid",
                )

            universe = data[0].get("universe", [])
            asset_ctxs = data[1]

            # Find matching asset in universe
            target_idx: Optional[int] = None
            for idx, asset in enumerate(universe):
                if asset.get("name") == raw_symbol:
                    target_idx = idx
                    break

            if target_idx is None or target_idx >= len(asset_ctxs):
                # Try allMids before giving up
                all_mids = self.get_all_mids()
                if raw_symbol in all_mids:
                    return LivePerpsQuote(
                        symbol=raw_symbol,
                        mark_price=all_mids[raw_symbol],
                        dex_namespace=dex_ns,
                        asset_class="synthetic_crypto_perp",
                        contract_type="perpetual_future",
                        source="Hyperliquid",
                        is_stale=False,
                    )
                raise DataUnavailableError(
                    f"Perpetual contract '{raw_symbol}' is not listed on Hyperliquid",
                    capability="perps_quote",
                    source="Hyperliquid",
                )

            ctx = asset_ctxs[target_idx]
            mark_px = float(ctx.get("markPx") or 0.0)
            oi = float(ctx.get("openInterest") or 0.0) if ctx.get("openInterest") else None
            funding = float(ctx.get("funding") or 0.0) if ctx.get("funding") else None
            ntl_vlm = float(ctx.get("dayNtlVlm") or 0.0) if ctx.get("dayNtlVlm") else None

            return LivePerpsQuote(
                symbol=raw_symbol,
                mark_price=mark_px,
                dex_namespace=dex_ns,
                asset_class="synthetic_crypto_perp",
                contract_type="perpetual_future",
                source="Hyperliquid",
                open_interest=oi,
                funding_rate=funding,
                day_ntl_vlm=ntl_vlm,
                is_stale=False,
            )

        try:
            return self._cache.get_or_set(
                cache_key,
                _loader,
                ttl_seconds=HL_TTL_SECONDS,
            )
        except Exception as exc:
            raise DataUnavailableError(
                f"Perpetual quote for '{raw_symbol}' is temporarily unavailable: {exc}",
                capability="perps_quote",
                source="Hyperliquid",
            ) from exc
