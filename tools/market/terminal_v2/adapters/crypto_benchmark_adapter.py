"""Crypto Benchmark & Cross-Asset Ratio Adapter.

Calculates Bitcoin spot prices, 24h/7d returns, and the BTC/Gold ratio
as a high-signal macro indicator for global risk appetite vs safe-haven demand.
Strict Rule: Keyless fallback; cached with ThreadSafeTTLCache.
"""
from datetime import datetime, timezone
import logging
import os
import time
from typing import Optional
import requests
import yfinance as yf

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import CryptoBenchmarkSnapshot
from tools.market.terminal_v2.ports.driven_ports import CryptoBenchmarkPort

logger = logging.getLogger(__name__)

BENCHMARK_TTL_SECONDS = 300.0          # 5 minutes
BENCHMARK_MAX_STALE_SECONDS = 7200.0    # 2 hours
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"


class CryptoBenchmarkAdapter(CryptoBenchmarkPort):
    """Adapter for Bitcoin spot benchmark and the BTC / Gold valuation ratio."""

    def __init__(
        self,
        cache: Optional[ThreadSafeTTLCache] = None,
        enabled: Optional[bool] = None,
    ):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=BENCHMARK_TTL_SECONDS)
        if enabled is not None:
            self._enabled = enabled
        else:
            env_val = os.getenv("ENABLE_CRYPTO_BENCHMARK", "true").strip().lower()
            self._enabled = env_val not in ("0", "false", "no", "off")

    def _fetch_snapshot(self) -> CryptoBenchmarkSnapshot:
        if not self._enabled:
            raise DataUnavailableError(
                "Crypto benchmark capability is disabled by feature flag",
                capability="crypto-benchmark",
                source="Benchmark",
            )

        btc_price: Optional[float] = None
        btc_24h_pct: Optional[float] = None
        btc_7d_pct: Optional[float] = None
        gold_price: Optional[float] = None
        source_used = "Yahoo Finance (BTC-USD, GC=F)"

        # 1. Try yfinance for BTC-USD and GC=F (Gold Futures USD/oz)
        try:
            btc_ticker = yf.Ticker("BTC-USD")
            btc_hist = btc_ticker.history(period="8d")
            if not btc_hist.empty and "Close" in btc_hist.columns:
                btc_price = float(btc_hist["Close"].iloc[-1])
                if len(btc_hist) >= 2:
                    prev_close = float(btc_hist["Close"].iloc[-2])
                    if prev_close > 0:
                        btc_24h_pct = round(((btc_price - prev_close) / prev_close) * 100.0, 2)
                first_close = float(btc_hist["Close"].iloc[0])
                if first_close > 0:
                    btc_7d_pct = round(((btc_price - first_close) / first_close) * 100.0, 2)

            gold_ticker = yf.Ticker("GC=F")
            gold_fi = gold_ticker.fast_info
            if hasattr(gold_fi, "last_price") and gold_fi.last_price is not None:
                gold_price = float(gold_fi.last_price)
        except Exception as exc:
            logger.warning("Primary yfinance fetch for crypto benchmark failed: %s; trying fallback", exc)

        # 2. Fallback to CoinGecko for BTC if yfinance failed
        if btc_price is None:
            try:
                resp = requests.get(COINGECKO_URL, headers=BROWSER_HEADERS, timeout=8)
                resp.raise_for_status()
                cg_data = resp.json().get("bitcoin", {})
                if "usd" in cg_data:
                    btc_price = float(cg_data["usd"])
                    btc_24h_pct = round(float(cg_data.get("usd_24h_change", 0.0)), 2)
                    source_used = "CoinGecko"
            except Exception as cg_exc:
                logger.warning("CoinGecko fallback failed: %s", cg_exc)

        if btc_price is None:
            raise ProviderError("Failed to obtain Bitcoin spot price from all providers", source="CryptoBenchmark")

        # 3. Calculate BTC / Gold ratio
        btc_gold_ratio: Optional[float] = None
        if btc_price is not None and gold_price is not None and gold_price > 0:
            btc_gold_ratio = round(btc_price / gold_price, 2)

        as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return CryptoBenchmarkSnapshot(
            symbol="BTC",
            price_usd=round(btc_price, 2),
            change_24h_pct=btc_24h_pct,
            change_7d_pct=btc_7d_pct,
            gold_price_usd=round(gold_price, 2) if gold_price is not None else None,
            btc_gold_ratio=btc_gold_ratio,
            as_of_date=as_of,
            fetched_at=time.time(),
            source=source_used,
        )

    def get_crypto_benchmark(self) -> CryptoBenchmarkSnapshot:
        """Fetch spot BTC price, returns, and BTC/Gold ratio."""
        cache_key = "crypto:benchmark:btc"
        return self._cache.get_or_set(
            key=cache_key,
            loader=self._fetch_snapshot,
            ttl_seconds=BENCHMARK_TTL_SECONDS,
            max_stale_seconds=BENCHMARK_MAX_STALE_SECONDS,
        )
