"""Application Query Service for OHLCV Market Data, Warmup, Pivots and Corporate Actions."""
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Literal
from zoneinfo import ZoneInfo
import pandas as pd

from tools.market.ohlcv.domain.models import (
    OHLCVCandleDTO,
    OHLCVResponseDTO,
    CorporateActionEventDTO,
    CorporateActionsMetadataDTO,
)
from tools.market.ohlcv.domain.validation import (
    TIMEFRAME_CAPABILITIES,
    validate_ticker,
    validate_interval_and_range,
)
from tools.market.ohlcv.domain.calculations import (
    get_fetch_period,
    calculate_warmup_metadata,
    calculate_52w,
    calculate_pivot_levels,
    map_corporate_actions,
)
from tools.market.ohlcv.ports.ohlcv_port import OhlcvProviderPort, CorporateActionProviderPort, AssetResolverPort

log = logging.getLogger(__name__)

EARNINGS_CACHE_TTL = 6 * 3600            # 6 Hours
DIVIDENDS_SPLITS_CACHE_TTL = 24 * 3600   # 24 Hours


class OhlcvRequestValidationError(ValueError):
    """Input validation failed before any outbound market call."""


class OHLCVQueryService:
    """Application Service for OHLCV Candles and Corporate Actions."""

    def __init__(
        self,
        ohlcv_provider: Optional[OhlcvProviderPort] = None,
        action_provider: Optional[CorporateActionProviderPort] = None,
        resolver: Optional[AssetResolverPort] = None,
    ):
        if ohlcv_provider is None or action_provider is None or resolver is None:
            raise ValueError("OHLCV application service requires provider and resolver ports")
        self._ohlcv_provider = ohlcv_provider
        self._action_provider = action_provider
        self._resolver = resolver

        self._action_lock = threading.Lock()
        self._action_key_locks: dict[str, threading.Lock] = {}
        self._action_cache: dict[str, dict[str, tuple]] = {}

    def _fetch_corporate_actions(
        self,
        ticker: str,
        provider_symbol: str,
        candles: list[OHLCVCandleDTO],
        interval: str,
        currency: str,
        tz_name: str,
    ) -> tuple[list[CorporateActionEventDTO], CorporateActionsMetadataDTO]:
        if not candles:
            meta = CorporateActionsMetadataDTO(
                status="unavailable",
                earnings_status="empty",
                dividends_status="empty",
                splits_status="empty",
                missing_sources=["earnings", "dividends", "splits"],
            )
            return [], meta

        now_mono = time.monotonic()

        with self._action_lock:
            if provider_symbol not in self._action_key_locks:
                self._action_key_locks[provider_symbol] = threading.Lock()
            ticker_key_lock = self._action_key_locks[provider_symbol]

        with ticker_key_lock:
            with self._action_lock:
                if provider_symbol not in self._action_cache:
                    self._action_cache[provider_symbol] = {}
                ticker_cache = self._action_cache[provider_symbol]

            # 1. Earnings
            raw_earnings, earnings_status, earnings_as_of = self._action_provider.fetch_earnings(
                provider_symbol, tz_name
            )

            # 2. Dividends
            if "dividends" in ticker_cache and now_mono < ticker_cache["dividends"][0]:
                raw_dividends = ticker_cache["dividends"][1]
                dividends_status = "ok" if raw_dividends else "empty"
                dividends_as_of = ticker_cache["dividends"][2] if len(ticker_cache["dividends"]) > 2 else None
            else:
                raw_dividends, dividends_status = self._action_provider.fetch_dividends(provider_symbol)
                dividends_as_of = datetime.now(ZoneInfo(tz_name)).isoformat()
                if dividends_status != "failed":
                    with self._action_lock:
                        ticker_cache["dividends"] = (now_mono + DIVIDENDS_SPLITS_CACHE_TTL, raw_dividends, dividends_as_of)

            # 3. Splits
            if "splits" in ticker_cache and now_mono < ticker_cache["splits"][0]:
                raw_splits = ticker_cache["splits"][1]
                splits_status = "ok" if raw_splits else "empty"
                splits_as_of = ticker_cache["splits"][2] if len(ticker_cache["splits"]) > 2 else None
            else:
                raw_splits, splits_status = self._action_provider.fetch_splits(provider_symbol)
                splits_as_of = datetime.now(ZoneInfo(tz_name)).isoformat()
                if splits_status != "failed":
                    with self._action_lock:
                        ticker_cache["splits"] = (now_mono + DIVIDENDS_SPLITS_CACHE_TTL, raw_splits, splits_as_of)

        statuses = [earnings_status, dividends_status, splits_status]
        missing_sources = []
        if earnings_status == "failed":
            missing_sources.append("earnings")
        if dividends_status == "failed":
            missing_sources.append("dividends")
        if splits_status == "failed":
            missing_sources.append("splits")

        if all(s == "failed" for s in statuses):
            overall_status: Literal["available", "partial", "unavailable"] = "unavailable"
        elif any(s == "failed" for s in statuses):
            overall_status = "partial"
        else:
            overall_status = "available"

        available_timestamps = [ts for ts in [earnings_as_of, dividends_as_of, splits_as_of] if ts is not None]
        oldest_as_of = min(available_timestamps) if available_timestamps else None

        metadata_dto = CorporateActionsMetadataDTO(
            status=overall_status,
            as_of=oldest_as_of,
            earnings_status=earnings_status,
            earnings_as_of=earnings_as_of,
            dividends_status=dividends_status,
            dividends_as_of=dividends_as_of,
            splits_status=splits_status,
            splits_as_of=splits_as_of,
            missing_sources=missing_sources,
            data_provenance="Yahoo Finance (yfinance)",
        )

        events = map_corporate_actions(
            raw_earnings, raw_dividends, raw_splits, candles, interval, currency, tz_name
        )

        return events, metadata_dto

    def get_ohlcv(self, ticker: str, range_str: str = "6mo", interval_str: str = "1d") -> OHLCVResponseDTO:
        try:
            clean_ticker = validate_ticker(ticker)
            validate_interval_and_range(interval_str, range_str)
        except ValueError as exc:
            raise OhlcvRequestValidationError(str(exc)) from exc

        resolved = self._resolver.resolve(clean_ticker)
        provider_symbol = resolved.provider_symbol if resolved else clean_ticker
        market: Literal["TH", "US"] = "TH" if (resolved and (resolved.market == "TH" or provider_symbol.endswith(".BK"))) else "US"
        currency: Literal["USD", "THB"] = "THB" if market == "TH" else "USD"
        tz_name = "Asia/Bangkok" if market == "TH" else "America/New_York"

        fetch_period = get_fetch_period(range_str, interval_str)
        chart_df = self._ohlcv_provider.fetch_history(provider_symbol, period=fetch_period, interval=interval_str, auto_adjust=True)

        if (chart_df is None or chart_df.empty) and fetch_period != range_str:
            chart_df = self._ohlcv_provider.fetch_history(provider_symbol, period=range_str, interval=interval_str, auto_adjust=True)

        if chart_df is None or chart_df.empty:
            raise ValueError(f"No OHLCV historical data found for {clean_ticker}")

        seen_timestamps = set()
        candles: list[OHLCVCandleDTO] = []
        for ts, row in chart_df.iterrows():
            try:
                epoch_ms = int(ts.timestamp() * 1000)
                if epoch_ms in seen_timestamps:
                    continue
                seen_timestamps.add(epoch_ms)
                candles.append(
                    OHLCVCandleDTO(
                        timestamp=epoch_ms,
                        open=round(float(row["Open"]), 4),
                        high=round(float(row["High"]), 4),
                        low=round(float(row["Low"]), 4),
                        close=round(float(row["Close"]), 4),
                        volume=round(float(row.get("Volume", 0)), 2),
                    )
                )
            except Exception:
                continue

        candles.sort(key=lambda c: c.timestamp)

        display_start_ts, avail_warmup, req_warmup, warmup_stat, indicator_warmup = calculate_warmup_metadata(
            candles, range_str, interval_str, tz_name
        )

        current_price: Optional[float] = None
        price_change: Optional[float] = None
        price_change_pct: Optional[float] = None
        price_as_of: Optional[str] = None
        daily_candles: list[OHLCVCandleDTO] = []

        try:
            if interval_str == "1d" and len(candles) >= 250:
                daily_candles = candles
            else:
                daily_df = self._ohlcv_provider.fetch_history(provider_symbol, period="1y", interval="1d", auto_adjust=True)
                if daily_df is not None and not daily_df.empty:
                    for ts, row in daily_df.iterrows():
                        daily_candles.append(
                            OHLCVCandleDTO(
                                timestamp=int(ts.timestamp() * 1000),
                                open=round(float(row["Open"]), 4),
                                high=round(float(row["High"]), 4),
                                low=round(float(row["Low"]), 4),
                                close=round(float(row["Close"]), 4),
                                volume=round(float(row.get("Volume", 0)), 2),
                            )
                        )

            ref_for_price = daily_candles if daily_candles else candles
            if ref_for_price:
                latest_c = ref_for_price[-1]
                current_price = latest_c.close
                latest_dt = datetime.fromtimestamp(latest_c.timestamp / 1000.0, tz=ZoneInfo(tz_name))
                price_as_of = latest_dt.isoformat()

                if len(ref_for_price) >= 2:
                    prev_c = ref_for_price[-2]
                    price_change = round(latest_c.close - prev_c.close, 4)
                    if prev_c.close > 0:
                        price_change_pct = round((price_change / prev_c.close) * 100.0, 4)
                    else:
                        price_change = 0.0
                        price_change_pct = 0.0
        except Exception as e:
            log.warning("Daily reference calculation error for %s: %s", provider_symbol, e)
            if candles:
                current_price = candles[-1].close

        w_high, w_low, cov_days = None, None, 0
        if candles:
            latest_dt = datetime.fromtimestamp(candles[-1].timestamp / 1000.0, tz=ZoneInfo(tz_name))
            w_high, w_low, cov_days = calculate_52w(daily_candles or candles, latest_dt, tz_name)

        pivot_levels, pivot_tf, pivot_as_of = None, None, None
        try:
            monthly_df = self._ohlcv_provider.fetch_history(provider_symbol, period="2y", interval="1mo", auto_adjust=True)
            pivot_levels, pivot_tf, pivot_as_of = calculate_pivot_levels(monthly_df, tz_name)
        except Exception as e:
            log.warning("Monthly pivot fetch failed for %s: %s", provider_symbol, e)

        # Corporate Actions
        events: list[CorporateActionEventDTO] = []
        events_metadata: Optional[CorporateActionsMetadataDTO] = None
        try:
            events, events_metadata = self._fetch_corporate_actions(
                clean_ticker,
                provider_symbol,
                candles,
                interval_str,
                currency,
                tz_name,
            )
        except Exception as e:
            log.warning("Corporate actions fetch error for %s: %s", provider_symbol, e)
            events_metadata = CorporateActionsMetadataDTO(
                status="unavailable",
                earnings_status="failed",
                dividends_status="failed",
                splits_status="failed",
                missing_sources=["earnings", "dividends", "splits"],
            )

        effective_caps = dict(TIMEFRAME_CAPABILITIES)
        allowed_for_interval = TIMEFRAME_CAPABILITIES.get(interval_str, [])

        return OHLCVResponseDTO(
            ticker=clean_ticker,
            market=market,
            currency=currency,
            price_basis="provider_proportional_adj_close_ratio",
            provider_name="yfinance",
            provider_tier="best_effort",
            feed_latency_model="delayed_15m",
            current_price=current_price,
            price_change=price_change,
            price_change_pct=price_change_pct,
            price_as_of=price_as_of,
            candles=candles,
            pivot_levels=pivot_levels,
            pivot_period=pivot_tf,
            pivot_as_of=pivot_as_of,
            requested_range=range_str,
            interval=interval_str,
            allowed_ranges=allowed_for_interval,
            effective_capabilities=effective_caps,
            capability_reasons={},
            display_start_timestamp=display_start_ts,
            available_warmup_bars=avail_warmup,
            required_warmup_bars=req_warmup,
            warmup_status=warmup_stat,
            indicator_warmup=indicator_warmup,
            events=events,
            events_metadata=events_metadata,
            week52_high=w_high,
            week52_low=w_low,
            week52_coverage_calendar_days=cov_days,
        )


class CachedOhlcvQueryService(OHLCVQueryService):
    """Decorator adding thread-safe TTL caching over OHLCVQueryService."""

    def __init__(
        self,
        ohlcv_provider: Optional[OhlcvProviderPort] = None,
        action_provider: Optional[CorporateActionProviderPort] = None,
        resolver: Optional[AssetResolverPort] = None,
        cache_ttl: float = 300.0,
    ):
        super().__init__(
            ohlcv_provider=ohlcv_provider,
            action_provider=action_provider,
            resolver=resolver,
        )
        self._cache_ttl = cache_ttl
        self._cache_lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._cache: dict[str, tuple[float, OHLCVResponseDTO]] = {}

    def _get_key_lock(self, cache_key: str) -> threading.Lock:
        with self._cache_lock:
            if cache_key not in self._key_locks:
                self._key_locks[cache_key] = threading.Lock()
            return self._key_locks[cache_key]

    def get_ohlcv(self, ticker: str, range_str: str = "6mo", interval_str: str = "1d") -> OHLCVResponseDTO:
        clean_ticker = ticker.upper().strip()
        resolved = self._resolver.resolve(clean_ticker)
        provider_symbol = resolved.provider_symbol if resolved else clean_ticker

        cache_key = f"{provider_symbol}:{range_str}:{interval_str}"
        now_mono = time.monotonic()

        with self._cache_lock:
            if cache_key in self._cache:
                expire_at, cached_dto = self._cache[cache_key]
                if now_mono < expire_at:
                    return cached_dto

        key_lock = self._get_key_lock(cache_key)
        with key_lock:
            with self._cache_lock:
                if cache_key in self._cache:
                    expire_at, cached_dto = self._cache[cache_key]
                    if time.monotonic() < expire_at:
                        return cached_dto

            dto = super().get_ohlcv(ticker=ticker, range_str=range_str, interval_str=interval_str)

            with self._cache_lock:
                self._cache[cache_key] = (now_mono + self._cache_ttl, dto)

            return dto
