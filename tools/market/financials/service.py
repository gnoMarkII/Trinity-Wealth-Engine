"""Financials Service (Facade & Orchestrator).

Orchestrates caching, tiered TTL, single-flight locking, stale recovery, and provider delegation.
"""
from datetime import datetime, timezone
import logging
import threading
import time
from typing import Literal, Optional

from tools.market.financials.adapters.composite_us_provider import CompositeUsFinancialProvider
from tools.market.financials.adapters.sqlite_cache_adapter import SQLiteCacheAdapter
from tools.market.financials.adapters.thai_set_provider import ThaiSetFinancialProvider
from tools.market.financials.domain.models import FinancialStatementsDTO
from tools.market.financials.ports.cache_port import CacheEntry, FinancialCachePort
from tools.market.financials.ports.provider_port import FinancialStatementProviderPort

log = logging.getLogger(__name__)

# Cache TTLs & Constants
EDGAR_CACHE_TTL_SECONDS = 7 * 86400.0        # 7 Days
US_FALLBACK_CACHE_TTL_SECONDS = 3600.0        # 1 Hour
TH_CACHE_TTL_SECONDS = 7 * 86400.0           # 7 Days
MAX_STALE_AGE_SECONDS = 30 * 86400.0         # 30 Days

_FETCH_SEMAPHORE = threading.BoundedSemaphore(value=3)
_BURST_FAIL_CACHE: dict[tuple[str, str], float] = {}  # (provider, provider_symbol) -> monotonic timestamp
_BURST_COOLDOWN_SECONDS = 15.0
_SYMBOL_LOCKS: dict[str, threading.Lock] = {}
_GLOBAL_LOCK = threading.Lock()


def get_symbol_lock(symbol: str) -> threading.Lock:
    with _GLOBAL_LOCK:
        sym = symbol.upper()
        if sym not in _SYMBOL_LOCKS:
            _SYMBOL_LOCKS[sym] = threading.Lock()
        return _SYMBOL_LOCKS[sym]


class FinancialsService:
    """Financials Service Facade orchestrating caching and multi-provider pipelines."""

    def __init__(
        self,
        cache_port: Optional[FinancialCachePort] = None,
        us_provider: Optional[FinancialStatementProviderPort] = None,
        th_provider: Optional[FinancialStatementProviderPort] = None,
    ):
        self._cache = cache_port or SQLiteCacheAdapter()
        self._us_provider = us_provider or CompositeUsFinancialProvider()
        self._th_provider = th_provider or ThaiSetFinancialProvider()

    def get_financial_statements(
        self,
        ticker: str,
        market: Literal["US", "TH"] = "US",
        provider_symbol: Optional[str] = None,
        force_refresh: bool = False,
    ) -> FinancialStatementsDTO:
        """ดึงงบการเงินพร้อมระบบ Cache, Tiered TTL, Single-Flight Lock, Fallback, และ Stale Recovery"""
        sym = (provider_symbol or ticker).upper()
        mkt = market.upper()
        now_ts = time.time()
        now_mono = time.monotonic()

        # 1. ตรวจสอบ Cache
        cached_entry: Optional[CacheEntry] = None
        if not force_refresh:
            cached_entry = self._cache.get(mkt, sym)
            if cached_entry:
                raw_synced = cached_entry.synced_at
                provider = cached_entry.provider
                ttl = (
                    EDGAR_CACHE_TTL_SECONDS
                    if provider == "edgartools"
                    else (US_FALLBACK_CACHE_TTL_SECONDS if mkt == "US" else TH_CACHE_TTL_SECONDS)
                )
                age = now_ts - raw_synced
                if age < ttl:
                    log.info("Returning fresh cached financial statements for %s:%s (age: %.1fh)", mkt, sym, age / 3600.0)
                    return cached_entry.statements
                else:
                    log.info("Cached financial statements expired for %s:%s (age: %.1fh, TTL: %.1fh). Refreshing.", mkt, sym, age / 3600.0, ttl / 3600.0)

        # 2. Concurrency Lock & Cooldown
        sym_lock = get_symbol_lock(sym)
        with sym_lock:
            # Re-check cache after acquiring lock
            if not force_refresh:
                cached_entry = self._cache.get(mkt, sym)
                if cached_entry:
                    raw_synced = cached_entry.synced_at
                    provider = cached_entry.provider
                    ttl = (
                        EDGAR_CACHE_TTL_SECONDS
                        if provider == "edgartools"
                        else (US_FALLBACK_CACHE_TTL_SECONDS if mkt == "US" else TH_CACHE_TTL_SECONDS)
                    )
                    age = now_ts - raw_synced
                    if age < ttl:
                        return cached_entry.statements

            # Check burst cooldown
            burst_key = ("us" if mkt == "US" else "th", sym)
            last_fail = _BURST_FAIL_CACHE.get(burst_key)
            if last_fail and (now_mono - last_fail < _BURST_COOLDOWN_SECONDS):
                log.warning("Burst cooldown active for %s:%s. Reusing cached or returning empty.", mkt, sym)
                if cached_entry:
                    return cached_entry.statements
                return self._create_empty_dto(ticker, mkt, sym, "Rate limit / Cooldown active")

            # 3. Live Fetch via Provider
            fresh_dto: Optional[FinancialStatementsDTO] = None
            acquire_ok = _FETCH_SEMAPHORE.acquire(blocking=True, timeout=10.0)
            if not acquire_ok:
                log.warning("Fetch semaphore timeout for %s:%s. Reusing cached.", mkt, sym)
                if cached_entry:
                    return cached_entry.statements
                return self._create_empty_dto(ticker, mkt, sym, "Server busy")

            try:
                if mkt == "US":
                    fresh_dto = self._us_provider.fetch_statements(ticker, sym)
                    if not fresh_dto:
                        log.info("US SEC EDGAR returned None. Trying US yfinance fallback for %s", sym)
                        fallback_provider = ThaiSetFinancialProvider(market="US")
                        fresh_dto = fallback_provider.fetch_statements(ticker, sym)
                else:
                    fresh_dto = self._th_provider.fetch_statements(ticker, sym)
            finally:
                _FETCH_SEMAPHORE.release()

            # 4. Save to Cache if fetch succeeded
            if fresh_dto and fresh_dto.annual and fresh_dto.quarterly:
                fresh_dto.synced_at = datetime.fromtimestamp(now_ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                entry = CacheEntry(
                    statements=fresh_dto,
                    provider=fresh_dto.provider or ("edgartools" if mkt == "US" else "yfinance"),
                    synced_at=now_ts,
                )
                self._cache.save(mkt, sym, entry)
                _BURST_FAIL_CACHE.pop(burst_key, None)
                return fresh_dto

            # 5. Stale Recovery Fallback if live fetch failed
            _BURST_FAIL_CACHE[burst_key] = now_mono
            if cached_entry:
                raw_synced = cached_entry.synced_at
                age = now_ts - raw_synced
                if age < MAX_STALE_AGE_SECONDS:
                    log.warning("Live fetch failed for %s:%s. Returning stale cached data (age: %.1fh)", mkt, sym, age / 3600.0)
                    stale_dto = cached_entry.statements
                    stale_dto.data_status = "stale"
                    stale_dto.warnings.append(f"Live fetch failed. Showing cached data from {age/86400.0:.1f} days ago.")
                    return stale_dto

            return self._create_empty_dto(ticker, mkt, sym, "Data fetch failed and no cache available")

    def _create_empty_dto(self, ticker: str, market: str, provider_symbol: str, error_msg: str) -> FinancialStatementsDTO:
        return FinancialStatementsDTO(
            schema_version=6,
            ticker=ticker.upper(),
            market="US" if market == "US" else "TH",
            currency="USD" if market == "US" else "THB",
            provider="edgartools" if market == "US" else "yfinance",
            provider_symbol=provider_symbol.upper(),
            data_status="empty",
            coverage_status="partial",
            core_coverage_status="partial",
            expanded_coverage_status="not_available",
            expanded_data_status="not_available",
            core_coverage_pct=0.0,
            expanded_coverage_pct=0.0,
            missing_required_items=["All periods missing"],
            missing_expanded_items=["All periods missing"],
            validation_warnings=[error_msg],
            expanded_validation_warnings=[],
            expanded_error_count=0,
            warnings=[error_msg],
            annual=[],
            quarterly=[],
            summary_chart_annual=[],
            summary_chart_quarterly=[],
            ratios_annual=[],
            ratios_quarterly=[],
            synced_at=None,
        )
