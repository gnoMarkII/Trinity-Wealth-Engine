import json
import os
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any

import requests

from core.logger import get_logger
from tools.portfolio.ports.thai_fund_port import ThaiFundPricePort, FundNavData

log = get_logger(__name__)

_CATALOG_URL = "https://www.finnomena.com/fn3/api/fund/public/list"
_NAV_URL_TEMPLATE = "https://www.finnomena.com/fn3/api/fund/v2/public/funds/{fund_id}/nav/q?range=1M"
_NAV_HISTORY_URL_TEMPLATE = "https://www.finnomena.com/fn3/api/fund/v2/public/funds/{fund_id}/nav/q?range=MAX"
_DEFAULT_TIMEOUT = 8.0
_CATALOG_TTL_SECONDS = 86400.0  # 24 hours
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


class FinnomenaFundAdapter(ThaiFundPricePort):
    """Adapter fetching Thai mutual fund catalog and NAV data from Finnomena Public API.
    
    Adheres to ThaiFundPricePort interface. Implements local cache for fund directory
    to minimize external HTTP calls.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        catalog_ttl: float = _CATALOG_TTL_SECONDS,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self._timeout = timeout
        self._catalog_ttl = catalog_ttl
        self._cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self._cache_file = self._cache_dir / "finnomena_funds_catalog.json"
        
        self._lock = threading.Lock()
        self._catalog: Dict[str, str] = {}  # normalized_code -> fund_id
        self._catalog_loaded_at: float = 0.0
        self._nav_history_cache: Dict[str, Dict[str, float]] = {}  # fund_id -> {date_iso: nav_value}
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    def _normalize_code(self, symbol: str) -> str:
        """Normalize symbol for consistent lookup."""
        return " ".join(symbol.strip().upper().split())

    def _load_catalog_from_cache(self) -> bool:
        """Attempt to read fund catalog from local disk cache."""
        try:
            if not self._cache_file.exists():
                return False
            mtime = self._cache_file.stat().st_mtime
            if time.time() - mtime > self._catalog_ttl:
                return False  # expired
            with open(self._cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    self._catalog = data
                    self._catalog_loaded_at = mtime
                    log.info("Loaded %d Thai funds from disk cache: %s", len(self._catalog), self._cache_file)
                    return True
        except Exception as e:
            log.warning("Failed to load fund catalog from cache: %s", e)
        return False

    def _save_catalog_to_cache(self) -> None:
        """Save fund catalog to disk cache."""
        try:
            from tools.archivist.maintenance_guard import assert_write_allowed
            assert_write_allowed(self._cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            temp_file = self._cache_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self._catalog, f, ensure_ascii=False, indent=2)
            temp_file.replace(self._cache_file)
            log.info("Saved %d Thai funds to disk cache: %s", len(self._catalog), self._cache_file)
        except Exception as e:
            log.warning("Failed to save fund catalog to cache: %s", e)

    def refresh_catalog(self, force: bool = False) -> int:
        """Fetch all funds from Finnomena public list endpoint and refresh catalog."""
        with self._lock:
            now = time.time()
            if not force and self._catalog and (now - self._catalog_loaded_at < self._catalog_ttl):
                return len(self._catalog)

            if not force and self._load_catalog_from_cache():
                return len(self._catalog)

            try:
                log.info("Fetching Thai mutual fund directory from Finnomena: %s", _CATALOG_URL)
                resp = self._session.get(_CATALOG_URL, timeout=self._timeout)
                resp.raise_for_status()
                raw_list = resp.json()

                funds_list = raw_list if isinstance(raw_list, list) else raw_list.get("data", [])
                new_catalog: Dict[str, str] = {}
                for item in funds_list:
                    short_code = item.get("short_code")
                    fund_id = item.get("id")
                    if short_code and fund_id:
                        norm = self._normalize_code(short_code)
                        new_catalog[norm] = str(fund_id).strip()
                        # Also index hyphen-separated or space-separated variations if they differ
                        hyphenated = norm.replace(" ", "-")
                        if hyphenated not in new_catalog:
                            new_catalog[hyphenated] = str(fund_id).strip()

                if new_catalog:
                    self._catalog = new_catalog
                    self._catalog_loaded_at = now
                    self._save_catalog_to_cache()
                    log.info("Successfully loaded %d Thai mutual funds into catalog.", len(self._catalog))
                    return len(self._catalog)
                else:
                    log.warning("Finnomena returned empty fund list.")
            except Exception as e:
                log.warning("Failed to fetch Thai mutual funds from Finnomena: %s", e)

            return len(self._catalog)

    def _ensure_catalog(self) -> None:
        if not self._catalog:
            self.refresh_catalog(force=False)

    def has_fund(self, symbol: str) -> bool:
        """Return True if symbol is found in the fund catalog."""
        self._ensure_catalog()
        norm = self._normalize_code(symbol)
        if norm in self._catalog:
            return True
        hyphenated = norm.replace(" ", "-")
        return hyphenated in self._catalog

    def get_fund_id(self, symbol: str) -> Optional[str]:
        """Look up fund id for a given symbol."""
        self._ensure_catalog()
        norm = self._normalize_code(symbol)
        fid = self._catalog.get(norm)
        if not fid:
            fid = self._catalog.get(norm.replace(" ", "-"))
        if not fid:
            # Try reloading once in case of a newly listed fund
            self.refresh_catalog(force=True)
            fid = self._catalog.get(norm) or self._catalog.get(norm.replace(" ", "-"))
        return fid

    def fetch_nav(self, symbol: str) -> Optional[FundNavData]:
        """Fetch latest daily NAV for a Thai mutual fund."""
        clean_sym = symbol.strip()
        fund_id = self.get_fund_id(clean_sym)
        if not fund_id:
            log.debug("Symbol '%s' not found in Thai mutual fund catalog.", clean_sym)
            return None

        url = _NAV_URL_TEMPLATE.format(fund_id=fund_id)
        try:
            log.debug("Fetching NAV for %s (id=%s) from %s", clean_sym, fund_id, url)
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
            res_data = resp.json()

            navs = []
            if isinstance(res_data, list):
                navs = res_data
            elif isinstance(res_data, dict):
                inner_data = res_data.get("data")
                if isinstance(inner_data, dict):
                    navs = inner_data.get("navs", [])
                elif isinstance(inner_data, list):
                    navs = inner_data

            if not navs:
                log.warning("No NAV records returned for %s (id=%s)", clean_sym, fund_id)
                return None

            latest = navs[-1]
            raw_val = latest.get("value")
            if raw_val is None:
                return None

            nav_val = float(raw_val)
            raw_date = latest.get("date", "")
            nav_date = str(raw_date)[:10] if raw_date else ""
            raw_pct = latest.get("percent_change")
            pct_val = float(raw_pct) if raw_pct is not None else None

            return FundNavData(
                symbol=clean_sym,
                nav=round(nav_val, 4),
                nav_date=nav_date,
                percent_change=pct_val,
                currency="THB",
            )
        except Exception as e:
            log.warning("Failed to fetch NAV for %s (id=%s): %s", clean_sym, fund_id, e)
            return None

    def fetch_historical_nav(self, symbol: str, target_date: str) -> Optional[float]:
        """Fetch historical NAV for a Thai mutual fund on a specific date (YYYY-MM-DD).
        
        If exact date is not available (e.g. weekend/holiday), returns the latest
        available NAV on or before target_date.
        """
        clean_sym = symbol.strip()
        fund_id = self.get_fund_id(clean_sym)
        if not fund_id:
            log.debug("Symbol '%s' not found for historical NAV query.", clean_sym)
            return None

        # Check cache
        if fund_id not in self._nav_history_cache:
            url = _NAV_HISTORY_URL_TEMPLATE.format(fund_id=fund_id)
            try:
                log.info("Fetching historical NAV series for %s (id=%s)...", clean_sym, fund_id)
                resp = self._session.get(url, timeout=self._timeout)
                resp.raise_for_status()
                res_data = resp.json()

                nav_items = []
                if isinstance(res_data, dict):
                    inner_data = res_data.get("data")
                    if isinstance(inner_data, dict):
                        nav_items = inner_data.get("navs", [])
                    elif isinstance(inner_data, list):
                        nav_items = inner_data

                history_dict: Dict[str, float] = {}
                for item in nav_items:
                    raw_d = item.get("date", "")
                    raw_v = item.get("value")
                    if raw_d and raw_v is not None:
                        d_str = str(raw_d)[:10]
                        history_dict[d_str] = float(raw_v)

                self._nav_history_cache[fund_id] = history_dict
            except Exception as e:
                log.warning("Failed to fetch historical NAV for %s (id=%s): %s", clean_sym, fund_id, e)
                return None

        history = self._nav_history_cache.get(fund_id, {})
        if not history:
            return None

        # 1. Exact match
        target_iso = target_date.strip()[:10]
        if target_iso in history:
            return round(history[target_iso], 4)

        # 2. Closest preceding date (e.g., if trade effective date falls on holiday)
        eligible_dates = [d for d in history.keys() if d <= target_iso]
        if eligible_dates:
            closest_date = max(eligible_dates)
            log.debug("No exact NAV for %s on %s; using closest prior date %s (NAV=%.4f)",
                      clean_sym, target_iso, closest_date, history[closest_date])
            return round(history[closest_date], 4)

        return None
