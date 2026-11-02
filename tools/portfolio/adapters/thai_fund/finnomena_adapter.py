import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests

from core.logger import get_logger
from core.portfolio.sector_taxonomy import normalize_sector_name
from tools._atomic_io import _atomic_write_to
from tools.portfolio.adapters.thai_fund.master_fund_catalog import find_master_fund
from tools.portfolio.adapters.thai_fund.pvd_symbol_mapper import resolve_pvd_symbol
from tools.portfolio.adapters.thai_fund.set_ticker_normalizer import normalize_set_company_name
from tools.portfolio.ports.thai_fund_port import (
    ThaiFundPricePort,
    FundNavData,
    FundConstituentsData,
    FundHoldingItem,
    FundSectorItem,
)

log = get_logger(__name__)

_CATALOG_URL = "https://www.finnomena.com/fn3/api/fund/public/list"
_NAV_URL_TEMPLATE = "https://www.finnomena.com/fn3/api/fund/v2/public/funds/{fund_id}/nav/q?range=1M"
_NAV_HISTORY_URL_TEMPLATE = "https://www.finnomena.com/fn3/api/fund/v2/public/funds/{fund_id}/nav/q?range=MAX"
_PORTFOLIO_URL_TEMPLATE = "https://www.finnomena.com/fn3/api/fund/v2/public/funds/{fund_id}/portfolio"
_DEFAULT_TIMEOUT = 8.0
_CATALOG_TTL_SECONDS = 86400.0  # 24 hours
_CONSTITUENTS_TTL_SECONDS = 7 * 86400.0  # 7 days
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
def _extract_elements(val: Any) -> tuple[Optional[str], list[dict]]:
    """Extract (data_date, elements_list) from either a dict with 'elements' or a raw list."""
    if isinstance(val, dict):
        date_str = str(val.get("data_date") or "")[:10] if val.get("data_date") else None
        elems = val.get("elements")
        if isinstance(elems, list):
            return date_str, [e for e in elems if isinstance(e, dict)]
        return date_str, []
    elif isinstance(val, list):
        return None, [e for e in val if isinstance(e, dict)]
    return None, []


class FinnomenaFundAdapter(ThaiFundPricePort):
    """Adapter fetching Thai mutual fund catalog and NAV data from Finnomena Public API.
    
    Adheres to ThaiFundPricePort interface. Implements local cache for fund directory
    and look-through constituents to minimize external HTTP calls.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        catalog_ttl: float = _CATALOG_TTL_SECONDS,
        constituents_ttl: float = _CONSTITUENTS_TTL_SECONDS,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self._timeout = timeout
        self._catalog_ttl = catalog_ttl
        self._constituents_ttl = constituents_ttl
        cache_dir_env = os.environ.get("FINNOMENA_CACHE_DIR")
        self._cache_dir = Path(cache_dir) if cache_dir else (Path(cache_dir_env) if cache_dir_env else Path("data/cache"))
        self._cache_file = self._cache_dir / "finnomena_funds_catalog.json"
        self._constituents_cache_file = self._cache_dir / "finnomena_constituents_cache.json"
        
        self._lock = threading.RLock()
        self._catalog: Dict[str, str] = {}  # normalized_code -> fund_id
        self._catalog_loaded_at: float = 0.0
        self._nav_history_cache: Dict[str, Dict[str, float]] = {}  # fund_id -> {date_iso: nav_value}
        self._constituents_cache: Dict[str, dict] = {}  # symbol -> cached dict
        self._constituents_loaded: bool = False
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    def _normalize_code(self, symbol: str) -> str:
        """Normalize symbol for consistent lookup."""
        return " ".join(symbol.strip().upper().split())

    def _load_catalog_from_cache(self) -> bool:
        """Attempt to read fund catalog from local disk cache."""
        with self._lock:
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
            _atomic_write_to(self._cache_file, json.dumps(self._catalog, ensure_ascii=False, indent=2))
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

    def _load_constituents_cache(self) -> None:
        """Load constituents from disk cache into memory."""
        with self._lock:
            if self._constituents_loaded:
                return
            if self._constituents_cache_file.exists():
                try:
                    with open(self._constituents_cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            self._constituents_cache = data
                except Exception as e:
                    log.warning("Failed to load constituents cache: %s", e)
            self._constituents_loaded = True

    def _save_constituents_cache(self) -> None:
        """Save in-memory constituents cache to disk using atomic swap."""
        try:
            content = json.dumps(self._constituents_cache, ensure_ascii=False, indent=2)
            _atomic_write_to(self._constituents_cache_file, content)
        except Exception as e:
            log.warning("Failed to save constituents cache to disk: %s", e)

    def fetch_constituents(self, symbol: str) -> Optional[FundConstituentsData]:
        """Fetch constituents breakdown (top holdings, sectors, asset allocation) for a fund."""
        clean_sym = symbol.strip()
        resolved_sym, is_pvd = resolve_pvd_symbol(clean_sym)
        cache_key = self._normalize_code(clean_sym)

        self._load_constituents_cache()
        now = time.time()

        # Check in-memory cache
        cached_entry = self._constituents_cache.get(cache_key)
        if cached_entry:
            cached_at = cached_entry.get("cached_at", 0.0)
            if now - cached_at < self._constituents_ttl:
                try:
                    return FundConstituentsData.model_validate(cached_entry["data"])
                except Exception as e:
                    log.warning("Failed to parse cached constituents for %s: %s", clean_sym, e)

        # Resolve fund_id
        fund_id = self.get_fund_id(resolved_sym)
        if not fund_id:
            # Try stripped variations (e.g. TLNDQINCOME-UH-X -> TLNDQINCOME, SCBS&P500E -> SCBS&P500)
            candidate_base = re.sub(r"(?:-(?:UH|X|A|D|E|SSF|RMF|P|I|B|\([A-Z0-9]+\)))+$", "", resolved_sym, flags=re.IGNORECASE)
            candidate_base = re.sub(r"\([A-Z0-9]+\)$", "", candidate_base).strip()
            fund_id = self.get_fund_id(candidate_base)
            if not fund_id and resolved_sym.endswith("E"):
                fund_id = self.get_fund_id(resolved_sym[:-1] + "(E)") or self.get_fund_id(resolved_sym[:-1] + "-E")

        # Fallback: if not found in Finnomena catalog or API, check if master fund catalog matches
        master_entry = find_master_fund(resolved_sym) or find_master_fund(clean_sym)

        if not fund_id and not master_entry:
            log.debug("No fund ID or master fund found for '%s' (resolved '%s')", clean_sym, resolved_sym)
            if cached_entry:
                return FundConstituentsData.model_validate(cached_entry["data"])
            return None

        data_payload: Optional[dict] = None
        if fund_id:
            url = _PORTFOLIO_URL_TEMPLATE.format(fund_id=fund_id)
            try:
                log.info("Fetching fund portfolio for %s (id=%s) from %s", resolved_sym, fund_id, url)
                resp = self._session.get(url, timeout=self._timeout)
                resp.raise_for_status()
                res_json = resp.json()
                if isinstance(res_json, dict):
                    data_payload = res_json.get("data") if "data" in res_json else res_json
            except Exception as e:
                log.warning("Failed to fetch fund portfolio for %s (id=%s): %s", resolved_sym, fund_id, e)
                if cached_entry:
                    log.info("Returning stale cached constituents for %s", clean_sym)
                    return FundConstituentsData.model_validate(cached_entry["data"])

        as_of_date = ""
        top_holdings: List[FundHoldingItem] = []
        sector_breakdown: List[FundSectorItem] = []
        asset_allocation: Dict[str, Decimal] = {}
        master_fund_name: Optional[str] = master_entry.master_name if master_entry else None
        master_fund_ticker: Optional[str] = master_entry.master_ticker if master_entry else None

        if data_payload and isinstance(data_payload, dict):
            h_date, raw_holdings = _extract_elements(data_payload.get("top_holdings"))
            s_date, raw_sector = _extract_elements(data_payload.get("sector") or data_payload.get("global_stock_sector"))
            a_date, alloc_elems = _extract_elements(data_payload.get("asset_allocation"))

            as_of_date = str(data_payload.get("as_of_date") or h_date or s_date or a_date or "")[:10]

            # Parse top holdings with 2-Tier Resolution
            for item in raw_holdings:
                h_name = str(item.get("name") or "").strip()
                try:
                    h_pct = Decimal(str(item.get("percent") or 0))
                except Exception:
                    h_pct = Decimal("0")
                h_short = str(item.get("short_code") or "").strip()
                h_type = str(item.get("asset_type") or "EQUITY").upper()

                # Check if this holding is a master fund
                m_sub = find_master_fund(h_name) or (find_master_fund(h_short) if h_short else None)
                if m_sub and h_pct > Decimal("40.0"):
                    master_fund_name = m_sub.master_name
                    master_fund_ticker = m_sub.master_ticker
                    feeder_ratio = h_pct / Decimal("100.0")
                    for m_ticker, m_cname, m_weight in m_sub.top_holdings:
                        effective_pct = (m_weight * feeder_ratio).quantize(Decimal("0.001"))
                        top_holdings.append(
                            FundHoldingItem(
                                name=m_cname,
                                percent=effective_pct,
                                ticker=m_ticker,
                                asset_type="EQUITY",
                            )
                        )
                    if not raw_sector and m_sub.sector_breakdown:
                        for s_en, s_weight in m_sub.sector_breakdown:
                            eff_sec_pct = (s_weight * feeder_ratio).quantize(Decimal("0.001"))
                            sec_en, sec_th = normalize_sector_name(s_en)
                            sector_breakdown.append(
                                FundSectorItem(
                                    sector_name_th=sec_th,
                                    sector_name_en=sec_en,
                                    percent=eff_sec_pct,
                                )
                            )
                else:
                    resolved_ticker = h_short if h_short and not re.search(r"[\u0E00-\u0E7F]", h_short) else None
                    if not resolved_ticker:
                        resolved_ticker = normalize_set_company_name(h_name)
                    top_holdings.append(
                        FundHoldingItem(
                            name=h_name,
                            percent=h_pct,
                            ticker=resolved_ticker,
                            asset_type=h_type,
                        )
                    )

            if not sector_breakdown and raw_sector:
                for sec in raw_sector:
                    sec_raw_name = str(sec.get("name") or sec.get("sector_name") or "").strip()
                    try:
                        sec_pct = Decimal(str(sec.get("percent") or 0))
                    except Exception:
                        sec_pct = Decimal("0")
                    sec_en, sec_th = normalize_sector_name(sec_raw_name)
                    sector_breakdown.append(
                        FundSectorItem(
                            sector_name_th=sec_th,
                            sector_name_en=sec_en,
                            percent=sec_pct,
                        )
                    )

            if alloc_elems:
                for item in alloc_elems:
                    k = str(item.get("name") or item.get("asset") or "").strip()
                    try:
                        asset_allocation[k] = Decimal(str(item.get("percent") or 0))
                    except Exception:
                        pass
            elif isinstance(data_payload.get("asset_allocation"), dict):
                for k, v in data_payload["asset_allocation"].items():
                    if k not in ("data_date", "elements"):
                        try:
                            asset_allocation[str(k)] = Decimal(str(v))
                        except Exception:
                            pass
            elif isinstance(data_payload.get("asset_allocation"), list):
                for item in data_payload["asset_allocation"]:
                    if isinstance(item, dict):
                        k = str(item.get("name") or item.get("asset") or "").strip()
                        try:
                            asset_allocation[k] = Decimal(str(item.get("percent") or 0))
                        except Exception:
                            pass

        # If top_holdings is still empty, and we have master_entry fallback:
        if not top_holdings and master_entry:
            master_fund_name = master_entry.master_name
            master_fund_ticker = master_entry.master_ticker
            for m_ticker, m_cname, m_weight in master_entry.top_holdings:
                top_holdings.append(
                    FundHoldingItem(
                        name=m_cname,
                        percent=m_weight,
                        ticker=m_ticker,
                        asset_type="EQUITY",
                    )
                )
            if not sector_breakdown:
                for s_en, s_weight in master_entry.sector_breakdown:
                    sec_en, sec_th = normalize_sector_name(s_en)
                    sector_breakdown.append(
                        FundSectorItem(
                            sector_name_th=sec_th,
                            sector_name_en=sec_en,
                            percent=s_weight,
                        )
                    )

        is_curated = False
        if not as_of_date:
            if master_entry and getattr(master_entry, "as_of_date", None):
                as_of_date = master_entry.as_of_date
                is_curated = getattr(master_entry, "is_curated", True)
            else:
                as_of_date = "UNKNOWN"

        result = FundConstituentsData(
            fund_id=fund_id or "MASTER_FALLBACK",
            symbol=clean_sym,
            as_of_date=as_of_date,
            master_fund_name=master_fund_name,
            master_fund_ticker=master_fund_ticker,
            top_holdings=top_holdings,
            sector_breakdown=sector_breakdown,
            asset_allocation=asset_allocation,
            is_curated=is_curated,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )

        with self._lock:
            self._constituents_cache[cache_key] = {
                "cached_at": now,
                "data": json.loads(result.model_dump_json()),
            }
            self._save_constituents_cache()

        return result
