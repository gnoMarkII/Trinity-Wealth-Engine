"""SEC Thailand Keyless Adapter for Mutual Fund Asset Allocations & Bond Statistics.

Parses official open data CSVs from SEC Thailand (MF_PORT_TH.csv, STAT_DEPT_TH.csv, OFFER_DEBT_COR_TH.csv).
Handles F5 WAF rejection responses, UTF-8 BOM, Buddhist Era offsets, and million THB scaling.
Strict Rule: Reports broad asset classes only; does NOT report equity sectors (Bank/Energy/Tech).
"""
import csv
import io
import logging
import time
from typing import Dict, List, Optional, Tuple
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import (
    ThaiBondMarketStats,
    ThaiCorporateBondIssuance,
    ThaiFundAssetAllocationRow,
    ThaiFundAssetAllocationSnapshot,
)
from tools.market.terminal_v2.ports.driven_ports import (
    ThaiBondMarketPort,
    ThaiFundAllocationPort,
)

logger = logging.getLogger(__name__)

SEC_TH_TTL_SECONDS = 43200.0           # 12 hours
SEC_TH_MAX_STALE_SECONDS = 7 * 86400.0  # 7 days ceiling
MILLIONS = 1e6
BE_OFFSET = 543

MF_PORT_URL = "https://dividend.sec.or.th/stat-report/MF_PORT_TH.csv"
STAT_DEPT_URL = "https://dividend.sec.or.th/stat-report/STAT_DEPT_TH.csv"
OFFER_DEBT_URL = "https://dividend.sec.or.th/stat-report/OFFER_DEBT_COR_TH.csv"

# WAF headers without Origin
SEC_HEADERS = {k: v for k, v in BROWSER_HEADERS.items() if k.lower() != "origin"}

ASSET_CLASS_TRANSLATIONS = {
    "หุ้นสามัญ": "Common stock",
    "หุ้นบุริมสิทธิ์": "Preferred stock",
    "หน่วยลงทุน": "Investment units / mutual funds",
    "ใบแสดงสิทธิ/ใบสำคัญแสดงสิทธิ": "Warrants & rights",
    "หุ้นกู้/ตั๋วแลกเงิน/ตั๋วสัญญาใช้เงิน": "Corporate debt & notes",
    "ตั๋วเงินคลัง/พันธบัตร": "Government bonds & treasury bills",
    "เงินฝาก/บัตรเงินฝาก/หนังสือยืนยันการรับฝากเงิน": "Bank deposits & certificates",
    "กองทรัสต์": "REITs & property funds",
    "สัญญาซื้อขายล่วงหน้า": "Derivatives",
    "Euro Commercial Paper/Euro Medium Term Note": "Euro notes & commercial paper",
    "ศุกูก": "Sukuk",
}


def _strip_bom(text: str) -> str:
    return text.lstrip("\ufeff")


def _check_waf_rejection(text: str, filename: str) -> None:
    if "<title>Request Rejected</title>" in text or "<p>Your support ID is:" in text:
        raise ProviderError(
            f"SEC Thailand upstream WAF rejected the request for {filename}",
            source="SEC Thailand",
            status_code=403,
        )


class SecThailandAdapter(ThaiFundAllocationPort, ThaiBondMarketPort):
    """Adapter reading regulatory statistical reports from SEC Thailand."""

    def __init__(self, cache: Optional[ThreadSafeTTLCache] = None):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=SEC_TH_TTL_SECONDS)

    def _fetch_csv(self, url: str, name: str) -> str:
        try:
            resp = requests.get(url, headers=SEC_HEADERS, timeout=25)
            _check_waf_rejection(resp.text, name)
            resp.raise_for_status()
            return _strip_bom(resp.text)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Failed to fetch {name} from SEC Thailand: {exc}", source="SEC Thailand") from exc

    def _parse_fund_allocation(self) -> ThaiFundAssetAllocationSnapshot:
        text = self._fetch_csv(MF_PORT_URL, "MF_PORT_TH.csv")
        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader if row]
        if len(rows) < 2:
            raise ProviderError("Empty or invalid MF_PORT_TH.csv from SEC Thailand", source="SEC Thailand")

        allocations: List[ThaiFundAssetAllocationRow] = []
        nav_thb: Optional[float] = None
        period_label = ""

        # Find NAV row and build allocations
        for cells in rows[1:]:
            if len(cells) < 6:
                continue
            as_of_date, group_type, desc, year, quarter, val_str = [c.strip() for c in cells[:6]]
            try:
                val_million = float(val_str)
            except ValueError:
                continue

            val_thb = val_million * MILLIONS
            if not period_label:
                try:
                    be_year = int(year)
                    ce_year = be_year - BE_OFFSET
                    period_label = f"{ce_year} {quarter}"
                except ValueError:
                    period_label = f"{year} {quarter}"

            # Check for Net Asset Value row
            if "มูลค่าทรัพย์สินสุทธิ (หลังหักมูลค่าการลงทุนในกองทุนภายใต้ บลจ.เดียวกัน)" in group_type:
                nav_thb = val_thb
                continue

            # Classify Domestic vs Foreign
            if "ต่างประเทศ" in group_type:
                dom_foreign = "Foreign"
            elif "ในประเทศ" in group_type:
                dom_foreign = "Domestic"
            else:
                dom_foreign = "Other"

            asset_name = ASSET_CLASS_TRANSLATIONS.get(desc, desc)
            if asset_name and asset_name != "-":
                allocations.append(
                    ThaiFundAssetAllocationRow(
                        asset_class=asset_name,
                        domestic_or_foreign=dom_foreign,
                        value_thb=val_thb,
                        share_of_nav_pct=None,
                    )
                )

        # Compute share of NAV pct if NAV is available
        enriched_allocations = []
        for row in allocations:
            share_pct = None
            if nav_thb is not None and nav_thb > 0:
                share_pct = round((row.value_thb / nav_thb) * 100.0, 2)
            enriched_allocations.append(
                ThaiFundAssetAllocationRow(
                    asset_class=row.asset_class,
                    domestic_or_foreign=row.domestic_or_foreign,
                    value_thb=row.value_thb,
                    share_of_nav_pct=share_pct,
                )
            )

        return ThaiFundAssetAllocationSnapshot(
            reporting_period=period_label,
            total_nav_thb=nav_thb,
            allocations=tuple(enriched_allocations),
            fetched_at=time.time(),
            source="SEC Thailand",
        )

    def get_fund_asset_allocation(self) -> ThaiFundAssetAllocationSnapshot:
        cache_key = "sec_th:mf_port:latest"
        return self._cache.get_or_set(
            key=cache_key,
            loader=self._parse_fund_allocation,
            ttl_seconds=SEC_TH_TTL_SECONDS,
            max_stale_seconds=SEC_TH_MAX_STALE_SECONDS,
        )

    def _parse_bond_stats(self) -> ThaiBondMarketStats:
        text = self._fetch_csv(STAT_DEPT_URL, "STAT_DEPT_TH.csv")
        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader if row]
        if len(rows) < 2:
            raise ProviderError("Empty or invalid STAT_DEPT_TH.csv from SEC Thailand", source="SEC Thailand")

        latest_year = 0
        latest_as_of = ""
        outstanding_thb = 0.0
        trading_val_thb = 0.0
        foreign_holding_pct = None

        for cells in rows[1:]:
            if len(cells) < 5:
                continue
            as_of, item, type_col, year_col, val_col = [c.strip() for c in cells[:5]]
            try:
                be_yr = int(year_col)
                ce_yr = be_yr - BE_OFFSET
                val = float(val_col)
            except ValueError:
                continue

            if ce_yr >= latest_year:
                latest_year = ce_yr
                latest_as_of = as_of

            if "มูลค่าหลักทรัพย์ขึ้นทะเบียนคงค้าง" in item and type_col == "ยอดรวม":
                if ce_yr == latest_year:
                    outstanding_thb = val * MILLIONS
            elif "มูลค่าซื้อขาย" in item and type_col == "ยอดรวม" and "เฉลี่ย" not in item and "สัดส่วน" not in item:
                if ce_yr == latest_year:
                    trading_val_thb = val * MILLIONS
            elif "สัดส่วนมูลค่าซื้อขาย" in item and "ต่างประเทศ" in type_col:
                if ce_yr == latest_year:
                    foreign_holding_pct = val

        foreign_holding_thb = (trading_val_thb * (foreign_holding_pct / 100.0)) if foreign_holding_pct else 0.0

        return ThaiBondMarketStats(
            reporting_period=f"{latest_year} (as of {latest_as_of})",
            outstanding_thb=outstanding_thb,
            trading_value_thb=trading_val_thb,
            foreign_holding_thb=foreign_holding_thb,
            foreign_holding_pct=foreign_holding_pct,
            fetched_at=time.time(),
            source="SEC Thailand",
        )

    def get_bond_market_stats(self) -> ThaiBondMarketStats:
        cache_key = "sec_th:bond_stats:latest"
        return self._cache.get_or_set(
            key=cache_key,
            loader=self._parse_bond_stats,
            ttl_seconds=SEC_TH_TTL_SECONDS,
            max_stale_seconds=SEC_TH_MAX_STALE_SECONDS,
        )

    def _parse_corporate_bond_issuance(self) -> ThaiCorporateBondIssuance:
        text = self._fetch_csv(OFFER_DEBT_URL, "OFFER_DEBT_COR_TH.csv")
        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader if row]
        if len(rows) < 2:
            raise ProviderError("Empty or invalid OFFER_DEBT_COR_TH.csv from SEC Thailand", source="SEC Thailand")

        latest_year = 0
        total_offering = 0.0
        long_term = 0.0
        short_term = 0.0

        for cells in rows[1:]:
            if len(cells) < 8:
                continue
            as_of, b_type, country, sec_id, instrument, year_col, q_col, val_col = [c.strip() for c in cells[:8]]
            try:
                yr = int(year_col)  # This file uses CE year already
                val = float(val_col) * MILLIONS
            except ValueError:
                continue

            if yr >= latest_year:
                latest_year = yr

            if yr == latest_year:
                total_offering += val
                if "ระยะสั้น" in instrument or "ตั๋วเงิน" in instrument:
                    short_term += val
                elif "ระยะยาว" in instrument:
                    long_term += val

        return ThaiCorporateBondIssuance(
            reporting_period=f"{latest_year}",
            total_offering_thb=total_offering,
            long_term_thb=long_term,
            short_term_thb=short_term,
            top_sectors=(),
            fetched_at=time.time(),
            source="SEC Thailand",
        )

    def get_corporate_bond_issuance(self) -> ThaiCorporateBondIssuance:
        cache_key = "sec_th:bond_issuance:latest"
        return self._cache.get_or_set(
            key=cache_key,
            loader=self._parse_corporate_bond_issuance,
            ttl_seconds=SEC_TH_TTL_SECONDS,
            max_stale_seconds=SEC_TH_MAX_STALE_SECONDS,
        )
