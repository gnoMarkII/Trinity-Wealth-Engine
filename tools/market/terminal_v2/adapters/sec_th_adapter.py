"""SEC Thailand Keyless Adapter for Mutual Fund Asset Allocations & Bond Statistics.

Parses official open data CSVs from SEC Thailand (MF_PORT_TH.csv, STAT_DEPT_TH.csv, OFFER_DEBT_COR_TH.csv).
Handles F5 WAF rejection responses, UTF-8 BOM, Buddhist Era offsets, and million THB scaling.
Strict Rule: Reports broad asset classes only; does NOT report equity sectors (Bank/Energy/Tech).
"""
import csv
import io
import logging
import time
import re
from typing import Dict, List, Optional, Tuple, Set
import requests

from schemas.macro_schemas import MarketObservable
from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import (
    ThaiBondMarketStats,
    ThaiCorporateBondIssuance,
    ThaiFundAssetAllocationRow,
    ThaiFundAssetAllocationSnapshot,
    ThaiIndustryGroupItem,
    ThaiIndustryMarketCapSnapshot,
    ThaiSectorMarketCapItem,
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
STAT_INDUSTRY_URL = "https://dividend.sec.or.th/stat-report/STAT_INDUSTRY_TH.csv"

# WAF headers without Origin or Accept-Language to prevent F5 WAF rejection
SEC_HEADERS = {
    k: v for k, v in BROWSER_HEADERS.items()
    if k.lower() not in ("origin", "accept-language")
}

# Standard SET Sector Hierarchy: 28 sectors -> 8 industry groups
SECTOR_GROUP: Dict[str, str] = {
    "AGRI": "AGRO",
    "FOOD": "AGRO",
    "FASHION": "CONSUMP",
    "HOME": "CONSUMP",
    "PERSON": "CONSUMP",
    "BANK": "FINCIAL",
    "FIN": "FINCIAL",
    "INSUR": "FINCIAL",
    "AUTO": "INDUS",
    "IMM": "INDUS",
    "PAPER": "INDUS",
    "PETRO": "INDUS",
    "PKG": "INDUS",
    "STEEL": "INDUS",
    "CONMAT": "PROPCON",
    "CONS": "PROPCON",
    "PROP": "PROPCON",
    "PF&REIT": "PROPCON",
    "ENERG": "RESOURC",
    "MINE": "RESOURC",
    "COMM": "SERVICE",
    "HELTH": "SERVICE",
    "MEDIA": "SERVICE",
    "PROF": "SERVICE",
    "TOURISM": "SERVICE",
    "TRANS": "SERVICE",
    "ETRON": "TECH",
    "ICT": "TECH",
}
GROUP_CODES: Set[str] = set(SECTOR_GROUP.values())

GROUP_NAMES: Dict[str, Tuple[str, str]] = {
    "AGRO": ("Agro & Food Industry", "เกษตรและอุตสาหกรรมอาหาร"),
    "CONSUMP": ("Consumer Products", "สินค้าอุปโภคบริโภค"),
    "FINCIAL": ("Financials", "ธุรกิจการเงิน"),
    "INDUS": ("Industrials", "สินค้าอุตสาหกรรม"),
    "PROPCON": ("Property & Construction", "อสังหาริมทรัพย์และก่อสร้าง"),
    "RESOURC": ("Resources", "ทรัพยากร"),
    "SERVICE": ("Services", "บริการ"),
    "TECH": ("Technology", "เทคโนโลยี"),
}

SECTOR_NAMES: Dict[str, Tuple[str, str]] = {
    "AGRI": ("Agribusiness", "ธุรกิจการเกษตร"),
    "FOOD": ("Food & Beverage", "อาหารและเครื่องดื่ม"),
    "FASHION": ("Fashion", "แฟชั่น"),
    "HOME": ("Home & Office Products", "ของใช้ในครัวเรือนและสำนักงาน"),
    "PERSON": ("Personal Products & Pharmaceuticals", "ของใช้ส่วนตัวและเวชภัณฑ์"),
    "BANK": ("Banking", "ธนาคาร"),
    "FIN": ("Finance & Securities", "เงินทุนและหลักทรัพย์"),
    "INSUR": ("Insurance", "ประกันภัยและประกันชีวิต"),
    "AUTO": ("Automotive", "ยานยนต์"),
    "IMM": ("Industrial Materials & Machinery", "วัสดุอุตสาหกรรมและเครื่องจักร"),
    "PAPER": ("Paper & Printing Materials", "กระดาษและวัสดุการพิมพ์"),
    "PETRO": ("Petrochemicals & Chemicals", "ปิโตรเคมีและเคมีภัณฑ์"),
    "PKG": ("Packaging", "บรรจุภัณฑ์"),
    "STEEL": ("Steel", "เหล็กและผลิตภัณฑ์โลหะ"),
    "CONMAT": ("Construction Materials", "วัสดุก่อสร้าง"),
    "CONS": ("Construction Services", "บริการรับเหมาก่อสร้าง"),
    "PROP": ("Property Development", "พัฒนาอสังหาริมทรัพย์"),
    "PF&REIT": ("Property Fund & REITs", "กองทุนรวมอสังหาริมทรัพย์และกองทรัสต์"),
    "ENERG": ("Energy & Utilities", "พลังงานและสาธารณูปโภค"),
    "MINE": ("Mining", "เหมืองแร่"),
    "COMM": ("Commerce", "พาณิชย์"),
    "HELTH": ("Health Care Services", "การแพทย์"),
    "MEDIA": ("Media & Publishing", "สื่อและสิ่งพิมพ์"),
    "PROF": ("Professional Services", "บริการเฉพาะกิจ"),
    "TOURISM": ("Tourism & Leisure", "การท่องเที่ยวและสันทนาการ"),
    "TRANS": ("Transportation & Logistics", "ขนส่งและโลจิสติกส์"),
    "ETRON": ("Electronic Components", "ชิ้นส่วนอิเล็กทรอนิกส์"),
    "ICT": ("Information & Communication Technology", "เทคโนโลยีสารสนเทศและการสื่อสาร"),
}

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

    def _parse_industry_market_cap(self, market: str = "SET") -> ThaiIndustryMarketCapSnapshot:
        text = self._fetch_csv(STAT_INDUSTRY_URL, "STAT_INDUSTRY_TH.csv")
        return parse_industry_csv(text, market=market)

    def get_industry_market_cap(self, market: str = "SET") -> ThaiIndustryMarketCapSnapshot:
        cache_key = f"sec_th:industry_market_cap:{market.upper()}:latest"
        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._parse_industry_market_cap(market=market),
            ttl_seconds=SEC_TH_TTL_SECONDS,
            max_stale_seconds=SEC_TH_MAX_STALE_SECONDS,
        )

    def as_macro_observables(self, snapshot: Optional[ThaiIndustryMarketCapSnapshot] = None) -> list[MarketObservable]:
        if snapshot is None:
            try:
                snapshot = self.get_industry_market_cap(market="SET")
            except Exception as e:
                logger.warning("Could not fetch SEC industry market cap for observables: %s", e)
                return []

        obs_date = snapshot.as_of if re.match(r"\d{4}-\d{2}-\d{2}", snapshot.as_of) else time.strftime("%Y-%m-%d")
        observables: list[MarketObservable] = []
        for grp in snapshot.groups:
            observables.append(
                MarketObservable(
                    observable_id=f"obs_th_sector_mcap_sec_{grp.group_code.lower()}",
                    asset_bucket="equities",
                    region="Thailand",
                    indicator=f"SET Industry Cap: {grp.group_name_en} ({grp.group_code})",
                    value=f"{grp.market_cap_thb / MILLIONS:,.1f}",
                    unit="Million THB",
                    observed_at=obs_date,
                    source_file="SEC_STAT_INDUSTRY_TH_CSV",
                    provider="SEC Thailand",
                    confidence="high",
                    is_valid=True,
                    status="verified",
                    period=snapshot.reporting_period,
                    metadata={
                        "group_code": grp.group_code,
                        "market_cap_thb": grp.market_cap_thb,
                        "share_of_market_pct": grp.share_of_market_pct,
                        "sector_codes": list(grp.sector_codes),
                    },
                )
            )
        return observables


def parse_industry_csv(csv_text: str, market: str = "SET") -> ThaiIndustryMarketCapSnapshot:
    """Parse STAT_INDUSTRY_TH.csv into structured sectors and parent industry groups."""
    _check_waf_rejection(csv_text, "STAT_INDUSTRY_TH.csv")
    text = _strip_bom(csv_text)
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if r]
    if len(rows) < 2:
        raise ProviderError("Empty or invalid STAT_INDUSTRY_TH.csv from SEC Thailand", source="SEC Thailand")

    target_market = market.strip().upper()
    latest_year = 0
    latest_q = 0
    latest_as_of = ""

    parsed_entries: List[Dict[str, Any]] = []

    for cells in rows[1:]:
        if len(cells) < 6:
            continue
        as_of_raw, mkt_col, code_desc, yr_col, q_col, val_col = [c.strip() for c in cells[:6]]
        if mkt_col.upper() != target_market:
            continue

        # Extract code (first whitespace-delimited token)
        code = code_desc.split()[0].strip() if code_desc else ""
        if not code:
            continue

        # Parse year (BE -> CE)
        try:
            be_yr = int(yr_col)
            ce_yr = be_yr - BE_OFFSET
        except ValueError:
            continue

        # Parse quarter
        q_match = re.search(r"\d+", q_col)
        quarter = int(q_match.group(0)) if q_match else 0

        # Parse value: '-' or empty is missing (never convert to 0!)
        clean_val = val_col.replace(",", "").strip()
        if not clean_val or clean_val == "-":
            continue
        try:
            val_million = float(clean_val)
        except ValueError:
            continue

        if ce_yr > latest_year or (ce_yr == latest_year and quarter > latest_q):
            latest_year = ce_yr
            latest_q = quarter
            # Parse as_of date if available
            date_match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", as_of_raw)
            if date_match:
                d, m, y = date_match.groups()
                latest_as_of = f"{int(y) - BE_OFFSET:04d}-{int(m):02d}-{int(d):02d}"
            else:
                latest_as_of = as_of_raw

        parsed_entries.append({
            "code": code,
            "year": ce_yr,
            "quarter": quarter,
            "val_thb": val_million * MILLIONS,
        })

    if not parsed_entries or latest_year == 0:
        raise ProviderError(f"No market cap data found for market {market} in STAT_INDUSTRY_TH.csv", source="SEC Thailand")

    # Filter to latest period
    latest_entries = [e for e in parsed_entries if e["year"] == latest_year and e["quarter"] == latest_q]
    by_code: Dict[str, float] = {e["code"]: e["val_thb"] for e in latest_entries}

    # Build sectors (28 sectors)
    sectors: List[ThaiSectorMarketCapItem] = []
    group_sums: Dict[str, float] = {g: 0.0 for g in GROUP_CODES}
    group_sectors_map: Dict[str, List[str]] = {g: [] for g in GROUP_CODES}

    for sector_code, group_code in SECTOR_GROUP.items():
        val = by_code.get(sector_code, 0.0)
        group_sums[group_code] += val
        group_sectors_map[group_code].append(sector_code)
        en_name, th_name = SECTOR_NAMES.get(sector_code, (sector_code, sector_code))
        sectors.append(
            ThaiSectorMarketCapItem(
                sector_code=sector_code,
                sector_name_en=en_name,
                sector_name_th=th_name,
                group_code=group_code,
                market_cap_thb=val,
            )
        )

    # Build parent groups (8 industry groups)
    groups: List[ThaiIndustryGroupItem] = []
    total_mcap = sum(group_sums.values())

    for group_code in sorted(GROUP_CODES):
        direct_val = by_code.get(group_code)
        g_val = direct_val if (direct_val is not None and direct_val > 0) else group_sums[group_code]
        en_name, th_name = GROUP_NAMES.get(group_code, (group_code, group_code))
        share_pct = round((g_val / total_mcap) * 100.0, 2) if total_mcap > 0 else 0.0
        groups.append(
            ThaiIndustryGroupItem(
                group_code=group_code,
                group_name_en=en_name,
                group_name_th=th_name,
                market_cap_thb=g_val,
                share_of_market_pct=share_pct,
                sector_codes=tuple(group_sectors_map[group_code]),
            )
        )

    # Enrich sector share_of_market_pct
    enriched_sectors: List[ThaiSectorMarketCapItem] = []
    for s in sectors:
        s_share = round((s.market_cap_thb / total_mcap) * 100.0, 2) if total_mcap > 0 else 0.0
        enriched_sectors.append(
            ThaiSectorMarketCapItem(
                sector_code=s.sector_code,
                sector_name_en=s.sector_name_en,
                sector_name_th=s.sector_name_th,
                group_code=s.group_code,
                market_cap_thb=s.market_cap_thb,
                share_of_market_pct=s_share,
            )
        )

    period_str = f"{latest_year} Q{latest_q}"

    return ThaiIndustryMarketCapSnapshot(
        market=target_market,
        as_of=latest_as_of or period_str,
        reporting_period=period_str,
        total_market_cap_thb=total_mcap,
        groups=tuple(groups),
        sectors=tuple(enriched_sectors),
        fetched_at=time.time(),
        source="SEC Thailand",
    )
