"""US Treasury Keyless Adapter for Yield Curve, Auctions, and Debt to the Penny.

Parses official Treasury.gov XML feed for par yield curves and Fiscal Data API
for completed auctions and daily close national debt accounting.
Strict Rule: Missing tenors are None, not 0.0. Debt to Penny is daily, not real-time.
"""
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.calculations import calculate_rate_spread_bps
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import (
    TreasuryAuctionResult,
    TreasuryYieldCurveSnapshot,
    TreasuryYieldPoint,
    UsNationalDebtSnapshot,
)
from tools.market.terminal_v2.ports.driven_ports import AuctionHistoryPort, TreasuryDataPort

logger = logging.getLogger(__name__)

TREASURY_TTL_SECONDS = 10800.0         # 3 hours
TREASURY_MAX_STALE_SECONDS = 4 * 86400.0  # 4 days ceiling

YIELD_CURVE_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?"
    "data=daily_treasury_yield_curve&field_tdr_date_value_month={yyyymm}"
)
AUCTIONS_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?"
    "sort=-auction_date&filter=bid_to_cover_ratio:gt:0&page[size]={size}"
)
DEBT_TO_PENNY_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny?"
    "sort=-record_date&page[size]={size}"
)

# XML Namespaces
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
    "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
}

TENOR_MAP = [
    ("1 Mo", "BC_1MONTH"),
    ("2 Mo", "BC_2MONTH"),
    ("3 Mo", "BC_3MONTH"),
    ("4 Mo", "BC_4MONTH"),
    ("6 Mo", "BC_6MONTH"),
    ("1 Yr", "BC_1YEAR"),
    ("2 Yr", "BC_2YEAR"),
    ("3 Yr", "BC_3YEAR"),
    ("5 Yr", "BC_5YEAR"),
    ("7 Yr", "BC_7YEAR"),
    ("10 Yr", "BC_10YEAR"),
    ("20 Yr", "BC_20YEAR"),
    ("30 Yr", "BC_30YEAR"),
]


class TreasuryAdapter(TreasuryDataPort, AuctionHistoryPort):
    """Adapter reading public US Treasury feeds."""

    def __init__(
        self,
        cache: Optional[ThreadSafeTTLCache] = None,
        fixture_dir: Optional[Path] = None,
    ):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=TREASURY_TTL_SECONDS)
        self._fixture_dir = fixture_dir

    def _fetch_yield_curve(self, month_yyyymm: Optional[str] = None) -> TreasuryYieldCurveSnapshot:
        now = datetime.utcnow()
        months_to_try = []
        if month_yyyymm:
            months_to_try.append(month_yyyymm)
        else:
            months_to_try.append(now.strftime("%Y%m"))
            # Fallback to previous month if early in current month
            prev_month = (now.replace(day=1) - timedelta(days=1)).strftime("%Y%m")
            months_to_try.append(prev_month)

        xml_text = None
        for m in months_to_try:
            url = YIELD_CURVE_URL.format(yyyymm=m)
            try:
                resp = requests.get(url, headers=BROWSER_HEADERS, timeout=15)
                if resp.status_code == 200 and "<entry>" in resp.text:
                    xml_text = resp.text
                    break
            except Exception as exc:
                logger.debug("Failed to fetch Treasury yield curve for %s: %s", m, exc)
                continue

        if not xml_text:
            raise DataUnavailableError("US Treasury yield curve unavailable", capability="yield-curve", source="US Treasury")

        try:
            root = ET.fromstring(xml_text)
        except Exception as exc:
            raise ProviderError(f"Malformed XML from Treasury yield curve: {exc}", source="US Treasury") from exc

        entries = root.findall("atom:entry", NS)
        if not entries:
            raise DataUnavailableError("No entries found in US Treasury yield curve XML", capability="yield-curve", source="US Treasury")

        latest_entry = entries[-1]
        props = latest_entry.find(".//m:properties", NS)
        if props is None:
            raise ProviderError("Missing m:properties in Treasury yield curve entry", source="US Treasury")

        date_elem = props.find("d:NEW_DATE", NS)
        raw_date = date_elem.text if date_elem is not None and date_elem.text else ""
        obs_date = raw_date[:10] if len(raw_date) >= 10 else raw_date

        yield_points: List[TreasuryYieldPoint] = []
        yields_by_tenor: Dict[str, float] = {}

        for label, tag in TENOR_MAP:
            elem = props.find(f"d:{tag}", NS)
            val: Optional[float] = None
            if elem is not None and elem.text and elem.text.strip():
                try:
                    val = float(elem.text.strip())
                    yields_by_tenor[label] = val
                except ValueError:
                    val = None
            yield_points.append(TreasuryYieldPoint(maturity=label, yield_percent=val))

        spread_10y_2y = None
        if "10 Yr" in yields_by_tenor and "2 Yr" in yields_by_tenor:
            spread_10y_2y = calculate_rate_spread_bps(yields_by_tenor["10 Yr"], yields_by_tenor["2 Yr"])

        spread_10y_3m = None
        if "10 Yr" in yields_by_tenor and "3 Mo" in yields_by_tenor:
            spread_10y_3m = calculate_rate_spread_bps(yields_by_tenor["10 Yr"], yields_by_tenor["3 Mo"])

        return TreasuryYieldCurveSnapshot(
            observation_date=obs_date,
            yields=tuple(yield_points),
            spread_10y_2y_bps=spread_10y_2y,
            spread_10y_3m_bps=spread_10y_3m,
            fetched_at=time.time(),
            source="US Treasury",
        )

    def get_yield_curve(self, month_yyyymm: Optional[str] = None) -> TreasuryYieldCurveSnapshot:
        cache_key = f"treasury:yield_curve:{month_yyyymm or 'latest'}"
        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._fetch_yield_curve(month_yyyymm),
            ttl_seconds=TREASURY_TTL_SECONDS,
            max_stale_seconds=TREASURY_MAX_STALE_SECONDS,
        )

    def _fetch_auctions(self, limit: int) -> Tuple[TreasuryAuctionResult, ...]:
        url = AUCTIONS_URL.format(size=limit)
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise ProviderError(f"Failed to fetch Treasury auctions: {exc}", source="US Treasury Fiscal Data") from exc

        rows = data.get("data", [])
        return self._parse_auction_rows(rows)

    def _parse_auction_rows(self, rows: List[dict]) -> Tuple[TreasuryAuctionResult, ...]:
        results: List[TreasuryAuctionResult] = []
        now_epoch = time.time()

        def _flt(v: Optional[object]) -> Optional[float]:
            if v is None:
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        for row in rows:
            results.append(
                TreasuryAuctionResult(
                    auction_date=row.get("auction_date", ""),
                    issue_date=row.get("issue_date", ""),
                    security_type=row.get("security_type", ""),
                    security_term=row.get("security_term", ""),
                    high_yield=_flt(row.get("high_yield")),
                    high_investment_rate=_flt(row.get("high_investment_rate")),
                    high_discount_rate=_flt(row.get("high_discnt_rate")),
                    bid_to_cover_ratio=_flt(row.get("bid_to_cover_ratio")),
                    offering_amount_usd=_flt(row.get("offering_amt")),
                    total_accepted_usd=_flt(row.get("total_accepted")),
                    fetched_at=now_epoch,
                    source="US Treasury Fiscal Data",
                )
            )

        return tuple(results)

    def get_auctions(self, limit: int = 10) -> Tuple[TreasuryAuctionResult, ...]:
        capped = min(max(limit, 1), 50)
        cache_key = f"treasury:auctions:{capped}"
        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._fetch_auctions(capped),
            ttl_seconds=TREASURY_TTL_SECONDS,
            max_stale_seconds=TREASURY_MAX_STALE_SECONDS,
        )

    def _fetch_auction_history(self, security_type: str, security_term: str, limit: int) -> Tuple[TreasuryAuctionResult, ...]:
        if self._fixture_dir:
            fname = f"treasury_auctions_{security_type.lower()}_{security_term.lower().replace('-', '_').replace(' ', '_')}_fixture.json"
            fpath = self._fixture_dir / fname
            if fpath.exists():
                import json
                data = json.loads(fpath.read_text(encoding="utf-8"))
                return self._parse_auction_rows(data.get("data", []))

        url = (
            "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?"
            f"filter=bid_to_cover_ratio:gt:0,security_type:eq:{security_type},security_term:eq:{security_term}&sort=-auction_date&page[size]={limit}"
        )
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise ProviderError(f"Failed to fetch Treasury auction history for {security_type} {security_term}: {exc}", source="US Treasury Fiscal Data") from exc

        return self._parse_auction_rows(data.get("data", []))

    def fetch_completed_auction_history(
        self,
        security_type: str,
        security_term: str,
        limit: int = 15,
    ) -> Tuple[TreasuryAuctionResult, ...]:
        capped = min(max(limit, 1), 50)
        cache_key = f"treasury:auction_history:{security_type}:{security_term}:{capped}"
        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._fetch_auction_history(security_type, security_term, capped),
            ttl_seconds=TREASURY_TTL_SECONDS,
            max_stale_seconds=TREASURY_MAX_STALE_SECONDS,
        )

    def _fetch_debt(self, limit: int) -> Tuple[UsNationalDebtSnapshot, ...]:
        url = DEBT_TO_PENNY_URL.format(size=limit)
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise ProviderError(f"Failed to fetch Debt to the Penny: {exc}", source="US Treasury Fiscal Data") from exc

        rows = data.get("data", [])
        results: List[UsNationalDebtSnapshot] = []
        now_epoch = time.time()

        def _flt(v: Optional[object]) -> Optional[float]:
            if v is None:
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        for row in rows:
            tot = _flt(row.get("tot_pub_debt_out_amt"))
            if tot is None:
                continue

            results.append(
                UsNationalDebtSnapshot(
                    record_date=row.get("record_date", ""),
                    total_public_debt_usd=tot,
                    debt_held_by_public_usd=_flt(row.get("debt_held_public_amt")),
                    intragovernmental_holdings_usd=_flt(row.get("intragov_hold_amt")),
                    is_daily_close=True,
                    fetched_at=now_epoch,
                    source="US Treasury Fiscal Data",
                )
            )

        return tuple(results)

    def get_national_debt(self, limit: int = 5) -> Tuple[UsNationalDebtSnapshot, ...]:
        capped = min(max(limit, 1), 30)
        cache_key = f"treasury:debt:{capped}"
        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._fetch_debt(capped),
            ttl_seconds=TREASURY_TTL_SECONDS,
            max_stale_seconds=TREASURY_MAX_STALE_SECONDS,
        )
