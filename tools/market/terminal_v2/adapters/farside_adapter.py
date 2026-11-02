"""Farside Investors Adapter for US Spot Bitcoin & Ethereum ETF Flows.

Scrapes daily ETF net inflow history from farside.co.uk.
Acts as robust secondary fallback when SoSoValue encounters Cloudflare 403 or network failure.
"""
from datetime import datetime, timezone
from pathlib import Path
import re
import time
from typing import List, Optional
from bs4 import BeautifulSoup
import requests

from core.logger import get_logger
from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import ProviderError
from tools.market.terminal_v2.domain.models import (
    SpotEtfFlowSnapshot,
    SpotEtfIssuerFlow,
)
from tools.market.terminal_v2.ports.driven_ports import SpotEtfFlowsPort

logger = get_logger(__name__)

URL_MAP = {
    "BTC": "https://farside.co.uk/bitcoin-etf-flow/",
    "ETH": "https://farside.co.uk/ethereum-etf-flow/",
}
FARSIDE_TTL_SECONDS = 21600.0  # 6 hours
FARSIDE_MAX_STALE_SECONDS = 2 * 86400.0


def _parse_farside_number(txt: Optional[str]) -> Optional[float]:
    """Parse Farside cell text into float in millions of USD.

    Examples:
        '136.0' -> 136.0
        '(44.8)' -> -44.8
        '12,017' -> 12017.0
        '9,199*' -> 9199.0
        '-' / '' / 'N/A' -> None
    """
    if not txt:
        return None
    s = str(txt).strip().replace(",", "").replace("*", "")
    if s in ("", "-", "—", "N/A", "null"):
        return None
    is_neg = False
    if s.startswith("(") and s.endswith(")"):
        is_neg = True
        s = s[1:-1].strip()
    try:
        val = float(s)
        return -val if is_neg else val
    except ValueError:
        return None


class FarsideEtfFlowsAdapter(SpotEtfFlowsPort):
    """Adapter scraping daily US Spot ETF flows from Farside Investors."""

    def __init__(
        self,
        cache: Optional[ThreadSafeTTLCache] = None,
        btc_fixture_path: Optional[Path] = None,
        eth_fixture_path: Optional[Path] = None,
        timeout: int = 15,
    ):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=FARSIDE_TTL_SECONDS)
        self._btc_fixture = btc_fixture_path
        self._eth_fixture = eth_fixture_path
        self._timeout = timeout

    def _fetch_html(self, asset: str) -> str:
        fixture = self._btc_fixture if asset == "BTC" else self._eth_fixture
        if fixture and fixture.exists():
            return fixture.read_text(encoding="utf-8")

        url = URL_MAP.get(asset)
        if not url:
            raise ValueError(f"Unsupported asset '{asset}'. Supported: BTC, ETH")

        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=self._timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            raise ProviderError(f"Failed to fetch {asset} ETF flows from Farside: {exc}", source="Farside") from exc

    def _parse_html(self, html: str, asset: str) -> SpotEtfFlowSnapshot:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            raise ProviderError(f"No table found in Farside {asset} HTML", source="Farside")

        rows = table.find_all("tr")
        if not rows:
            raise ProviderError(f"Empty table in Farside {asset} HTML", source="Farside")

        # 1. Identify Ticker Headers
        ticker_headers: List[str] = []
        for r in rows[:6]:
            cells = [c.get_text(strip=True) for c in r.find_all(["th", "td"])]
            valid_tickers = [c for c in cells if re.fullmatch(r"[A-Z]{3,5}", c)]
            if len(valid_tickers) >= 3:
                ticker_headers = cells
                break

        # 2. Extract Data Rows and Summary Total
        date_rows: List[List[str]] = []
        cum_total_m: Optional[float] = None

        for r in rows:
            cells = [c.get_text(strip=True) for c in r.find_all(["th", "td"])]
            if not cells:
                continue
            first_col = cells[0].strip()
            if first_col.lower() == "total" and len(cells) > 1:
                cum_total_m = _parse_farside_number(cells[-1])
            elif re.fullmatch(r"\d{1,2}\s+[A-Za-z]{3}\s+\d{4}", first_col):
                date_rows.append(cells)

        if not date_rows:
            raise ProviderError(f"No date rows found in Farside {asset} table", source="Farside")

        last_row = date_rows[-1]
        raw_date_str = last_row[0]
        try:
            iso_date = datetime.strptime(raw_date_str, "%d %b %Y").strftime("%Y-%m-%d")
        except Exception:
            iso_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        daily_total_m = _parse_farside_number(last_row[-1])
        daily_total_usd = (daily_total_m * 1_000_000.0) if daily_total_m is not None else None
        cum_total_usd = (cum_total_m * 1_000_000.0) if cum_total_m is not None else None

        # 3. Build Issuer Flows
        issuers: List[SpotEtfIssuerFlow] = []
        if ticker_headers and len(ticker_headers) == len(last_row):
            for i, ticker in enumerate(ticker_headers):
                t_clean = ticker.strip().upper()
                if not t_clean or t_clean in ("DATE", "TOTAL", "FEE", "SEED"):
                    continue
                flow_m = _parse_farside_number(last_row[i])
                flow_usd = (flow_m * 1_000_000.0) if flow_m is not None else None
                issuers.append(
                    SpotEtfIssuerFlow(
                        ticker=t_clean,
                        institute="",
                        daily_net_inflow_usd=flow_usd,
                        cumulative_net_inflow_usd=None,
                        total_net_assets_usd=None,
                    )
                )

        return SpotEtfFlowSnapshot(
            asset=asset,
            report_date=iso_date,
            daily_total_usd=daily_total_usd,
            cumulative_total_usd=cum_total_usd,
            issuers=tuple(issuers),
            is_partial=(daily_total_usd is None),
            completeness_notes=f"Parsed from Farside Investors; {len(issuers)} issuers." if issuers else "Farside table",
            fetched_at=time.time(),
            source="Farside",
        )

    def get_spot_etf_flows(self, asset: str) -> SpotEtfFlowSnapshot:
        norm = asset.strip().upper()
        if norm not in ("BTC", "ETH"):
            raise ValueError(f"Unsupported ETF asset '{asset}'. Supported assets are BTC and ETH.")
        cache_key = f"farside:etf:{norm}"
        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._parse_html(self._fetch_html(norm), norm),
            ttl_seconds=FARSIDE_TTL_SECONDS,
            max_stale_seconds=FARSIDE_MAX_STALE_SECONDS,
        )

