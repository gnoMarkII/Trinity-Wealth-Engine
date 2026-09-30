"""FINRA Keyless Adapter for Consolidated Daily Short Sale Volume.

Fetches and parses FINRA consolidated TRF/ADF daily short sale volume files.
Caches parsed daily files to serve multiple ticker queries efficiently.
Strict Regulatory Rule: Reported short volume is NOT short interest.
"""
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, Optional, Sequence
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import FinraShortVolumeSnapshot
from tools.market.terminal_v2.ports.driven_ports import ShortVolumePort

logger = logging.getLogger(__name__)

FINRA_BASE_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"
FINRA_TTL_SECONDS = 3600.0        # 1 hour
FINRA_MAX_STALE_SECONDS = 7 * 86400.0  # 7 days ceiling
MAX_LOOKBACK_DAYS = 7


def _strip_ticker(symbol: str) -> str:
    """Normalize symbol: strip DEX prefixes like 'xyz:TSLA' -> 'TSLA'."""
    idx = symbol.find(":")
    clean = symbol[idx + 1 :] if idx != -1 else symbol
    return clean.strip().upper()


class FinraAdapter(ShortVolumePort):
    """Adapter reading FINRA's public daily Reg SHO short sale volume files."""

    def __init__(self, cache: Optional[ThreadSafeTTLCache] = None):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=FINRA_TTL_SECONDS)

    def _parse_report(self, text: str, fallback_date: str) -> Dict[str, FinraShortVolumeSnapshot]:
        """Parse pipe-delimited FINRA daily file."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            raise ProviderError("Empty or invalid FINRA short sale volume file", source="FINRA")

        by_symbol: Dict[str, FinraShortVolumeSnapshot] = {}
        now_epoch = time.time()
        file_iso_date = ""

        # Line 0 is header: Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
        for line in lines[1:]:
            parts = line.split("|")
            if len(parts) < 5:
                continue

            raw_date, raw_symbol, raw_short, raw_exempt, raw_total = parts[:5]
            symbol = raw_symbol.strip().upper()
            if not symbol or symbol == "SYMBOL":
                continue

            try:
                short_vol = int(float(raw_short))
                exempt_vol = int(float(raw_exempt))
                total_vol = int(float(raw_total))
            except (ValueError, TypeError):
                continue

            if not file_iso_date and len(raw_date) == 8 and raw_date.isdigit():
                file_iso_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"

            report_date = file_iso_date or fallback_date
            short_pct = (100.0 * short_vol / total_vol) if total_vol > 0 else None

            by_symbol[symbol] = FinraShortVolumeSnapshot(
                symbol=symbol,
                report_date=report_date,
                short_volume=short_vol,
                short_exempt_volume=exempt_vol,
                finra_reported_total_volume=total_vol,
                short_pct=short_pct,
                fetched_at=now_epoch,
                coverage="FINRA consolidated TRF/ADF",
                unit="shares",
                source="FINRA",
            )

        if not by_symbol:
            raise ProviderError("No valid symbol entries parsed from FINRA report", source="FINRA")

        return by_symbol

    def _fetch_latest_report(self) -> Dict[str, FinraShortVolumeSnapshot]:
        """Walk back up to 7 days from today to find the latest published daily file."""
        now = datetime.utcnow()
        for i in range(MAX_LOOKBACK_DAYS + 1):
            target_date = now - timedelta(days=i)
            day_str = target_date.strftime("%Y%m%d")
            iso_date = target_date.strftime("%Y-%m-%d")
            url = FINRA_BASE_URL.format(yyyymmdd=day_str)

            try:
                resp = requests.get(url, headers=BROWSER_HEADERS, timeout=15)
                if resp.status_code == 200 and len(resp.text) > 100:
                    return self._parse_report(resp.text, fallback_date=iso_date)
            except Exception as exc:
                logger.debug("FINRA probe for %s failed: %s", day_str, exc)
                continue

        raise DataUnavailableError("No recent FINRA short volume files found within 7 days", capability="short-volume", source="FINRA")

    def get_short_volume(self, symbols: Sequence[str]) -> Dict[str, FinraShortVolumeSnapshot]:
        """Fetch consolidated short volume for requested symbols."""
        if not symbols:
            return {}

        cache_key = "finra:shortvol:latest"
        report_map = self._cache.get_or_set(
            key=cache_key,
            loader=self._fetch_latest_report,
            ttl_seconds=FINRA_TTL_SECONDS,
            max_stale_seconds=FINRA_MAX_STALE_SECONDS,
        )

        out: Dict[str, FinraShortVolumeSnapshot] = {}
        for sym in symbols:
            clean = _strip_ticker(sym)
            if clean in report_map:
                out[sym] = report_map[clean]

        return out
