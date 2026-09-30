"""New York Fed Markets API Adapter for Overnight Reference Rates.

Fetches benchmark reference rates (SOFR, EFFR, TGCR, BGCR, OBFR) and calculates
pair spreads deterministically in basis points.
Strict Rule: Spreads are explicitly named by pair; rates do not represent ON RRP.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.calculations import calculate_rate_spread_bps
from tools.market.terminal_v2.domain.errors import ProviderError
from tools.market.terminal_v2.domain.models import (
    ReferenceRatePoint,
    ReferenceRateSnapshot,
)
from tools.market.terminal_v2.ports.driven_ports import ReferenceRatesPort

logger = logging.getLogger(__name__)

NYFED_RATES_URL = "https://markets.newyorkfed.org/api/rates/all/latest.json"
NYFED_TTL_SECONDS = 720.0             # 12 minutes
NYFED_MAX_STALE_SECONDS = 4 * 86400.0  # 4 days ceiling

RATE_LABELS: Dict[str, str] = {
    "SOFR": "Secured Overnight Financing Rate",
    "EFFR": "Effective Federal Funds Rate",
    "OBFR": "Overnight Bank Funding Rate",
    "TGCR": "Tri-Party General Collateral Rate",
    "BGCR": "Broad General Collateral Rate",
    "SOFRAI": "SOFR Index / Averages",
}


class NyFedAdapter(ReferenceRatesPort):
    """Adapter reading overnight reference rates from the New York Fed."""

    def __init__(self, cache: Optional[ThreadSafeTTLCache] = None):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=NYFED_TTL_SECONDS)

    def _fetch_rates(self) -> ReferenceRateSnapshot:
        try:
            resp = requests.get(NYFED_RATES_URL, headers=BROWSER_HEADERS, timeout=12)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise ProviderError(f"Failed to fetch NY Fed reference rates: {exc}", source="New York Fed") from exc

        raw_rates = data.get("refRates", [])
        if not raw_rates:
            raise ProviderError("Empty refRates list returned by NY Fed API", source="New York Fed")

        points: List[ReferenceRatePoint] = []
        rates_by_code: Dict[str, ReferenceRatePoint] = {}
        as_of = ""

        for item in raw_rates:
            code = item.get("type", "").strip().upper()
            if not code:
                continue

            eff_date = item.get("effectiveDate", "")
            if not as_of and eff_date:
                as_of = eff_date

            def _flt(v: Optional[object]) -> Optional[float]:
                if v is None:
                    return None
                try:
                    val = float(v)
                    return val
                except (ValueError, TypeError):
                    return None

            pt = ReferenceRatePoint(
                code=code,
                label=RATE_LABELS.get(code, code),
                effective_date=eff_date,
                rate_percent=_flt(item.get("percentRate")),
                volume_in_billions=_flt(item.get("volumeInBillions")),
                target_rate_from=_flt(item.get("targetRateFrom")),
                target_rate_to=_flt(item.get("targetRateTo")),
            )
            points.append(pt)
            rates_by_code[code] = pt

        # Calculate deterministic spreads between matching dates in basis points
        spreads: Dict[str, float] = {}
        effr = rates_by_code.get("EFFR")
        sofr = rates_by_code.get("SOFR")
        tgcr = rates_by_code.get("TGCR")
        bgcr = rates_by_code.get("BGCR")

        if sofr and effr and sofr.rate_percent is not None and effr.rate_percent is not None:
            spreads["SOFR-EFFR"] = calculate_rate_spread_bps(sofr.rate_percent, effr.rate_percent)
        if tgcr and effr and tgcr.rate_percent is not None and effr.rate_percent is not None:
            spreads["TGCR-EFFR"] = calculate_rate_spread_bps(tgcr.rate_percent, effr.rate_percent)
        if bgcr and effr and bgcr.rate_percent is not None and effr.rate_percent is not None:
            spreads["BGCR-EFFR"] = calculate_rate_spread_bps(bgcr.rate_percent, effr.rate_percent)
        if sofr and tgcr and sofr.rate_percent is not None and tgcr.rate_percent is not None:
            spreads["SOFR-TGCR"] = calculate_rate_spread_bps(sofr.rate_percent, tgcr.rate_percent)

        return ReferenceRateSnapshot(
            as_of=as_of,
            rates=tuple(points),
            spreads_bps=spreads,
            fetched_at=time.time(),
            source="New York Fed",
        )

    def get_reference_rates(self) -> ReferenceRateSnapshot:
        """Fetch latest benchmark reference rates and deterministic pair spreads."""
        cache_key = "nyfed:rates:latest"
        return self._cache.get_or_set(
            key=cache_key,
            loader=self._fetch_rates,
            ttl_seconds=NYFED_TTL_SECONDS,
            max_stale_seconds=NYFED_MAX_STALE_SECONDS,
        )
