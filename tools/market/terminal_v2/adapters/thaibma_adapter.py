"""ThaiBMA Public Web API Adapter for Thai Government Bond Yield Curve.

Fetches official end-of-day model government bond yield curve directly from
Thai Bond Market Association (ThaiBMA) public web endpoints.
Sole official pricing center for Thai fixed income securities.
"""
import logging
import time
from typing import Dict, List, Optional, Tuple
import requests

from schemas.macro_schemas import MarketObservable
from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import ProviderError
from tools.market.terminal_v2.domain.models import (
    ThaiYieldCurveSnapshot,
    ThaiYieldPoint,
)
from tools.market.terminal_v2.ports.driven_ports import ThaiYieldCurvePort

logger = logging.getLogger(__name__)

THAIBMA_AVAIL_URL = "https://www.thaibma.or.th/yieldcurve/avail"
THAIBMA_GOV_URL_TEMPLATE = "https://www.thaibma.or.th/yieldcurve/gov/{date}"
THAIBMA_TTL_SECONDS = 14400.0         # 4 hours
THAIBMA_MAX_STALE_SECONDS = 7 * 86400.0  # 7 days ceiling

# Standard Tenor Mapping from TTM years
_STANDARD_TENORS: Dict[float, str] = {
    1.0: "1Y",
    2.0: "2Y",
    3.0: "3Y",
    4.0: "4Y",
    5.0: "5Y",
    6.0: "6Y",
    7.0: "7Y",
    8.0: "8Y",
    9.0: "9Y",
    10.0: "10Y",
    15.0: "15Y",
    20.0: "20Y",
    30.0: "30Y",
    50.0: "50Y",
}


def _match_standard_tenor(ttm: float) -> Optional[str]:
    """Map TTM year value to standard tenor label."""
    # Check bills (1M ~ 0.08, 3M ~ 0.25, 6M ~ 0.50)
    if 0.07 <= ttm <= 0.09:
        return "1M"
    if 0.24 <= ttm <= 0.26:
        return "3M"
    if 0.48 <= ttm <= 0.51:
        return "6M"
    
    # Check exact integer tenors
    for yr, label in _STANDARD_TENORS.items():
        if abs(ttm - yr) < 0.01:
            return label
    return None


class ThaiBmaPublicAdapter(ThaiYieldCurvePort):
    """Keyless public web adapter for ThaiBMA Government Bond Yield Curve."""

    def __init__(self, cache: Optional[ThreadSafeTTLCache] = None):
        self._cache = cache or ThreadSafeTTLCache()

    def _get_headers(self) -> Dict[str, str]:
        headers = dict(BROWSER_HEADERS)
        headers.update({
            "Referer": "https://www.thaibma.or.th/EN/Market/YieldCurve/Government.aspx",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        })
        return headers

    def get_latest_available_date(self) -> str:
        """Fetch the latest available date from ThaiBMA."""
        try:
            resp = requests.get(THAIBMA_AVAIL_URL, headers=self._get_headers(), timeout=10)
            if resp.status_code != 200:
                raise ProviderError(f"ThaiBMA avail returned HTTP {resp.status_code}", source="ThaiBMA")
            dates = resp.json()
            if isinstance(dates, list) and len(dates) >= 2:
                # Latest date is second element (ISO string like '2026-10-02T00:00:00')
                return str(dates[1])[:10]
            raise ProviderError("Invalid date array returned by ThaiBMA avail", source="ThaiBMA")
        except Exception as exc:
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(f"Failed to fetch available dates from ThaiBMA: {exc}", source="ThaiBMA") from exc

    def _fetch_yield_curve_raw(self, target_date: Optional[str] = None) -> ThaiYieldCurveSnapshot:
        obs_date = target_date
        if not obs_date:
            obs_date = self.get_latest_available_date()

        url = THAIBMA_GOV_URL_TEMPLATE.format(date=obs_date)
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=12)
            if resp.status_code != 200:
                raise ProviderError(f"ThaiBMA gov yield curve returned HTTP {resp.status_code}", source="ThaiBMA")
            data = resp.json()
        except Exception as exc:
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(f"Failed to fetch ThaiBMA gov curve for {obs_date}: {exc}", source="ThaiBMA") from exc

        curve_rows = data.get("Curve", [])
        if not curve_rows:
            raise ProviderError(f"Empty yield curve returned by ThaiBMA for {obs_date}", source="ThaiBMA")

        yield_points: List[ThaiYieldPoint] = []
        yields_by_tenor: Dict[str, float] = {}

        for row in curve_rows:
            ttm = row.get("X")
            yld = row.get("Y")
            if ttm is None or yld is None:
                continue
            ttm_float = float(ttm)
            yld_float = float(yld)
            tenor_lbl = _match_standard_tenor(ttm_float)
            if tenor_lbl:
                yield_points.append(
                    ThaiYieldPoint(
                        tenor=tenor_lbl,
                        ttm_years=round(ttm_float, 4),
                        yield_percent=round(yld_float, 4),
                    )
                )
                yields_by_tenor[tenor_lbl] = yld_float

        y2 = yields_by_tenor.get("2Y")
        y10 = yields_by_tenor.get("10Y")
        y1 = yields_by_tenor.get("1Y")

        spread_10y_2y = round((y10 - y2) * 100.0, 1) if (y10 is not None and y2 is not None) else None
        spread_10y_1y = round((y10 - y1) * 100.0, 1) if (y10 is not None and y1 is not None) else None

        return ThaiYieldCurveSnapshot(
            observation_date=obs_date,
            yields=tuple(yield_points),
            spread_10y_2y_bps=spread_10y_2y,
            spread_10y_1y_bps=spread_10y_1y,
            fetched_at=time.time(),
            source="ThaiBMA",
            unit="percent / basis points",
            is_stale=False,
            stale_reason="",
        )

    def get_government_yield_curve(self, as_of_date: Optional[str] = None) -> ThaiYieldCurveSnapshot:
        cache_key = f"thaibma:gov_curve:{as_of_date or 'latest'}"
        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._fetch_yield_curve_raw(as_of_date),
            ttl_seconds=THAIBMA_TTL_SECONDS,
            max_stale_seconds=THAIBMA_MAX_STALE_SECONDS,
        )

    def as_macro_observables(self, snapshot: Optional[ThaiYieldCurveSnapshot] = None) -> List[MarketObservable]:
        """Convert ThaiBMA yield curve snapshot into canonical MarketObservable items."""
        if snapshot is None:
            try:
                snapshot = self.get_government_yield_curve()
            except Exception as e:
                logger.warning("Could not fetch ThaiBMA yield curve for observables: %s", e)
                return []

        obs_date = snapshot.observation_date
        observables: List[MarketObservable] = []

        yields_map = {p.tenor: p.yield_percent for p in snapshot.yields if p.yield_percent is not None}

        # 1. 2Y Yield
        y2 = yields_map.get("2Y")
        if y2 is not None:
            observables.append(
                MarketObservable(
                    observable_id="obs_th_gov_yield_2y",
                    asset_bucket="fixed_income",
                    region="Thailand",
                    indicator="Thailand 2Y Gov Bond Yield",
                    value=f"{y2:.2f}",
                    unit="%",
                    observed_at=obs_date,
                    source_file="ThaiBMA_Yield_Curve",
                    provider="ThaiBMA",
                    confidence="high",
                    is_valid=True,
                    status="verified",
                    metadata={"val": y2, "tenor": "2Y"},
                )
            )

        # 2. 10Y Yield
        y10 = yields_map.get("10Y")
        if y10 is not None:
            observables.append(
                MarketObservable(
                    observable_id="obs_th_gov_yield_10y",
                    asset_bucket="fixed_income",
                    region="Thailand",
                    indicator="Thailand 10Y Gov Bond Yield",
                    value=f"{y10:.2f}",
                    unit="%",
                    observed_at=obs_date,
                    source_file="ThaiBMA_Yield_Curve",
                    provider="ThaiBMA",
                    confidence="high",
                    is_valid=True,
                    status="verified",
                    metadata={"val": y10, "tenor": "10Y"},
                )
            )

        # 3. 10Y - 2Y Spread
        if snapshot.spread_10y_2y_bps is not None:
            observables.append(
                MarketObservable(
                    observable_id="obs_th_gov_10y_2y_spread",
                    asset_bucket="fixed_income",
                    region="Thailand",
                    indicator="Thailand Gov Bond 10Y-2Y Spread",
                    value=f"{snapshot.spread_10y_2y_bps:.1f}",
                    unit="bps",
                    observed_at=obs_date,
                    source_file="ThaiBMA_Yield_Curve",
                    provider="ThaiBMA",
                    confidence="high",
                    is_valid=True,
                    status="verified",
                    metadata={
                        "val": snapshot.spread_10y_2y_bps,
                        "diff_bps": snapshot.spread_10y_2y_bps,
                        "yield_10y": y10,
                        "yield_2y": y2,
                    },
                )
            )

        # 4. 1Y Yield (for Policy Spread corroboration)
        y1 = yields_map.get("1Y")
        if y1 is not None:
            observables.append(
                MarketObservable(
                    observable_id="obs_th_gov_yield_1y",
                    asset_bucket="fixed_income",
                    region="Thailand",
                    indicator="Thailand 1Y Gov Bond Yield",
                    value=f"{y1:.2f}",
                    unit="%",
                    observed_at=obs_date,
                    source_file="ThaiBMA_Yield_Curve",
                    provider="ThaiBMA",
                    confidence="high",
                    is_valid=True,
                    status="verified",
                    metadata={"val": y1, "tenor": "1Y"},
                )
            )

        # 5. 5Y Yield
        y5 = yields_map.get("5Y")
        if y5 is not None:
            observables.append(
                MarketObservable(
                    observable_id="obs_th_gov_yield_5y",
                    asset_bucket="fixed_income",
                    region="Thailand",
                    indicator="Thailand 5Y Gov Bond Yield",
                    value=f"{y5:.2f}",
                    unit="%",
                    observed_at=obs_date,
                    source_file="ThaiBMA_Yield_Curve",
                    provider="ThaiBMA",
                    confidence="high",
                    is_valid=True,
                    status="verified",
                    metadata={"val": y5, "tenor": "5Y"},
                )
            )

        return observables
