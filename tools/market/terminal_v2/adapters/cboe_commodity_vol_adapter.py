"""Cboe Commodity Volatility Index Adapter (GVZ, VXSLV, OVX).

Parses Cboe's official daily history CSV files:
- GVZ: SPDR Gold Shares (GLD) ETF Volatility
- VXSLV: iShares Silver Trust (SLV) ETF Volatility
- OVX: United States Oil Fund (USO) ETF Volatility

Strict Invariants:
1. Underlying instruments are ETF options, not physical futures.
2. 1D change in index points requires two most recent trading closes.
3. 52-week percentile computed from historical closes (min 100 samples).
4. Thread-safe TTL cache with bounded stale fallback.
"""
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.calculations import (
    calculate_commodity_vol_percentile,
    classify_commodity_vol_regime,
)
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import CommodityVolPoint, CommodityVolSnapshot
from tools.market.terminal_v2.ports.driven_ports import CommodityVolPort

logger = logging.getLogger(__name__)

CBOE_BASE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices"
VOL_TTL_SECONDS = 14400.0          # 4 hours
VOL_MAX_STALE_SECONDS = 86400.0    # 24 hours

INDEX_METADATA = {
    "GVZ": {
        "name": "Cboe Gold ETF Volatility Index",
        "underlying": "SPDR Gold Shares (GLD) ETF Options",
        "filename": "GVZ_History.csv",
    },
    "VXSLV": {
        "name": "Cboe Silver ETF Volatility Index",
        "underlying": "iShares Silver Trust (SLV) ETF Options",
        "filename": "VXSLV_History.csv",
    },
    "OVX": {
        "name": "Cboe Crude Oil ETF Volatility Index",
        "underlying": "United States Oil Fund (USO) ETF Options",
        "filename": "OVX_History.csv",
    },
}


class CboeCommodityVolAdapter(CommodityVolPort):
    """Adapter fetching commodity implied volatility indices from Cboe."""

    def __init__(
        self,
        cache: Optional[ThreadSafeTTLCache] = None,
        fixture_dir: Optional[Path] = None,
    ):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=VOL_TTL_SECONDS)
        self._fixture_dir = fixture_dir

    def _fetch_csv(self, symbol: str) -> str:
        sym = symbol.upper()
        if sym not in INDEX_METADATA:
            raise ProviderError(f"Unsupported commodity volatility index: {symbol}", source="Cboe")

        if self._fixture_dir:
            fpath = self._fixture_dir / f"cboe_{sym.lower()}_fixture.csv"
            if fpath.exists():
                return fpath.read_text(encoding="utf-8")

        filename = INDEX_METADATA[sym]["filename"]
        url = f"{CBOE_BASE_URL}/{filename}"
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            raise ProviderError(f"Failed to fetch Cboe {sym} CSV from {url}: {exc}", source="Cboe") from exc

    def _parse_csv_history(self, symbol: str, csv_text: str) -> List[CommodityVolPoint]:
        sym = symbol.upper()
        lines = [line.strip() for line in csv_text.splitlines() if line.strip()]
        if len(lines) < 2:
            raise DataUnavailableError(f"Cboe {sym} CSV contains insufficient data", capability="commodity-vol", source="Cboe")

        header = [h.strip().upper() for h in lines[0].split(",")]
        # Formats:
        # 1. Close-only (GVZ, OVX): DATE,GVZ or DATE,OVX
        # 2. OHLC (VXSLV): DATE,OPEN,HIGH,LOW,CLOSE
        date_idx = 0
        close_idx = -1
        if "CLOSE" in header:
            close_idx = header.index("CLOSE")
        elif sym in header:
            close_idx = header.index(sym)
        else:
            close_idx = 1  # default 2nd column

        points: List[CommodityVolPoint] = []
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) <= max(date_idx, close_idx):
                continue

            raw_date = parts[date_idx]
            raw_close = parts[close_idx]

            # Parse date MM/DD/YYYY -> YYYY-MM-DD
            try:
                dt = datetime.strptime(raw_date, "%m/%d/%Y")
                iso_date = dt.strftime("%Y-%m-%d")
            except ValueError:
                iso_date = raw_date

            try:
                close_val = float(raw_close)
                if close_val > 0:
                    points.append(CommodityVolPoint(date=iso_date, close=close_val))
            except ValueError:
                continue

        if not points:
            raise DataUnavailableError(f"No valid historical points parsed for Cboe {sym}", capability="commodity-vol", source="Cboe")

        return points

    def get_commodity_vol(self, symbol: str) -> CommodityVolSnapshot:
        sym = symbol.upper()
        if sym not in INDEX_METADATA:
            raise ProviderError(f"Unsupported commodity volatility symbol: {symbol}", source="Cboe")

        cache_key = f"cboe:commodity_vol:{sym}"
        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._load_commodity_vol(sym),
            ttl_seconds=VOL_TTL_SECONDS,
            max_stale_seconds=VOL_MAX_STALE_SECONDS,
        )

    def _load_commodity_vol(self, symbol: str) -> CommodityVolSnapshot:
        csv_text = self._fetch_csv(symbol)
        history = self._parse_csv_history(symbol, csv_text)

        latest = history[-1]
        prior = history[-2] if len(history) >= 2 else None

        change_1d = round(latest.close - prior.close, 4) if prior else None
        all_closes = [p.close for p in history]

        percentile, sample_count = calculate_commodity_vol_percentile(latest.close, all_closes, min_samples=100)
        regime = classify_commodity_vol_regime(percentile)

        return CommodityVolSnapshot(
            index_symbol=symbol,
            underlying_instrument=INDEX_METADATA[symbol]["underlying"],
            close_date=latest.date,
            implied_volatility=round(latest.close, 2),
            change_1d_points=change_1d,
            percentile_52w=percentile,
            sample_count=sample_count,
            regime_label=regime,
            source="Cboe",
            as_of_date=latest.date,
            fetched_at=time.time(),
            is_stale=False,
            stale_reason=None,
        )
