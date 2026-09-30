"""FRED Keyless CSV Adapter for Macroeconomic Series.

Downloads public macroeconomic time series directly from FRED's chart export endpoint:
https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>
Requires NO API key, NO registration, and NO token.
Cache TTL: 3600 seconds (1 hour).
"""
import csv
import io
import logging
from typing import Dict, Optional, Tuple
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import MacroSeries, MacroSeriesPoint
from tools.market.terminal_v2.ports.driven_ports import MacroSeriesPort

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_TTL_SECONDS = 3600.0
TIMEOUT_SECONDS = 15.0

FRED_HEADERS = {
    "User-Agent": "curl/8.4.0",
    "Accept": "*/*",
}

SERIES_METADATA: Dict[str, Dict[str, str]] = {
    "CPIAUCSL": {
        "label": "Consumer Price Index for All Urban Consumers (CPI-U)",
        "frequency": "Monthly",
        "unit": "Index 1982-1984=100",
    },
    "FEDFUNDS": {
        "label": "Federal Funds Effective Rate",
        "frequency": "Monthly",
        "unit": "Percent",
    },
    "DGS10": {
        "label": "10-Year Treasury Constant Maturity Rate",
        "frequency": "Daily",
        "unit": "Percent",
    },
    "T10Y2Y": {
        "label": "10-Year Treasury Minus 2-Year Treasury Yield Spread",
        "frequency": "Daily",
        "unit": "Percent",
    },
    "MORTGAGE30US": {
        "label": "30-Year Fixed Rate Mortgage Average in the United States",
        "frequency": "Weekly",
        "unit": "Percent",
    },
    "SP500": {
        "label": "S&P 500 Index Level",
        "frequency": "Daily",
        "unit": "Index",
    },
    "VIXCLS": {
        "label": "CBOE Volatility Index (VIX)",
        "frequency": "Daily",
        "unit": "Index",
    },
    "UNRATE": {
        "label": "Unemployment Rate",
        "frequency": "Monthly",
        "unit": "Percent",
    },
    "GDP": {
        "label": "Gross Domestic Product",
        "frequency": "Quarterly",
        "unit": "Billions of Dollars",
    },
}


class FredCsvAdapter(MacroSeriesPort):
    """Native Python keyless adapter for FRED CSV time series."""

    def __init__(self, cache: Optional[ThreadSafeTTLCache] = None):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=FRED_TTL_SECONDS)

    def get_macro_series(self, series_id: str) -> MacroSeries:
        """Fetch macro series via keyless fredgraph.csv.

        Parses CSV, skips missing '.' dots, and returns strongly-typed MacroSeries.
        """
        sid = series_id.strip().upper()
        cache_key = f"fred:csv:{sid}"

        def _loader() -> MacroSeries:
            url = f"{FRED_CSV_URL}?id={sid}"
            try:
                resp = requests.get(url, headers=FRED_HEADERS, timeout=TIMEOUT_SECONDS)
                resp.raise_for_status()
                text = resp.text
            except Exception as exc:
                raise ProviderError(
                    f"Failed to fetch FRED series '{sid}': {exc}",
                    source="FRED Keyless CSV",
                ) from exc

            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            if not rows or len(rows) < 2:
                raise ProviderError(
                    f"FRED series '{sid}' returned empty or malformed CSV",
                    source="FRED Keyless CSV",
                )

            # Header check: DATE / OBSERVATION_DATE, <SERIES_ID>
            header = rows[0]
            first_col = header[0].strip().upper() if header else ""
            if len(header) < 2 or first_col not in ("DATE", "OBSERVATION_DATE"):
                raise ProviderError(
                    f"FRED series '{sid}' returned unrecognized CSV header: {header}",
                    source="FRED Keyless CSV",
                )

            points: list[MacroSeriesPoint] = []
            for row in rows[1:]:
                if len(row) < 2:
                    continue
                date_str = row[0].strip()
                val_str = row[1].strip()
                if val_str == "." or not val_str:
                    continue
                try:
                    val = float(val_str)
                    points.append(MacroSeriesPoint(date=date_str, value=val))
                except ValueError:
                    continue

            meta = SERIES_METADATA.get(sid, {
                "label": f"FRED Series {sid}",
                "frequency": "Varies",
                "unit": "Value",
            })

            return MacroSeries(
                series_id=sid,
                label=meta["label"],
                source="FRED",
                frequency=meta["frequency"],
                unit=meta["unit"],
                points=tuple(points),
                is_stale=False,
            )

        try:
            return self._cache.get_or_set(
                cache_key,
                _loader,
                ttl_seconds=FRED_TTL_SECONDS,
            )
        except Exception as exc:
            raise DataUnavailableError(
                f"Macro series '{sid}' is temporarily unavailable from FRED: {exc}",
                capability="macro_series",
                source="FRED",
            ) from exc
