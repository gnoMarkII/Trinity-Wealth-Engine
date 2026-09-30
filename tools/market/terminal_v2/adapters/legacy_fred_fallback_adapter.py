"""Legacy Fallback Adapter for FRED Macro Series.

Uses the official fredapi client if FRED_API_KEY is available in the environment.
Strict Rule: Same-type fallback only. Only serves exact macro series IDs.
Does NOT perform cross-asset substitution.
"""
import logging
import os
from typing import Optional
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import MacroSeries, MacroSeriesPoint
from tools.market.terminal_v2.ports.driven_ports import MacroSeriesPort

logger = logging.getLogger(__name__)


class LegacyFredFallbackAdapter(MacroSeriesPort):
    """Fallback adapter using legacy fredapi package when an API key is available."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("FRED_API_KEY")

    def get_macro_series(self, series_id: str) -> MacroSeries:
        """Fetch series using legacy fredapi if API key is present."""
        sid = series_id.strip().upper()
        if not self._api_key:
            raise DataUnavailableError(
                f"Legacy FRED fallback is disabled (no FRED_API_KEY configured for '{sid}')",
                capability="macro_series",
                source="legacy_fredapi",
            )

        try:
            from fredapi import Fred  # Lazy import
            fred = Fred(api_key=self._api_key)
            series_data = fred.get_series(sid).dropna()
        except Exception as exc:
            raise ProviderError(
                f"Legacy fredapi failed for series '{sid}': {exc}",
                source="legacy_fredapi",
            ) from exc

        points: list[MacroSeriesPoint] = []
        for dt, val in series_data.items():
            try:
                date_str = dt.strftime("%Y-%m-%d")
                points.append(MacroSeriesPoint(date=date_str, value=float(val)))
            except Exception:
                continue

        return MacroSeries(
            series_id=sid,
            label=f"FRED Series {sid} (Fallback)",
            source="legacy_fredapi",
            frequency="Varies",
            unit="Value",
            points=tuple(points),
            is_stale=False,
            stale_reason="Served via legacy fredapi fallback",
        )
