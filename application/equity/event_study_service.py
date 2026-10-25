"""Application Service for Equity Event Study & Historical Scenario Analysis.

Coordinates between market data (OHLCV candles and corporate action earnings dates),
curated event catalogs, and the pure-domain EventStudyEngine.
"""
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
from zoneinfo import ZoneInfo

from core.event_study.engine import EventStudyEngine
from core.event_study.models import DailyPriceBar, EventStudyReport
from tools.market.event_study_presets import get_preset_catalog, list_event_presets

log = logging.getLogger(__name__)


class OhlcvDataServicePort(Protocol):
    """Port interface for querying OHLCV candles and corporate action events."""

    def get_ohlcv(self, ticker: str, range_str: str, interval_str: str) -> Any:
        ...


class EventStudyApplicationService:
    """Orchestrates event study backtests across presets and custom dates."""

    def __init__(self, ohlcv_service: OhlcvDataServicePort):
        self._ohlcv_service = ohlcv_service

    def get_available_presets(self) -> List[Dict[str, str]]:
        """Returns catalog of preset scenarios."""
        return list_event_presets()

    def run_event_study(
        self,
        ticker: str,
        preset_id: str = "earnings",
        custom_dates: Optional[List[Union[str, Tuple[str, Optional[str]]]]] = None,
        custom_event_name: Optional[str] = None,
        horizons: Optional[List[int]] = None,
    ) -> EventStudyReport:
        """Executes an event study for the given ticker and scenario."""
        clean_ticker = ticker.strip().upper()
        norm_preset = preset_id.strip().lower()

        # 1. Fetch OHLCV daily data (try max, 5y, then 1y for deep historical sample)
        ohlcv_res = None
        for r_str in ("max", "5y", "1y"):
            try:
                ohlcv_res = self._ohlcv_service.get_ohlcv(clean_ticker, range_str=r_str, interval_str="1d")
                if ohlcv_res and getattr(ohlcv_res, "candles", None):
                    break
            except Exception as exc:
                log.debug("Failed to fetch OHLCV for %s with range %s: %s", clean_ticker, r_str, exc)

        bars: List[DailyPriceBar] = []
        if ohlcv_res and hasattr(ohlcv_res, "candles") and ohlcv_res.candles:
            tz = ZoneInfo("Asia/Bangkok") if clean_ticker.endswith(".BK") else timezone.utc
            for c in ohlcv_res.candles:
                # Convert timestamp ms to YYYY-MM-DD
                dt = datetime.fromtimestamp(c.timestamp / 1000.0, tz=tz)
                date_str = dt.strftime("%Y-%m-%d")
                bars.append(
                    DailyPriceBar(
                        date=date_str,
                        open=float(c.open),
                        high=float(c.high),
                        low=float(c.low),
                        close=float(c.close),
                        volume=float(c.volume),
                    )
                )

        # 2. Resolve event dates and event name
        event_items: List[Union[str, Tuple[str, Optional[str]]]] = []
        event_name = custom_event_name or ""

        if norm_preset == "earnings":
            event_name = event_name or "Quarterly Earnings Announcements (PEAD)"
            # Extract historical earnings dates from corporate actions if present
            # OHLCVResponseDTO provides events: list[CorporateActionEventDTO]
            events_source = getattr(ohlcv_res, "events", None) or getattr(ohlcv_res, "corporate_actions", None)
            if events_source:
                for ca in events_source:
                    if getattr(ca, "event_type", None) == "earnings" and getattr(ca, "date_str", None):
                        raw_tooltip = getattr(ca, "tooltip", "")
                        # e.g. "Reported Date: 2021-04-20 (Mapped: reported_date) | Earnings Beat: EPS $0.38 vs Est $0.30"
                        if raw_tooltip and "|" in raw_tooltip:
                            desc = raw_tooltip.split("|")[-1].strip()
                        elif raw_tooltip:
                            desc = raw_tooltip.strip()
                        else:
                            desc = "Quarterly Earnings Announcement"
                        event_items.append((ca.date_str, desc))

        elif norm_preset == "custom":
            event_name = event_name or "Custom Event Scenario"
            if custom_dates:
                event_items.extend(custom_dates)

        else:
            catalog = get_preset_catalog(norm_preset)
            if catalog:
                event_name = event_name or catalog.name
                event_items = [(item.date, item.title) for item in catalog.items]
            else:
                event_name = event_name or f"Scenario: {preset_id}"
                if custom_dates:
                    event_items.extend(custom_dates)

        # 3. Execute pure-domain calculation engine
        report = EventStudyEngine.evaluate(
            ticker=clean_ticker,
            event_name=event_name,
            bars=bars,
            event_items=event_items,
            horizons=horizons or [5, 10, 20, 60],
        )

        # 4. Add domain warning if evaluating a specific company preset on a different ticker
        if norm_preset not in ("earnings", "custom"):
            catalog = get_preset_catalog(norm_preset)
            if catalog and catalog.recommended_ticker and catalog.recommended_ticker.upper() != clean_ticker:
                from core.event_study.models import EventStudyGuardrailWarning
                report.warnings.append(
                    EventStudyGuardrailWarning(
                        code="CROSS_ASSET_SCENARIO",
                        message=f"Preset '{catalog.name}' is tailored for {catalog.recommended_ticker}. Evaluated as a cross-asset scenario on {clean_ticker}.",
                    )
                )

        return report
