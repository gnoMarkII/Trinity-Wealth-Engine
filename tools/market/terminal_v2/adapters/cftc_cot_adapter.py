"""HTTP Adapter for CFTC Commitments of Traders (COT) Disaggregated Report.

Source: US Commodity Futures Trading Commission (publicreporting.cftc.gov)
Keyless Socrata API for weekly disaggregated futures and options positioning.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from core.logger import get_logger
from tools.market.terminal_v2.application.cache import TerminalTtlCache
from tools.market.terminal_v2.domain.calculations import calculate_cot_percentile
from tools.market.terminal_v2.domain.models import (
    MetalsCotPositioningSnapshot,
    TraderClassPosition,
)
from tools.market.terminal_v2.ports.driven_ports import MetalsPositioningPort

logger = get_logger(__name__)

COT_DISAGG_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"

COMMODITY_CODES: Dict[str, str] = {
    "gold": "088691",
    "silver": "084691",
    "copper": "085692",
    "platinum": "076651",
}


def _safe_int(val: Any) -> int:
    try:
        return int(float(str(val).replace(",", "").strip()))
    except (ValueError, TypeError):
        return 0


class CftcCotHttpAdapter(MetalsPositioningPort):
    """Fetches and parses CFTC Disaggregated COT data for precious and industrial metals."""

    def __init__(
        self,
        cache: Optional[TerminalTtlCache] = None,
        fixture_path: Optional[Path] = None,
        timeout: int = 15,
    ) -> None:
        self._cache = cache or TerminalTtlCache(
            default_ttl_seconds=21600,  # 6 hours
            max_stale_seconds=86400 * 7,
            max_entries=50,
        )
        self._fixture_path = fixture_path
        self._timeout = timeout

    def fetch_metals_cot(self, commodity: str = "gold") -> MetalsCotPositioningSnapshot:
        comm_key = commodity.lower().strip()
        code = COMMODITY_CODES.get(comm_key, COMMODITY_CODES["gold"])
        cache_key = f"cftc:cot:disagg:{code}"

        def _fetch() -> MetalsCotPositioningSnapshot:
            if self._fixture_path and self._fixture_path.exists():
                text = self._fixture_path.read_text(encoding="utf-8")
                raw_data = json.loads(text)
            else:
                params = {
                    "cftc_contract_market_code": code,
                    "$order": "report_date_as_yyyy_mm_dd DESC",
                    "$limit": "52",
                }
                resp = requests.get(
                    COT_DISAGG_URL,
                    params=params,
                    timeout=self._timeout,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                resp.raise_for_status()
                raw_data = resp.json()

            return self._parse_disaggregated(raw_data, comm_key, code)

        return self._cache.get_or_compute(cache_key, _fetch, ttl_seconds=21600)

    def _parse_disaggregated(
        self,
        rows: List[Dict[str, Any]],
        commodity: str,
        code: str,
    ) -> MetalsCotPositioningSnapshot:
        if not rows:
            raise ValueError(f"No COT rows returned for commodity {commodity} ({code})")

        # Newest row first
        latest = rows[0]
        raw_date = str(latest.get("report_date_as_yyyy_mm_dd", ""))
        as_of_date = raw_date[:10] if len(raw_date) >= 10 else datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Trader classes in Disaggregated report
        mm_long = _safe_int(latest.get("m_money_positions_long_all"))
        mm_short = _safe_int(latest.get("m_money_positions_short_all"))
        mm_spread = _safe_int(latest.get("m_money_positions_spread"))
        mm_change_l = _safe_int(latest.get("change_in_m_money_long_all"))
        mm_change_s = _safe_int(latest.get("change_in_m_money_short_all"))
        managed_money = TraderClassPosition(
            class_name="Managed Money",
            long_contracts=mm_long,
            short_contracts=mm_short,
            net_contracts=mm_long - mm_short,
            spread_contracts=mm_spread,
            change_long=mm_change_l,
            change_short=mm_change_s,
        )

        swap_l = _safe_int(latest.get("swap_positions_long_all"))
        # Notice double underscore in CFTC schema: swap__positions_short_all
        swap_s = _safe_int(latest.get("swap__positions_short_all") or latest.get("swap_positions_short_all"))
        swap_sp = _safe_int(latest.get("swap__positions_spread_all") or latest.get("swap_positions_spread_all"))
        swap_dealers = TraderClassPosition(
            class_name="Swap Dealers",
            long_contracts=swap_l,
            short_contracts=swap_s,
            net_contracts=swap_l - swap_s,
            spread_contracts=swap_sp,
        )

        prod_l = _safe_int(latest.get("prod_merc_positions_long"))
        prod_s = _safe_int(latest.get("prod_merc_positions_short"))
        producer_merchant = TraderClassPosition(
            class_name="Producer/Merchant/Processor/User",
            long_contracts=prod_l,
            short_contracts=prod_s,
            net_contracts=prod_l - prod_s,
        )

        other_l = _safe_int(latest.get("other_rept_positions_long"))
        other_s = _safe_int(latest.get("other_rept_positions_short"))
        other_sp = _safe_int(latest.get("other_rept_positions_spread"))
        other_reportables = TraderClassPosition(
            class_name="Other Reportables",
            long_contracts=other_l,
            short_contracts=other_s,
            net_contracts=other_l - other_s,
            spread_contracts=other_sp,
        )

        nonrept_l = _safe_int(latest.get("nonrept_positions_long_all"))
        nonrept_s = _safe_int(latest.get("nonrept_positions_short_all"))
        non_reportables = TraderClassPosition(
            class_name="Non-Reportable",
            long_contracts=nonrept_l,
            short_contracts=nonrept_s,
            net_contracts=nonrept_l - nonrept_s,
        )

        # 52-week percentile calculation on net managed money
        history_net: List[int] = []
        for r in rows:
            l_val = _safe_int(r.get("m_money_positions_long_all"))
            s_val = _safe_int(r.get("m_money_positions_short_all"))
            history_net.append(l_val - s_val)

        net_mm = managed_money.net_contracts
        percentile = calculate_cot_percentile(net_mm, history_net)
        published_now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return MetalsCotPositioningSnapshot(
            commodity=commodity.upper(),
            commodity_code=code,
            as_of_date=as_of_date,
            published_at=published_now,
            report_type="disaggregated",
            open_interest=_safe_int(latest.get("open_interest_all")),
            managed_money=managed_money,
            swap_dealers=swap_dealers,
            producer_merchant=producer_merchant,
            other_reportables=other_reportables,
            non_reportables=non_reportables,
            net_managed_money=net_mm,
            percentile_52w=percentile,
            source="CFTC",
            is_stale=False,
        )
