"""HTTP Adapter for BIS Central Bank Policy Rates.

Source: Bank for International Settlements (stats.bis.org)
Keyless, CORS-open SDMX v2 REST API covering 12 major central banks.
"""
import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import requests

from core.logger import get_logger
from tools.market.terminal_v2.application.cache import TerminalTtlCache
from tools.market.terminal_v2.domain.calculations import calculate_policy_rate_spreads
from tools.market.terminal_v2.domain.models import (
    GlobalPolicyRateSnapshot,
    PolicyRateItem,
)
from tools.market.terminal_v2.ports.driven_ports import GlobalPolicyRatesPort

logger = get_logger(__name__)

BIS_CBPOL_URL = (
    "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/"
    "M.TH+US+XM+JP+GB+CN+IN+KR+ID+MY+PH+AU?format=csv"
)

COUNTRY_METADATA = {
    "TH": {"central_bank": "Bank of Thailand", "currency": "THB", "fallback_rate_type": "1-Day Bilateral Repo Rate"},
    "US": {"central_bank": "Federal Reserve", "currency": "USD", "fallback_rate_type": "Federal Funds Target Rate"},
    "XM": {"central_bank": "European Central Bank", "currency": "EUR", "fallback_rate_type": "Deposit Facility Rate"},
    "JP": {"central_bank": "Bank of Japan", "currency": "JPY", "fallback_rate_type": "Uncollateralized Overnight Call Rate"},
    "GB": {"central_bank": "Bank of England", "currency": "GBP", "fallback_rate_type": "Official Bank Rate"},
    "CN": {"central_bank": "People's Bank of China", "currency": "CNY", "fallback_rate_type": "1-Year Loan Prime Rate (LPR)"},
    "IN": {"central_bank": "Reserve Bank of India", "currency": "INR", "fallback_rate_type": "Policy Repo Rate"},
    "KR": {"central_bank": "Bank of Korea", "currency": "KRW", "fallback_rate_type": "Base Rate"},
    "ID": {"central_bank": "Bank Indonesia", "currency": "IDR", "fallback_rate_type": "BI-Rate (7-Day Reverse Repo)"},
    "MY": {"central_bank": "Bank Negara Malaysia", "currency": "MYR", "fallback_rate_type": "Overnight Policy Rate (OPR)"},
    "PH": {"central_bank": "Bangko Sentral ng Pilipinas", "currency": "PHP", "fallback_rate_type": "Overnight RRP Rate"},
    "AU": {"central_bank": "Reserve Bank of Australia", "currency": "AUD", "fallback_rate_type": "Cash Rate Target"},
}


class BisPolicyRatesHttpAdapter(GlobalPolicyRatesPort):
    """Fetches and parses central bank policy rates across 12 countries from the BIS."""

    def __init__(
        self,
        cache: Optional[TerminalTtlCache] = None,
        fixture_path: Optional[Path] = None,
        timeout: int = 20,
    ) -> None:
        self._cache = cache or TerminalTtlCache(
            default_ttl_seconds=43200,  # 12 hours
            max_stale_seconds=86400 * 7,
            max_entries=10,
        )
        self._fixture_path = fixture_path
        self._timeout = timeout

    def fetch_global_policy_rates(self) -> GlobalPolicyRateSnapshot:
        cache_key = "bis:policy_rates:latest"

        def _fetch() -> GlobalPolicyRateSnapshot:
            if self._fixture_path and self._fixture_path.exists():
                text = self._fixture_path.read_text(encoding="utf-8")
            else:
                resp = requests.get(
                    BIS_CBPOL_URL,
                    timeout=self._timeout,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                resp.raise_for_status()
                text = resp.text

            return self._parse_csv(text)

        return self._cache.get_or_compute(cache_key, _fetch, ttl_seconds=43200)

    def _parse_csv(self, csv_text: str) -> GlobalPolicyRateSnapshot:
        reader = csv.DictReader(io.StringIO(csv_text.strip()))
        rows_by_country: Dict[str, List[Dict[str, str]]] = {}

        for row in reader:
            country = row.get("REF_AREA", "").strip().upper()
            if not country:
                continue
            if country not in rows_by_country:
                rows_by_country[country] = []
            rows_by_country[country].append(row)

        if not rows_by_country:
            raise ValueError("No country observations found in BIS CSV")

        policy_items: List[PolicyRateItem] = []

        # Maintain consistent ordering starting with TH
        country_order = ["TH", "US", "XM", "JP", "GB", "CN", "IN", "KR", "ID", "MY", "PH", "AU"]

        for ctry in country_order:
            rows = rows_by_country.get(ctry)
            if not rows:
                continue

            # Sort chronological by TIME_PERIOD
            rows.sort(key=lambda r: r.get("TIME_PERIOD", ""))
            latest_row = rows[-1]

            try:
                rate_val = float(latest_row.get("OBS_VALUE", "0"))
            except ValueError:
                continue

            meta = COUNTRY_METADATA.get(ctry, {
                "central_bank": "Central Bank",
                "currency": "N/A",
                "fallback_rate_type": "Policy Rate",
            })

            rate_type = latest_row.get("COMPILATION", "").strip() or meta["fallback_rate_type"]
            effective_date = latest_row.get("TIME_PERIOD", "").strip()

            # Find previous different rate
            prev_rate: Optional[float] = None
            last_change: Optional[str] = None
            for past_row in reversed(rows[:-1]):
                try:
                    past_val = float(past_row.get("OBS_VALUE", "0"))
                    if past_val != rate_val:
                        prev_rate = past_val
                        last_change = past_row.get("TIME_PERIOD")
                        break
                except ValueError:
                    continue

            # Release freshness check: verify observation period age (TH-F08)
            item_stale = False
            if effective_date:
                try:
                    if len(effective_date) == 7:
                        p_dt = datetime.strptime(effective_date, "%Y-%m").replace(tzinfo=timezone.utc)
                    else:
                        p_dt = datetime.strptime(effective_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    if (datetime.now(timezone.utc) - p_dt).days > 180:
                        item_stale = True
                except Exception:
                    pass

            item = PolicyRateItem(
                country=ctry,
                rate_value=rate_val,
                rate_type=rate_type,
                effective_date=effective_date,
                currency=meta["currency"],
                central_bank=meta["central_bank"],
                previous_rate=prev_rate,
                last_change_date=last_change,
                is_stale=item_stale,
            )
            policy_items.append(item)

        spreads = calculate_policy_rate_spreads(policy_items, benchmark_country="TH")
        as_of = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Snapshot is stale if all items are stale or empty
        snapshot_stale = bool(policy_items and all(item.is_stale for item in policy_items))

        return GlobalPolicyRateSnapshot(
            as_of_date=as_of,
            rates=tuple(policy_items),
            spreads_vs_bot_repo=spreads,
            source="BIS",
            is_stale=snapshot_stale,
        )
