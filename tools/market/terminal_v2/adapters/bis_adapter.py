"""HTTP Adapter for BIS Central Bank Policy Rates.

Source: Bank for International Settlements (stats.bis.org)
Keyless, CORS-open SDMX v2 REST API covering 12 major central banks.
"""
import csv
import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import requests

from core.logger import get_logger
from schemas.macro_schemas import MarketObservable
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
        as_of: Optional[datetime] = None,
    ) -> None:
        self._cache = cache or TerminalTtlCache(
            default_ttl_seconds=43200,  # 12 hours
            max_stale_seconds=86400 * 7,
            max_entries=10,
        )
        self._fixture_path = fixture_path
        self._timeout = timeout
        self._as_of = as_of

    def fetch_global_policy_rates(self, as_of: Optional[datetime] = None) -> GlobalPolicyRateSnapshot:
        ref_dt = as_of if as_of is not None else self._as_of
        cache_key = (
            f"bis:policy_rates:{ref_dt.strftime('%Y-%m-%d')}"
            if ref_dt is not None
            else "bis:policy_rates:latest"
        )

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

            return self._parse_csv(text, as_of=ref_dt)

        return self._cache.get_or_compute(cache_key, _fetch, ttl_seconds=43200)

    def _parse_csv(self, csv_text: str, as_of: Optional[datetime] = None) -> GlobalPolicyRateSnapshot:
        ref_dt = as_of if as_of is not None else self._as_of
        if ref_dt is None:
            now_utc = datetime.now(timezone.utc)
        elif ref_dt.tzinfo is not None:
            now_utc = ref_dt.astimezone(timezone.utc)
        else:
            now_utc = ref_dt.replace(tzinfo=timezone.utc)
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
                    if (now_utc - p_dt).days > 180:
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
        as_of_str = now_utc.strftime("%Y-%m-%d")

        # Snapshot is stale if all items are stale or empty
        snapshot_stale = bool(policy_items and all(item.is_stale for item in policy_items))

        return GlobalPolicyRateSnapshot(
            as_of_date=as_of_str,
            rates=tuple(policy_items),
            spreads_vs_bot_repo=spreads,
            source="BIS",
            is_stale=snapshot_stale,
        )

    def as_macro_observables(
        self,
        snapshot: Optional[GlobalPolicyRateSnapshot] = None,
        as_of_date: Optional[str] = None,
    ) -> List[MarketObservable]:
        """Convert BIS policy rates snapshot into typed MarketObservables."""
        if snapshot is None:
            try:
                snapshot = self.fetch_global_policy_rates()
            except Exception as exc:
                logger.warning("Could not fetch BIS policy rates for observables: %s", exc)
                return []

        today_str = as_of_date or (
            self._as_of.strftime("%Y-%m-%d")
            if self._as_of is not None
            else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
        obs_date = snapshot.as_of_date or today_str
        if len(obs_date) == 7:
            obs_date = f"{obs_date}-01"
        elif not re.fullmatch(r"\d{4}-\d{2}-\d{2}", obs_date):
            obs_date = today_str

        observables: List[MarketObservable] = []
        is_snap_stale = snapshot.is_stale

        region_map = {
            "US": "United States",
            "TH": "Thailand",
            "XM": "Euro Area",
            "JP": "Japan",
            "CN": "China",
            "IN": "India",
            "GB": "United Kingdom",
            "KR": "South Korea",
            "ID": "Indonesia",
            "MY": "Malaysia",
            "PH": "Philippines",
            "AU": "Australia",
        }

        us_rate_val: Optional[float] = None
        us_rate_valid = False
        th_rate_val: Optional[float] = None
        th_rate_valid = False

        for item in snapshot.rates:
            ctry = (item.country or "").strip().upper()
            rate = item.rate_value
            eff_date = item.effective_date or obs_date
            if len(eff_date) == 7:
                eff_date = f"{eff_date}-01"
            elif not re.fullmatch(r"\d{4}-\d{2}-\d{2}", eff_date):
                eff_date = obs_date

            item_valid = (not is_snap_stale) and (not item.is_stale)
            region = region_map.get(ctry, ctry)

            if ctry == "US":
                obs_id = "obs_us_policy_rate_bis"
                indicator = "US Fed Funds Policy Rate (BIS)"
                us_rate_val = rate
                us_rate_valid = item_valid
            elif ctry == "TH":
                obs_id = "obs_thai_policy_rate_bis"
                indicator = "Bank of Thailand Policy Rate (BIS)"
                th_rate_val = rate
                th_rate_valid = item_valid
            else:
                obs_id = f"obs_{ctry.lower()}_policy_rate_bis"
                indicator = f"{item.central_bank} Policy Rate (BIS)"

            val_str = f"{rate:.3f}" if rate != round(rate, 2) else f"{rate:.2f}"
            observables.append(
                MarketObservable(
                    observable_id=obs_id,
                    asset_bucket="cash",
                    region=region,
                    indicator=indicator,
                    value=val_str,
                    unit="%",
                    observed_at=eff_date,
                    source_file="Terminal_V2_BIS",
                    provider="Terminal V2 (BIS)",
                    confidence="high" if item_valid else "low",
                    is_valid=item_valid,
                    status="verified" if item_valid else "stale",
                    stale_reason="Observation age exceeds fresh threshold" if not item_valid else "",
                    metadata={
                        "raw_country": ctry,
                        "numeric_value": rate,
                        "previous_rate": item.previous_rate,
                        "rate_type": item.rate_type,
                        "central_bank": item.central_bank,
                        "currency": item.currency,
                        "last_change_date": item.last_change_date,
                    },
                )
            )

        # Differential: US - TH
        if us_rate_val is not None and th_rate_val is not None and us_rate_valid and th_rate_valid:
            diff_pct = us_rate_val - th_rate_val
            diff_bps = round(diff_pct * 100.0, 1)
            observables.append(
                MarketObservable(
                    observable_id="obs_diff_us_th_policy_rate_bis",
                    asset_bucket="cash",
                    region="Global",
                    indicator="US-Thailand Policy Rate Differential (BIS)",
                    value=f"{diff_bps:.1f}",
                    unit="bps",
                    observed_at=obs_date,
                    source_file="Terminal_V2_BIS",
                    provider="Terminal V2 (BIS)",
                    confidence="high",
                    is_valid=True,
                    status="verified",
                    metadata={
                        "diff_bps": diff_bps,
                        "diff_pct": round(diff_pct, 4),
                        "us_rate": us_rate_val,
                        "th_rate": th_rate_val,
                    },
                )
            )

        return observables
