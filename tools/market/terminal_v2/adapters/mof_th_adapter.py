"""Ministry of Finance Thailand Keyless Adapter for Public Debt to GDP.

Parses monthly public debt report from MOF Thailand (id=4).
Extracts statutory debt ceiling metrics: components 1-5, Debt:GDP ratio, and exchange rate.
Strict Rule: Values are in THB; does not map to US Treasury intragovernmental buckets.
"""
import calendar
import csv
import io
import logging
import re
import time
from typing import Dict, List, Optional, Tuple
import requests

from schemas.macro_schemas import MarketObservable
from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import ProviderError
from tools.market.terminal_v2.domain.models import (
    ThaiPublicDebtComponent,
    ThaiPublicDebtSnapshot,
)
from tools.market.terminal_v2.ports.driven_ports import ThaiPublicDebtPort

logger = logging.getLogger(__name__)

MOF_DEBT_URL = "https://dataservices.mof.go.th/export/csv/menu5?id=4"
MOF_TTL_SECONDS = 43200.0             # 12 hours
MOF_MAX_STALE_SECONDS = 7 * 86400.0    # 7 days ceiling
MILLION = 1e6
BE_OFFSET = 543

THAI_MONTHS = [
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
]

COMPONENT_LABELS = {
    1: ("Government debt", "หนี้รัฐบาล"),
    2: ("State-enterprise debt", "หนี้รัฐวิสาหกิจ"),
    3: ("Financial state-enterprise debt (guaranteed)", "หนี้รัฐวิสาหกิจที่ทำธุรกิจในภาคการเงินฯ (รัฐบาลค้ำประกัน)"),
    4: ("FIDF debt", "หนี้กองทุนเพื่อการฟื้นฟูฯ"),
    5: ("Other government agencies", "หนี้หน่วยงานของรัฐ"),
}


def _parse_thai_month_header(header: str) -> str:
    """Convert 'กุมภาพันธ์ 2569' to ISO month-end date '2026-02-28'."""
    clean = header.strip()
    for idx, m_name in enumerate(THAI_MONTHS, start=1):
        if m_name in clean:
            match = re.search(r"(\d{4})", clean)
            if match:
                be_yr = int(match.group(1))
                ce_yr = be_yr - BE_OFFSET
                _, last_day = calendar.monthrange(ce_yr, idx)
                return f"{ce_yr}-{idx:02d}-{last_day:02d}"
    return clean


def _parse_num(val_str: str) -> Optional[float]:
    clean = val_str.replace(",", "").strip()
    try:
        return float(clean)
    except ValueError:
        return None


def parse_public_debt_csv(csv_text: str) -> ThaiPublicDebtSnapshot:
    """Parse MOF Thailand Public Debt CSV text into a structured snapshot."""
    text = csv_text.lstrip("\ufeff")
    reader = csv.reader(io.StringIO(text))
    rows = [r for r in reader if r]
    if len(rows) < 7:
        raise ProviderError("Incomplete public debt CSV returned by MOF Thailand", source="MOF Thailand")

    # Header determines latest month column (index 1 is latest month)
    headers = rows[0]
    if len(headers) < 2:
        raise ProviderError("Missing date column headers in MOF public debt CSV", source="MOF Thailand")

    latest_month_col = 1
    month_label = _parse_thai_month_header(headers[latest_month_col])

    components_by_num: Dict[int, float] = {}
    debt_to_gdp_pct: Optional[float] = None
    fx_rate: Optional[float] = None

    for row in rows[1:]:
        if len(row) <= latest_month_col:
            continue
        item_name = row[0].strip()
        val_num = _parse_num(row[latest_month_col])
        if val_num is None:
            continue

        # Match top-level line items: "1. หนี้รัฐบาล", "2. หนี้รัฐวิสาหกิจ", ..., "5. หนื้..."
        m = re.match(r"^\s*([1-5])\.(?!\d)", item_name)
        if m:
            c_num = int(m.group(1))
            components_by_num[c_num] = val_num * MILLION

        # Match Debt:GDP
        if re.search(r"debt\s*:\s*gdp", item_name, re.IGNORECASE):
            debt_to_gdp_pct = val_num

        # Match FX rate
        if "อัตราแลกเปลี่ยน" in item_name:
            fx_rate = val_num

    components: List[ThaiPublicDebtComponent] = []
    total_debt = 0.0

    for num in range(1, 6):
        amt = components_by_num.get(num, 0.0)
        total_debt += amt
        en_lbl, th_lbl = COMPONENT_LABELS[num]
        components.append(
            ThaiPublicDebtComponent(
                component_number=num,
                label_en=en_lbl,
                label_th=th_lbl,
                amount_thb=amt,
            )
        )

    return ThaiPublicDebtSnapshot(
        reporting_month=month_label,
        total_debt_thb=total_debt,
        debt_to_gdp_pct=debt_to_gdp_pct,
        fx_rate_usd_thb=fx_rate,
        components=tuple(components),
        fetched_at=time.time(),
        source="MOF Thailand",
    )


class MofThailandAdapter(ThaiPublicDebtPort):
    """Adapter reading monthly Public Debt reports from MOF Thailand."""

    def __init__(self, cache: Optional[ThreadSafeTTLCache] = None):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=MOF_TTL_SECONDS)

    def _fetch_report(self) -> ThaiPublicDebtSnapshot:
        try:
            resp = requests.get(MOF_DEBT_URL, headers=BROWSER_HEADERS, timeout=20)
            resp.raise_for_status()
            return parse_public_debt_csv(resp.text)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Failed to fetch public debt from MOF Thailand: {exc}", source="MOF Thailand") from exc

    def get_public_debt(self) -> ThaiPublicDebtSnapshot:
        cache_key = "mof_th:public_debt:latest"
        return self._cache.get_or_set(
            key=cache_key,
            loader=self._fetch_report,
            ttl_seconds=MOF_TTL_SECONDS,
            max_stale_seconds=MOF_MAX_STALE_SECONDS,
        )

    def as_macro_observables(self, snapshot: Optional[ThaiPublicDebtSnapshot] = None) -> list[MarketObservable]:
        """Convert public debt snapshot into MarketObservable records for macro evaluation."""
        if snapshot is None:
            try:
                snapshot = self.get_public_debt()
            except Exception as e:
                logger.warning("Could not fetch MOF public debt for observables: %s", e)
                return []

        obs_date = snapshot.reporting_month
        observables: list[MarketObservable] = []

        # 1. Total Public Debt
        observables.append(
            MarketObservable(
                observable_id="obs_th_public_debt_mof",
                asset_bucket="cash",
                region="Thailand",
                indicator="Thailand Public Debt Total",
                value=f"{snapshot.total_debt_thb / MILLION:,.2f}",
                unit="Million THB",
                observed_at=obs_date,
                source_file="MOF_Public_Debt_CSV",
                provider="MOF Thailand",
                confidence="high",
                is_valid=True,
                status="verified",
                metadata={
                    "val": snapshot.total_debt_thb / MILLION,
                    "total_debt_thb": snapshot.total_debt_thb,
                    "fx_rate_usd_thb": snapshot.fx_rate_usd_thb,
                    "reporting_month": snapshot.reporting_month,
                },
            )
        )

        # 2. Debt to GDP %
        if snapshot.debt_to_gdp_pct is not None:
            observables.append(
                MarketObservable(
                    observable_id="obs_th_debt_to_gdp_mof",
                    asset_bucket="cash",
                    region="Thailand",
                    indicator="Thailand Debt to GDP Ratio",
                    value=f"{snapshot.debt_to_gdp_pct:.2f}",
                    unit="%",
                    observed_at=obs_date,
                    source_file="MOF_Public_Debt_CSV",
                    provider="MOF Thailand",
                    confidence="high",
                    is_valid=True,
                    status="verified",
                    metadata={
                        "val": snapshot.debt_to_gdp_pct,
                        "statutory_limit_pct": 70.0,
                        "is_within_limit": snapshot.debt_to_gdp_pct <= 70.0,
                    },
                )
            )

        return observables
