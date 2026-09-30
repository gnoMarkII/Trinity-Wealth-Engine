"""HTTP Adapter for US OFR Financial Stress Index.

Source: US Office of Financial Research (financialresearch.gov)
Keyless, daily CSV publication with 5 decomposed categories and T-2 business day lag.
"""
import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
import requests

from core.logger import get_logger
from tools.market.terminal_v2.application.cache import TerminalTtlCache
from tools.market.terminal_v2.domain.models import (
    FinancialStressCategory,
    FinancialStressPoint,
    FinancialStressSnapshot,
)
from tools.market.terminal_v2.ports.driven_ports import OfrFinancialStressPort

logger = get_logger(__name__)

OFR_CSV_URL = "https://www.financialresearch.gov/financial-stress-index/data/fsi.csv"
CATEGORY_COLUMNS = [
    (2, "Credit"),
    (3, "Equity valuation"),
    (4, "Safe assets"),
    (5, "Funding"),
    (6, "Volatility"),
]


class OfrHttpAdapter(OfrFinancialStressPort):
    """Fetches and parses the OFR Financial Stress Index from public CSV."""

    def __init__(
        self,
        cache: Optional[TerminalTtlCache] = None,
        fixture_path: Optional[Path] = None,
        timeout: int = 15,
    ) -> None:
        self._cache = cache or TerminalTtlCache(
            default_ttl_seconds=3600,
            max_stale_seconds=86400,
            max_entries=20,
        )
        self._fixture_path = fixture_path
        self._timeout = timeout

    def fetch_financial_stress(self) -> FinancialStressSnapshot:
        cache_key = "ofr:fsi:latest"

        def _fetch() -> FinancialStressSnapshot:
            if self._fixture_path and self._fixture_path.exists():
                text = self._fixture_path.read_text(encoding="utf-8")
            else:
                resp = requests.get(
                    OFR_CSV_URL,
                    timeout=self._timeout,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                resp.raise_for_status()
                text = resp.text

            return self._parse_csv(text)

        return self._cache.get_or_compute(cache_key, _fetch, ttl_seconds=3600)

    def _parse_csv(self, csv_text: str) -> FinancialStressSnapshot:
        reader = csv.reader(io.StringIO(csv_text.strip()))
        rows = list(reader)
        if len(rows) < 2:
            raise ValueError("OFR FSI CSV contains insufficient rows")

        points: List[FinancialStressPoint] = []
        last_row: Optional[List[str]] = None

        for row in rows[1:]:
            if len(row) < 7:
                continue
            date_str = row[0].strip()
            try:
                val = float(row[1])
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                time_ms = int(dt.timestamp() * 1000)

                credit = float(row[2]) if row[2] else None
                equity_val = float(row[3]) if row[3] else None
                safe_assets = float(row[4]) if row[4] else None
                funding = float(row[5]) if row[5] else None
                volatility = float(row[6]) if row[6] else None

                point = FinancialStressPoint(
                    time_ms=time_ms,
                    date=date_str,
                    value=val,
                    credit=credit,
                    equity_valuation=equity_val,
                    safe_assets=safe_assets,
                    funding=funding,
                    volatility=volatility,
                )
                points.append(point)
                last_row = row
            except (ValueError, IndexError):
                continue

        if not points or not last_row:
            raise ValueError("No valid data points found in OFR CSV")

        latest = points[-1]
        categories: List[FinancialStressCategory] = []
        for col_idx, label in CATEGORY_COLUMNS:
            try:
                cat_val = float(last_row[col_idx])
            except (ValueError, IndexError):
                cat_val = 0.0
            categories.append(FinancialStressCategory(label=label, value=cat_val))

        trend_90d = tuple(points[-90:])
        published_now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        return FinancialStressSnapshot(
            as_of_date=latest.date,
            published_at=published_now,
            fsi_value=latest.value,
            categories=tuple(categories),
            trend_90d=trend_90d,
            source="OFR",
            data_lag_days=2,
            is_stale=False,
        )
