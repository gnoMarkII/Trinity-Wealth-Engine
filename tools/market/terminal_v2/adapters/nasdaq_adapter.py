"""HTTP Adapter for Nasdaq Equity Intelligence (Surprise, Calendar, Consensus).

Source: api.nasdaq.com
Exchange internal keyless API with coverage verification and estimated/confirmed date tracking.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests

from core.logger import get_logger
from tools.market.terminal_v2.application.cache import TerminalTtlCache
from tools.market.terminal_v2.domain.models import (
    AnalystRatingConsensus,
    EarningsDateItem,
    EarningsSurpriseItem,
    NasdaqEarningsConsensusSnapshot,
)
from tools.market.terminal_v2.ports.driven_ports import NasdaqEquityIntelligencePort

logger = get_logger(__name__)

BASE_URL = "https://api.nasdaq.com"
DESKTOP_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).replace("$", "").replace("%", "").replace(",", "").strip()
    if s in ("", "--", "N/A", "null"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_us_date_to_iso(date_str: str) -> str:
    """Parse US date '8/26/2026' or '2/25/2026' into ISO '2026-08-26'."""
    s = date_str.strip()
    if not s:
        return ""
    try:
        parts = s.split("/")
        if len(parts) == 3:
            m, d, y = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{y:04d}-{m:02d}-{d:02d}"
    except Exception:
        pass
    return s


class NasdaqHttpAdapter(NasdaqEquityIntelligencePort):
    """Fetches and parses earnings surprise history and analyst ratings from Nasdaq."""

    def __init__(
        self,
        cache: Optional[TerminalTtlCache] = None,
        surprise_fixture_path: Optional[Path] = None,
        ratings_fixture_path: Optional[Path] = None,
        calendar_fixture_path: Optional[Path] = None,
        timeout: int = 15,
    ) -> None:
        self._cache = cache or TerminalTtlCache(
            default_ttl_seconds=14400,  # 4 hours
            max_stale_seconds=86400 * 2,
            max_entries=200,
        )
        self._surprise_fixture = surprise_fixture_path
        self._ratings_fixture = ratings_fixture_path
        self._calendar_fixture = calendar_fixture_path
        self._timeout = timeout

    def fetch_earnings_consensus(self, symbol: str) -> NasdaqEarningsConsensusSnapshot:
        clean_sym = symbol.upper().strip()
        cache_key = f"nasdaq:consensus:{clean_sym}"

        def _fetch() -> NasdaqEarningsConsensusSnapshot:
            surprise_data = self._get_payload(
                endpoint=f"/api/company/{clean_sym}/earnings-surprise",
                fixture=self._surprise_fixture,
            )
            ratings_data = self._get_payload(
                endpoint=f"/api/analyst/{clean_sym}/ratings",
                fixture=self._ratings_fixture,
            )

            return self._build_snapshot(clean_sym, surprise_data, ratings_data)

        return self._cache.get_or_compute(cache_key, _fetch, ttl_seconds=14400)

    def _get_payload(self, endpoint: str, fixture: Optional[Path]) -> Optional[Dict[str, Any]]:
        if fixture and fixture.exists():
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to read Nasdaq fixture {fixture}: {e}")
                return None

        url = f"{BASE_URL}{endpoint}"
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": DESKTOP_UA, "Accept": "application/json"},
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception as e:
            logger.warning(f"Nasdaq request error for {endpoint}: {e}")
            return None

    def _build_snapshot(
        self,
        symbol: str,
        surprise_payload: Optional[Dict[str, Any]],
        ratings_payload: Optional[Dict[str, Any]],
    ) -> NasdaqEarningsConsensusSnapshot:
        # 1. Parse Earnings Surprises
        surprises: List[EarningsSurpriseItem] = []
        if surprise_payload and isinstance(surprise_payload.get("data"), dict):
            s_data = surprise_payload["data"]
            table = s_data.get("earningsSurpriseTable") or {}
            rows = table.get("rows") or []
            for r in rows:
                eps_val = _safe_float(r.get("eps"))
                cons_val = _safe_float(r.get("consensusForecast"))
                surp_val = _safe_float(r.get("percentageSurprise"))
                date_rep = _parse_us_date_to_iso(r.get("dateReported", ""))

                if eps_val is not None and cons_val is not None:
                    surprises.append(
                        EarningsSurpriseItem(
                            fiscal_quarter_end=str(r.get("fiscalQtrEnd", "")).strip(),
                            date_reported=date_rep,
                            eps=eps_val,
                            consensus_eps=cons_val,
                            surprise_pct=surp_val if surp_val is not None else 0.0,
                        )
                    )

        # 2. Parse Analyst Ratings
        ratings_obj: Optional[AnalystRatingConsensus] = None
        if ratings_payload and isinstance(ratings_payload.get("data"), dict):
            r_data = ratings_payload["data"]
            consensus = str(r_data.get("meanRatingType") or "").strip()
            summary_str = str(r_data.get("ratingsSummary") or "")
            count_match = re.search(r"Based on (\d+) analysts", summary_str)
            analyst_count = int(count_match.group(1)) if count_match else 0
            brokers = tuple(str(b).strip() for b in (r_data.get("brokerNames") or []))

            if consensus:
                ratings_obj = AnalystRatingConsensus(
                    symbol=symbol,
                    consensus=consensus,
                    analyst_count=analyst_count,
                    broker_names=brokers,
                )

        has_surprise = len(surprises) > 0
        has_ratings = ratings_obj is not None

        # Determine Coverage Status
        if has_surprise and has_ratings:
            coverage_status = "full"
        elif has_surprise or has_ratings:
            coverage_status = "partial"
        else:
            coverage_status = "no_coverage"

        return NasdaqEarningsConsensusSnapshot(
            symbol=symbol,
            coverage_status=coverage_status,
            has_earnings_surprise=has_surprise,
            has_analyst_ratings=has_ratings,
            upcoming_earnings=None,
            surprise_history=tuple(surprises),
            ratings=ratings_obj,
            source="Nasdaq",
            is_stale=False,
        )
