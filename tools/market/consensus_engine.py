"""Institutional Best-Effort Consensus & Expectation Gap Engine (Phase 2 & v3.1).

Probes runtime analyst consensus capabilities (EPS revisions, growth forecasts),
enforces horizon matching for Expectation Gap against Reverse DCF implied metrics,
and applies graceful sparse-coverage degradation without synthetic proxies.
"""
import time
from typing import Any, Dict, Literal, Optional, Tuple

import pandas as pd
import yfinance as yf
from langsmith import traceable

from core.logger import get_logger
from core.retry import with_retry as _with_retry
from schemas.micro_quant_schemas import DataStatus

log = get_logger(__name__)

_CONSENSUS_CACHE: Dict[str, Tuple[Optional[dict], float]] = {}
_CONSENSUS_ERROR_CACHE: Dict[str, float] = {}
_CONSENSUS_SUCCESS_TTL_SECONDS = 6 * 3600
_CONSENSUS_ERROR_TTL_SECONDS = 60.0


def probe_consensus_runtime(provider_symbol: str) -> Tuple[Optional[dict], list[str]]:
    """Probes YFinance consensus tables (eps_revisions, eps_trend, growth_estimates).

    Returns:
        tuple[Optional[dict], list[str]]: (consensus_payload, quality_flags)
    """
    flags: list[str] = []
    try:
        tk = yf.Ticker(provider_symbol)
        revisions = _with_retry(lambda: tk.eps_revisions)
        trend = _with_retry(lambda: tk.eps_trend)
        growth = _with_retry(lambda: getattr(tk, "growth_estimates", None))
    except Exception as e:
        log.warning("probe_consensus_runtime: yfinance fetch failed for %s: %s", provider_symbol, e)
        return None, ["consensus_fetch_error"]

    data: dict[str, Any] = {
        "provider_symbol": provider_symbol,
        "revisions_0y": None,
        "trend_0y": None,
        "revenue_growth_next_year_pct": None,
        "earnings_growth_next_5y_pct": None,
        "analyst_count": 0,
        "coverage_level": "zero",
    }

    # 1. EPS Revisions
    if revisions is not None and not revisions.empty and "0y" in revisions.index:
        rev_row = revisions.loc["0y"]
        up_30d = rev_row.get("upLast30days")
        down_30d = rev_row.get("downLast30days")
        up_int = int(up_30d) if up_30d is not None and pd.notna(up_30d) else 0
        down_int = int(down_30d) if down_30d is not None and pd.notna(down_30d) else 0
        data["revisions_0y"] = {
            "up_last_30d": up_int,
            "down_last_30d": down_int,
            "net_30d": up_int - down_int,
        }
        data["analyst_count"] = max(data["analyst_count"], up_int + down_int)

    # 2. EPS Trend
    if trend is not None and not trend.empty and "0y" in trend.index:
        trend_row = trend.loc["0y"]
        curr_est = trend_row.get("current")
        ago_30d = trend_row.get("30daysAgo")
        curr_f = float(curr_est) if curr_est is not None and pd.notna(curr_est) else None
        ago_f = float(ago_30d) if ago_30d is not None and pd.notna(ago_30d) else None
        
        change_pct = None
        if curr_f is not None and ago_f is not None and ago_f != 0:
            change_pct = round(((curr_f - ago_f) / abs(ago_f)) * 100.0, 2)

        data["trend_0y"] = {
            "current_estimate": curr_f,
            "estimate_30d_ago": ago_f,
            "change_30d_pct": change_pct,
        }

    # 3. Growth Estimates (Horizon matching)
    if growth is not None and not growth.empty:
        # Check Next Year revenue/earnings growth
        if "+1y" in growth.index:
            g_1y = growth.loc["+1y"].get(provider_symbol)
            if g_1y is not None and pd.notna(g_1y):
                try:
                    data["revenue_growth_next_year_pct"] = float(g_1y) * 100.0 if float(g_1y) < 2.0 else float(g_1y)
                except (ValueError, TypeError):
                    pass

    if data["analyst_count"] >= 5:
        data["coverage_level"] = "high"
    elif data["analyst_count"] >= 2:
        data["coverage_level"] = "low"
    else:
        data["coverage_level"] = "sparse"
        flags.append("sparse_analyst_coverage")

    return data, flags


def compute_horizon_matched_expectation_gap(
    consensus_data: Optional[dict],
    market_implied_growth_pct: Optional[float],
) -> Tuple[Optional[float], Optional[float], DataStatus, list[str]]:
    """Calculates Horizon-Matched Expectation Gap between Market Implied Growth & Consensus.

    Invariant:
        Calculated ONLY when consensus revenue growth matches horizon.
        If unavailable, returns status='unavailable' with eps_momentum preserved.

    Returns:
        tuple: (expectation_gap_pct, eps_momentum_score, status, flags)
    """
    flags: list[str] = []
    if consensus_data is None:
        return None, None, "unavailable", ["missing_consensus_data"]

    # 1. 30D EPS Momentum Score (0 - 100)
    eps_momentum_score = None
    trend = consensus_data.get("trend_0y")
    if trend and trend.get("change_30d_pct") is not None:
        chg = trend["change_30d_pct"]
        # Linear scale: -5% -> 0, +5% -> 100
        score = 50.0 + (chg / 5.0) * 50.0
        eps_momentum_score = max(0.0, min(100.0, round(score, 1)))

    # 2. Horizon-Matched Expectation Gap
    consensus_growth = consensus_data.get("revenue_growth_next_year_pct")
    if consensus_growth is not None and market_implied_growth_pct is not None:
        gap = round(market_implied_growth_pct - consensus_growth, 2)
        status: DataStatus = "available"
        return gap, eps_momentum_score, status, flags
    else:
        flags.append("unmatched_consensus_horizon:expectation_gap_unavailable")
        return None, eps_momentum_score, "partial" if eps_momentum_score is not None else "unavailable", flags
