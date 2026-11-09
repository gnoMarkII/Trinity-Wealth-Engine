"""Earnings Momentum & Revisions — ติดตามการปรับประมาณการกำไรของนักวิเคราะห์ (EPS Revisions/Trend)

ใช้ yfinance Ticker.eps_revisions + Ticker.eps_trend (verified ว่ามีให้ฟรีจริง) — โฟกัสที่ '0y'
(ปีบัญชีปัจจุบัน) เพราะเป็น signal ที่ actionable สุดสำหรับการตัดสินใจระยะสั้น-กลาง

ไม่รวมใน composite_score — เป็น contextual indicator แยกต่างหาก (ดู schemas/micro_quant_schemas.py)
"""
import time
from typing import Dict, Optional, Tuple

import pandas as pd
import yfinance as yf
from langsmith import traceable

from core.logger import get_logger
from core.retry import with_retry as _with_retry
from .quant_scoring import _linear_score

log = get_logger(__name__)

_FOCUS_PERIOD = "0y"  # ปีบัญชีปัจจุบัน

_REVISION_CACHE: Dict[str, Tuple[Optional[dict], float]] = {}
_REVISION_ERROR_CACHE: Dict[str, float] = {}
_REVISION_SUCCESS_TTL_SECONDS = 6 * 3600
_REVISION_ERROR_TTL_SECONDS = 60.0


def _fetch_earnings_revision_raw(provider_symbol: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        tk = yf.Ticker(provider_symbol)
        revisions = _with_retry(lambda: tk.eps_revisions)
        trend = _with_retry(lambda: tk.eps_trend)
    except Exception as e:
        log.warning("fetch_earnings_revision_data: fetch failed for %s: %s", provider_symbol, e)
        return None, "fetch_error:earnings_momentum"

    if revisions is None or revisions.empty or _FOCUS_PERIOD not in revisions.index:
        return None, "missing_earnings_revision_data:earnings_momentum"
    if trend is None or trend.empty or _FOCUS_PERIOD not in trend.index:
        return None, "missing_earnings_revision_data:earnings_momentum"

    rev_row = revisions.loc[_FOCUS_PERIOD]
    trend_row = trend.loc[_FOCUS_PERIOD]

    def _safe_num(row, key, cast):
        val = row.get(key)
        return cast(val) if val is not None and pd.notna(val) else None

    data = {
        "up_last_30d": _safe_num(rev_row, "upLast30days", int),
        "down_last_30d": _safe_num(rev_row, "downLast30days", int),
        "estimate_current": _safe_num(trend_row, "current", float),
        "estimate_30d_ago": _safe_num(trend_row, "30daysAgo", float),
    }
    try:
        from tools.market.consensus_engine import probe_consensus_runtime

        consensus_payload, _consensus_flags = probe_consensus_runtime(
            provider_symbol,
            ticker_obj=tk,
            revisions=revisions,
            trend=trend,
        )
        if isinstance(consensus_payload, dict):
            if consensus_payload.get("revenue_growth_next_year_pct") is not None:
                data["revenue_growth_next_year_pct"] = consensus_payload["revenue_growth_next_year_pct"]
            if consensus_payload.get("analyst_count") is not None:
                data["analyst_count"] = consensus_payload["analyst_count"]
            if consensus_payload.get("coverage_level") is not None:
                data["coverage_level"] = consensus_payload["coverage_level"]
            if consensus_payload.get("trend_0y") is not None:
                data["trend_0y"] = consensus_payload["trend_0y"]
    except Exception as e:
        log.debug("probe_consensus_runtime enrichment skipped for %s: %s", provider_symbol, e)
    return data, None


@traceable(run_type="tool")
def fetch_earnings_revision_data(provider_symbol: str) -> Tuple[Optional[dict], Optional[str]]:
    """ดึง EPS revision/trend ของปีบัญชีปัจจุบัน พร้อม TTL cache (success 6h / error 60s)

    Returns:
        (data, flag) — data มี keys: up_last_30d, down_last_30d, estimate_current, estimate_30d_ago
        (ทุก key อาจเป็น None แยกกันได้ถ้า yfinance ไม่มีข้อมูลบางส่วน)
    """
    now = time.time()
    if provider_symbol in _REVISION_CACHE:
        data, ts = _REVISION_CACHE[provider_symbol]
        if now - ts < _REVISION_SUCCESS_TTL_SECONDS:
            return data, None
        del _REVISION_CACHE[provider_symbol]

    if provider_symbol in _REVISION_ERROR_CACHE:
        ts = _REVISION_ERROR_CACHE[provider_symbol]
        if now - ts < _REVISION_ERROR_TTL_SECONDS:
            return None, "fetch_error:earnings_momentum"
        del _REVISION_ERROR_CACHE[provider_symbol]

    data, flag = _fetch_earnings_revision_raw(provider_symbol)
    if flag is not None:
        _REVISION_ERROR_CACHE[provider_symbol] = now
        return None, flag

    _REVISION_CACHE[provider_symbol] = (data, now)
    return data, None


@traceable(run_type="parser")
def compute_earnings_revision_score(
    data: Optional[dict],
) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[str]]:
    """คำนวณ net_revisions (up-down ใน 30 วัน) + estimate_change_pct (drift ของ EPS estimate 30 วัน) + score

    Threshold: estimate_change_pct >=5% → 100, <=-5% → 0 (เชิงเส้น) — ให้น้ำหนักที่ทิศทางการปรับ
    ประมาณการมากกว่าจำนวนนักวิเคราะห์ดิบ เพราะแปลความหมายตรงไปตรงมากว่า

    Returns:
        (eps_revision_net_30d, eps_estimate_change_30d_pct, earnings_momentum_score, flag)
    """
    if data is None:
        return None, None, None, "missing_earnings_revision_data:earnings_momentum"

    up = data.get("up_last_30d")
    down = data.get("down_last_30d")
    net_revisions = (up - down) if (up is not None and down is not None) else None

    current = data.get("estimate_current")
    ago = data.get("estimate_30d_ago")
    estimate_change_pct = None
    if current is not None and ago is not None and ago != 0:
        estimate_change_pct = round((current - ago) / abs(ago) * 100, 2)

    if estimate_change_pct is None:
        return net_revisions, None, None, "missing_earnings_revision_data:earnings_momentum"

    score = _linear_score(estimate_change_pct, best=5, worst=-5)
    score = round(score, 1) if score is not None else None

    return net_revisions, estimate_change_pct, score, None
