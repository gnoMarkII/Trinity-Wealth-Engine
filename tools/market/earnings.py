"""Shared earnings dates fetch — single-flight lock, in-memory TTL cache (6h),
in-memory burst failure cooldown (10s), and single-point numeric sanitization.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo
import math
import threading
import time
import logging
import pandas as pd
import yfinance as yf
from core.retry import with_retry as _with_retry

log = logging.getLogger(__name__)
EARNINGS_CACHE_TTL = 6 * 3600  # 6 Hours
BURST_FAIL_TTL = 10.0          # 10 Seconds


def finite_or_none(value) -> float | None:
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


@dataclass
class EarningsFetchResult:
    rows: list[dict]          # [{date_str, timestamp_ms, eps_actual, eps_estimate}]
    status: Literal["ok", "empty", "failed"]
    source_as_of: str | None  # ISO datetime string


_CACHE: dict[str, tuple[float, EarningsFetchResult]] = {}
_CACHE_LOCK = threading.Lock()
_KEY_LOCKS: dict[str, threading.Lock] = {}
_BURST_FAIL_CACHE: dict[str, float] = {}


def _get_key_lock(symbol: str) -> threading.Lock:
    with _CACHE_LOCK:
        if symbol not in _KEY_LOCKS:
            _KEY_LOCKS[symbol] = threading.Lock()
        return _KEY_LOCKS[symbol]


def fetch_earnings_dates(provider_symbol: str, tz_name: str) -> EarningsFetchResult:
    """Single-flight earnings fetch พร้อม in-memory TTL cache (6h) และ burst failure cooldown (10s)"""
    now_mono = time.monotonic()
    clean_sym = provider_symbol.strip().upper()

    with _CACHE_LOCK:
        if clean_sym in _CACHE:
            expire_at, cached = _CACHE[clean_sym]
            if now_mono < expire_at:
                return cached
        if clean_sym in _BURST_FAIL_CACHE:
            if now_mono - _BURST_FAIL_CACHE[clean_sym] < BURST_FAIL_TTL:
                return EarningsFetchResult(rows=[], status="failed", source_as_of=None)

    key_lock = _get_key_lock(clean_sym)
    with key_lock:
        with _CACHE_LOCK:
            if clean_sym in _CACHE:
                expire_at, cached = _CACHE[clean_sym]
                if now_mono < expire_at:
                    return cached
            if clean_sym in _BURST_FAIL_CACHE and (now_mono - _BURST_FAIL_CACHE[clean_sym] < BURST_FAIL_TTL):
                return EarningsFetchResult(rows=[], status="failed", source_as_of=None)

        try:
            tk = yf.Ticker(clean_sym)
            ed_df = _with_retry(lambda: tk.get_earnings_dates(limit=24))
            rows = []
            if ed_df is not None and not ed_df.empty:
                for ts, row in ed_df.iterrows():
                    actual = finite_or_none(row.get("Reported EPS"))
                    estimate = finite_or_none(row.get("EPS Estimate"))
                    date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
                    rows.append({
                        "date_str": date_str,
                        "timestamp_ms": int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else 0,
                        "eps_actual": actual,
                        "eps_estimate": estimate,
                    })
            status: Literal["ok", "empty", "failed"] = "ok" if rows else "empty"
            source_as_of = datetime.now(ZoneInfo(tz_name)).isoformat()
            result = EarningsFetchResult(rows=rows, status=status, source_as_of=source_as_of)
            with _CACHE_LOCK:
                _CACHE[clean_sym] = (now_mono + EARNINGS_CACHE_TTL, result)
            return result
        except Exception as e:
            log.warning("Earnings dates fetch failed for %s: %s", clean_sym, e)
            with _CACHE_LOCK:
                _BURST_FAIL_CACHE[clean_sym] = now_mono
            return EarningsFetchResult(rows=[], status="failed", source_as_of=None)
