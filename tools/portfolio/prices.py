from langsmith import traceable
import concurrent.futures
import csv
import json
import os
import re
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import frontmatter
import yfinance as yf
from filelock import FileLock, Timeout
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.logger import get_logger

log = get_logger(__name__)

_USDTHB_TICKER = "USDTHB=X"


from tools._atomic_io import _atomic_write_to
from tools.tool_errors import LOCK_TIMEOUT, validation_error
from .core import _load_or_init, _save, _recalc_all, _recalc_holding, _recalc_summary, _find_holding, _require_cash, _require_fx, get_portfolio_state, _holding_currency, compute_allocation_breakdown
from .models import _now_iso, _coerce_iso_string, Holding, Summary, PortfolioState, WatchlistItem, WatchlistState, GoalItem, GoalsState


from .constants import *

CASH_THB_SYMBOL = "CASH_THB"
CASH_USD_SYMBOL = "CASH_USD"
_CASH_SYMBOLS = (CASH_THB_SYMBOL, CASH_USD_SYMBOL)
# Back-compat alias — call sites and tests still reference CASH_SYMBOL
CASH_SYMBOL = CASH_THB_SYMBOL

_FLOAT_EPS = 1e-6
_MONEY_DP = 2
_COST_DP = 6
_PCT_DP = 2

_LOCK_TIMEOUT = 15  # seconds — wait up to 15s for another process to release
_PRICE_FETCH_TIMEOUT = 6  # seconds per symbol when refreshing

_PORTFOLIO_LOCK_PATH = str(PORTFOLIO_PATH) + ".lock"
_portfolio_lock = FileLock(_PORTFOLIO_LOCK_PATH, timeout=_LOCK_TIMEOUT)



def _yf_symbol(symbol: str, currency: str) -> str:
    """แปลง symbol → ticker ที่ yfinance รู้จัก (THB → เติม .BK)"""
    if currency == "THB" and not symbol.endswith(".BK"):
        return f"{symbol}.BK"
    return symbol


def _fetch_last_price(symbol: str) -> float | None:
    """ดึง last_price จาก yfinance — คืน None ถ้า fail"""
    try:
        tk = yf.Ticker(symbol)
        fi = tk.fast_info
        last = getattr(fi, "last_price", None)
        if last is not None:
            return float(last)
        hist = tk.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        log.warning("fetch price failed for %s: %s", symbol, e)
    return None


def _fetch_fx_rate() -> float | None:
    """ดึง USD/THB exchange rate ล่าสุดจาก yfinance (ticker USDTHB=X)
    Pattern เดียวกับ _fetch_last_price: fast_info ก่อน → history fallback
    คืน None ถ้าดึงไม่ได้ (log warning อัตโนมัติ)
    """
    return _fetch_last_price(_USDTHB_TICKER)


def fetch_latest_price(symbol: str, currency: Literal["THB", "USD"]) -> float | None:
    """Public helper: ดึงราคาล่าสุดจาก yfinance (แปลง symbol THB → .BK ให้)

    Args:
        symbol: ticker เช่น 'AAPL', 'PTT' (ห้ามมี .BK suffix สำหรับ THB — ระบบเติมให้)
        currency: 'THB' (เติม .BK) หรือ 'USD' (ใช้ symbol ตรงๆ) — *required*
                  ไม่มี default เพื่อกัน silent fail กับหุ้นไทยที่ต้องเติม .BK

    Returns:
        last_price (float) หรือ None ถ้าดึงไม่ได้ (log warning อัตโนมัติ)
    """
    if currency not in ("THB", "USD"):
        raise ValueError(f"currency ต้องเป็น 'THB' หรือ 'USD' (got '{currency}')")
    return _fetch_last_price(_yf_symbol(symbol, currency))


def _refresh_prices(state: PortfolioState) -> dict[str, str]:
    """Refresh current_price_* ของทุก holding ที่ไม่ใช่ Cash — best-effort

    Returns: dict ของ {symbol: status_msg} สำหรับ logging/observability
    """
    targets: list[tuple[Holding, str, str]] = []
    for h in state.holdings:
        if h.asset_type == "Cash":
            continue
        if h.avg_cost_usd is not None:
            targets.append((h, h.symbol, "USD"))
        elif h.avg_cost_thb is not None:
            targets.append((h, h.symbol, "THB"))

    if not targets:
        return {}

    results: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(targets))) as ex:
        future_map = {
            ex.submit(_fetch_last_price, _yf_symbol(sym, cur)): (h, sym, cur)
            for h, sym, cur in targets
        }
        try:
            for future in concurrent.futures.as_completed(
                future_map, timeout=_PRICE_FETCH_TIMEOUT * 2
            ):
                h, sym, cur = future_map[future]
                try:
                    price = future.result()
                except Exception as e:
                    results[sym] = f"error: {e}"
                    continue
                if price is None:
                    results[sym] = "no_data"
                    continue
                if cur == "USD":
                    h.current_price_usd = price
                else:
                    h.current_price_thb = price
                results[sym] = "ok"
        except concurrent.futures.TimeoutError:
            for f, (h, sym, cur) in future_map.items():
                if not f.done():
                    results.setdefault(sym, "timeout")
    return results


def _fetch_fundamentals(state: PortfolioState, force: bool = False) -> dict[str, str]:
    """ดึงข้อมูล Fundamentals (PE, EPS, Payout, MarketCap, Dividend, LongName) จาก yfinance.info

    มี TTL Cache (FUNDAMENTALS_TTL_SECONDS) เก็บ timestamp ใน h.fundamentals_updated_at
    แยกจาก _refresh_prices เพื่อไม่ให้กระทบ hot path / timeout
    """
    targets: list[tuple[Holding, str, str]] = []
    now = time.time()
    for h in state.holdings:
        if h.asset_type == "Cash":
            continue
        if not force and h.fundamentals_updated_at is not None:
            if now - h.fundamentals_updated_at < FUNDAMENTALS_TTL_SECONDS:
                continue
        if h.avg_cost_usd is not None or h.current_price_usd is not None:
            targets.append((h, _yf_symbol(h.symbol, "USD"), "USD"))
        else:
            targets.append((h, _yf_symbol(h.symbol, "THB"), "THB"))

    if not targets:
        return {}

    results: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(targets))) as ex:
        def _get_info(sym: str) -> dict | None:
            try:
                tk = yf.Ticker(sym)
                info = getattr(tk, "info", None)
                return info if isinstance(info, dict) else None
            except Exception as e:
                log.warning("fetch fundamentals failed for %s: %s", sym, e)
                return None

        future_map = {ex.submit(_get_info, yf_sym): (h, yf_sym) for h, yf_sym, _ in targets}
        try:
            for future in concurrent.futures.as_completed(future_map, timeout=_PRICE_FETCH_TIMEOUT * 4):
                h, yf_sym = future_map[future]
                try:
                    info = future.result()
                except Exception as e:
                    results[h.symbol] = f"error: {e}"
                    continue

                if not info:
                    results[h.symbol] = "no_data"
                    h.fundamentals_updated_at = time.time()
                    continue

                pe = info.get("trailingPE") or info.get("peRatio")
                eps = info.get("trailingEps") or info.get("epsTrailingTwelveMonths")
                payout = info.get("payoutRatio")
                mcap = info.get("marketCap")
                div_rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
                div_yield = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
                long_name = info.get("longName") or info.get("shortName")

                if pe is not None and pe > 0:
                    setattr(h, "pe_ratio", float(pe))
                if eps is not None:
                    setattr(h, "eps", float(eps))
                if payout is not None and payout > 0:
                    setattr(h, "payout_ratio", float(payout * 100))
                if mcap is not None and mcap > 0:
                    setattr(h, "market_cap_value", float(mcap))
                if div_rate is not None and div_rate >= 0:
                    setattr(h, "dividend_per_share", float(div_rate))
                if div_yield is not None and div_yield >= 0:
                    setattr(h, "dividend_yield", float(div_yield * 100))
                if long_name and isinstance(long_name, str):
                    setattr(h, "company_name", long_name)

                h.fundamentals_updated_at = time.time()
                results[h.symbol] = "ok"
        except concurrent.futures.TimeoutError:
            for f, (h, _) in future_map.items():
                if not f.done():
                    results.setdefault(h.symbol, "timeout")
    return results


@tool
def sync_market_prices() -> str:
    """ดึงราคาตลาดล่าสุดของทุกสินทรัพย์ในพอร์ตโฟลิโอ

    [Usage/When to use]
    ใช้เพื่ออัปเดตราคาตลาดปัจจุบันและคำนวณ NAV + Unrealized P/L ของพอร์ตใหม่ทั้งหมด
    - ดึงราคาจาก yfinance ให้กับทุก Holding (ยกเว้น Cash)

    [Caution]
    - อาจใช้เวลาสักพักหากมีสินทรัพย์จำนวนมาก

    Returns:
        str: สรุปผลการอัปเดตราคาตลาด
    """
    try:
        with _portfolio_lock:
            post, state = _load_or_init()
            nav_before = state.summary.total_value_thb
            refresh_info = _refresh_prices(state)
            _save(post, state)
            nav_after = state.summary.total_value_thb
            unrealized_after = state.summary.total_unrealized_profit
    except Timeout:
        return LOCK_TIMEOUT.format(detail=f"portfolio lock {_LOCK_TIMEOUT}s")
    except ValueError as e:
        return f"Error: {e}"

    total = len(refresh_info)
    if total == 0:
        return f"[SYNC] | no non-cash holdings | NAV: {nav_after:,.2f} THB"

    ok_count = sum(1 for s in refresh_info.values() if s == "ok")
    issues = {sym: s for sym, s in refresh_info.items() if s != "ok"}
    issue_note = ""
    if issues:
        sample = ", ".join(f"{s}={st}" for s, st in list(issues.items())[:3])
        more = f" +{len(issues) - 3} more" if len(issues) > 3 else ""
        issue_note = f" [issues: {sample}{more}]"

    return (
        f"[SYNC] | refreshed {ok_count}/{total}{issue_note} | "
        f"NAV: {nav_before:,.2f} → {nav_after:,.2f} THB | "
        f"unrealized: {unrealized_after:+,.2f} THB"
    )


