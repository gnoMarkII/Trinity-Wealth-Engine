from langsmith import traceable
import concurrent.futures
import csv
import json
import os
import re
import tempfile
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

def _compute_total_cost(state, current_fx: float) -> float:
    total = 0.0
    for h in state.holdings:
        if h.asset_type == "Cash":
            if h.symbol == "CASH_USD":
                total += h.units * current_fx
            else:
                total += h.units
            continue
        if h.avg_cost_usd is not None and h.current_price_usd is not None:
            total += h.units * h.avg_cost_usd * current_fx
        elif h.avg_cost_thb is not None and h.current_price_thb is not None:
            total += h.units * h.avg_cost_thb
    return round(total, 2)

from tools._atomic_io import _atomic_write_to
from tools.tool_errors import LOCK_TIMEOUT, validation_error
from .models import _now_iso, _coerce_iso_string, Holding, Summary, PortfolioState, WatchlistItem, WatchlistState, GoalItem, GoalsState
from .core import _load_or_init, _save, _recalc_all, _recalc_holding, _recalc_summary, _find_holding, _require_cash, _require_fx, get_portfolio_state, _holding_currency, compute_allocation_breakdown
from .prices import fetch_latest_price, _fetch_last_price, _fetch_fx_rate, _refresh_prices, sync_market_prices
from .journal import append_trading_journal, _inject_journal_wikilinks, _write_journal_entry


from .constants import *
from .constants import _PERFORMANCE_LOG_HEADER

CASH_THB_SYMBOL = "CASH_THB"
CASH_USD_SYMBOL = "CASH_USD"
_CASH_SYMBOLS = (CASH_THB_SYMBOL, CASH_USD_SYMBOL)
# Back-compat alias — call sites and tests still reference CASH_SYMBOL
CASH_SYMBOL = CASH_THB_SYMBOL

_FLOAT_EPS = 1e-6
_MONEY_DP = 2
_COST_DP = 4
_PCT_DP = 2

_LOCK_TIMEOUT = 15  # seconds — wait up to 15s for another process to release
_PRICE_FETCH_TIMEOUT = 6  # seconds per symbol when refreshing

_PORTFOLIO_LOCK_PATH = str(PORTFOLIO_PATH) + ".lock"
_portfolio_lock = FileLock(_PORTFOLIO_LOCK_PATH, timeout=_LOCK_TIMEOUT)



@tool
def record_performance_snapshot(refresh_prices: bool = True) -> str:
    """บันทึก Snapshot สถานะพอร์ตโฟลิโอ ณ สิ้นวัน (Performance Logging)

    [Usage/When to use]
    ใช้บันทึกประวัติการเติบโตของพอร์ตประจำวัน (Time-series) ลงใน CSV
    - บันทึก Date, Total_NAV, Total_Cost, Unrealized_PnL, Cash_Balance

    [Caution]
    - ควรเรียกใช้งานแค่วันละครั้งเพื่อป้องกันข้อมูลซ้ำซ้อนเกินไป

    Args:
        refresh_prices (bool): True (อัปเดตราคาล่าสุดก่อนบันทึก, default)
    """
    try:
        with _portfolio_lock:
            post, state = _load_or_init()
            if refresh_prices:
                _refresh_prices(state)
                _save(post, state)
            else:
                _recalc_all(state)

            current_fx = _require_fx(state)
            total_nav = state.summary.total_value_thb
            unrealized = state.summary.total_unrealized_profit
            total_cost = _compute_total_cost(state, current_fx)

            # Cash_Balance ต้องรวมทั้ง CASH_THB และ CASH_USD ใน THB equivalent
            # ไม่งั้น Cash_Balance + non_cash_assets ≠ Total_NAV (drift)
            cash_balance = round(
                sum(h.market_value_thb for h in state.holdings if h.asset_type == "Cash"),
                _MONEY_DP,
            )
    except Timeout:
        return LOCK_TIMEOUT.format(detail=f"portfolio lock {_LOCK_TIMEOUT}s")
    except ValueError as e:
        return f"Error: {e}"

    today = datetime.now().strftime("%Y-%m-%d")
    row = [
        today,
        f"{total_nav:.2f}",
        f"{total_cost:.2f}",
        f"{unrealized:.2f}",
        f"{cash_balance:.2f}",
    ]

    PERFORMANCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    needs_header = (
        not PERFORMANCE_LOG_PATH.exists() or PERFORMANCE_LOG_PATH.stat().st_size == 0
    )
    with PERFORMANCE_LOG_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(_PERFORMANCE_LOG_HEADER)
        writer.writerow(row)

    return (
        f"[PERF] {today} | NAV: {total_nav:,.2f} | "
        f"Cost: {total_cost:,.2f} | PnL: {unrealized:+,.2f} | "
        f"Cash: {cash_balance:,.2f}"
    )


@tool
def read_performance_history(days: int = 30) -> str:
    """อ่านประวัติและวิเคราะห์ผลตอบแทนของพอร์ตโฟลิโอ (Performance Analytics)

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์การเติบโต NAV, Drawdown, หรือผลตอบแทนย้อนหลัง
    - ระบบจะคำนวณ metrics เช่น P&L, Drawdown ให้อัตโนมัติ

    Args:
        days (int): จำนวนวันย้อนหลังที่ต้องการวิเคราะห์ (default 30)

    Returns:
        str: สรุปผลตอบแทนและ Metrics ในรูปแบบ Markdown
    """

    if days <= 0:
        return validation_error("days ต้องมากกว่า 0")

    if not PERFORMANCE_LOG_PATH.exists():
        return json.dumps(
            {"error": "ยังไม่มี Performance_Log.csv — ใช้ record_performance_snapshot ก่อน"},
            ensure_ascii=False,
        )

    with PERFORMANCE_LOG_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if not all_rows:
        return json.dumps(
            {"error": "Performance_Log.csv ว่างเปล่า"},
            ensure_ascii=False,
        )

    rows = all_rows[-days:]

    navs = [float(r["Total_NAV"]) for r in rows]
    first_nav = navs[0]
    latest_nav = navs[-1]
    change_abs = latest_nav - first_nav
    change_pct = (change_abs / first_nav * 100) if first_nav > 0 else 0.0

    # Max drawdown: running peak จาก left → right, เทียบ current value
    peak = navs[0]
    max_dd = 0.0
    for nav in navs:
        if nav > peak:
            peak = nav
        dd = ((nav - peak) / peak * 100) if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    return json.dumps(
        {
            "window_days": days,
            "n_observations": len(rows),
            "first_date": rows[0]["Date"],
            "latest_date": rows[-1]["Date"],
            "first_nav": round(first_nav, _MONEY_DP),
            "latest_nav": round(latest_nav, _MONEY_DP),
            "change_abs": round(change_abs, _MONEY_DP),
            "change_pct": round(change_pct, _PCT_DP),
            "max_nav": round(max(navs), _MONEY_DP),
            "min_nav": round(min(navs), _MONEY_DP),
            "max_drawdown_pct": round(max_dd, _PCT_DP),
            "rows": rows,
        },
        ensure_ascii=False,
        indent=2,
    )


