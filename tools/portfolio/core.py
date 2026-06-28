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

_TOP_LEVEL_KEY_ORDER = ("doc_type", "last_updated", "holdings", "cash")

from tools._atomic_io import _atomic_write_to
from tools.tool_errors import LOCK_TIMEOUT, validation_error
from .models import _now_iso, _coerce_iso_string, Holding, Summary, PortfolioState, WatchlistItem, WatchlistState, GoalItem, GoalsState


VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
PORTFOLIO_REL = os.getenv("PORTFOLIO_FILE", "20_Portfolio_Management/Current_Holdings/Portfolio_Holdings.md")
PORTFOLIO_PATH = VAULT_PATH / PORTFOLIO_REL
TRADING_JOURNAL_REL = os.getenv(
    "TRADING_JOURNAL_FILE",
    "20_Portfolio_Management/Journals_and_Reports/Trading_Journal.md",
)
TRADING_JOURNAL_PATH = VAULT_PATH / TRADING_JOURNAL_REL
WATCHLIST_REL = os.getenv(
    "WATCHLIST_FILE",
    "20_Portfolio_Management/Current_Holdings/Watchlist.md",
)
WATCHLIST_PATH = VAULT_PATH / WATCHLIST_REL
PERFORMANCE_LOG_REL = os.getenv(
    "PERFORMANCE_LOG_FILE",
    "20_Portfolio_Management/Journals_and_Reports/Performance_Log.csv",
)
PERFORMANCE_LOG_PATH = VAULT_PATH / PERFORMANCE_LOG_REL
_PERFORMANCE_LOG_HEADER = ["Date", "Total_NAV", "Total_Cost", "Unrealized_PnL", "Cash_Balance"]

# Derived sidecar folders (master = Portfolio_Holdings.md / Watchlist.md)
HOLDINGS_DIR = VAULT_PATH / "20_Portfolio_Management/Current_Holdings/Holdings"
WATCHLIST_ITEMS_DIR = VAULT_PATH / "20_Portfolio_Management/Current_Holdings/WatchlistItems"

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





def _initial_state() -> PortfolioState:
    return PortfolioState(
        last_updated=_now_iso(),
        summary=Summary(),
        fx_rates={"USDTHB": 36.5},
        holdings=[
            Holding(
                symbol=CASH_THB_SYMBOL,
                asset_type="Cash",
                units=0.0,
                market_value_thb=0.0,
            ),
            Holding(
                symbol=CASH_USD_SYMBOL,
                asset_type="Cash",
                units=0.0,
                market_value_thb=0.0,
            ),
        ],
    )


def _load_or_init() -> tuple[frontmatter.Post, PortfolioState]:
    if not PORTFOLIO_PATH.exists():
        PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(content="")  # YAML state only — no body
        state = _initial_state()
        _save(post, state)
        return post, state

    with PORTFOLIO_PATH.open("r", encoding="utf-8") as f:
        post = frontmatter.load(f)

    if not post.metadata:
        log.warning("Portfolio_Holdings.md ไม่มี YAML frontmatter — บูตข้อมูลใหม่")
        state = _initial_state()
        _save(post, state)
        return post, state

    state = PortfolioState.model_validate(post.metadata)
    return post, state


def _holding_to_md(h: Holding) -> str:
    """สร้าง YAML frontmatter สำหรับ sidecar file ของ Holding รายตัว"""
    if h.avg_cost_usd is not None:
        currency = "USD"
        avg_cost = h.avg_cost_usd
        current_price = h.current_price_usd
    else:
        currency = "THB"
        avg_cost = h.avg_cost_thb
        current_price = h.current_price_thb

    lines = [
        "---",
        "entity_type: holding",
        f"symbol: {h.symbol}",
        f"asset_type: {h.asset_type}",
        f"currency: {currency}",
        f"units: {h.units}",
    ]
    if avg_cost is not None:
        lines.append(f"avg_cost: {avg_cost}")
    if current_price is not None:
        lines.append(f"current_price: {current_price}")
    lines.append(f"market_value_thb: {h.market_value_thb}")
    if h.unrealized_pnl_percent is not None:
        lines.append(f"unrealized_pnl_pct: {h.unrealized_pnl_percent}")
    if h.accumulated_dividend_thb is not None:
        lines.append(f"dividend_thb: {h.accumulated_dividend_thb}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _sync_holding_sidecars(state: "PortfolioState") -> None:
    """Sync derived sidecar files ใน HOLDINGS_DIR จาก master PortfolioState
    เขียน 1 ไฟล์ต่อ holding (ข้าม Cash) — ลบ sidecar เก่าที่ holding ถูกลบแล้ว
    """
    HOLDINGS_DIR.mkdir(parents=True, exist_ok=True)
    live: set[str] = set()

    for h in state.holdings:
        if h.asset_type == "Cash":
            continue
        safe = h.symbol.replace("/", "_")
        _atomic_write_to(HOLDINGS_DIR / f"{safe}.md", _holding_to_md(h))
        live.add(safe)

    for old in HOLDINGS_DIR.glob("*.md"):
        if old.stem not in live:
            old.unlink(missing_ok=True)
            log.debug("[SIDECAR DEL] | holdings/%s", old.name)


def _recalc_holding(h: Holding, current_fx: float) -> None:
    from .prices import _fetch_last_price

    """คำนวณ market_value_thb และ unrealized_pnl_percent ของ holding รายตัว

    fx_rate ใน Holding คือ FX ตอน trade (cost-basis); market_value ใช้ current_fx ของพอร์ต
    """
    if h.asset_type == "Cash":
        if h.symbol == CASH_USD_SYMBOL:
            h.market_value_thb = round(h.units * current_fx, _MONEY_DP)
        else:
            h.market_value_thb = round(h.units, _MONEY_DP)
        h.unrealized_pnl_percent = None
        h.accumulated_dividend_thb = None
        return

    if h.avg_cost_usd is not None and h.current_price_usd is not None:
        h.market_value_thb = round(h.units * h.current_price_usd * current_fx, _MONEY_DP)
        h.unrealized_pnl_percent = (
            round((h.current_price_usd - h.avg_cost_usd) / h.avg_cost_usd * 100, _PCT_DP)
            if h.avg_cost_usd
            else 0.0
        )
    elif h.avg_cost_thb is not None and h.current_price_thb is not None:
        h.market_value_thb = round(h.units * h.current_price_thb, _MONEY_DP)
        h.unrealized_pnl_percent = (
            round((h.current_price_thb - h.avg_cost_thb) / h.avg_cost_thb * 100, _PCT_DP)
            if h.avg_cost_thb
            else 0.0
        )
    else:
        log.warning(
            "Holding %s has incomplete cost/price pair — market value reset to 0", h.symbol
        )
        h.market_value_thb = 0.0
        h.unrealized_pnl_percent = None
        return


def _recalc_summary(state: PortfolioState, current_fx: float) -> None:
    from .prices import _fetch_last_price

    """รวม total_value_thb และ total_unrealized_profit จาก holdings (ใช้ current_fx)"""
    total_value = 0.0
    total_unrealized = 0.0

    for h in state.holdings:
        total_value += h.market_value_thb
        if h.asset_type == "Cash":
            continue
        if h.avg_cost_usd is not None and h.current_price_usd is not None:
            total_unrealized += (h.current_price_usd - h.avg_cost_usd) * h.units * current_fx
        elif h.avg_cost_thb is not None and h.current_price_thb is not None:
            total_unrealized += (h.current_price_thb - h.avg_cost_thb) * h.units

    state.summary.total_value_thb = round(total_value, _MONEY_DP)
    state.summary.total_unrealized_profit = round(total_unrealized, _MONEY_DP)


def _recalc_all(state: PortfolioState) -> None:
    """Anti-Drift: คำนวณใหม่ทั้งหมดโดยใช้ fx_rates.USDTHB ปัจจุบันของพอร์ต"""
    current_fx = state.fx_rates.get("USDTHB", 0.0) or 0.0
    if current_fx <= 0:
        log.warning("fx_rates.USDTHB missing or invalid — USD holdings will compute as 0")
    for h in state.holdings:
        _recalc_holding(h, current_fx)
    _recalc_summary(state, current_fx)


def _save(post: frontmatter.Post, state: PortfolioState) -> None:
    """[CRITICAL] Atomic save พร้อม Anti-Drift recalculation
    เขียน temp file ก่อนเสมอ → os.replace() เป็น atomic operation
    """
    _recalc_all(state)
    state.last_updated = _now_iso()

    dump = state.model_dump(exclude_none=True)

    ordered: dict = {}
    for key in _TOP_LEVEL_KEY_ORDER:
        if key in dump:
            ordered[key] = dump.pop(key)
    # คงค่า extra fields ที่ผู้ใช้อาจเติมเองใน YAML ไว้ท้าย
    ordered.update(dump)

    post.metadata.clear()
    post.metadata.update(ordered)
    post.content = ""  # YAML-only contract — เนื้อหาอื่นห้ามปนใน Portfolio_Holdings.md

    serialized = frontmatter.dumps(post, sort_keys=False)
    _atomic_write_to(PORTFOLIO_PATH, serialized)
    _sync_holding_sidecars(state)


def _find_holding(state: PortfolioState, symbol: str) -> Holding | None:
    return next((h for h in state.holdings if h.symbol == symbol), None)


def _require_cash(state: PortfolioState, currency: Literal["THB", "USD"] = "THB") -> Holding:
    """หา cash holding ตาม currency — lazy-create ถ้าไฟล์เก่ายังไม่มี CASH_USD"""
    sym = CASH_THB_SYMBOL if currency == "THB" else CASH_USD_SYMBOL
    cash = _find_holding(state, sym)
    if cash is None:
        cash = Holding(symbol=sym, asset_type="Cash", units=0.0, market_value_thb=0.0)
        state.holdings.append(cash)
    return cash


def _require_fx(state: PortfolioState) -> float:
    fx = state.fx_rates.get("USDTHB")
    if fx is None or fx <= 0:
        raise ValueError("ไม่พบ fx_rates.USDTHB ที่ valid ใน portfolio")
    return fx


@tool
def get_portfolio_state(refresh_prices: bool = True) -> str:
    """อ่านสถานะ Portfolio ปัจจุบันคืนเป็น JSON string (Read-only)

    [Usage/When to use]
    ใช้เมื่อต้องการสรุปภาพรวมพอร์ตโฟลิโอ สินทรัพย์ที่ถือครอง หรือคำนวณ NAV
    - ดึงราคาตลาดล่าสุดจาก yfinance อัตโนมัติ (ยกเว้นสั่ง refresh_prices=False)

    [Caution]
    - ไม่ทำการเปลี่ยนแปลงสถานะพอร์ต

    Args:
        refresh_prices (bool): True (ดึงราคาตลาดล่าสุด, default), False (ใช้ราคาเดิม)

    Returns:
        str: JSON string ของ PortfolioState พร้อมสถานะการอัปเดตราคา (_price_refresh)
    """
    try:
        with _portfolio_lock:
            post, state = _load_or_init()
            refresh_info: dict[str, str] = {}
            if refresh_prices:
                from .prices import _refresh_prices
                refresh_info = _refresh_prices(state)
            if refresh_info:
                _save(post, state)
            else:
                _recalc_all(state)
    except Timeout:
        return json.dumps(
            {"error": f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — another op in progress"},
            ensure_ascii=False,
        )

    dump = state.model_dump(exclude_none=True)
    if refresh_info:
        dump["_price_refresh"] = refresh_info
    return json.dumps(dump, ensure_ascii=False, indent=2)


def _holding_currency(h: Holding) -> str:
    """Derive currency tag ของ holding — Cash ใช้ symbol, asset อื่นใช้ avg_cost_* field ที่มี"""
    if h.symbol == CASH_USD_SYMBOL:
        return "USD"
    if h.symbol == CASH_THB_SYMBOL:
        return "THB"
    if h.avg_cost_usd is not None:
        return "USD"
    if h.avg_cost_thb is not None:
        return "THB"
    return "UNKNOWN"


@tool
def compute_allocation_breakdown(
    group_by: Literal["asset_type", "currency"] = "asset_type",
) -> str:
    """Calculate portfolio Asset Allocation breakdown.

    Args:
        group_by: Dimension to group by ('asset_type' or 'currency'). Defaults to 'asset_type'.

    Returns:
        JSON string: {group_by, total_nav_thb, breakdown: [{group, value_thb, pct, count}], generated_at}
    """
    if group_by not in ("asset_type", "currency"):
        return validation_error("group_by ต้องเป็น 'asset_type' หรือ 'currency'")

    try:
        with _portfolio_lock:
            post, state = _load_or_init()
            _recalc_all(state)
            total_nav = state.summary.total_value_thb

            buckets: dict[str, dict] = {}
            for h in state.holdings:
                key = h.asset_type if group_by == "asset_type" else _holding_currency(h)
                b = buckets.setdefault(key, {"value_thb": 0.0, "count": 0})
                b["value_thb"] += h.market_value_thb
                b["count"] += 1
    except Timeout:
        return LOCK_TIMEOUT.format(detail=f"portfolio lock {_LOCK_TIMEOUT}s")
    except ValueError as e:
        return f"Error: {e}"

    breakdown = [
        {
            "group": k,
            "value_thb": round(v["value_thb"], _MONEY_DP),
            "pct": round((v["value_thb"] / total_nav * 100) if total_nav > 0 else 0.0, _PCT_DP),
            "count": v["count"],
        }
        for k, v in buckets.items()
    ]
    breakdown.sort(key=lambda x: x["value_thb"], reverse=True)

    return json.dumps(
        {
            "group_by": group_by,
            "total_nav_thb": round(total_nav, _MONEY_DP),
            "breakdown": breakdown,
            "generated_at": _now_iso(),
        },
        ensure_ascii=False,
        indent=2,
    )


