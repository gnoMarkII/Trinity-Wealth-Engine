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

_EDITABLE_HOLDING_FIELDS = ("units", "avg_cost", "accumulated_dividend_thb", "asset_type")


from tools._atomic_io import _atomic_write_to
from .models import _now_iso, _coerce_iso_string, Holding, Summary, PortfolioState, WatchlistItem, WatchlistState, GoalItem, GoalsState
from .core import _load_or_init, _save, _recalc_all, _recalc_holding, _recalc_summary, _find_holding, _require_cash, _require_fx, get_portfolio_state, _holding_currency, compute_allocation_breakdown
from .prices import fetch_latest_price, _fetch_last_price, _fetch_fx_rate, _refresh_prices, sync_market_prices, _USDTHB_TICKER
from .journal import append_trading_journal, _inject_journal_wikilinks, _write_journal_entry


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



@tool
@traceable(run_type="tool")
def execute_trade(
    symbol: str,
    asset_type: str,
    action: Literal["buy", "sell"],
    units: float,
    price: float,
    currency: Literal["THB", "USD"] = "THB",
) -> str:
    """ดำเนินการเทรดซื้อหรือขายสินทรัพย์ พร้อมจัดการเงินสด weighted-avg cost และ realized P&L

    [Buy] เช็คเงินสด → หักจาก CASH_THB → ถ้า holding ใหม่ให้สร้างพร้อม fields ครบ
          ถ้ามีอยู่แล้วให้คำนวณ weighted-average cost ตามสกุลเงิน
    [Sell] เช็คว่าขายไม่เกินที่มี → คำนวณ realized profit (USD คูณ fx_rate ปัจจุบัน)
           บวกเข้า summary.total_realized_profit_ytd → คืนเงินเข้า CASH_THB
           ถ้า units เหลือ 0 จะลบ holding ออกจาก list

    Args:
        symbol: ticker หรือชื่อย่อสินทรัพย์ เช่น 'AAPL', 'PTT' (ห้ามใช้ CASH_THB)
        asset_type: ประเภทสินทรัพย์ เช่น 'Stock', 'ETF', 'REIT', 'Bond'
        action: 'buy' หรือ 'sell'
        units: จำนวนหน่วยที่เทรด (>0)
        price: ราคาต่อหน่วยในสกุลเงินของสินทรัพย์
        currency: 'THB' (ดีฟอลต์) หรือ 'USD' — ระบุสกุลเงินของ price/avg_cost ของสินทรัพย์นี้

    Raises:
        ValueError: ถ้าซื้อเกินเงินสด, ขายเกินที่มี, หรือ currency ไม่ตรงกับ holding เดิม
    """
    if symbol.strip().upper() in _CASH_SYMBOLS:
        raise ValueError(
            f"ห้ามเทรด cash sentinel ({'/'.join(_CASH_SYMBOLS)}) ผ่าน execute_trade "
            f"— ใช้ manage_cash_flow แทน"
        )
    if units <= 0:
        raise ValueError("units ต้องมากกว่า 0")
    if price <= 0:
        raise ValueError("price ต้องมากกว่า 0")
    if action not in ("buy", "sell"):
        raise ValueError("action ต้องเป็น 'buy' หรือ 'sell'")

    try:
        with _portfolio_lock:
            return _execute_trade_locked(symbol, asset_type, action, units, price, currency)
    except Timeout:
        raise ValueError(
            f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นกำลังทำงาน"
        )


def _execute_trade_locked(
    symbol: str,
    asset_type: str,
    action: Literal["buy", "sell"],
    units: float,
    price: float,
    currency: Literal["THB", "USD"],
) -> str:
    post, state = _load_or_init()
    cash = _require_cash(state, currency)
    target = _find_holding(state, symbol)

    fx_rate = _require_fx(state) if currency == "USD" else None
    # ทุก trade เคลื่อนเงินสดในสกุลเงินของตัวเอง (USD trade → CASH_USD, THB trade → CASH_THB)
    amount_native = units * price

    if action == "buy":
        if cash.units + _FLOAT_EPS < amount_native:
            raise ValueError(
                f"Insufficient cash balance — มี {cash.units:,.2f} {currency} "
                f"ต้องใช้ {amount_native:,.2f} {currency}"
            )

        if target is None:
            new_h = Holding(
                symbol=symbol,
                asset_type=asset_type,
                units=units,
                accumulated_dividend_thb=0.0,
            )
            if currency == "USD":
                new_h.avg_cost_usd = round(price, _COST_DP)
                new_h.current_price_usd = price
                new_h.fx_rate = fx_rate
            else:
                new_h.avg_cost_thb = round(price, _COST_DP)
                new_h.current_price_thb = price
            state.holdings.append(new_h)
            target = new_h
            avg_cost_note = ""
        else:
            if currency == "USD":
                if target.avg_cost_usd is None:
                    raise ValueError(
                        f"{symbol} เป็นสินทรัพย์ THB อยู่แล้ว ไม่สามารถซื้อเพิ่มเป็น USD ได้"
                    )
                old_basis = target.units * target.avg_cost_usd
                new_basis = units * price
                target.units += units
                target.avg_cost_usd = round((old_basis + new_basis) / target.units, _COST_DP)
                target.current_price_usd = price
                # fx_rate ของ holding คือ historical cost-basis FX — ห้าม update ตอน weighted-avg buy
                # (ถ้าต้องการ true weighted-avg FX ต้อง refactor ที่ Holding model ทั้งหมด)
                avg_cost_note = f" (Avg cost updated to ${target.avg_cost_usd:.2f})"
            else:
                if target.avg_cost_thb is None:
                    raise ValueError(
                        f"{symbol} เป็นสินทรัพย์ USD อยู่แล้ว ไม่สามารถซื้อเพิ่มเป็น THB ได้"
                    )
                old_basis = target.units * target.avg_cost_thb
                new_basis = units * price
                target.units += units
                target.avg_cost_thb = round((old_basis + new_basis) / target.units, _COST_DP)
                target.current_price_thb = price
                avg_cost_note = f" (Avg cost updated to ฿{target.avg_cost_thb:.2f})"

        cash.units = round(cash.units - amount_native, _MONEY_DP)
        realized = 0.0
        action_tag = "[BUY]"
        cashflow_sign = "-"

    else:  # sell
        if target is None:
            raise ValueError(f"Insufficient units to sell — {symbol} ไม่มีในพอร์ต")
        if units > target.units + _FLOAT_EPS:
            raise ValueError(
                f"Insufficient units to sell — มี {symbol} {target.units} หน่วย แต่สั่งขาย {units} หน่วย"
            )

        if currency == "USD":
            if target.avg_cost_usd is None:
                raise ValueError(f"{symbol} cost เป็น THB ไม่สามารถขายด้วย USD ได้")
            realized = (price - target.avg_cost_usd) * units * fx_rate
            target.current_price_usd = price
            # ไม่ update target.fx_rate — เป็น historical cost-basis ของหน่วยที่เหลือ
        else:
            if target.avg_cost_thb is None:
                raise ValueError(f"{symbol} cost เป็น USD ไม่สามารถขายด้วย THB ได้")
            realized = (price - target.avg_cost_thb) * units
            target.current_price_thb = price

        target.units = round(target.units - units, _COST_DP)
        cash.units = round(cash.units + amount_native, _MONEY_DP)
        state.summary.total_realized_profit_ytd = round(
            state.summary.total_realized_profit_ytd + realized, _MONEY_DP
        )

        if target.units < _FLOAT_EPS:
            state.holdings.remove(target)

        avg_cost_note = ""
        action_tag = "[SELL]"
        cashflow_sign = "+"

    _save(post, state)

    cash_sym = CASH_USD_SYMBOL if currency == "USD" else CASH_THB_SYMBOL
    cash_after = _find_holding(state, cash_sym)
    cash_after_units = cash_after.units if cash_after else 0.0

    price_str = f"${price:.2f}" if currency == "USD" else f"฿{price:.2f}"
    parts = [
        f"{action_tag} {symbol} {units:g} units @ {price_str}{avg_cost_note}",
        f"กระแสเงินสด: {cashflow_sign}{amount_native:,.2f} {currency}",
    ]
    if action == "sell":
        parts.append(f"Realized P/L: {realized:+,.2f} THB")
    parts.append(f"{cash_sym} คงเหลือ: {cash_after_units:,.2f} {currency}")
    return " | ".join(parts)


@tool
@traceable(run_type="tool")
def record_income(
    income_type: Literal["Dividend", "Interest", "Rental", "Other"],
    amount_thb: float,
    source_symbol: str | None = None,
) -> str:
    """บันทึกรายรับ passive (เงินปันผล/ดอกเบี้ย/ค่าเช่า/อื่นๆ) เข้า portfolio

    Effects:
        - บวกเงินเข้า CASH_THB.units
        - บวกเข้า summary.passive_income_ytd
        - บวกเข้า summary.total_accumulated_dividend
        - ถ้าระบุ source_symbol → บวกเข้า holdings[source].accumulated_dividend_thb ด้วย

    Args:
        income_type: ประเภทรายรับ — 'Dividend' / 'Interest' / 'Rental' / 'Other'
        amount_thb: จำนวนเงินใน THB (>0)
        source_symbol: ticker ของสินทรัพย์ที่จ่ายรายรับนี้ (ถ้ามี) เช่น 'PTT'
                       ห้ามใส่ CASH_THB

    Raises:
        ValueError: ถ้า amount ไม่ถูกต้อง, ไม่พบ source_symbol, หรือ source เป็น CASH_THB
    """
    if amount_thb <= 0:
        raise ValueError("amount_thb ต้องมากกว่า 0")

    try:
        with _portfolio_lock:
            return _record_income_locked(income_type, amount_thb, source_symbol)
    except Timeout:
        raise ValueError(
            f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นกำลังทำงาน"
        )


def _record_income_locked(
    income_type: Literal["Dividend", "Interest", "Rental", "Other"],
    amount_thb: float,
    source_symbol: str | None,
) -> str:
    post, state = _load_or_init()
    cash = _require_cash(state)

    target: Holding | None = None
    if source_symbol:
        src_norm = source_symbol.strip().upper()
        if src_norm in _CASH_SYMBOLS:
            raise ValueError(f"source_symbol ห้ามเป็น cash sentinel ({'/'.join(_CASH_SYMBOLS)})")
        target = _find_holding(state, source_symbol)
        if target is None:
            raise ValueError(f"ไม่พบ {source_symbol} ใน portfolio")

    cash.units = round(cash.units + amount_thb, _MONEY_DP)
    state.summary.passive_income_ytd = round(
        state.summary.passive_income_ytd + amount_thb, _MONEY_DP
    )
    state.summary.total_accumulated_dividend = round(
        state.summary.total_accumulated_dividend + amount_thb, _MONEY_DP
    )

    if target is not None:
        current = target.accumulated_dividend_thb or 0.0
        target.accumulated_dividend_thb = round(current + amount_thb, _MONEY_DP)

    _save(post, state)

    tag = {"Dividend": "[DIV]", "Interest": "[INT]", "Rental": "[RENT]"}.get(income_type, "[INCOME]")
    src_note = f" จาก {source_symbol}" if source_symbol else ""
    return (
        f"{tag} +{amount_thb:,.2f} THB{src_note} | "
        f"passive_income_ytd: {state.summary.passive_income_ytd:,.2f} | "
        f"เงินสดคงเหลือ: {cash.units:,.2f} บาท"
    )


@tool
@traceable(run_type="tool")
def batch_import_holdings(
    assets_list: list[dict],
    mode: Literal["overwrite", "merge"] = "merge",
    reset_cash_usd: bool = False,
) -> str:
    """Smart Paste Import: บันทึก holdings หลายรายการพร้อมกัน + auto-fetch ราคาตลาดให้ที่ขาด

    รับ list ของ asset dict แต่ละตัวต้องมี keys:
      symbol (str), asset_type (str), units (float), avg_cost (float), currency ('THB'|'USD')
      current_price (float, optional) — ถ้าไม่ส่งมา ระบบจะ parallel-fetch จาก yfinance ให้
                                         ถ้าดึงไม่ได้ใช้ avg_cost เป็น fallback (unrealized P/L = 0 ชั่วคราว)

    Mode:
      'merge' (default) — อัปเดต holding ที่มี symbol ตรงกัน, ที่ไม่มีให้เพิ่ม (เก็บ accumulated_dividend_thb ของเดิม)
      'overwrite' — ล้าง holdings ที่ไม่ใช่ Cash ทั้งหมดก่อน

    reset_cash_usd: True = ศูนย์ CASH_USD.units ด้วย (ใช้คู่กับ mode='overwrite' ตอน full portfolio reset)
                    False (default) = คง CASH_USD ไว้ (backward-compatible)

    Anti-Drift: หลัง mutate จะรัน _recalc_all bottom-up อัตโนมัติผ่าน _save (Market Value → Summary)
    Atomic Storage: บันทึก Portfolio_Holdings.md ผ่าน tempfile + os.replace

    Args:
        assets_list: รายการสินทรัพย์ที่จะ import (ห้ามมี CASH_THB/CASH_USD, ห้ามมี symbol ซ้ำใน list)
        mode: 'merge' หรือ 'overwrite'
        reset_cash_usd: True = ล้าง CASH_USD เป็น 0 ด้วย (สำหรับ full reset เท่านั้น)

    Raises:
        ValueError: ถ้า list ว่าง, validation ล้มเหลว, symbol ซ้ำ, หรือมี cash sentinel
    """
    if mode not in ("overwrite", "merge"):
        raise ValueError("mode ต้องเป็น 'overwrite' หรือ 'merge'")
    if not isinstance(assets_list, list):
        raise ValueError("assets_list ต้องเป็น list")
    if not assets_list and mode != "overwrite":
        raise ValueError("assets_list ต้องเป็น list ที่ไม่ว่าง")

    # ─── 1. Pre-validate + normalize + duplicate check (ไม่ต้องล็อก) ───
    normalized: list[dict] = []
    seen: set[str] = set()
    for i, a in enumerate(assets_list):
        if not isinstance(a, dict):
            raise ValueError(f"item ลำดับ {i} ไม่ใช่ dict")
        try:
            sym = str(a["symbol"]).strip().upper()
            asset_type = str(a["asset_type"]).strip()
            units = float(a["units"])
            avg_cost = float(a["avg_cost"])
            currency = str(a["currency"]).strip().upper()
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"item ลำดับ {i}: field ขาดหรือ format ผิด — {e}")

        if not sym:
            raise ValueError(f"item ลำดับ {i}: symbol ว่าง")
        if sym in _CASH_SYMBOLS:
            raise ValueError(
                f"ห้าม import cash sentinel ({'/'.join(_CASH_SYMBOLS)}) ผ่าน batch_import "
                f"— ใช้ manage_cash_flow แทน"
            )
        if units <= 0:
            raise ValueError(f"{sym}: units ต้องมากกว่า 0")
        if avg_cost <= 0:
            raise ValueError(f"{sym}: avg_cost ต้องมากกว่า 0")
        if currency not in ("THB", "USD"):
            raise ValueError(f"{sym}: currency ต้องเป็น 'THB' หรือ 'USD' (got '{currency}')")
        if sym in seen:
            raise ValueError(f"symbol '{sym}' ซ้ำใน assets_list")
        seen.add(sym)

        cp_raw = a.get("current_price")
        current_price: float | None = None
        if cp_raw is not None:
            try:
                current_price = float(cp_raw)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{sym}: current_price format ผิด — {e}")
            if current_price <= 0:
                raise ValueError(f"{sym}: current_price ต้องมากกว่า 0 (หรือส่ง None เพื่อให้ระบบดึงให้)")

        normalized.append({
            "symbol": sym,
            "asset_type": asset_type,
            "units": units,
            "avg_cost": avg_cost,
            "currency": currency,
            "current_price": current_price,
            "_source": "provided" if current_price is not None else "pending",
        })

    # ─── 2. Parallel fetch ราคาที่ขาด (นอก lock เพื่อไม่บล็อก ops อื่น) ───
    to_fetch = [a for a in normalized if a["_source"] == "pending"]
    if to_fetch:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(to_fetch))) as ex:
            future_map = {
                ex.submit(fetch_latest_price, a["symbol"], a["currency"]): a
                for a in to_fetch
            }
            try:
                for future in concurrent.futures.as_completed(
                    future_map, timeout=_PRICE_FETCH_TIMEOUT * 2
                ):
                    asset = future_map[future]
                    try:
                        price = future.result()
                    except Exception:
                        price = None
                    if price is not None:
                        asset["current_price"] = price
                        asset["_source"] = "fetched"
                    else:
                        asset["current_price"] = asset["avg_cost"]
                        asset["_source"] = "fallback_avg_cost"
            except concurrent.futures.TimeoutError:
                for f, asset in future_map.items():
                    if not f.done():
                        asset["current_price"] = asset["avg_cost"]
                        asset["_source"] = "fallback_timeout"

    # ─── 3. Acquire lock → mutate → save (atomic + recalc) ───
    try:
        with _portfolio_lock:
            post, state = _load_or_init()
            current_fx = _require_fx(state)

            if mode == "overwrite":
                state.holdings = [h for h in state.holdings if h.asset_type == "Cash"]
                if reset_cash_usd:
                    cash_usd = _find_holding(state, CASH_USD_SYMBOL)
                    if cash_usd is not None:
                        cash_usd.units = 0.0

            for asset in normalized:
                sym = asset["symbol"]
                existing = next((h for h in state.holdings if h.symbol == sym), None)
                preserved_div = (existing.accumulated_dividend_thb if existing else None) or 0.0

                new_h = Holding(
                    symbol=sym,
                    asset_type=asset["asset_type"],
                    units=round(asset["units"], _COST_DP),
                    accumulated_dividend_thb=preserved_div,
                )
                if asset["currency"] == "USD":
                    new_h.avg_cost_usd = round(asset["avg_cost"], _COST_DP)
                    new_h.current_price_usd = float(asset["current_price"])
                    new_h.fx_rate = current_fx
                else:
                    new_h.avg_cost_thb = round(asset["avg_cost"], _COST_DP)
                    new_h.current_price_thb = float(asset["current_price"])

                if existing is not None:
                    state.holdings[state.holdings.index(existing)] = new_h
                else:
                    state.holdings.append(new_h)

            _save(post, state)  # ← _recalc_all bottom-up + atomic write
            total_nav = state.summary.total_value_thb
            non_cash = sum(1 for h in state.holdings if h.asset_type != "Cash")
    except Timeout:
        raise ValueError(f"portfolio lock timeout ({_LOCK_TIMEOUT}s)")

    # ─── 4. รายงานผล (Channel B format §4.2) ───
    counts = {"provided": 0, "fetched": 0, "fallback_avg_cost": 0, "fallback_timeout": 0}
    for a in normalized:
        counts[a["_source"]] = counts.get(a["_source"], 0) + 1
    fallback_total = counts["fallback_avg_cost"] + counts["fallback_timeout"]

    return (
        f"[IMPORT {mode.upper()}] {len(normalized)} assets | "
        f"prices(provided={counts['provided']}, fetched={counts['fetched']}, "
        f"fallback={fallback_total}) | "
        f"holdings: {non_cash} non-cash | NAV: {total_nav:,.2f} THB"
    )


@tool
@traceable(run_type="tool")
def manage_cash_flow(
    amount: float,
    action: Literal["deposit", "withdraw"],
    currency: Literal["THB", "USD"] = "THB",
) -> str:
    """ฝาก/ถอนเงินสดเข้า/ออก cash pot ของพอร์ต (CASH_THB หรือ CASH_USD)

    Effects:
        deposit  → +amount เข้า CASH_{currency} (เม็ดเงินใหม่เข้าพอร์ต)
        withdraw → -amount จาก CASH_{currency} (ต้องมีเงินพอ)

    Anti-Drift: ฐานเงินทุน (cost basis) คำนวณ bottom-up อัตโนมัติผ่าน _compute_total_cost
    — cash sentinel นับเป็น cost = units (THB) หรือ units × fx (USD)
    ดังนั้นการ mutate cash.units อย่างเดียวก็ทำให้ทั้ง NAV และ cost basis sync กัน
    Unrealized P/L ไม่เพี้ยน (เพราะ NAV เพิ่ม = cost เพิ่ม เท่ากัน)

    Args:
        amount: จำนวนเงิน (>0) ในสกุล currency
        action: 'deposit' หรือ 'withdraw'
        currency: 'THB' (default) หรือ 'USD'

    Raises:
        ValueError: amount < 0, action/currency invalid, withdraw เกิน cash ที่มี
    """
    if amount <= 0:
        raise ValueError("amount ต้องมากกว่า 0")
    if action not in ("deposit", "withdraw"):
        raise ValueError("action ต้องเป็น 'deposit' หรือ 'withdraw'")
    if currency not in ("THB", "USD"):
        raise ValueError(f"currency ต้องเป็น 'THB' หรือ 'USD' (got '{currency}')")

    try:
        with _portfolio_lock:
            return _manage_cash_flow_locked(amount, action, currency)
    except Timeout:
        raise ValueError(
            f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน"
        )


def _manage_cash_flow_locked(
    amount: float,
    action: Literal["deposit", "withdraw"],
    currency: Literal["THB", "USD"],
) -> str:
    post, state = _load_or_init()
    cash = _require_cash(state, currency)

    if action == "deposit":
        cash.units = round(cash.units + amount, _MONEY_DP)
        tag = "[DEPOSIT]"
        sign = "+"
    else:  # withdraw
        if cash.units + _FLOAT_EPS < amount:
            raise ValueError(
                f"Insufficient cash balance — มี {cash.units:,.2f} {currency} "
                f"ต้องถอน {amount:,.2f} {currency}"
            )
        cash.units = round(cash.units - amount, _MONEY_DP)
        tag = "[WITHDRAW]"
        sign = "-"

    _save(post, state)

    cash_sym = CASH_USD_SYMBOL if currency == "USD" else CASH_THB_SYMBOL
    return (
        f"{tag} {currency} | {sign}{amount:,.2f} {currency} | "
        f"{cash_sym} คงเหลือ: {cash.units:,.2f} {currency}"
    )


@tool
@traceable(run_type="tool")
def update_fx_rate(rate: float | None = None) -> str:
    """อัปเดต USD/THB exchange rate ของพอร์ต — manual rate หรือ auto-fetch จาก yfinance

    เรียกเมื่อผู้ใช้บอกค่าเงินบาทเปลี่ยน หรือ "อัปเดต FX / refresh อัตราแลกเปลี่ยน"

    Effects:
        - mutate state.fx_rates['USDTHB']
        - _save → _recalc_all bottom-up: ทุก USD market_value, CASH_USD value,
          total NAV, unrealized P/L ถูกคำนวณใหม่ด้วย fx ใหม่ทันที (Anti-Drift §3.2)

    Args:
        rate: USDTHB rate ใหม่ (>0). ถ้า None → auto-fetch จาก yfinance (USDTHB=X)

    Raises:
        ValueError: rate <= 0, auto-fetch ไม่สำเร็จ, หรือ lock timeout

    Returns:
        prefix-token — [FX] | USDTHB: old → new (±X%) | NAV: before → after | unrealized: before → after
    """
    if rate is not None and rate <= 0:
        raise ValueError("rate ต้องมากกว่า 0")

    try:
        with _portfolio_lock:
            return _update_fx_rate_locked(rate)
    except Timeout:
        raise ValueError(
            f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน"
        )


def _update_fx_rate_locked(rate: float | None) -> str:
    post, state = _load_or_init()
    old_rate = state.fx_rates.get("USDTHB", 0.0) or 0.0
    nav_before = state.summary.total_value_thb
    unrealized_before = state.summary.total_unrealized_profit

    if rate is None:
        fetched = _fetch_fx_rate()
        if fetched is None or fetched <= 0:
            raise ValueError(
                f"auto-fetch FX ล้มเหลว ({_USDTHB_TICKER}) — ลองระบุ rate manual"
            )
        new_rate = fetched
        source = "yfinance"
    else:
        new_rate = float(rate)
        source = "manual"

    state.fx_rates["USDTHB"] = new_rate
    _save(post, state)

    nav_after = state.summary.total_value_thb
    unrealized_after = state.summary.total_unrealized_profit
    pct_change = ((new_rate - old_rate) / old_rate * 100) if old_rate > 0 else 0.0

    return (
        f"[FX {source}] | USDTHB: {old_rate:.4f} → {new_rate:.4f} ({pct_change:+.2f}%) | "
        f"NAV: {nav_before:,.2f} → {nav_after:,.2f} THB | "
        f"unrealized: {unrealized_before:+,.2f} → {unrealized_after:+,.2f} THB"
    )


@tool
@traceable(run_type="tool")
def edit_holding(
    symbol: str,
    units: float | None = None,
    avg_cost: float | None = None,
    accumulated_dividend_thb: float | None = None,
    asset_type: str | None = None,
    reason: str = "",
) -> str:
    """แก้ไขข้อมูล holding ที่บันทึกผิด (correction tool) — เช่น พิมพ์ผิด, ลืมหารโบนัสหุ้น

    เปิดให้แก้เฉพาะ fields ปลอดภัย — ไม่แตะ:
        - market_value_thb / unrealized_pnl_percent → computed (auto-recalc)
        - current_price_* → ใช้ sync_market_prices
        - fx_rate ของ holding → historical cost-basis (ห้ามเขียนทับ)
        - symbol → ใช้ ลบ + import ใหม่แทน
        - summary.total_realized_profit_ytd → historical, ห้าม rewrite (per-trade booking)

    avg_cost auto-route ไป avg_cost_usd หรือ avg_cost_thb ตามสกุลที่ holding เดิมใช้

    Auto-side-effect: append เหตุผลลง Trading_Journal เพื่อ audit trail

    Args:
        symbol: ticker ของ holding ที่จะแก้ (ห้ามเป็น CASH_THB / CASH_USD — ใช้ manage_cash_flow)
        units: จำนวนหน่วยใหม่ (>0)
        avg_cost: avg cost ใหม่ (>0) — ระบบเลือก THB/USD field ตาม holding เดิม
        accumulated_dividend_thb: ยอดสะสมปันผลใหม่ (>=0)
        asset_type: เปลี่ยนหมวด (Stock/ETF/REIT/Bond ฯลฯ)
        reason: เหตุผลที่ต้องแก้ (audit log) — ควรระบุชัดเสมอ

    Raises:
        ValueError: symbol ไม่พบ, symbol เป็น cash, ไม่ส่ง field ไหนมาแก้, validation ค่าไม่ถูกต้อง
    """
    sym = symbol.strip().upper()
    if sym in _CASH_SYMBOLS:
        raise ValueError(
            f"ห้ามแก้ {sym} ผ่าน edit_holding — ใช้ manage_cash_flow สำหรับ cash"
        )
    if all(v is None for v in (units, avg_cost, accumulated_dividend_thb, asset_type)):
        raise ValueError(f"ต้องระบุอย่างน้อย 1 field ที่จะแก้ ({', '.join(_EDITABLE_HOLDING_FIELDS)})")
    if units is not None and units <= 0:
        raise ValueError("units ต้องมากกว่า 0")
    if avg_cost is not None and avg_cost <= 0:
        raise ValueError("avg_cost ต้องมากกว่า 0")
    if accumulated_dividend_thb is not None and accumulated_dividend_thb < 0:
        raise ValueError("accumulated_dividend_thb ต้อง >= 0")

    try:
        with _portfolio_lock:
            return _edit_holding_locked(
                sym, units, avg_cost, accumulated_dividend_thb, asset_type, reason
            )
    except Timeout:
        raise ValueError(
            f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน"
        )


def _edit_holding_locked(
    symbol: str,
    units: float | None,
    avg_cost: float | None,
    accumulated_dividend_thb: float | None,
    asset_type: str | None,
    reason: str,
) -> str:
    post, state = _load_or_init()
    target = _find_holding(state, symbol)
    if target is None:
        raise ValueError(f"ไม่พบ {symbol} ใน portfolio")
    if target.asset_type == "Cash":
        raise ValueError(f"{symbol} เป็น Cash holding — ใช้ manage_cash_flow")

    nav_before = state.summary.total_value_thb
    changes: list[str] = []

    if units is not None and units != target.units:
        changes.append(f"units: {target.units:g} → {units:g}")
        target.units = round(units, _COST_DP)

    if avg_cost is not None:
        if target.avg_cost_usd is not None:
            if avg_cost != target.avg_cost_usd:
                changes.append(f"avg_cost_usd: ${target.avg_cost_usd:.4f} → ${avg_cost:.4f}")
                target.avg_cost_usd = round(avg_cost, _COST_DP)
        elif target.avg_cost_thb is not None:
            if avg_cost != target.avg_cost_thb:
                changes.append(f"avg_cost_thb: ฿{target.avg_cost_thb:.4f} → ฿{avg_cost:.4f}")
                target.avg_cost_thb = round(avg_cost, _COST_DP)
        else:
            raise ValueError(
                f"{symbol} ไม่มี avg_cost_thb/usd เดิม — สร้าง holding ใหม่ผ่าน execute_trade แทน"
            )

    if accumulated_dividend_thb is not None:
        current = target.accumulated_dividend_thb or 0.0
        if accumulated_dividend_thb != current:
            changes.append(
                f"accumulated_dividend_thb: {current:,.2f} → {accumulated_dividend_thb:,.2f}"
            )
            target.accumulated_dividend_thb = round(accumulated_dividend_thb, _MONEY_DP)

    if asset_type is not None:
        new_type = asset_type.strip()
        if new_type and new_type != target.asset_type:
            changes.append(f"asset_type: {target.asset_type} → {new_type}")
            target.asset_type = new_type

    if not changes:
        raise ValueError("ค่าใหม่เหมือนเดิมทั้งหมด — ไม่มีอะไรต้องแก้")

    _save(post, state)
    nav_after = state.summary.total_value_thb

    reason_text = reason.strip() or "(no reason given)"
    _write_journal_entry(
        f"**[EDIT {symbol}]** {' | '.join(changes)}\n\nReason: {reason_text}"
    )

    return (
        f"[EDIT {symbol}] | {' | '.join(changes)} | reason: {reason_text} | "
        f"NAV: {nav_before:,.2f} → {nav_after:,.2f} THB"
    )


def _compute_total_cost(state: PortfolioState, current_fx: float) -> float:
    """รวมต้นทุนทุก holding ใน THB — ใช้ current_fx แปลง USD cost ให้สอดคล้องกับ market_value
    (Anti-Drift: NAV - Cost = Unrealized P/L แบบเป๊ะตามที่ _recalc_summary คำนวณ)

    *Pair-required guard*: นับ holding เฉพาะตอนมี cost+price ครบคู่
    (สอดคล้องกับ _recalc_summary ที่ skip holding incomplete เช่นกัน)
    """
    total = 0.0
    for h in state.holdings:
        if h.asset_type == "Cash":
            if h.symbol == CASH_USD_SYMBOL:
                total += h.units * current_fx
            else:
                total += h.units
            continue
        if h.avg_cost_usd is not None and h.current_price_usd is not None:
            total += h.units * h.avg_cost_usd * current_fx
        elif h.avg_cost_thb is not None and h.current_price_thb is not None:
            total += h.units * h.avg_cost_thb
    return round(total, _MONEY_DP)


