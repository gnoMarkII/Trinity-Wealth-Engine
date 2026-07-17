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

_TOP_LEVEL_KEY_ORDER = ("doc_type", "last_updated", "allocation_targets", "holdings", "cash")

from tools._atomic_io import _atomic_write_to
from tools.tool_errors import LOCK_TIMEOUT, validation_error
from .models import _now_iso, _coerce_iso_string, AllocationTarget, default_allocation_targets, Holding, Summary, PortfolioState, WatchlistItem, WatchlistState, GoalItem, GoalsState



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
    state.summary.total_cost_basis_thb = _compute_total_cost(state, current_fx)
    state.summary.total_unrealized_profit = round(total_unrealized, _MONEY_DP)


def _recalc_fundamentals_derived(state: PortfolioState) -> None:
    """คำนวณ derived fundamentals/metrics ให้ทุก holding (market_cap_tier, yield_on_cost, unrealized_pnl_value)"""
    current_fx = state.fx_rates.get("USDTHB", 0.0) or 0.0
    for h in state.holdings:
        if h.asset_type == "Cash":
            setattr(h, "market_cap_tier", "N/A")
            setattr(h, "yield_on_cost", None)
            setattr(h, "unrealized_pnl_value", 0.0)
            if h.bucket_id is None:
                h.bucket_id = "cash"
            continue

        # Market cap tier
        mcap = getattr(h, "market_cap_value", None)
        if mcap is not None and isinstance(mcap, (int, float)) and mcap > 0:
            if mcap >= MARKET_CAP_MEGA_USD:
                setattr(h, "market_cap_tier", "Mega")
            elif mcap >= MARKET_CAP_LARGE_USD:
                setattr(h, "market_cap_tier", "Large")
            elif mcap >= MARKET_CAP_MID_USD:
                setattr(h, "market_cap_tier", "Mid")
            else:
                setattr(h, "market_cap_tier", "Small")
        else:
            setattr(h, "market_cap_tier", "N/A")

        # Yield on cost
        div_rate = getattr(h, "dividend_per_share", None)
        if div_rate is not None and isinstance(div_rate, (int, float)) and div_rate >= 0:
            if h.avg_cost_usd is not None and h.avg_cost_usd > 0:
                setattr(h, "yield_on_cost", round((div_rate / h.avg_cost_usd) * 100, _PCT_DP))
            elif h.avg_cost_thb is not None and h.avg_cost_thb > 0:
                setattr(h, "yield_on_cost", round((div_rate / h.avg_cost_thb) * 100, _PCT_DP))
            else:
                setattr(h, "yield_on_cost", None)
        else:
            setattr(h, "yield_on_cost", None)

        # Unrealized PnL Value (THB)
        if h.avg_cost_usd is not None and h.current_price_usd is not None:
            cost_thb = h.units * h.avg_cost_usd * current_fx
            setattr(h, "unrealized_pnl_value", round(h.market_value_thb - cost_thb, _MONEY_DP))
        elif h.avg_cost_thb is not None and h.current_price_thb is not None:
            cost_thb = h.units * h.avg_cost_thb
            setattr(h, "unrealized_pnl_value", round(h.market_value_thb - cost_thb, _MONEY_DP))
        else:
            setattr(h, "unrealized_pnl_value", None)


def _recalc_all(state: PortfolioState) -> None:
    """Anti-Drift: คำนวณใหม่ทั้งหมดโดยใช้ fx_rates.USDTHB ปัจจุบันของพอร์ต"""
    current_fx = state.fx_rates.get("USDTHB", 0.0) or 0.0
    if current_fx <= 0:
        log.warning("fx_rates.USDTHB missing or invalid — USD holdings will compute as 0")
    for h in state.holdings:
        _recalc_holding(h, current_fx)
    _recalc_summary(state, current_fx)
    _recalc_fundamentals_derived(state)



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


def get_structured_bucket_allocation(
    state: PortfolioState | None = None,
) -> tuple[list[dict], str | None]:
    """คำนวณการกระจายสัดส่วนตาม Strategy Buckets (Target vs Actual) และหา variance + warning ถ้า target_sum != 100%"""
    if state is None:
        with _portfolio_lock:
            _, state = _load_or_init()
            _recalc_all(state)

    total_nav = state.summary.total_value_thb
    bucket_totals: dict[str, float] = {}
    unassigned_thb = 0.0

    target_ids = {t.bucket_id for t in state.allocation_targets}
    for h in state.holdings:
        val = h.market_value_thb
        b_id = h.bucket_id if h.bucket_id else ("cash" if h.asset_type == "Cash" else None)
        if b_id and b_id in target_ids:
            bucket_totals[b_id] = bucket_totals.get(b_id, 0.0) + val
        else:
            unassigned_thb += val

    target_sum = sum(t.target_percent for t in state.allocation_targets)
    warning_flag = None
    if abs(target_sum - 100.0) > 0.01:
        warning_flag = f"ผลรวมเป้าหมายสัดส่วนปัจจุบันไม่เท่ากับ 100% (รวมได้ {target_sum:.1f}%)"

    summaries: list[dict] = []
    for t in state.allocation_targets:
        actual_thb = bucket_totals.get(t.bucket_id, 0.0)
        actual_pct = round((actual_thb / total_nav) * 100, _PCT_DP) if total_nav > 0 else 0.0
        variance = round(actual_pct - t.target_percent, _PCT_DP)
        summaries.append({
            "bucket_id": t.bucket_id,
            "name": t.name,
            "target_percent": t.target_percent,
            "actual_value_thb": round(actual_thb, _MONEY_DP),
            "actual_percent": actual_pct,
            "variance": variance,
            "color": t.color,
        })

    if unassigned_thb > _FLOAT_EPS:
        actual_pct = round((unassigned_thb / total_nav) * 100, _PCT_DP) if total_nav > 0 else 0.0
        summaries.append({
            "bucket_id": "unassigned",
            "name": "Unassigned",
            "target_percent": 0.0,
            "actual_value_thb": round(unassigned_thb, _MONEY_DP),
            "actual_percent": actual_pct,
            "variance": actual_pct,
            "color": "#64748B",
        })

    return summaries, warning_flag


def get_structured_portfolio_state(
    refresh_prices: bool = False, fetch_fundamentals: bool = False
) -> PortfolioState:
    """Structured read accessor สำหรับ Phase 1 Hub คืนค่า Pydantic PortfolioState พร้อม derived metrics/TTL cache"""
    info = None
    f_info = None

    if refresh_prices or fetch_fundamentals:
        with _portfolio_lock:
            _, state_snapshot = _load_or_init()
            state_clone = state_snapshot.model_copy(deep=True)

        if refresh_prices:
            from .prices import _refresh_prices
            info = _refresh_prices(state_clone)
        if fetch_fundamentals:
            from .prices import _fetch_fundamentals
            f_info = _fetch_fundamentals(state_clone, force=False)

        with _portfolio_lock:
            post, state = _load_or_init()
            modified = False
            if info is not None:
                clone_price_map = {
                    h.symbol: (h.current_price_usd, h.current_price_thb)
                    for h in state_clone.holdings
                }
                for h in state.holdings:
                    if h.symbol in clone_price_map:
                        p_usd, p_thb = clone_price_map[h.symbol]
                        if p_usd is not None:
                            h.current_price_usd = p_usd
                        if p_thb is not None:
                            h.current_price_thb = p_thb
                state.price_refresh_info = info
                modified = True
            if f_info is not None:
                clone_f_map = {
                    h.symbol: (
                        h.pe_ratio, h.eps, h.payout_ratio, h.market_cap_value,
                        h.dividend_per_share, h.dividend_yield, h.fundamentals_updated_at
                    )
                    for h in state_clone.holdings
                }
                for h in state.holdings:
                    if h.symbol in clone_f_map:
                        (pe, eps, pay, mcap, dps, dy, f_at) = clone_f_map[h.symbol]
                        if pe is not None: h.pe_ratio = pe
                        if eps is not None: h.eps = eps
                        if pay is not None: h.payout_ratio = pay
                        if mcap is not None: h.market_cap_value = mcap
                        if dps is not None: h.dividend_per_share = dps
                        if dy is not None: h.dividend_yield = dy
                        if f_at is not None: h.fundamentals_updated_at = f_at
                modified = True

            _recalc_all(state)
            if modified:
                _save(post, state)
                if refresh_prices:
                    try:
                        from .performance import record_performance_snapshot
                        record_performance_snapshot.func(refresh_prices=False)
                    except Exception as e:
                        log.warning("Failed to record performance snapshot: %s", e)
            return state
    else:
        with _portfolio_lock:
            post, state = _load_or_init()
            _recalc_all(state)
            return state


def structured_upsert_allocation_targets(targets: list[AllocationTarget]) -> PortfolioState:
    """อัปเดตเป้าหมายสัดส่วนกลยุทธ์ (Allocation Targets) และบันทึกลง master file"""
    total_pct = sum(t.target_percent for t in targets)
    if abs(total_pct - 100.0) > 0.01:
        raise ValueError(f"ผลรวมเป้าหมายสัดส่วน (target_percent) ต้องเท่ากับ 100% (ปัจจุบันได้ {total_pct:.1f}%)")
    with _portfolio_lock:
        post, state = _load_or_init()
        state.allocation_targets = targets
        _save(post, state)
        return state


def structured_assign_holding_bucket(symbol: str, bucket_id: str | None) -> PortfolioState:
    """เปลี่ยน/ระบุ bucket_id ของ Holding รายตัว"""
    with _portfolio_lock:
        post, state = _load_or_init()
        if bucket_id and bucket_id not in ("unassigned", "cash"):
            valid_ids = {t.bucket_id for t in state.allocation_targets}
            if bucket_id not in valid_ids:
                raise ValueError(f"ไม่พบ bucket_id '{bucket_id}' ใน Allocation Targets ของพอร์ต")
        h = _find_holding(state, symbol)
        if not h:
            raise ValueError(f"ไม่พบสินทรัพย์ {symbol} ในพอร์ต")
        h.bucket_id = bucket_id
        _save(post, state)
        return state


def structured_batch_assign_holding_buckets(symbols: list[str], bucket_id: str | None) -> PortfolioState:
    """เปลี่ยน bucket_id พร้อมกันหลายรายการ (Batch action)"""
    with _portfolio_lock:
        post, state = _load_or_init()
        if bucket_id and bucket_id not in ("unassigned", "cash"):
            valid_ids = {t.bucket_id for t in state.allocation_targets}
            if bucket_id not in valid_ids:
                raise ValueError(f"ไม่พบ bucket_id '{bucket_id}' ใน Allocation Targets ของพอร์ต")
        sym_set = set(symbols)
        found = 0
        for h in state.holdings:
            if h.symbol in sym_set:
                h.bucket_id = bucket_id
                found += 1
        if found == 0 and symbols:
            raise ValueError(f"ไม่พบสินทรัพย์ตามที่ระบุในพอร์ต")
        _save(post, state)
        return state


def structured_batch_remove_holdings(symbols: list[str]) -> PortfolioState:
    """ลบสินทรัพย์ออกจากพอร์ตพร้อมกันหลายรายการ (Batch action)"""
    with _portfolio_lock:
        post, state = _load_or_init()
        sym_set = set(symbols)
        before_len = len(state.holdings)
        state.holdings = [h for h in state.holdings if h.symbol not in sym_set]
        if len(state.holdings) == before_len and symbols:
            raise ValueError(f"ไม่พบสินทรัพย์ตามที่ระบุเพื่อลบ")
        _save(post, state)
        return state


def structured_reset_clean_slate() -> PortfolioState:
    """ล้างข้อมูลพอร์ตทั้งหมด (Clean Slate) เพื่อเริ่มต้นใหม่ตาม user preference: 'ลบข้อมูลทั้งหมด เดี๋ยวใส่ใหม่เอง'

    ลบไฟล์ sidecars เฉพาะ Holdings/*.md และรีเซ็ต PortfolioState (ไม่ลบ Watchlist, Goals, Trading Journal, และ Performance history)
    """
    with _portfolio_lock:
        # 1. Reset master portfolio state
        post, _ = _load_or_init()
        new_state = PortfolioState(
            last_updated=_now_iso(),
            allocation_targets=default_allocation_targets(),
            holdings=[],
            summary=Summary(
                total_value_thb=0.0,
                total_unrealized_profit=0.0,
                passive_income_ytd=0.0,
            ),
        )
        _save(post, new_state)

        # 2. Clean sidecar directory Holdings/ (เฉพาะสินทรัพย์ในพอร์ต)
        if HOLDINGS_DIR.exists():
            for f in HOLDINGS_DIR.glob("*.md"):
                f.unlink(missing_ok=True)

        return new_state


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


