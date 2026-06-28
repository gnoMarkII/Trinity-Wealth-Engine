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

_WATCHLIST_KEY_ORDER = ("doc_type", "last_updated", "items")


from tools._atomic_io import _atomic_write_to
from tools.tool_errors import LOCK_TIMEOUT, validation_error
from .core import _load_or_init, _save, _recalc_all, _recalc_holding, _recalc_summary, _find_holding, _require_cash, _require_fx, get_portfolio_state, _holding_currency, compute_allocation_breakdown
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



def _atomic_write_watchlist(serialized: str) -> None:
    """Atomic write สำหรับ Watchlist.md — pattern เดียวกับ portfolio _atomic_write"""
    parent = WATCHLIST_PATH.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".watchlist_", suffix=".md.tmp", dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
        os.replace(tmp_path, WATCHLIST_PATH)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _save_watchlist(post: frontmatter.Post, state: WatchlistState) -> None:
    state.last_updated = _now_iso()
    dump = state.model_dump(exclude_none=True)

    ordered: dict = {}
    for key in _WATCHLIST_KEY_ORDER:
        if key in dump:
            ordered[key] = dump.pop(key)
    ordered.update(dump)

    post.metadata.clear()
    post.metadata.update(ordered)
    post.content = ""

    _atomic_write_watchlist(frontmatter.dumps(post, sort_keys=False))
    _sync_watchlist_sidecars(state)


def _watchlist_item_to_md(item: WatchlistItem) -> str:
    """สร้าง YAML frontmatter สำหรับ sidecar file ของ WatchlistItem"""
    lines = [
        "---",
        "entity_type: watchlist_item",
        f"symbol: {item.symbol}",
        f"asset_type: {item.asset_type}",
        f"added_date: {item.added_date}",
    ]
    if item.target_price is not None:
        lines.append(f"target_price: {item.target_price}")
    if item.notes is not None:
        notes_escaped = item.notes.replace('"', '\\"')
        lines.append(f'notes: "{notes_escaped}"')
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _sync_watchlist_sidecars(state: WatchlistState) -> None:
    """Sync derived sidecar files ใน WATCHLIST_ITEMS_DIR จาก master WatchlistState"""
    WATCHLIST_ITEMS_DIR.mkdir(parents=True, exist_ok=True)
    live: set[str] = set()

    for item in state.items:
        safe = item.symbol.replace("/", "_")
        _atomic_write_to(WATCHLIST_ITEMS_DIR / f"{safe}.md", _watchlist_item_to_md(item))
        live.add(safe)

    for old in WATCHLIST_ITEMS_DIR.glob("*.md"):
        if old.stem not in live:
            old.unlink(missing_ok=True)
            log.debug("[SIDECAR DEL] | watchlist/%s", old.name)


def _load_or_init_watchlist() -> tuple[frontmatter.Post, WatchlistState]:
    if not WATCHLIST_PATH.exists():
        WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(content="")
        state = WatchlistState(last_updated=_now_iso())
        _save_watchlist(post, state)
        return post, state

    with WATCHLIST_PATH.open("r", encoding="utf-8") as f:
        post = frontmatter.load(f)

    if not post.metadata:
        log.warning("Watchlist.md ไม่มี YAML frontmatter — บูตข้อมูลใหม่")
        state = WatchlistState(last_updated=_now_iso())
        _save_watchlist(post, state)
        return post, state

    state = WatchlistState.model_validate(post.metadata)
    return post, state


@tool
def add_to_watchlist(
    symbol: str,
    asset_type: str,
    target_price: float | None = None,
    notes: str | None = None,
) -> str:
    """เพิ่มหรืออัปเดตสินทรัพย์ใน Watchlist

    [Usage/When to use]
    ใช้เมื่อต้องการจับตาสินทรัพย์ที่สนใจลงทุนในอนาคต พร้อมระบุราคาเป้าหมาย (Target Price)
    - สามารถอัปเดต Target Price สำหรับสินทรัพย์ที่มีอยู่แล้วได้ (Upsert)

    Args:
        symbol (str): Ticker ของสินทรัพย์
        asset_type (str): ประเภทสินทรัพย์
        target_price (float | None): ราคาที่ต้องการแจ้งเตือนเมื่อถึงเป้า
        notes (str | None): บันทึกเตือนความจำเพิ่มเติม
    """
    sym = symbol.strip().upper()
    if not sym:
        return validation_error("symbol ต้องไม่ว่าง")
    if target_price is not None and target_price <= 0:
        return validation_error("target_price ต้องมากกว่า 0")

    try:
        with _watchlist_lock:
            post, state = _load_or_init_watchlist()
            today = datetime.now().strftime("%Y-%m-%d")

            existing_idx = next(
                (i for i, it in enumerate(state.items) if it.symbol == sym),
                None,
            )
            preserved_date = (
                state.items[existing_idx].added_date if existing_idx is not None else today
            )
            new_item = WatchlistItem(
                symbol=sym,
                asset_type=asset_type,
                target_price=target_price,
                notes=notes,
                added_date=preserved_date,
            )
            if existing_idx is not None:
                state.items[existing_idx] = new_item
                action = "[WATCH UPD]"
            else:
                state.items.append(new_item)
                action = "[WATCH ADD]"
            _save_watchlist(post, state)
            total_items = len(state.items)
    except Timeout:
        return LOCK_TIMEOUT.format(detail=f"watchlist lock {_LOCK_TIMEOUT}s")
    except ValueError as e:
        return f"Error: {e}"

    tp_note = f" target ฿{target_price:.2f}" if target_price else ""
    return f"{action} {sym} ({asset_type}){tp_note} | total: {total_items}"


@tool
def remove_from_watchlist(symbol: str) -> str:
    """ลบสินทรัพย์ออกจาก Watchlist

    [Usage/When to use]
    ใช้ลบสินทรัพย์ที่เลิกสนใจติดตามแล้ว

    Args:
        symbol (str): Ticker ที่ต้องการลบ
    """
    sym = symbol.strip().upper()
    if not sym:
        return validation_error("symbol ต้องไม่ว่าง")

    try:
        with _watchlist_lock:
            post, state = _load_or_init_watchlist()
            existing = next((it for it in state.items if it.symbol == sym), None)
            if existing is None:
                return validation_error(f"ไม่พบ {sym} ใน Watchlist")
            state.items.remove(existing)
            _save_watchlist(post, state)
            remaining = len(state.items)
    except Timeout:
        return LOCK_TIMEOUT.format(detail=f"watchlist lock {_LOCK_TIMEOUT}s")
    except ValueError as e:
        return f"Error: {e}"

    return f"[WATCH DEL] {sym} | remaining: {remaining}"


@tool
def read_watchlist() -> str:
    """อ่านรายการสินทรัพย์ที่อยู่ใน Watchlist

    [Usage/When to use]
    ใช้เพื่อดูรายการสินทรัพย์ที่จับตาดูอยู่และราคาเป้าหมาย

    Returns:
        str: ข้อมูล Watchlist ในรูปแบบ JSON String
    """
    try:
        with _watchlist_lock:
            _, state = _load_or_init_watchlist()
            dump = state.model_dump(exclude_none=True)
    except Timeout:
        return json.dumps(
            {"error": f"watchlist lock timeout ({_LOCK_TIMEOUT}s)"},
            ensure_ascii=False,
        )

    items = dump.get("items", [])
    return json.dumps(
        {
            "n_items": len(items),
            "last_updated": dump.get("last_updated"),
            "items": items,
        },
        ensure_ascii=False,
        indent=2,
    )




_WATCHLIST_LOCK_PATH = str(WATCHLIST_PATH) + ".lock"
_watchlist_lock = FileLock(_WATCHLIST_LOCK_PATH, timeout=_LOCK_TIMEOUT)
