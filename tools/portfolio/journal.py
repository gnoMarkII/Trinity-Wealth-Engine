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

_TRADE_TITLE_RE = re.compile(r'(\*\*\[[\w\s]+\]\*\*\s+)([A-Z][\w.\-]*)([^\]]*\]\*\*)(?!\s*—\s*\[\[)')
_JOURNAL_BLOCK_RE = re.compile(
    r"^##\s+\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*\n(?P<body>.*?)(?=\n##\s+\[\d{4}-\d{2}-\d{2}|\Z)",
    re.DOTALL | re.MULTILINE,
)

from tools._atomic_io import _atomic_write_to
from tools.tool_errors import validation_error
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



def _inject_journal_wikilinks(content: str) -> str:
    """เติม ' — [[SYMBOL]]' หลัง trade title เพื่อเชื่อม Layer 3 → Layer 1 ใน Graph
    Skip cash holdings (CASH_THB/CASH_USD) — ไม่ใช่ entity ที่ต้องการ hub node
    """
    def _replace(m: re.Match) -> str:
        symbol = m.group(2)
        if symbol in _CASH_SYMBOLS:
            return m.group(0)
        return f"{m.group(1)}{symbol}{m.group(3)} — [[{symbol}]]"
    return _TRADE_TITLE_RE.sub(_replace, content)


def _write_journal_entry(content: str) -> str:
    """Append timestamped block ลง Trading_Journal.md — คืน timestamp ที่ใช้
    ใช้ได้ทั้งจาก @tool append_trading_journal และจากภายใน locked ops อื่น (เช่น edit_holding)
    """
    TRADING_JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linked = _inject_journal_wikilinks(content)
    block = f"\n## [{timestamp}]\n\n{linked}\n"
    with TRADING_JOURNAL_PATH.open("a", encoding="utf-8") as f:
        f.write(block)
    return timestamp


@tool
def append_trading_journal(entry: str) -> str:
    """บันทึกการเทรดและข้อคิดเห็น (Trading Journal)

    [Usage/When to use]
    ใช้จดบันทึกเหตุผลที่ซื้อ/ขาย สภาพตลาด บทเรียนที่ได้ หรือข้อผิดพลาด (Mistakes)
    - เพื่อแยกข้อมูล Qualitative (เหตุผล) ออกจากข้อมูล Quantitative (ตัวเลข Portfolio)

    [Caution]
    - ไม่ใช้เพื่อแก้ข้อมูลพอร์ต ให้ใช้บันทึกเป็น Text เท่านั้น

    Args:
        entry (str): เนื้อหาที่จะบันทึก (ระบบจะลง Timestamp ให้อัตโนมัติ)
    """
    content = (entry or "").strip()
    if not content:
        return validation_error("entry ต้องไม่ว่าง")

    timestamp = _write_journal_entry(content)
    return f"[JOURNAL] บันทึกสำเร็จ | [{timestamp}] | {len(content)} chars"


@tool
def read_trading_journal(
    days: int = 30,
    keyword: str | None = None,
    limit: int = 20,
) -> str:
    """อ่านบันทึกการเทรด (Trading Journal) ย้อนหลัง

    [Usage/When to use]
    ใช้ดึงประวัติการบันทึกการลงทุน (Journal) เพื่อทบทวนเหตุผล ข้อคิด หรือสรุปบทเรียน
    - สามารถระบุ keyword เพื่อกรองเฉพาะบันทึกที่เกี่ยวข้องได้

    Args:
        days (int): จำนวนวันย้อนหลังที่ต้องการดึง
        keyword (str | None): คำที่ต้องการค้นหา
        limit (int): จำนวนบันทึกสูงสุดที่จะแสดง
    Returns:
        str: JSON string: {n_total_in_window, n_returned, filters_used, entries:[{timestamp, content}]}
        entries เรียงจากใหม่สุดไปเก่าสุด
    """
    if days <= 0:
        return validation_error("days ต้องมากกว่า 0")
    if limit <= 0:
        return validation_error("limit ต้องมากกว่า 0")

    if not TRADING_JOURNAL_PATH.exists():
        return json.dumps(
            {"error": "ยังไม่มี Trading_Journal.md — ใช้ append_trading_journal บันทึกก่อน"},
            ensure_ascii=False,
        )

    text = TRADING_JOURNAL_PATH.read_text(encoding="utf-8")
    cutoff = datetime.now() - timedelta(days=days)
    kw = keyword.strip().lower() if keyword else None

    entries: list[dict] = []
    for m in _JOURNAL_BLOCK_RE.finditer(text):
        ts_str = m.group("ts")
        try:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if ts < cutoff:
            continue
        body = m.group("body").strip()
        if kw and kw not in body.lower():
            continue
        entries.append({"timestamp": ts_str, "content": body})

    # ใหม่สุดก่อน — ใช้ file order (append-only) ไม่ใช่ timestamp
    # เพราะ entries ที่เขียนติดกันใน 1 วินาทีจะ tie-break ไม่ได้ด้วย sort(timestamp)
    entries.reverse()
    n_total = len(entries)
    returned = entries[:limit]

    return json.dumps(
        {
            "n_total_in_window": n_total,
            "n_returned": len(returned),
            "filters_used": {"days": days, "keyword": keyword, "limit": limit},
            "entries": returned,
        },
        ensure_ascii=False,
        indent=2,
    )


