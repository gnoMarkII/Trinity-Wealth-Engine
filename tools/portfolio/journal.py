import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

from filelock import Timeout
from langchain_core.tools import tool

from core.logger import get_logger
from tools._atomic_io import _atomic_write_to
from tools.tool_errors import LOCK_TIMEOUT, validation_error
from .core import _get_portfolio_lock, _normalize_portfolio_id, _portfolio_exists
from .constants import (
    CASH_THB_SYMBOL,
    CASH_USD_SYMBOL,
    CASH_SYMBOL,
    _CASH_SYMBOLS,
    _LOCK_TIMEOUT,
    PORTFOLIOS_DIR,
    get_journal_filepath as _get_journal_filepath,
)

log = get_logger(__name__)

_TRADE_TITLE_RE = re.compile(r'(\*\*\[[\w\s]+\]\*\*\s+)([A-Z][\w.\-]*)([^\]]*\]\*\*)(?!\s*—\s*\[\[)')
_JOURNAL_BLOCK_RE = re.compile(
    r"^##\s+\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*\n(?P<body>.*?)(?=\n##\s+\[\d{4}-\d{2}-\d{2}|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _inject_journal_wikilinks(content: str) -> str:
    def _replace(m: re.Match) -> str:
        symbol = m.group(2)
        if symbol in _CASH_SYMBOLS:
            return m.group(0)
        return f"{m.group(1)}{symbol}{m.group(3)} — [[{symbol}]]"
    return _TRADE_TITLE_RE.sub(_replace, content)


def _write_journal_entry(content: str, date_str: str | None = None, portfolio_id: str = "default") -> str:
    pid = _normalize_portfolio_id(portfolio_id)
    if pid != "default" and not _portfolio_exists(pid):
        raise ValueError(f"ไม่พบพอร์ตไอดี '{pid}' ในระบบ — ใช้ tool_create_portfolio ก่อน")
    jpath = _get_journal_filepath(pid)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    if date_str and date_str.strip():
        val = date_str.strip()
        if len(val) == 10:
            timestamp = f"{val} 12:00:00"
        else:
            timestamp = val
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linked = _inject_journal_wikilinks(content)
    block = f"\n## [{timestamp}]\n\n{linked}\n"
    existing = jpath.read_text(encoding="utf-8") if jpath.exists() else ""
    _atomic_write_to(jpath, existing + block)
    return timestamp


@tool
def append_trading_journal(entry: str, portfolio_id: str = "default") -> str:
    """บันทึกการเทรดและข้อคิดเห็น (Trading Journal)"""
    content = (entry or "").strip()
    if not content:
        return validation_error("entry ต้องไม่ว่าง")

    lock = _get_portfolio_lock(portfolio_id)
    try:
        with lock:
            timestamp = _write_journal_entry(content, portfolio_id=portfolio_id)
    except Timeout:
        return LOCK_TIMEOUT.format(detail=f"journal lock {_LOCK_TIMEOUT}s")
    except ValueError as e:
        return f"Error: {e}"

    return f"[JOURNAL] บันทึกสำเร็จ | [{timestamp}] | {len(content)} chars"


def get_structured_journal(days: int | None = 365, keyword: str | None = None, limit: int = 100, portfolio_id: str = "default") -> list[dict]:
    if days is None:
        days = 365
    jpath = _get_journal_filepath(portfolio_id)
    if not jpath.exists():
        return []
    text = jpath.read_text(encoding="utf-8")
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

    entries.reverse()
    return entries[:limit]


@tool
def read_trading_journal(
    days: int = 30,
    keyword: str | None = None,
    limit: int = 20,
    portfolio_id: str = "default",
) -> str:
    """อ่านบันทึกการเทรด (Trading Journal) ย้อนหลัง"""
    if days <= 0:
        return validation_error("days ต้องมากกว่า 0")
    if limit <= 0:
        return validation_error("limit ต้องมากกว่า 0")

    jpath = _get_journal_filepath(portfolio_id)
    if not jpath.exists():
        return json.dumps(
            {"error": "ยังไม่มี Trading_Journal.md — ใช้ append_trading_journal บันทึกก่อน"},
            ensure_ascii=False,
        )

    returned = get_structured_journal(days=days, keyword=keyword, limit=limit, portfolio_id=portfolio_id)
    all_in_window = get_structured_journal(days=days, keyword=keyword, limit=1000000, portfolio_id=portfolio_id)

    return json.dumps(
        {
            "n_total_in_window": len(all_in_window),
            "n_returned": len(returned),
            "filters_used": {"days": days, "keyword": keyword, "limit": limit},
            "entries": returned,
        },
        ensure_ascii=False,
        indent=2,
    )


def structured_append_journal(entry: str, portfolio_id: str = "default") -> list[dict]:
    content = (entry or "").strip()
    if not content:
        raise ValueError("entry ต้องไม่ว่าง")
    lock = _get_portfolio_lock(portfolio_id)
    with lock:
        _write_journal_entry(content, portfolio_id=portfolio_id)
    return get_structured_journal(days=365, limit=100, portfolio_id=portfolio_id)
