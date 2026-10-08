import csv
from io import StringIO
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, List, Dict

import frontmatter
from filelock import FileLock, Timeout
from langchain_core.tools import tool

from core.logger import get_logger
from tools._atomic_io import _atomic_write_to
from tools.archivist.maintenance_guard import assert_write_allowed
from tools.tool_errors import LOCK_TIMEOUT, validation_error
from .models import _now_iso, PortfolioState, Holding, Summary
from .core import _load_or_init, _save, _recalc_all, _compute_total_cost, _require_fx, _get_portfolio_lock
from .prices import _refresh_prices
from .constants import (
    _MONEY_DP,
    _PCT_DP,
    _LOCK_TIMEOUT,
    _PERFORMANCE_LOG_HEADER,
)
from .adapters.markdown.paths import get_performance_filepath as _get_performance_filepath

log = get_logger(__name__)


@tool
def record_performance_snapshot(refresh_prices: bool = True, portfolio_id: str = "default") -> str:
    """บันทึก Snapshot สถานะพอร์ตโฟลิโอ ณ สิ้นวัน (Performance Logging)"""
    lock = _get_portfolio_lock(portfolio_id)
    try:
        with lock:
            post, state = _load_or_init(portfolio_id=portfolio_id)
            if refresh_prices:
                _refresh_prices(state)
                _save(post, state, portfolio_id=portfolio_id)
            else:
                _recalc_all(state)

            current_fx = _require_fx(state)
            total_nav = state.summary.total_value_thb
            unrealized = state.summary.total_unrealized_profit
            total_cost = _compute_total_cost(state, current_fx)

            cash_balance = round(
                sum(h.market_value_thb for h in state.holdings if h.asset_type == "Cash"),
                _MONEY_DP,
            )
            realized_ytd = state.summary.total_realized_profit_ytd
            passive_ytd = state.summary.passive_income_ytd

            today = datetime.now().strftime("%Y-%m-%d")
            asset_class_values = {}
            for h in state.holdings:
                at = h.asset_type or "Unknown"
                val = round(float(h.market_value_thb or 0.0), _MONEY_DP)
                asset_class_values[at] = round(asset_class_values.get(at, 0.0) + val, _MONEY_DP)
            ac_json = json.dumps(asset_class_values, ensure_ascii=False) if asset_class_values else ""

            row = [
                today,
                f"{total_nav:.2f}",
                f"{total_cost:.2f}",
                f"{unrealized:.2f}",
                f"{cash_balance:.2f}",
                f"{realized_ytd:.2f}",
                f"{passive_ytd:.2f}",
                ac_json,
            ]

            perf_path = _get_performance_filepath(portfolio_id)
            assert_write_allowed(perf_path)
            perf_path.parent.mkdir(parents=True, exist_ok=True)
            existing_rows: list[list[str]] = []
            if perf_path.exists() and perf_path.stat().st_size > 0:
                with perf_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    header_read = False
                    for r in reader:
                        if not header_read:
                            header_read = True
                            if not r or (r and r[0] == "Date"):
                                continue
                        if r and len(r) >= 5:
                            if len(r) < len(_PERFORMANCE_LOG_HEADER):
                                r.extend([""] * (len(_PERFORMANCE_LOG_HEADER) - len(r)))
                            existing_rows.append(r[:len(_PERFORMANCE_LOG_HEADER)])

            replaced = False
            for idx, r in enumerate(existing_rows):
                if r[0] == today:
                    existing_rows[idx] = row
                    replaced = True
                    break
            if not replaced:
                existing_rows.append(row)

            output = StringIO()
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(_PERFORMANCE_LOG_HEADER)
            writer.writerows(existing_rows)
            _atomic_write_to(perf_path, output.getvalue())

    except Timeout:
        return LOCK_TIMEOUT.format(detail=f"portfolio lock {_LOCK_TIMEOUT}s")
    except ValueError as e:
        return f"Error: {e}"

    action_label = "updated" if replaced else "recorded"
    return (
        f"[PERF] {today} | {action_label} | NAV: {total_nav:,.2f} | "
        f"Cost: {total_cost:,.2f} | PnL: {unrealized:+,.2f} | "
        f"Cash: {cash_balance:,.2f}"
    )


def get_structured_performance_history(days: int | None = None, portfolio_id: str = "default") -> list[dict]:
    perf_path = _get_performance_filepath(portfolio_id)
    if not perf_path.exists():
        return []
    with perf_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if days is not None and days > 0:
        rows = rows[-days:]
    result = []
    for r in rows:
        try:
            val_realized = r.get("Realized_PnL_YTD")
            val_passive = r.get("Passive_Income_YTD")
            realized_float = float(val_realized) if val_realized not in (None, "") else None
            passive_float = float(val_passive) if val_passive not in (None, "") else None
            result.append({
                "Date": r["Date"],
                "Total_NAV": float(r["Total_NAV"]),
                "Total_Cost": float(r["Total_Cost"]),
                "Unrealized_PnL": float(r["Unrealized_PnL"]),
                "Cash_Balance": float(r["Cash_Balance"]),
                "Realized_PnL_YTD": realized_float,
                "Passive_Income_YTD": passive_float,
                "realized_pnl_ytd": realized_float,
                "passive_income_ytd": passive_float,
            })
        except (KeyError, ValueError):
            continue
    return result


@tool
def read_performance_history(days: int = 30, portfolio_id: str = "default") -> str:
    """อ่านประวัติและวิเคราะห์ผลตอบแทนของพอร์ตโฟลิโอ (Performance Analytics)"""
    if days <= 0:
        return validation_error("days ต้องมากกว่า 0")

    perf_path = _get_performance_filepath(portfolio_id)
    if not perf_path.exists():
        return json.dumps(
            {"error": "ยังไม่มี Performance_Log.csv — ใช้ record_performance_snapshot ก่อน"},
            ensure_ascii=False,
        )

    with perf_path.open("r", encoding="utf-8", newline="") as f:
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
