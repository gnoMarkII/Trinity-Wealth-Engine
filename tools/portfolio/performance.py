import csv
from io import StringIO
import json
import os
from datetime import datetime, timezone
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
    """Append a versioned end-of-day portfolio snapshot."""
    from decimal import Decimal, InvalidOperation
    from uuid import uuid4
    from tools.portfolio.adapters.markdown.performance_adapter import MarkdownPerformanceAdapter

    lock = _get_portfolio_lock(portfolio_id)
    try:
        with lock:
            post, state = _load_or_init(portfolio_id=portfolio_id)
            if refresh_prices:
                _refresh_prices(state)
                _save(post, state, portfolio_id=portfolio_id)
            else:
                _recalc_all(state)

            def decimal_value(value) -> Decimal:
                try:
                    result = Decimal(str(value if value is not None else 0))
                except (InvalidOperation, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid portfolio performance value: {value!r}") from exc
                if not result.is_finite():
                    raise ValueError("portfolio performance values must be finite")
                return result

            current_fx = _require_fx(state)
            fx_dec = decimal_value(current_fx)
            total_nav = decimal_value(state.summary.total_value_thb)
            total_cost = decimal_value(_compute_total_cost(state, current_fx))
            unrealized = decimal_value(state.summary.total_unrealized_profit)
            cash_balance = sum(
                (decimal_value(h.market_value_thb) for h in state.holdings if h.asset_type == "Cash"),
                Decimal("0"),
            )
            realized_ytd = decimal_value(state.summary.total_realized_profit_ytd)
            passive_ytd = decimal_value(state.summary.passive_income_ytd)
            cutoff = datetime.now(timezone.utc)
            cutoff_utc = cutoff.isoformat().replace("+00:00", "Z")
            from tools.portfolio.adapters.markdown.paths import get_trades_log_filepath
            from tools.portfolio.adapters.markdown.repository_adapter import _read_and_migrate_trade_log_locked
            from tools.portfolio.services.performance_service import build_accrued_dividend_context

            trade_rows = _read_and_migrate_trade_log_locked(get_trades_log_filepath(portfolio_id))
            accrual_ctx = build_accrued_dividend_context(
                state=state,
                trade_rows=trade_rows,
                as_of_date=cutoff.date().isoformat(),
                fx_dec=fx_dec,
            )
            accrued_receivable = accrual_ctx["accrued_receivable_thb"]
            economic_nav = total_nav + accrued_receivable
            asset_class_values: dict[str, Decimal] = {}
            for holding in state.holdings:
                asset_type = holding.asset_type or "Unknown"
                value = decimal_value(holding.market_value_thb).quantize(Decimal("0.01"))
                asset_class_values[asset_type] = asset_class_values.get(asset_type, Decimal("0")) + value
            row = {
                "Date": cutoff.date().isoformat(),
                "Total_NAV_THB": str(total_nav.quantize(Decimal("0.01"))),
                "Accrued_Dividend_Receivable_THB": str(accrued_receivable.quantize(Decimal("0.01"))),
                "Economic_NAV_THB": str(economic_nav.quantize(Decimal("0.01"))),
                "Total_Cost": str(total_cost.quantize(Decimal("0.01"))),
                "Unrealized_PnL": str(unrealized.quantize(Decimal("0.01"))),
                "Cash_Balance": str(cash_balance.quantize(Decimal("0.01"))),
                "Realized_PnL_YTD": str(realized_ytd.quantize(Decimal("0.01"))),
                "Passive_Income_YTD": str(passive_ytd.quantize(Decimal("0.01"))),
                "Asset_Class_Values_THB": {
                    key: str(value.quantize(Decimal("0.01")))
                    for key, value in asset_class_values.items()
                },
                "Snapshot_ID": f"PERF-{uuid4().hex}",
                "Cutoff_UTC": cutoff_utc,
                "Recorded_At_UTC": cutoff_utc,
                "Flow_Coverage_Status": "MISSING_FLOW_HISTORY",
                "Net_External_Flow_THB": None,
                "Cash_Flow_Event_IDs_JSON": [],
            }
    except Timeout:
        return LOCK_TIMEOUT.format(detail=f"portfolio lock {_LOCK_TIMEOUT}s")
    except ValueError as e:
        return f"Error: {e}"

    try:
        MarkdownPerformanceAdapter().upsert_snapshot(portfolio_id, row)
    except (Timeout, ValueError, OSError) as e:
        return f"Error: {e}"

    return (
        f"[PERF] {row['Date']} | recorded | NAV: {total_nav:,.2f} | "
        f"Cost: {total_cost:,.2f} | PnL: {unrealized:+,.2f} | "
        f"Cash: {cash_balance:,.2f}"
    )


def get_structured_performance_history(days: int | None = None, portfolio_id: str = "default") -> list[dict]:
    from tools.portfolio.adapters.markdown.performance_adapter import MarkdownPerformanceAdapter

    rows = MarkdownPerformanceAdapter().read_history(portfolio_id=portfolio_id, days=days)
    rows.sort(key=lambda row: (str(row.get("Date") or ""), int(row.get("Revision") or 0)))
    numeric_fields = (
        "Total_NAV", "Total_Cost", "Unrealized_PnL", "Cash_Balance",
        "Realized_PnL_YTD", "Passive_Income_YTD", "Total_NAV_THB",
        "Accrued_Dividend_Receivable_THB", "Economic_NAV_THB",
        "Net_External_Flow_THB",
    )
    result = []
    for row in rows:
        item = dict(row)
        for field in numeric_fields:
            value = item.get(field)
            if value in (None, ""):
                item[field] = None
            else:
                try:
                    item[field] = float(value)
                except (TypeError, ValueError):
                    item[field] = None
        for field in ("Revision", "Source_State_Sequence"):
            value = item.get(field)
            if value not in (None, ""):
                try:
                    item[field] = int(value)
                except (TypeError, ValueError):
                    item[field] = None
        item["realized_pnl_ytd"] = item.get("Realized_PnL_YTD")
        item["passive_income_ytd"] = item.get("Passive_Income_YTD")
        result.append(item)
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

    from decimal import Decimal
    from tools.portfolio.adapters.markdown.performance_adapter import MarkdownPerformanceAdapter

    all_rows = MarkdownPerformanceAdapter().read_history(portfolio_id=portfolio_id)
    latest_by_date: dict[str, dict] = {}
    for item in all_rows:
        date_key = str(item.get("Date") or "")[:10]
        prior = latest_by_date.get(date_key)
        if prior is None or int(item.get("Revision") or 0) >= int(prior.get("Revision") or 0):
            latest_by_date[date_key] = item
    rows = [latest_by_date[key] for key in sorted(latest_by_date)][-days:]
    if not rows:
        return json.dumps({"error": "Performance_Log.csv ว่างเปล่า"}, ensure_ascii=False)

    settled_navs = [
        Decimal(str(row.get("Total_NAV_THB") or row.get("Total_NAV")))
        for row in rows
    ]
    navs = [
        Decimal(str(row.get("Economic_NAV_THB") or row.get("Total_NAV_THB") or row.get("Total_NAV")))
        for row in rows
    ]
    if any(not nav.is_finite() for nav in settled_navs) or any(not nav.is_finite() for nav in navs):
        return json.dumps({"error": "Performance_Log.csv contains an invalid NAV"}, ensure_ascii=False)
    first_nav = navs[0]
    latest_nav = navs[-1]
    change_abs = latest_nav - first_nav
    change_pct = (change_abs / first_nav * Decimal("100")) if first_nav > 0 else Decimal("0")

    peak = navs[0]
    max_dd = Decimal("0")
    for nav in navs:
        if nav > peak:
            peak = nav
        drawdown = ((nav - peak) / peak * Decimal("100")) if peak > 0 else Decimal("0")
        if drawdown < max_dd:
            max_dd = drawdown

    return json.dumps(
        {
            "window_days": days,
            "n_observations": len(rows),
            "nav_basis": "economic_nav",
            "first_date": rows[0]["Date"],
            "latest_date": rows[-1]["Date"],
            "first_nav": float(first_nav.quantize(Decimal("0.01"))),
            "latest_nav": float(latest_nav.quantize(Decimal("0.01"))),
            "first_settled_nav": float(settled_navs[0].quantize(Decimal("0.01"))),
            "latest_settled_nav": float(settled_navs[-1].quantize(Decimal("0.01"))),
            "first_economic_nav": float(first_nav.quantize(Decimal("0.01"))),
            "latest_economic_nav": float(latest_nav.quantize(Decimal("0.01"))),
            "change_abs": float(change_abs.quantize(Decimal("0.01"))),
            "change_pct": float(change_pct.quantize(Decimal("0.01"))),
            "max_nav": float(max(navs).quantize(Decimal("0.01"))),
            "min_nav": float(min(navs).quantize(Decimal("0.01"))),
            "max_drawdown_pct": float(max_dd.quantize(Decimal("0.01"))),
            "rows": rows,
        },
        ensure_ascii=False,
        indent=2,
    )
