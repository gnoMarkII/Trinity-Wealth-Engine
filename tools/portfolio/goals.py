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

_GOALS_KEY_ORDER = ("schema_version", "doc_type", "last_updated", "goals")


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



def _atomic_write_goals(serialized: str) -> None:
    parent = GOALS_PATH.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".goals_", suffix=".md.tmp", dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
        os.replace(tmp_path, GOALS_PATH)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _goal_item_to_md(goal: GoalItem) -> str:
    lines = [
        "---",
        f"schema_version: {goal.schema_version}",
        "entity_type: goal",
        "derived: true",
        f"name: {goal.name}",
        f"goal_type: {goal.goal_type}",
    ]
    lines.append(f"target_amount_thb: {goal.target_amount_thb}")
    lines.append(f"created_date: {goal.created_date}")
    if goal.deadline is not None:
        lines.append(f"deadline: {goal.deadline}")
    if goal.notes is not None:
        notes_escaped = goal.notes.replace('"', '\\"')
        lines.append(f'notes: "{notes_escaped}"')
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _sync_goals_sidecars(state: GoalsState) -> None:
    GOALS_ITEMS_DIR.mkdir(parents=True, exist_ok=True)
    live: set[str] = set()

    for goal in state.goals:
        safe = goal.name.replace("/", "_").replace(" ", "_")
        _atomic_write_to(GOALS_ITEMS_DIR / f"{safe}.md", _goal_item_to_md(goal))
        live.add(safe)

    for old in GOALS_ITEMS_DIR.glob("*.md"):
        if old.stem not in live:
            old.unlink(missing_ok=True)
            log.debug("[SIDECAR DEL] | goals/%s", old.name)


def _save_goals(post: frontmatter.Post, state: GoalsState) -> None:
    state.last_updated = _now_iso()
    dump = state.model_dump(exclude_none=True)

    ordered: dict = {}
    for key in _GOALS_KEY_ORDER:
        if key in dump:
            ordered[key] = dump.pop(key)
    ordered.update(dump)

    post.metadata.clear()
    post.metadata.update(ordered)
    post.content = ""

    _atomic_write_goals(frontmatter.dumps(post, sort_keys=False))
    _sync_goals_sidecars(state)


def _load_or_init_goals() -> tuple[frontmatter.Post, GoalsState]:
    if not GOALS_PATH.exists():
        GOALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(content="")
        state = GoalsState(last_updated=_now_iso())
        _save_goals(post, state)
        return post, state

    with GOALS_PATH.open("r", encoding="utf-8") as f:
        post = frontmatter.load(f)

    if not post.metadata:
        log.warning("Goals.md ไม่มี YAML frontmatter — บูตข้อมูลใหม่")
        state = GoalsState(last_updated=_now_iso())
        _save_goals(post, state)
        return post, state

    state = GoalsState.model_validate(post.metadata)
    return post, state


@tool
def set_goal(
    name: str,
    goal_type: Literal["nav_target", "cash_target", "passive_income_ytd"],
    target_amount_thb: float,
    deadline: str | None = None,
    years_from_now: int | None = None,
    notes: str | None = None,
) -> str:
    """บันทึกหรืออัปเดตเป้าหมายทางการเงิน (Financial Goals)

    [Usage/When to use]
    ใช้ตั้งเป้าหมายทางการเงิน เช่น เป้าหมายมูลค่าพอร์ต (NAV), จำนวนเงินสดสำรอง, หรือรายได้ Passive Income

    [Caution]
    - ข้อมูลเป้าหมายจะถูกใช้เมื่อสั่งคำนวณ Progress

    Args:
        name (str): ชื่อเป้าหมาย
        goal_type (Literal["nav_target", "cash_target", "passive_income_ytd"]): ประเภทเป้าหมาย
        target_amount_thb (float): จำนวนเป้าหมาย (บาท)
        deadline (str | None): กำหนดเวลา (ถ้ามี)
        years_from_now (int | None): จำนวนปีจากปัจจุบัน (ถ้ามี)
        notes (str | None): บันทึกเพิ่มเติม
    Returns:
        str: ข้อความยืนยันการบันทึกเป้าหมาย

    Raises:
        ValueError: name ว่าง, target_amount_thb <= 0, deadline format ผิด
    """
    nm = name.strip()
    if not nm:
        return validation_error("name ต้องไม่ว่าง")
    if target_amount_thb <= 0:
        return validation_error("target_amount_thb ต้องมากกว่า 0")
    if years_from_now is not None:
        if years_from_now <= 0:
            return validation_error("years_from_now ต้องมากกว่า 0")
        if deadline is not None:
            return validation_error("ห้ามระบุทั้ง deadline และ years_from_now พร้อมกัน")
        
        # วันสิ้นปีของปีเป้าหมาย
        target_year = datetime.now().year + years_from_now
        deadline = f"{target_year}-12-31"
        
    if deadline is not None:
        try:
            datetime.strptime(deadline, "%Y-%m-%d")
        except ValueError:
            return validation_error(f"deadline ต้องอยู่ในรูปแบบ 'YYYY-MM-DD' (got '{deadline}')")

    try:
        with _goals_lock:
            post, state = _load_or_init_goals()
            today = datetime.now().strftime("%Y-%m-%d")

            existing_idx = next(
                (i for i, g in enumerate(state.goals) if g.name == nm), None
            )
            preserved_date = (
                state.goals[existing_idx].created_date if existing_idx is not None else today
            )
            new_goal = GoalItem(
                name=nm,
                goal_type=goal_type,
                target_amount_thb=target_amount_thb,
                deadline=deadline,
                notes=notes,
                created_date=preserved_date,
            )
            if existing_idx is not None:
                state.goals[existing_idx] = new_goal
                action = "[GOAL UPD]"
            else:
                state.goals.append(new_goal)
                action = "[GOAL SET]"
            _save_goals(post, state)
            total = len(state.goals)
    except Timeout:
        return LOCK_TIMEOUT.format(detail=f"goals lock {_LOCK_TIMEOUT}s")
    except ValueError as e:
        return f"Error: {e}"

    dl_note = f" | deadline: {deadline}" if deadline else ""
    return f"{action} {nm} ({goal_type}) target: {target_amount_thb:,.2f} THB{dl_note} | total: {total}"


@tool
def remove_goal(name: str) -> str:
    """ลบเป้าหมายทางการเงินออกจากระบบ

    [Usage/When to use]
    ใช้เมื่อต้องการยกเลิกหรือลบเป้าหมายที่ไม่ต้องการติดตามแล้ว

    Args:
        name (str): ชื่อเป้าหมายที่ต้องการลบ
    """
    nm = name.strip()
    if not nm:
        return validation_error("name ต้องไม่ว่าง")

    try:
        with _goals_lock:
            post, state = _load_or_init_goals()
            existing = next((g for g in state.goals if g.name == nm), None)
            if existing is None:
                return validation_error(f"ไม่พบเป้าหมาย '{nm}'")
            state.goals.remove(existing)
            _save_goals(post, state)
            remaining = len(state.goals)
    except Timeout:
        return LOCK_TIMEOUT.format(detail=f"goals lock {_LOCK_TIMEOUT}s")
    except ValueError as e:
        return f"Error: {e}"

    return f"[GOAL DEL] {nm} | remaining: {remaining}"


def get_structured_goals() -> list[dict]:
    """Structured read accessor คืนค่าความคืบหน้าของเป้าหมายทั้งหมดเป็น list of dicts"""
    with _portfolio_lock:
        _, port_state = _load_or_init()
        _recalc_all(port_state)
        current_fx = port_state.fx_rates.get("USDTHB", 0.0) or 0.0
        cash_thb = next(
            (h.units for h in port_state.holdings if h.symbol == CASH_THB_SYMBOL), 0.0
        )
        cash_usd = next(
            (h.units for h in port_state.holdings if h.symbol == CASH_USD_SYMBOL), 0.0
        )
        total_cash_thb = round(cash_thb + cash_usd * current_fx, _MONEY_DP)
        nav = port_state.summary.total_value_thb
        passive_ytd = port_state.summary.passive_income_ytd

        with _goals_lock:
            _, goals_state = _load_or_init_goals()

    now = datetime.now()
    results = []
    for g in goals_state.goals:
        if g.goal_type == "nav_target":
            current = nav
        elif g.goal_type == "cash_target":
            current = total_cash_thb
        else:  # passive_income_ytd
            current = passive_ytd

        pct = round(
            (current / g.target_amount_thb * 100) if g.target_amount_thb > 0 else 0.0,
            _PCT_DP,
        )

        entry: dict = {
            "name": g.name,
            "goal_type": g.goal_type,
            "target_amount_thb": g.target_amount_thb,
            "current_amount_thb": round(current, _MONEY_DP),
            "progress_pct": pct,
        }
        if g.deadline:
            try:
                dl = datetime.strptime(g.deadline, "%Y-%m-%d")
                entry["deadline"] = g.deadline
                entry["deadline_days_left"] = (dl - now).days
            except ValueError:
                entry["deadline"] = g.deadline
        if g.notes:
            entry["notes"] = g.notes
        results.append(entry)
    return results


@tool
def get_goals_progress() -> str:
    """เรียกดูความคืบหน้าของเป้าหมายทั้งหมด

    [Usage/When to use]
    ใช้เมื่อต้องการคำนวณ Progress (%) เทียบยอดเงินใน Portfolio กับเป้าหมายที่บันทึกไว้

    Returns:
        str: JSON string ประกอบด้วยสถานะของแต่ละเป้าหมาย
    """
    try:
        results = get_structured_goals()
    except Timeout:
        return json.dumps(
            {"error": f"lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน"},
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "n_goals": len(results),
            "goals": results,
            "generated_at": _now_iso(),
        },
        ensure_ascii=False,
        indent=2,
    )


def structured_upsert_goal(
    name: str,
    goal_type: Literal["nav_target", "cash_target", "passive_income_ytd"],
    target_amount_thb: float,
    deadline: str | None = None,
    years_from_now: int | None = None,
    notes: str | None = None,
) -> list[dict]:
    """Structured mutation accessor สำหรับเพิ่มหรืออัปเดตเป้าหมาย"""
    nm = name.strip()
    if not nm:
        raise ValueError("name ต้องไม่ว่าง")
    if target_amount_thb <= 0:
        raise ValueError("target_amount_thb ต้องมากกว่า 0")
    if years_from_now is not None:
        if years_from_now <= 0:
            raise ValueError("years_from_now ต้องมากกว่า 0")
        if deadline is not None:
            raise ValueError("ห้ามระบุทั้ง deadline และ years_from_now พร้อมกัน")
        target_year = datetime.now().year + years_from_now
        deadline = f"{target_year}-12-31"
    if deadline is not None:
        try:
            datetime.strptime(deadline, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"deadline ต้องอยู่ในรูปแบบ 'YYYY-MM-DD' (got '{deadline}')")

    with _goals_lock:
        post, state = _load_or_init_goals()
        today = datetime.now().strftime("%Y-%m-%d")
        existing_idx = next(
            (i for i, g in enumerate(state.goals) if g.name == nm), None
        )
        preserved_date = (
            state.goals[existing_idx].created_date if existing_idx is not None else today
        )
        new_goal = GoalItem(
            name=nm,
            goal_type=goal_type,
            target_amount_thb=target_amount_thb,
            deadline=deadline,
            notes=notes,
            created_date=preserved_date,
        )
        if existing_idx is not None:
            state.goals[existing_idx] = new_goal
        else:
            state.goals.append(new_goal)
        _save_goals(post, state)

    return get_structured_goals()


def structured_remove_goal(name: str) -> list[dict]:
    """Structured mutation accessor สำหรับลบเป้าหมาย"""
    nm = name.strip()
    if not nm:
        raise ValueError("name ต้องไม่ว่าง")
    with _goals_lock:
        post, state = _load_or_init_goals()
        existing = next((g for g in state.goals if g.name == nm), None)
        if existing is None:
            raise ValueError(f"ไม่พบเป้าหมาย '{nm}'")
        state.goals.remove(existing)
        _save_goals(post, state)
    return get_structured_goals()


GOALS_ITEMS_DIR = VAULT_PATH / "20_Portfolio_Management/Goals/Items"
_GOALS_LOCK_PATH = str(GOALS_PATH) + ".lock"
_goals_lock = FileLock(_GOALS_LOCK_PATH, timeout=_LOCK_TIMEOUT)
