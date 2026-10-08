import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, List, Dict

import frontmatter
from filelock import FileLock, Timeout
from langchain_core.tools import tool

from core.logger import get_logger
from tools._atomic_io import _atomic_write_to
from tools.archivist.metadata import dump_note
from tools.archivist.maintenance_guard import assert_write_allowed
from tools.portfolio.adapters.markdown.identity import portfolio_note_identity
from tools.portfolio.adapters.markdown.paths import get_vault_path
from tools.tool_errors import LOCK_TIMEOUT, validation_error
from .core import _load_or_init, _recalc_all, _get_portfolio_lock
from .models import _now_iso, GoalItem, GoalsState, PortfolioState
from .constants import (
    CASH_THB_SYMBOL,
    CASH_USD_SYMBOL,
    _MONEY_DP,
    _PCT_DP,
    _LOCK_TIMEOUT,
    VAULT_PATH,
    GOALS_PATH,
)

log = get_logger(__name__)

_GOALS_KEY_ORDER = ("schema_version", "doc_type", "last_updated", "goals")
GOALS_ITEMS_DIR = VAULT_PATH / "20_Portfolio_Management/Goals/Items"
_GOALS_LOCK_PATH = str(GOALS_PATH) + ".lock"
_goals_lock = FileLock(_GOALS_LOCK_PATH, timeout=_LOCK_TIMEOUT)


def _get_goals_filepath() -> Path:
    return GOALS_PATH


def _atomic_write_goals(serialized: str) -> None:
    assert_write_allowed(GOALS_PATH)
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
    item_key = f"{goal.name}:{goal.goal_type}:{goal.created_date}"
    note_id, document_key = portfolio_note_identity(
        get_vault_path(), goal.portfolio_id, "goal_item", item_key
    )
    metadata = {
        "schema_version": 2,
        "note_id": note_id,
        "document_key": document_key,
        "title": goal.name,
        "entity_type": "goal",
        "document_role": "portfolio_item",
        "portfolio_id": goal.portfolio_id,
        "search_scope": "excluded",
        "derived": True,
        "name": goal.name,
        "goal_type": goal.goal_type,
        "target_amount_thb": goal.target_amount_thb,
        "created_date": goal.created_date,
    }
    if goal.deadline is not None:
        metadata["deadline"] = goal.deadline
    if goal.notes is not None:
        metadata["notes"] = goal.notes
    if goal.bucket_id is not None:
        metadata["bucket_id"] = goal.bucket_id
    return dump_note(metadata, "")


def _sync_goals_sidecars(state: GoalsState) -> None:
    assert_write_allowed(GOALS_ITEMS_DIR)
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
    note_id, document_key = portfolio_note_identity(get_vault_path(), "default", "goals")
    post.metadata.update(
        {
            "schema_version": 2,
            "note_id": note_id,
            "document_key": document_key,
            "title": "Portfolio Goals",
            "entity_type": "portfolio_state",
            "document_role": "goals",
            "portfolio_id": "default",
            "search_scope": "excluded",
        }
    )
    post.content = ""

    _atomic_write_goals(frontmatter.dumps(post, sort_keys=False))
    _sync_goals_sidecars(state)


def _load_or_init_goals() -> tuple[frontmatter.Post, GoalsState]:
    if not GOALS_PATH.exists():
        assert_write_allowed(GOALS_PATH)
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
    goal_type: Literal["nav_target", "cash_target", "passive_income_ytd", "bucket_target"],
    target_amount_thb: float,
    deadline: str | None = None,
    years_from_now: int | None = None,
    notes: str | None = None,
    portfolio_id: str = "default",
    bucket_id: str | None = None,
) -> str:
    """บันทึกหรืออัปเดตเป้าหมายทางการเงิน (Financial Goals)"""
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
                portfolio_id=portfolio_id,
                bucket_id=bucket_id,
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
    """ลบเป้าหมายทางการเงินออกจากระบบ"""
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


def _compute_structured_goals(
    goals_state: GoalsState,
    portfolio_id: str | None = None,
) -> list[dict]:
    now = datetime.now()
    results = []
    port_states: dict[str, PortfolioState] = {}

    for g in goals_state.goals:
        pid = g.portfolio_id if g.portfolio_id else "default"
        if portfolio_id and pid != portfolio_id:
            continue

        if pid not in port_states:
            try:
                p_lock = _get_portfolio_lock(pid)
                with p_lock:
                    _, p_state = _load_or_init(portfolio_id=pid)
                    _recalc_all(p_state)
                    port_states[pid] = p_state
            except Exception:
                p_state = PortfolioState(last_updated=_now_iso())
                port_states[pid] = p_state
        else:
            p_state = port_states[pid]

        current_fx = p_state.fx_rates.get("USDTHB", 0.0) or 0.0
        cash_thb = next(
            (h.units for h in p_state.holdings if h.symbol == CASH_THB_SYMBOL), 0.0
        )
        cash_usd = next(
            (h.units for h in p_state.holdings if h.symbol == CASH_USD_SYMBOL), 0.0
        )
        total_cash_thb = round(cash_thb + cash_usd * current_fx, _MONEY_DP)
        nav = p_state.summary.total_value_thb
        passive_ytd = p_state.summary.passive_income_ytd

        if g.goal_type == "nav_target":
            current = nav
        elif g.goal_type == "cash_target":
            current = total_cash_thb
        elif g.goal_type == "bucket_target":
            current = (
                round(
                    sum(h.market_value_thb for h in p_state.holdings if h.bucket_id == g.bucket_id),
                    _MONEY_DP,
                )
                if g.bucket_id
                else 0.0
            )
        else:
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
            "portfolio_id": pid,
            "bucket_id": g.bucket_id,
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


def get_structured_goals(
    portfolio_id: Optional[str] = None,
) -> List[Dict]:
    with _goals_lock:
        _, goals_state = _load_or_init_goals()
    return _compute_structured_goals(goals_state=goals_state, portfolio_id=portfolio_id)


@tool
def get_goals_progress(portfolio_id: str | None = None) -> str:
    """เรียกดูความคืบหน้าของเป้าหมายทั้งหมด"""
    try:
        results = get_structured_goals(portfolio_id=portfolio_id)
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
    goal_type: Literal["nav_target", "cash_target", "passive_income_ytd", "bucket_target"],
    target_amount_thb: float,
    deadline: str | None = None,
    years_from_now: int | None = None,
    notes: str | None = None,
    portfolio_id: str = "default",
    bucket_id: str | None = None,
) -> list[dict]:
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
            portfolio_id=portfolio_id,
            bucket_id=bucket_id,
        )
        if existing_idx is not None:
            state.goals[existing_idx] = new_goal
        else:
            state.goals.append(new_goal)
        _save_goals(post, state)

    return get_structured_goals(portfolio_id=None)


def structured_remove_goal(name: str, portfolio_id: str | None = None) -> list[dict]:
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
    return get_structured_goals(portfolio_id=portfolio_id)
