from pathlib import Path
from typing import Optional
import frontmatter
from filelock import FileLock

from core.logger import get_logger
from tools._atomic_io import _atomic_write_to
from tools.archivist.maintenance_guard import assert_write_allowed
from tools.portfolio.domain.models import GoalsState, GoalItem, _now_iso
from tools.portfolio.ports.goals_port import GoalsRepositoryPort
from .paths import GOALS_PATH, GOALS_ITEMS_DIR, _LOCK_TIMEOUT, get_vault_path
from .identity import portfolio_note_identity

log = get_logger(__name__)

_GOALS_KEY_ORDER = ("schema_version", "doc_type", "last_updated", "goals")
_goals_lock = FileLock(str(GOALS_PATH.parent / "Goals.md.lock"), timeout=_LOCK_TIMEOUT)


def _initial_goals() -> GoalsState:
    return GoalsState(
        schema_version=1,
        doc_type="goals",
        last_updated=_now_iso(),
        goals=[],
    )


def _goal_item_to_md(goal: GoalItem, vault_root: Path) -> str:
    item_key = f"{goal.name}:{goal.goal_type}:{goal.created_date}"
    note_id, document_key = portfolio_note_identity(
        vault_root, goal.portfolio_id, "goal_item", item_key
    )
    lines = [
        "---",
        "schema_version: 2",
        f"note_id: {note_id}",
        f"document_key: {document_key}",
        f"title: {goal.name}",
        "entity_type: goal",
        "document_role: portfolio_item",
        f"portfolio_id: {goal.portfolio_id}",
        "search_scope: excluded",
        "derived: true",
        f"name: {goal.name}",
        f"goal_type: {goal.goal_type}",
        f"target_amount_thb: {goal.target_amount_thb}",
        f'created_date: "{goal.created_date}"',
    ]
    if goal.deadline is not None:
        lines.append(f'deadline: "{goal.deadline}"')
    if goal.portfolio_id:
        lines.append(f'portfolio_id: "{goal.portfolio_id}"')
    if goal.bucket_id:
        lines.append(f'bucket_id: "{goal.bucket_id}"')
    if goal.notes is not None:
        notes_escaped = goal.notes.replace('"', '\\"')
        lines.append(f'notes: "{notes_escaped}"')
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


class MarkdownGoalsAdapter(GoalsRepositoryPort):
    """Markdown Vault storage adapter for Goals."""

    def load_goals(self, portfolio_id: Optional[str] = None) -> GoalsState:
        with _goals_lock:
            if not GOALS_PATH.exists():
                state = _initial_goals()
                self._save_goals_locked(state)
                return self._filter_goals(state, portfolio_id)

            with GOALS_PATH.open("r", encoding="utf-8") as f:
                post = frontmatter.load(f)

            if not post.metadata:
                state = _initial_goals()
                self._save_goals_locked(state)
                return self._filter_goals(state, portfolio_id)

            state = GoalsState.model_validate(post.metadata)
            return self._filter_goals(state, portfolio_id)

    def save_goals(self, state: GoalsState) -> None:
        with _goals_lock:
            self._save_goals_locked(state)

    def _filter_goals(self, state: GoalsState, portfolio_id: Optional[str]) -> GoalsState:
        if not portfolio_id:
            return state
        clone = state.model_copy(deep=True)
        clone.goals = [g for g in clone.goals if g.portfolio_id == portfolio_id]
        return clone

    def _save_goals_locked(self, state: GoalsState) -> None:
        state.last_updated = _now_iso()
        dump = state.model_dump(exclude_none=True)

        ordered = {}
        for key in _GOALS_KEY_ORDER:
            if key in dump:
                ordered[key] = dump.pop(key)
        ordered.update(dump)
        note_id, document_key = portfolio_note_identity(
            get_vault_path(), "default", "goals"
        )
        ordered.update({
            "schema_version": 2,
            "note_id": note_id,
            "document_key": document_key,
            "title": "Portfolio Goals",
            "entity_type": "portfolio_state",
            "document_role": "goals",
            "portfolio_id": "default",
            "search_scope": "excluded",
        })

        post = frontmatter.Post(content="", **ordered)
        serialized = frontmatter.dumps(post, sort_keys=False)
        _atomic_write_to(GOALS_PATH, serialized)

        # Sync items sidecars
        assert_write_allowed(GOALS_ITEMS_DIR)
        GOALS_ITEMS_DIR.mkdir(parents=True, exist_ok=True)
        live: set[str] = set()

        for goal in state.goals:
            safe = goal.name.replace("/", "_").replace(" ", "_")
            _atomic_write_to(GOALS_ITEMS_DIR / f"{safe}.md", _goal_item_to_md(goal, get_vault_path()))
            live.add(safe)

        for old in GOALS_ITEMS_DIR.glob("*.md"):
            if old.stem not in live:
                old.unlink(missing_ok=True)
