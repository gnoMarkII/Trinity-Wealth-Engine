from pathlib import Path
import frontmatter

from core.logger import get_logger
from tools._atomic_io import _atomic_write_to
from tools.portfolio.domain.models import WatchlistState, WatchlistItem, _now_iso
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.watchlist_port import WatchlistRepositoryPort
from .paths import (
    get_watchlist_filepath,
    get_watchlist_items_dir,
    get_portfolio_lock_path,
    _LOCK_TIMEOUT,
)
from tools.portfolio.domain.errors import PortfolioNotFoundError
from .repository_adapter import _get_portfolio_lock, _portfolio_exists

log = get_logger(__name__)

_WATCHLIST_KEY_ORDER = ("schema_version", "doc_type", "last_updated", "items")


def _initial_watchlist() -> WatchlistState:
    return WatchlistState(
        schema_version=1,
        doc_type="watchlist",
        last_updated=_now_iso(),
        items=[],
    )


def _item_to_md(item: WatchlistItem) -> str:
    lines = [
        "---",
        f"schema_version: {item.schema_version}",
        "entity_type: watchlist_item",
        f"symbol: {item.symbol}",
        f"asset_type: {item.asset_type}",
        "derived: true",
    ]
    if item.target_price is not None:
        lines.append(f"target_price: {item.target_price}")
    lines.append(f'added_date: "{item.added_date}"')
    lines.append("---")
    lines.append("")
    lines.append(f"# {item.symbol} (Watchlist)")
    lines.append("")
    if item.notes:
        lines.append(item.notes)
        lines.append("")
    return "\n".join(lines)


class MarkdownWatchlistAdapter(WatchlistRepositoryPort):
    """Markdown Vault storage adapter for Watchlist."""

    def load_watchlist(self, portfolio_id: str = "default") -> WatchlistState:
        pid = validate_portfolio_id(portfolio_id)
        if pid != "default" and not _portfolio_exists(pid):
            raise PortfolioNotFoundError(f"ไม่พบพอร์ตไอดี '{pid}' ในระบบ — ใช้ tool_create_portfolio ก่อน")
        lock = _get_portfolio_lock(pid)
        with lock:
            wpath = get_watchlist_filepath(pid)
            if not wpath.exists():
                state = _initial_watchlist()
                self._save_watchlist_locked(state, pid)
                return state

            with wpath.open("r", encoding="utf-8") as f:
                post = frontmatter.load(f)

            if not post.metadata:
                state = _initial_watchlist()
                self._save_watchlist_locked(state, pid)
                return state

            return WatchlistState.model_validate(post.metadata)

    def save_watchlist(self, state: WatchlistState, portfolio_id: str = "default") -> None:
        pid = validate_portfolio_id(portfolio_id)
        lock = _get_portfolio_lock(pid)
        with lock:
            self._save_watchlist_locked(state, pid)

    def _save_watchlist_locked(self, state: WatchlistState, portfolio_id: str = "default") -> None:
        state.last_updated = _now_iso()
        dump = state.model_dump(exclude_none=True)

        ordered = {}
        for key in _WATCHLIST_KEY_ORDER:
            if key in dump:
                ordered[key] = dump.pop(key)
        ordered.update(dump)

        post = frontmatter.Post(content="", **ordered)
        serialized = frontmatter.dumps(post, sort_keys=False)
        wpath = get_watchlist_filepath(portfolio_id)
        _atomic_write_to(wpath, serialized)

        # Sync items sidecars
        items_dir = get_watchlist_items_dir(portfolio_id)
        items_dir.mkdir(parents=True, exist_ok=True)
        live: set[str] = set()

        for it in state.items:
            safe = it.symbol.replace("/", "_")
            _atomic_write_to(items_dir / f"{safe}.md", _item_to_md(it))
            live.add(safe)

        for old in items_dir.glob("*.md"):
            if old.stem not in live:
                old.unlink(missing_ok=True)
