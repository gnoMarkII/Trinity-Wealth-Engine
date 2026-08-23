from typing import Optional, Tuple
import frontmatter
from tools.portfolio import get_default_service
from tools.portfolio.domain.models import WatchlistState, WatchlistItem, _now_iso
from tools.portfolio.adapters.markdown.paths import (
    get_watchlist_filepath as _get_watchlist_filepath,
    get_watchlist_items_dir as _get_watchlist_items_dir,
)
from tools.portfolio.adapters.markdown.repository_adapter import _get_portfolio_lock
from tools.portfolio.agent_tools import add_to_watchlist, remove_from_watchlist, read_watchlist

def _load_or_init_watchlist(portfolio_id: str = "default") -> Tuple[frontmatter.Post, WatchlistState]:
    wpath = _get_watchlist_filepath(portfolio_id)
    if not wpath.exists():
        state = WatchlistState(schema_version=1, doc_type="watchlist", last_updated=_now_iso(), items=[])
        post = frontmatter.Post(content="", metadata=state.model_dump())
        return post, state
    with wpath.open("r", encoding="utf-8") as f:
        post = frontmatter.load(f)
    if not post.metadata:
        state = WatchlistState(schema_version=1, doc_type="watchlist", last_updated=_now_iso(), items=[])
        return post, state
    state = WatchlistState.model_validate(post.metadata)
    return post, state

def get_structured_watchlist(portfolio_id: str = "default") -> WatchlistState:
    return get_default_service().get_structured_watchlist(portfolio_id=portfolio_id)

def structured_upsert_watchlist_item(
    symbol: str, asset_type: str, target_price: Optional[float] = None, notes: str = "", portfolio_id: str = "default"
) -> WatchlistState:
    return get_default_service().structured_upsert_watchlist_item(
        symbol=symbol, asset_type=asset_type, target_price=target_price, notes=notes, portfolio_id=portfolio_id
    )

def structured_remove_watchlist_item(symbol: str, portfolio_id: str = "default") -> WatchlistState:
    return get_default_service().structured_remove_watchlist_item(symbol=symbol, portfolio_id=portfolio_id)
