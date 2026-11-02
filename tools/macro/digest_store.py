"""Unified persistence store for Daily/Weekly Market Intelligence and Ticker News Index."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock
from tools._atomic_io import _atomic_write_to
from core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DIGEST_STORE_PATH = os.path.join("data", "digest_store.json")


def resolve_digest_store_path(store_path: Optional[str] = None) -> str:
    """Resolve digest store path with fallback to data/digest_store.json."""
    if store_path:
        return store_path
    env_path = os.getenv("DIGEST_STORE_PATH")
    if env_path:
        return env_path
    return os.path.abspath(DEFAULT_DIGEST_STORE_PATH)


def _get_lock_path(store_path: str) -> str:
    return f"{store_path}.lock"


def _empty_digest_store() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "last_candidates_fetched_at": None,
        "cached_news_candidates": [],
        "cached_youtube_candidates": [],
        "stale_dropped_count": 0,
        "latest_daily": None,
        "daily_history": {},
        "latest_weekly": None,
        "weekly_history": {},
        "ticker_intelligence_index": {},
        "read_item_ids": [],
        "last_user_visit_at": None,
    }


def _read_digest_store_unlocked(target_path: str) -> Dict[str, Any]:
    """Read the digest state JSON without acquiring lock (caller must hold lock or accept dirty read)."""
    if not os.path.exists(target_path):
        return _empty_digest_store()
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                default_store = _empty_digest_store()
                for k, v in default_store.items():
                    data.setdefault(k, v)
                return data
    except Exception as e:
        logger.warning("Could not read digest store from %s (%s), returning empty state", target_path, e)
    return _empty_digest_store()


def load_digest_store(store_path: Optional[str] = None) -> Dict[str, Any]:
    """Read the digest state JSON under file lock."""
    target_path = resolve_digest_store_path(store_path)
    if not os.path.exists(target_path):
        return _empty_digest_store()

    lock_file = _get_lock_path(target_path)
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    with FileLock(lock_file, timeout=10):
        return _read_digest_store_unlocked(target_path)


def _prune_digest_store(state: Dict[str, Any]) -> None:
    """Keep digest_store bounded in size by pruning historical records and indexed items."""
    # 1. daily_history: keep top 60 newest records
    daily_history = state.get("daily_history")
    if isinstance(daily_history, dict) and len(daily_history) > 60:
        sorted_keys = sorted(daily_history.keys(), reverse=True)
        keep_keys = set(sorted_keys[:60])
        state["daily_history"] = {k: daily_history[k] for k in sorted_keys if k in keep_keys}

    # 2. weekly_history: keep top 26 newest records
    weekly_history = state.get("weekly_history")
    if isinstance(weekly_history, dict) and len(weekly_history) > 26:
        sorted_keys = sorted(weekly_history.keys(), reverse=True)
        keep_keys = set(sorted_keys[:26])
        state["weekly_history"] = {k: weekly_history[k] for k in sorted_keys if k in keep_keys}

    # 3. ticker_intelligence_index: cap news_items at 25, youtube_mentions at 15 per symbol
    ticker_index = state.get("ticker_intelligence_index")
    if isinstance(ticker_index, dict):
        for sym, entry in ticker_index.items():
            if isinstance(entry, dict):
                news_items = entry.get("news_items")
                if isinstance(news_items, list) and len(news_items) > 25:
                    entry["news_items"] = news_items[:25]
                yt_mentions = entry.get("youtube_mentions")
                if isinstance(yt_mentions, list) and len(yt_mentions) > 15:
                    entry["youtube_mentions"] = yt_mentions[:15]

    # 4. read_item_ids: cap at latest 1,000 entries
    read_ids = state.get("read_item_ids")
    if isinstance(read_ids, list) and len(read_ids) > 1000:
        state["read_item_ids"] = read_ids[-1000:]


def save_pre_llm_candidates(
    news_candidates: List[Dict[str, Any]],
    youtube_candidates: List[Dict[str, Any]],
    stale_dropped_count: int,
    store_path: Optional[str] = None,
    cache_bucket: str = "daily",
) -> None:
    """Cache fresh candidates for the interactive Pre-LLM Selection Gate."""
    target_path = resolve_digest_store_path(store_path)
    lock_file = _get_lock_path(target_path)
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    with FileLock(lock_file, timeout=10):
        state = _read_digest_store_unlocked(target_path)
        now_iso = datetime.now(timezone.utc).isoformat()
        state[f"last_{cache_bucket}_candidates_fetched_at"] = now_iso
        state[f"cached_{cache_bucket}_news_candidates"] = news_candidates
        state[f"cached_{cache_bucket}_youtube_candidates"] = youtube_candidates
        state[f"stale_dropped_count_{cache_bucket}"] = stale_dropped_count

        # Maintain backwards compatibility
        if cache_bucket == "daily":
            state["last_candidates_fetched_at"] = now_iso
            state["cached_news_candidates"] = news_candidates
            state["cached_youtube_candidates"] = youtube_candidates
            state["stale_dropped_count"] = stale_dropped_count

        _prune_digest_store(state)
        _atomic_write_to(Path(target_path), json.dumps(state, ensure_ascii=False, indent=2))


def save_daily_digest(daily_payload: Dict[str, Any], store_path: Optional[str] = None) -> None:
    """Save synthesized daily digest and automatically index by ticker."""
    target_path = resolve_digest_store_path(store_path)
    lock_file = _get_lock_path(target_path)
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    with FileLock(lock_file, timeout=10):
        state = _read_digest_store_unlocked(target_path)
        date_key = daily_payload.get("date") or datetime.now().strftime("%Y-%m-%d")

        state["latest_daily"] = daily_payload
        state.setdefault("daily_history", {})[date_key] = daily_payload

        # Update ticker intelligence index for instant /equity/:ticker display
        ticker_index: Dict[str, Dict[str, Any]] = state.setdefault("ticker_intelligence_index", {})
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in daily_payload.get("items", []):
            affected_tickers = item.get("affected_tickers", [])
            for ticker in affected_tickers:
                clean_sym = ticker.strip().upper().replace(".BK", "")
                if not clean_sym:
                    continue
                entry = ticker_index.setdefault(clean_sym, {
                    "last_updated": now_iso,
                    "news_items": [],
                    "youtube_mentions": [],
                })
                entry["last_updated"] = now_iso

                # Deduplicate by source_candidate_id or link
                news_list = entry.setdefault("news_items", [])
                item_id = item.get("source_candidate_id") or item.get("link")
                if not any((x.get("source_candidate_id") or x.get("link")) == item_id for x in news_list):
                    news_list.insert(0, item)

        for y in daily_payload.get("youtube_highlights", []):
            mentioned = y.get("mentioned_tickers", [])
            for ticker in mentioned:
                clean_sym = ticker.strip().upper().replace(".BK", "")
                if not clean_sym:
                    continue
                entry = ticker_index.setdefault(clean_sym, {
                    "last_updated": now_iso,
                    "news_items": [],
                    "youtube_mentions": [],
                })
                entry["last_updated"] = now_iso

                yt_list = entry.setdefault("youtube_mentions", [])
                vid = y.get("video_id")
                if not any(x.get("video_id") == vid for x in yt_list):
                    yt_list.insert(0, y)

        _prune_digest_store(state)
        _atomic_write_to(Path(target_path), json.dumps(state, ensure_ascii=False, indent=2))


def save_weekly_digest(weekly_payload: Dict[str, Any], store_path: Optional[str] = None) -> None:
    """Save synthesized weekly digest and index items by ticker."""
    target_path = resolve_digest_store_path(store_path)
    lock_file = _get_lock_path(target_path)
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    with FileLock(lock_file, timeout=10):
        state = _read_digest_store_unlocked(target_path)
        date_key = weekly_payload.get("to_date") or datetime.now().strftime("%Y-%m-%d")
        state["latest_weekly"] = weekly_payload
        state.setdefault("weekly_history", {})[date_key] = weekly_payload

        # Update ticker intelligence index for instant /equity/:ticker display
        ticker_index: Dict[str, Dict[str, Any]] = state.setdefault("ticker_intelligence_index", {})
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in weekly_payload.get("items", []):
            affected_tickers = item.get("affected_tickers", [])
            for ticker in affected_tickers:
                clean_sym = ticker.strip().upper().replace(".BK", "")
                if not clean_sym:
                    continue
                entry = ticker_index.setdefault(clean_sym, {
                    "last_updated": now_iso,
                    "news_items": [],
                    "youtube_mentions": [],
                })
                entry["last_updated"] = now_iso

                news_list = entry.setdefault("news_items", [])
                item_id = item.get("source_candidate_id") or item.get("link")
                if not any((x.get("source_candidate_id") or x.get("link")) == item_id for x in news_list):
                    news_list.insert(0, item)

        for y in weekly_payload.get("youtube_highlights", []):
            mentioned = y.get("mentioned_tickers", [])
            for ticker in mentioned:
                clean_sym = ticker.strip().upper().replace(".BK", "")
                if not clean_sym:
                    continue
                entry = ticker_index.setdefault(clean_sym, {
                    "last_updated": now_iso,
                    "news_items": [],
                    "youtube_mentions": [],
                })
                entry["last_updated"] = now_iso

                yt_list = entry.setdefault("youtube_mentions", [])
                vid = y.get("video_id")
                if not any(x.get("video_id") == vid for x in yt_list):
                    yt_list.insert(0, y)

        _prune_digest_store(state)
        _atomic_write_to(Path(target_path), json.dumps(state, ensure_ascii=False, indent=2))


def mark_digest_items_read(item_ids: List[str], store_path: Optional[str] = None) -> List[str]:
    """Record read items to track user acknowledgment."""
    target_path = resolve_digest_store_path(store_path)
    lock_file = _get_lock_path(target_path)
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    with FileLock(lock_file, timeout=10):
        state = _read_digest_store_unlocked(target_path)
        read_set = set(state.setdefault("read_item_ids", []))
        for it in item_ids:
            if it:
                read_set.add(str(it))
        updated_list = sorted(read_set)
        state["read_item_ids"] = updated_list
        _prune_digest_store(state)
        _atomic_write_to(Path(target_path), json.dumps(state, ensure_ascii=False, indent=2))
        return state["read_item_ids"]


def get_ticker_intelligence(ticker: str, store_path: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve pre-indexed news and YouTube insights for an individual ticker."""
    clean_sym = ticker.strip().upper().replace(".BK", "")
    state = load_digest_store(store_path)
    ticker_index = state.get("ticker_intelligence_index", {})
    return ticker_index.get(clean_sym, {"news_items": [], "youtube_mentions": []})
