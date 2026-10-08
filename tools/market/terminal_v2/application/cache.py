"""Thread-Safe TTL Cache with Double-Checked Locking & Stale-on-Error Grace Period.

Implements anti-ban protection, exact TTL limits, bounded entry eviction,
and bounded stale-on-error fallback for all Keyless Terminal V2 adapters.
"""
import dataclasses
import logging
import threading
import time
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from tools.market.terminal_v2.domain.errors import (
    DataUnavailableError,
    InvalidCapabilityError,
    ProviderError,
    SymbolMarketMismatchError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

BROWSER_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,th;q=0.8",
    "Connection": "keep-alive",
}


def is_transient_upstream_error(exc: Exception) -> bool:
    """Return True if exception indicates a transient network or server error.

    Validation, domain mismatch, or client errors (400, 404) are NOT transient
    and must never be masked with stale data.
    """
    if isinstance(exc, (SymbolMarketMismatchError, InvalidCapabilityError, ValueError, KeyError, TypeError)):
        return False
    if isinstance(exc, DataUnavailableError):
        return True
    if isinstance(exc, ProviderError):
        if exc.status_code is not None and 400 <= exc.status_code < 500 and exc.status_code != 429:
            return False
        return True
    try:
        import requests
        if isinstance(exc, (requests.ConnectionError, requests.Timeout, requests.RequestException)):
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                code = exc.response.status_code
                if 400 <= code < 500 and code != 429:
                    return False
            return True
    except ImportError:
        pass
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    # General upstream failures (e.g. mock side_effect exceptions) are transient
    return True


@dataclasses.dataclass
class CacheEntry(Generic[T]):
    """Internal cache entry storing payload, expiration clock, and timestamps."""
    data: T
    expires_at: float           # monotonic expiration time
    cached_at: float            # wall-clock epoch seconds
    monotonic_cached_at: float = dataclasses.field(default_factory=time.monotonic)


class ThreadSafeTTLCache:
    """Thread-safe in-memory cache with per-key double-checked locking,
    bounded LRU/FIFO eviction, and bounded stale-on-error fallback.

    Key guarantees:
    1. Double-Checked Locking: Prevents thundering herd / cache stampede on upstream.
    2. Bounded Stale-on-Error: Serves last-known-good snapshot only for transient errors
       and only within max_stale_seconds.
    3. Bounded Eviction: Enforces max_entries to prevent unbounded memory growth.
    4. Per-source strict TTL: Prevents IP bans from aggressive polling.
    """

    def __init__(
        self,
        default_ttl_seconds: float = 60.0,
        max_entries: int = 500,
        max_stale_seconds: Optional[float] = None,
    ):
        self._default_ttl = default_ttl_seconds
        self._max_entries = max_entries
        self._default_max_stale = max_stale_seconds
        self._entries: Dict[str, CacheEntry[Any]] = {}
        self._global_lock = threading.Lock()
        self._key_locks: Dict[str, threading.Lock] = {}

    def _get_key_lock(self, key: str) -> threading.Lock:
        with self._global_lock:
            if key not in self._key_locks:
                self._key_locks[key] = threading.Lock()
            return self._key_locks[key]

    def _evict_if_needed(self) -> None:
        """Evict expired entries first, then oldest entries if capacity exceeded."""
        if len(self._entries) <= self._max_entries:
            return
        now = time.monotonic()
        expired_keys = [k for k, v in self._entries.items() if now >= v.expires_at]
        for k in expired_keys:
            self._entries.pop(k, None)
            self._key_locks.pop(k, None)

        if len(self._entries) > self._max_entries:
            sorted_by_age = sorted(
                self._entries.keys(),
                key=lambda k: self._entries[k].monotonic_cached_at,
            )
            overflow = len(self._entries) - self._max_entries
            for k in sorted_by_age[:overflow]:
                self._entries.pop(k, None)
                self._key_locks.pop(k, None)

    def get(self, key: str) -> Optional[Any]:
        """Direct retrieval if present and not expired."""
        entry = self._entries.get(key)
        if entry is not None and time.monotonic() < entry.expires_at:
            return entry.data
        return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """Direct store with TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        now_mono = time.monotonic()
        now_epoch = time.time()
        with self._global_lock:
            self._entries[key] = CacheEntry(
                data=value,
                expires_at=now_mono + ttl,
                cached_at=now_epoch,
                monotonic_cached_at=now_mono,
            )
            self._evict_if_needed()

    def peek(self, key: str) -> Optional[Dict[str, Any]]:
        """Read-only inspection of a cache entry without triggering any loader.

        Returns metadata and data if present (even if expired). Returns None if absent.
        """
        with self._global_lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            now = time.monotonic()
            return {
                "data": entry.data,
                "cached_at": entry.cached_at,
                "expires_at": entry.expires_at,
                "is_expired": now >= entry.expires_at,
            }

    def peek_data(self, key: str) -> Optional[Any]:
        """Return the cached payload if present, regardless of expiration, without triggering loader."""
        with self._global_lock:
            entry = self._entries.get(key)
            return entry.data if entry is not None else None

    def snapshot_entries(self) -> Dict[str, Dict[str, Any]]:
        """Return a read-only snapshot dictionary of all stored cache entries."""
        with self._global_lock:
            now = time.monotonic()
            return {
                k: {
                    "data": entry.data,
                    "cached_at": entry.cached_at,
                    "expires_at": entry.expires_at,
                    "is_expired": now >= entry.expires_at,
                }
                for k, entry in self._entries.items()
            }

    def get_or_compute(
        self,
        key: str,
        loader: Callable[[], T],
        ttl_seconds: Optional[float] = None,
        max_stale_seconds: Optional[float] = None,
    ) -> T:
        """Alias for get_or_set."""
        effective_stale = max_stale_seconds if max_stale_seconds is not None else self._default_max_stale
        return self.get_or_set(key, loader, ttl_seconds=ttl_seconds, max_stale_seconds=effective_stale)

    def get_or_set(
        self,
        key: str,
        loader: Callable[[], T],
        ttl_seconds: Optional[float] = None,
        max_stale_seconds: Optional[float] = None,
    ) -> T:
        """Double-checked locking getter with bounded stale-on-error fallback."""
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        effective_max_stale = max_stale_seconds if max_stale_seconds is not None else self._default_max_stale

        # First check (lock-free fast path)
        entry = self._entries.get(key)
        now = time.monotonic()
        if entry is not None and now < entry.expires_at:
            return entry.data

        # Double-check inside per-key lock
        key_lock = self._get_key_lock(key)
        with key_lock:
            entry = self._entries.get(key)
            now = time.monotonic()
            if entry is not None and now < entry.expires_at:
                return entry.data

            # Expired or missing: invoke loader with bounded stale-on-error protection
            try:
                data = loader()
                with self._global_lock:
                    self._entries[key] = CacheEntry(
                        data=data,
                        expires_at=time.monotonic() + ttl,
                        cached_at=time.time(),
                        monotonic_cached_at=time.monotonic(),
                    )
                    self._evict_if_needed()
                return data
            except Exception as exc:
                if entry is not None and is_transient_upstream_error(exc):
                    stale_age = time.monotonic() - entry.monotonic_cached_at
                    if effective_max_stale is not None and stale_age > effective_max_stale:
                        logger.error(
                            "Terminal V2 Cache: Stale snapshot for key '%s' expired (age %.1fs > max %.1fs).",
                            key,
                            stale_age,
                            effective_max_stale,
                        )
                        raise DataUnavailableError(
                            f"Upstream unavailable and stale snapshot expired (age {stale_age:.0f}s > max {effective_max_stale:.0f}s)",
                        ) from exc

                    logger.warning(
                        "Terminal V2 Cache: Upstream error for key '%s' (%s: %s). Serving stale snapshot (age %.1fs).",
                        key,
                        type(exc).__name__,
                        exc,
                        stale_age,
                    )
                    stale_reason = (
                        f"Upstream error ({type(exc).__name__}: {str(exc)}); "
                        f"serving last known valid snapshot (age {stale_age:.0f}s)"
                    )
                    stale_data = entry.data
                    if (
                        dataclasses.is_dataclass(stale_data)
                        and not isinstance(stale_data, type)
                        and "is_stale" in stale_data.__dataclass_fields__
                    ):
                        try:
                            stale_data = dataclasses.replace(
                                stale_data,
                                is_stale=True,
                                stale_reason=stale_reason,
                            )
                        except Exception:
                            pass
                    elif isinstance(stale_data, dict):
                        new_dict = {}
                        for k, v in stale_data.items():
                            if (
                                dataclasses.is_dataclass(v)
                                and not isinstance(v, type)
                                and "is_stale" in v.__dataclass_fields__
                            ):
                                try:
                                    new_dict[k] = dataclasses.replace(
                                        v,
                                        is_stale=True,
                                        stale_reason=stale_reason,
                                    )
                                except Exception:
                                    new_dict[k] = v
                            else:
                                new_dict[k] = v
                        stale_data = new_dict
                    return stale_data

                # Non-transient error or no cache exists: re-raise
                logger.error(
                    "Terminal V2 Cache: Upstream failed for key '%s' and cannot serve stale (%s: %s).",
                    key,
                    type(exc).__name__,
                    exc,
                )
                raise exc

    def invalidate(self, key: str) -> None:
        """Invalidate a specific key immediately."""
        with self._global_lock:
            self._entries.pop(key, None)
            self._key_locks.pop(key, None)

    def clear(self) -> None:
        """Clear all entries."""
        with self._global_lock:
            self._entries.clear()
            self._key_locks.clear()


# Alias for backwards compatibility with adapters
TerminalTtlCache = ThreadSafeTTLCache
