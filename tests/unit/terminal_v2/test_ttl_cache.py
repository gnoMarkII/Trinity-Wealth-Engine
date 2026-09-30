"""Unit tests verifying ThreadSafeTTLCache, Double-Checked Locking & Stale-on-Error."""
import concurrent.futures
from dataclasses import dataclass
import time
import pytest

from tools.market.terminal_v2.application.cache import ThreadSafeTTLCache


@dataclass(frozen=True)
class DummyData:
    val: int
    is_stale: bool = False
    stale_reason: str = ""


def test_cache_hits_within_ttl():
    cache = ThreadSafeTTLCache(default_ttl_seconds=10.0)
    calls = 0

    def loader():
        nonlocal calls
        calls += 1
        return DummyData(val=calls)

    val1 = cache.get_or_set("key1", loader, ttl_seconds=10.0)
    assert val1.val == 1
    assert calls == 1

    # Second call should hit cache, not invoke loader
    val2 = cache.get_or_set("key1", loader, ttl_seconds=10.0)
    assert val2.val == 1
    assert calls == 1


def test_double_checked_locking_prevents_thundering_herd():
    """Verify that 20 threads requesting the same cold key execute the loader exactly ONCE."""
    cache = ThreadSafeTTLCache(default_ttl_seconds=10.0)
    calls = 0

    def slow_loader():
        nonlocal calls
        calls += 1
        time.sleep(0.05)  # Simulate network latency
        return DummyData(val=42)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(cache.get_or_set, "stampede_key", slow_loader, 10.0)
            for _ in range(20)
        ]
        results = [f.result() for f in futures]

    assert calls == 1
    for r in results:
        assert r.val == 42


def test_stale_on_error_grace_period():
    """When upstream fails, cache returns previous good entry flagged as stale."""
    cache = ThreadSafeTTLCache(default_ttl_seconds=0.01)

    # First load succeeds
    res1 = cache.get_or_set("stale_test", lambda: DummyData(val=100), ttl_seconds=0.01)
    assert res1.val == 100
    assert res1.is_stale is False

    # Force expiration
    cache._entries["stale_test"].expires_at = 0.0

    # Loader now fails
    def failing_loader():
        raise ConnectionResetError("Remote server closed connection")

    res2 = cache.get_or_set("stale_test", failing_loader, ttl_seconds=0.01)
    assert res2.val == 100
    assert res2.is_stale is True
    assert "Remote server closed connection" in res2.stale_reason


def test_cache_invalidation():
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0)
    cache.get_or_set("k1", lambda: "v1")
    assert cache.get("k1") == "v1"

    cache.invalidate("k1")
    assert cache.get("k1") is None


def test_stale_not_served_for_non_transient_error():
    """Non-transient errors (e.g. ValueError, domain mismatches) must not serve stale."""
    from tools.market.terminal_v2.domain.errors import SymbolMarketMismatchError

    cache = ThreadSafeTTLCache(default_ttl_seconds=0.01)
    cache.get_or_set("mismatch_key", lambda: DummyData(val=999), ttl_seconds=0.01)
    cache._entries["mismatch_key"].expires_at = 0.0

    def bad_request_loader():
        raise SymbolMarketMismatchError("Perp symbol not valid for cash equity")

    with pytest.raises(SymbolMarketMismatchError):
        cache.get_or_set("mismatch_key", bad_request_loader, ttl_seconds=0.01)


def test_stale_exceeding_max_stale_seconds_raises_data_unavailable():
    """Stale snapshot older than max_stale_seconds must raise DataUnavailableError."""
    from tools.market.terminal_v2.domain.errors import DataUnavailableError

    cache = ThreadSafeTTLCache(default_ttl_seconds=0.01)
    cache.get_or_set("old_key", lambda: DummyData(val=123), ttl_seconds=0.01)
    cache._entries["old_key"].expires_at = 0.0
    # Simulate entry cached 100 seconds ago
    cache._entries["old_key"].monotonic_cached_at = time.monotonic() - 100.0

    def transient_failure():
        raise TimeoutError("Connection timed out")

    with pytest.raises(DataUnavailableError) as exc_info:
        cache.get_or_set("old_key", transient_failure, ttl_seconds=0.01, max_stale_seconds=30.0)

    assert "expired" in str(exc_info.value).lower()


def test_bounded_eviction_when_max_entries_exceeded():
    """Cache must prune old entries when exceeding max_entries."""
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0, max_entries=3)

    cache.get_or_set("k1", lambda: "v1")
    cache.get_or_set("k2", lambda: "v2")
    cache.get_or_set("k3", lambda: "v3")
    assert len(cache._entries) == 3

    # Adding 4th entry must evict the oldest entry ("k1")
    cache.get_or_set("k4", lambda: "v4")
    assert len(cache._entries) <= 3
    assert cache.get("k1") is None
    assert cache.get("k4") == "v4"
