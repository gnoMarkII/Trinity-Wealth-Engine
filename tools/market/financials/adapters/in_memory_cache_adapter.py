"""In-Memory Cache Adapter for Testing and Isolation."""
from typing import Optional
from tools.market.financials.ports.cache_port import CacheEntry, FinancialCachePort


class InMemoryCacheAdapter(FinancialCachePort):
    """In-Memory Storage Adapter สำหรับการทำ Unit Tests โดยไม่ต้องต่อ Database จริง"""

    def __init__(self):
        self._store: dict[tuple[str, str], CacheEntry] = {}

    def get(self, market: str, provider_symbol: str) -> Optional[CacheEntry]:
        return self._store.get((market.upper(), provider_symbol.upper()))

    def save(self, market: str, provider_symbol: str, entry: CacheEntry) -> None:
        self._store[(market.upper(), provider_symbol.upper())] = entry

    def delete(self, market: str, provider_symbol: str) -> None:
        self._store.pop((market.upper(), provider_symbol.upper()), None)

    def clear(self) -> None:
        self._store.clear()
