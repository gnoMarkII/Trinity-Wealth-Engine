"""Unit tests for SQLite and InMemory Cache Adapters."""
import sqlite3
from tools.market.financials.adapters.in_memory_cache_adapter import InMemoryCacheAdapter
from tools.market.financials.adapters.sqlite_cache_adapter import SQLiteCacheAdapter
from tools.market.financials.domain.models import FinancialStatementsDTO
from tools.market.financials.ports.cache_port import CacheEntry


def _make_sample_dto(ticker: str = "FTNT") -> FinancialStatementsDTO:
    return FinancialStatementsDTO(
        schema_version=6,
        ticker=ticker,
        market="US",
        currency="USD",
        provider="edgartools",
        provider_symbol=ticker,
        data_status="ok",
        coverage_status="complete",
        core_coverage_status="complete",
        expanded_coverage_status="complete",
        expanded_data_status="complete",
    )


def test_in_memory_cache_adapter():
    adapter = InMemoryCacheAdapter()
    assert adapter.get("US", "FTNT") is None

    dto = _make_sample_dto("FTNT")
    entry = CacheEntry(statements=dto, provider="edgartools", synced_at=1000.0)
    adapter.save("US", "FTNT", entry)

    cached = adapter.get("US", "FTNT")
    assert cached is not None
    assert cached.statements.ticker == "FTNT"
    assert cached.synced_at == 1000.0

    adapter.delete("US", "FTNT")
    assert adapter.get("US", "FTNT") is None


def test_sqlite_cache_adapter_crud_and_schema_version():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE financial_statements_cache (
            market TEXT NOT NULL,
            provider_symbol TEXT NOT NULL,
            provider TEXT NOT NULL,
            data_json TEXT NOT NULL,
            synced_at REAL NOT NULL,
            PRIMARY KEY (market, provider_symbol)
        )
        """
    )

    adapter = SQLiteCacheAdapter(conn_factory=lambda: conn)
    assert adapter.get("US", "FTNT") is None

    dto = _make_sample_dto("FTNT")
    entry = CacheEntry(statements=dto, provider="edgartools", synced_at=1700000000.0)
    adapter.save("US", "FTNT", entry)

    cached = adapter.get("US", "FTNT")
    assert cached is not None
    assert cached.statements.ticker == "FTNT"
    assert cached.provider == "edgartools"

    # Invalidate corrupted / old schema version (e.g. schema_version=5)
    corrupted_dto = _make_sample_dto("OLD")
    corrupted_dto.schema_version = 5
    corrupted_entry = CacheEntry(statements=corrupted_dto, provider="edgartools", synced_at=1700000000.0)
    adapter.save("US", "OLD", corrupted_entry)

    # get() must automatically reject and delete schema_version != 6
    assert adapter.get("US", "OLD") is None
