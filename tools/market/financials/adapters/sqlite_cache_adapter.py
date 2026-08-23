"""SQLite Persistence Adapter for Financial Statements Cache."""
from datetime import datetime, timezone
import json
import logging
from typing import Any, Optional
from tools.market.financials.domain.models import FinancialStatementsDTO
from tools.market.financials.ports.cache_port import CacheEntry, FinancialCachePort

log = logging.getLogger(__name__)


class SQLiteCacheAdapter(FinancialCachePort):
    """Driven Adapter: จัดเก็บและดึงงบการเงินจาก SQLite table `financial_statements_cache`"""

    def __init__(self, conn_factory: Optional[Any] = None):
        self._conn_factory = conn_factory

    def _get_conn(self) -> Any:
        if self._conn_factory:
            return self._conn_factory()
        from api.state_db import get_connection
        return get_connection()

    def get(self, market: str, provider_symbol: str) -> Optional[CacheEntry]:
        conn = self._get_conn()
        cur = conn.execute(
            """
            SELECT data_json, synced_at, provider
            FROM financial_statements_cache
            WHERE market = ? AND provider_symbol = ?
            """,
            (market.upper(), provider_symbol.upper()),
        )
        row = cur.fetchone()
        if not row:
            return None

        data_json = row["data_json"]
        synced_at = float(row["synced_at"])
        provider = str(row["provider"])

        try:
            parsed = json.loads(data_json)
            # ตรวจสอบ Schema Version V6
            if parsed.get("schema_version") != 6:
                log.info("Invalidating legacy/corrupted cache (schema_version != 6) for %s:%s", market, provider_symbol)
                self.delete(market, provider_symbol)
                return None

            dto = FinancialStatementsDTO.model_validate(parsed)
            if not dto.synced_at and synced_at > 0:
                dto.synced_at = datetime.fromtimestamp(synced_at, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            return CacheEntry(statements=dto, provider=provider, synced_at=synced_at)
        except Exception as e:
            log.warning("Failed to deserialize financial statements cache for %s:%s: %s", market, provider_symbol, e)
            self.delete(market, provider_symbol)
            return None

    def save(self, market: str, provider_symbol: str, entry: CacheEntry) -> None:
        conn = self._get_conn()
        data_json = entry.statements.model_dump_json()
        with conn:
            conn.execute(
                """
                INSERT INTO financial_statements_cache (market, provider_symbol, provider, data_json, synced_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(market, provider_symbol) DO UPDATE SET
                    provider = excluded.provider,
                    data_json = excluded.data_json,
                    synced_at = excluded.synced_at
                """,
                (market.upper(), provider_symbol.upper(), entry.provider, data_json, entry.synced_at),
            )

    def delete(self, market: str, provider_symbol: str) -> None:
        conn = self._get_conn()
        with conn:
            conn.execute(
                """
                DELETE FROM financial_statements_cache
                WHERE market = ? AND provider_symbol = ?
                """,
                (market.upper(), provider_symbol.upper()),
            )
