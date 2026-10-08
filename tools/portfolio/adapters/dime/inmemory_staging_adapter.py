"""In-Memory Staging Driven Adapter (Hexagonal Architecture).

Stores staged TradeImportItem lists with thread-safe session-bound TTL eviction.
Raises domain exceptions (StagedScanExpiredError, StagedScanForbiddenError, StagedScanNotFoundError)
which the driving adapter (FastAPI router) maps to HTTP status codes.
"""
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from tools.portfolio.domain.errors import (
    StagedScanExpiredError,
    StagedScanForbiddenError,
    StagedScanNotFoundError,
)
from tools.portfolio.domain.models import TradeImportItem
from tools.portfolio.ports.trade_ingestion_port import TradeStagingPort


class _StagedBatch:
    def __init__(self, items: List[TradeImportItem], expires_at: float, session_id: Optional[str] = None):
        self.items = items
        self.expires_at = expires_at
        self.session_id = session_id
        self.provenance: Optional[Dict[str, Any]] = None


class InMemoryStagingAdapter(TradeStagingPort):
    """In-memory thread-safe staging implementation."""

    def __init__(self):
        self._store: Dict[str, _StagedBatch] = {}
        self._lock = threading.Lock()

    def stage_items(
        self,
        items: List[TradeImportItem],
        ttl_seconds: int = 1800,
        session_id: Optional[str] = None,
    ) -> str:
        scan_id = f"scan_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        expires_at = time.time() + ttl_seconds
        with self._lock:
            # Clean up expired batches while holding lock
            now = time.time()
            expired_keys = [k for k, v in self._store.items() if v.expires_at < now]
            for k in expired_keys:
                del self._store[k]

            self._store[scan_id] = _StagedBatch(items=items, expires_at=expires_at, session_id=session_id)
        return scan_id

    def get_staged_items(
        self,
        scan_id: str,
        session_id: Optional[str] = None,
    ) -> List[TradeImportItem]:
        with self._lock:
            batch = self._store.get(scan_id)
            if not batch:
                raise StagedScanNotFoundError(f"ไม่พบข้อมูลที่ staged ไว้สำหรับ scan_id '{scan_id}'")

            if time.time() > batch.expires_at:
                del self._store[scan_id]
                raise StagedScanExpiredError(f"ข้อมูล scan_id '{scan_id}' หมดอายุแล้ว กรุณาสแกนใหม่")

            if batch.session_id is not None:
                if not session_id or batch.session_id != session_id:
                    raise StagedScanForbiddenError("ไม่อนุญาตให้เข้าถึงข้อมูล staged ของ session อื่น")

            return list(batch.items)

    def delete_staged(
        self,
        scan_id: str,
        session_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            batch = self._store.get(scan_id)
            if not batch:
                return

            if batch.session_id is not None and session_id and batch.session_id != session_id:
                raise StagedScanForbiddenError("ไม่อนุญาตให้ลบข้อมูล staged ของ session อื่น")

            self._store.pop(scan_id, None)

    def stage_provenance(
        self,
        scan_id: str,
        provenance: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            batch = self._store.get(scan_id)
            if not batch:
                return
            if batch.session_id is not None and session_id and batch.session_id != session_id:
                raise StagedScanForbiddenError("ไม่อนุญาตให้เข้าถึงข้อมูล staged ของ session อื่น")
            batch.provenance = dict(provenance)

    def get_provenance(
        self,
        scan_id: str,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            batch = self._store.get(scan_id)
            if not batch:
                return None
            if time.time() > batch.expires_at:
                return None
            if batch.session_id is not None and session_id and batch.session_id != session_id:
                raise StagedScanForbiddenError("ไม่อนุญาตให้เข้าถึงข้อมูล staged ของ session อื่น")
            return dict(batch.provenance) if batch.provenance is not None else None

    def pop_provenance(
        self,
        scan_id: str,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            batch = self._store.get(scan_id)
            if not batch:
                return None
            if batch.session_id is not None and session_id and batch.session_id != session_id:
                raise StagedScanForbiddenError("ไม่อนุญาตให้เข้าถึงข้อมูล staged ของ session อื่น")
            prov = batch.provenance
            batch.provenance = None
            return dict(prov) if prov is not None else None
