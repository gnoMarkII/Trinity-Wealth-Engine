"""Durable Entity Registry for Obsidian Vault V2.

Resolves financial assets and tickers to stable, durable entity_ids
independent of symbol renames, relistings, or ticker reuse.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import threading
from filelock import FileLock

from application.knowledge.identity import EntityRegistryPort
from tools.archivist.core import _atomic_write_text
from tools.archivist.maintenance_guard import assert_write_allowed

log = logging.getLogger(__name__)

_ENTITY_LOCKS: dict[str, threading.RLock] = {}
_ENTITY_LOCKS_GUARD = threading.Lock()


def _get_entity_lock(path_str: str) -> threading.RLock:
    import os
    norm = os.path.normcase(os.path.abspath(path_str))
    with _ENTITY_LOCKS_GUARD:
        if norm not in _ENTITY_LOCKS:
            _ENTITY_LOCKS[norm] = threading.RLock()
        return _ENTITY_LOCKS[norm]


class DurableEntityRegistry(EntityRegistryPort):
    """File-backed entity registry with cross-process locking."""

    def __init__(self, root: Union[str, Path, None] = None) -> None:
        if root is not None:
            self._root = Path(root).resolve()
        else:
            from tools.archivist.vault_paths import VaultPaths
            self._root = VaultPaths().root

        self._store_dir = self._root / ".system" / "identities" / "entities"
        self._store_file = self._store_dir / "entities.json"
        self._lock_file = self._store_dir / "entities.lock"

    def _ensure_store(self) -> None:
        assert_write_allowed(self._store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)
        if not self._store_file.exists():
            _atomic_write_text(self._store_file, "{}")

    def _read_entities(self) -> dict[str, dict[str, Any]]:
        if not self._store_file.exists():
            return {}
        try:
            content = self._store_file.read_text(encoding="utf-8").strip()
            if not content:
                return {}
            data = json.loads(content)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_entities(self, data: dict[str, dict[str, Any]]) -> None:
        _atomic_write_text(
            self._store_file,
            json.dumps(data, indent=2, ensure_ascii=False),
        )

    def resolve_stable_entity_id(
        self,
        ticker: str,
        market: Optional[str] = None,
    ) -> str:
        """Returns a stable entity_id for ticker:market."""
        clean_ticker = ticker.upper().strip()
        clean_market = (market or "US").upper().strip()
        lookup_key = f"{clean_market}:{clean_ticker}"

        thread_lock = _get_entity_lock(str(self._lock_file))
        file_lock = FileLock(str(self._lock_file), timeout=15)

        with thread_lock:
            with file_lock:
                self._ensure_store()
                entities = self._read_entities()
                if lookup_key in entities:
                    return entities[lookup_key]["entity_id"]

                new_entity_id = f"ent_{clean_ticker}_{uuid.uuid4().hex[:8]}"
                entities[lookup_key] = {
                    "entity_id": new_entity_id,
                    "ticker": clean_ticker,
                    "market": clean_market,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                self._save_entities(entities)
                return new_entity_id
