"""Durable, vault-scoped selection bindings for Macro sector evidence."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from filelock import FileLock


class SectorRunBindingStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve() / "run_bindings"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        digest = hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def lock_path(self, run_id: str) -> Path:
        return self._path(run_id).with_suffix(".lock")

    def load(self, run_id: str) -> Optional[dict[str, Any]]:
        try:
            value = json.loads(self._path(run_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict) or value.get("run_id") != str(run_id):
            raise RuntimeError("sector_run_binding_identity_mismatch")
        return value

    def create(self, run_id: str, record: dict[str, Any]) -> None:
        if record.get("run_id") != str(run_id):
            raise ValueError("sector_run_binding_identity_mismatch")
        path = self._path(run_id)
        if path.exists():
            raise RuntimeError("sector_run_binding_already_exists")
        self._atomic_write(path, record)

    def update_publication(self, run_id: str, status: str, receipt: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        record = self.load(run_id)
        if record is None:
            raise RuntimeError("sector_run_binding_not_found")
        record["publication_status"] = status
        record["publication_receipt"] = receipt
        if status in {"committed", "duplicate_reused", "recovered_from_evidence"}:
            # The evidence broker now holds the exact immutable inputs.
            record.pop("pending_snapshot", None)
            record.pop("pending_prices", None)
            record.pop("pending_expected_sessions", None)
        self._atomic_write(self._path(run_id), record)
        return record

    @staticmethod
    def _atomic_write(path: Path, record: dict[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".sector-run-binding-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                json.dump(record, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
