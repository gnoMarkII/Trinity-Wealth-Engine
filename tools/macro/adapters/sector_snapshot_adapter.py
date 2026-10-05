"""Atomic runtime cache for immutable sector snapshots and refresh state."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from filelock import FileLock
from schemas.sector_rotation_schemas import SectorRotationSnapshot


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SectorSnapshotStore:
    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.getenv("SECTOR_ROTATION_RUNTIME_DIR") or "data/sector_rotation"
        self.root = Path(configured).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._state_path = self.root / "state.json"
        self._state_lock = self.root / ".state.lock"
        self._refresh_lock = self.root / ".refresh.lock"

    @property
    def refresh_lock_path(self) -> Path:
        return self._refresh_lock

    def read_state(self) -> dict[str, Any]:
        with FileLock(str(self._state_lock), timeout=10):
            try:
                value = json.loads(self._state_path.read_text(encoding="utf-8"))
                return value if isinstance(value, dict) else {}
            except (OSError, json.JSONDecodeError):
                return {}

    def update_state(self, **changes: Any) -> dict[str, Any]:
        with FileLock(str(self._state_lock), timeout=10):
            try:
                state = json.loads(self._state_path.read_text(encoding="utf-8"))
                if not isinstance(state, dict):
                    state = {}
            except (OSError, json.JSONDecodeError):
                state = {}
            state.update(changes)
            self._atomic_json(self._state_path, state)
            return state

    def save(self, snapshot: SectorRotationSnapshot, evidence_ref: dict[str, Any]) -> None:
        path = self.root / "snapshots" / f"{snapshot.snapshot_id}.json"
        self._atomic_json(path, {"snapshot": snapshot.model_dump(mode="json"), "evidence_ref": evidence_ref})
        self.update_state(latest_snapshot_id=snapshot.snapshot_id, latest_as_of=snapshot.as_of_date)

    def load(self, snapshot_id: str) -> Optional[tuple[SectorRotationSnapshot, dict[str, Any]]]:
        path = self.root / "snapshots" / f"{snapshot_id}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return SectorRotationSnapshot.model_validate(value["snapshot"]), value.get("evidence_ref", {})
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def latest(self) -> Optional[tuple[SectorRotationSnapshot, dict[str, Any]]]:
        snapshot_id = str(self.read_state().get("latest_snapshot_id") or "")
        return self.load(snapshot_id) if snapshot_id else None

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".sector-rotation-", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise

