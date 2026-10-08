"""System adapters for Clock and IdGenerator ports."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from application.investor_essence.ports import ClockPort, IdGeneratorPort


class SystemClockAdapter(ClockPort):
    """Real system clock implementation."""

    def now_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def now_epoch(self) -> float:
        return time.time()


class UuidGeneratorAdapter(IdGeneratorPort):
    """UUID4-based ID generator implementation."""

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix

    def new_id(self) -> str:
        uid = str(uuid.uuid4())
        if self._prefix:
            return f"{self._prefix}_{uid}"
        return uid
