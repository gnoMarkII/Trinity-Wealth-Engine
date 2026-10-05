"""Logical Macro run ID propagated to tools invoked inside the same agent call."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional


_ACTIVE_SECTOR_RUN_ID: ContextVar[Optional[str]] = ContextVar("active_sector_rotation_run_id", default=None)


def current_sector_run_id() -> Optional[str]:
    return _ACTIVE_SECTOR_RUN_ID.get()


@contextmanager
def sector_run_scope(run_id: str) -> Iterator[None]:
    token = _ACTIVE_SECTOR_RUN_ID.set(str(run_id))
    try:
        yield
    finally:
        _ACTIVE_SECTOR_RUN_ID.reset(token)
