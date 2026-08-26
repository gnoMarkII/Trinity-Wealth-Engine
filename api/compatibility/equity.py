"""Compatibility helpers for historical Equity imports and test patches.

New routers use application ports.  This module exists only for direct legacy
imports from ``api.routes_equity`` and will be removed after downstream callers
move to dependency overrides.
"""
from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from api.schemas import EquityNewsDTO
from schemas.micro_quant_schemas import MicroQuantOutput
from tools.archivist.parser import extract_yaml_frontmatter_value
from tools.market.adapters.equity_vault_query_adapter import EquityVaultQueryAdapter


def _validate_ticker(ticker: str) -> str:
    clean = ticker.upper()
    if not re.match(r"^[A-Z0-9.\-_]+$", clean):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid ticker format")
    if ".." in clean or "/" in clean or "\\" in clean:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Path traversal not allowed")
    return clean


def _validate_schema(data: dict, expected_ticker: str) -> MicroQuantOutput | None:
    return EquityVaultQueryAdapter._validate_model(data, expected_ticker)


def _get_equity_files(ticker: Optional[str] = None) -> list[Path]:
    return EquityVaultQueryAdapter()._sidecar_files(ticker)


def _extract_date_key(file_path: Path) -> tuple[str, str]:
    return EquityVaultQueryAdapter._date_key(file_path)


def _get_latest_sidecar_for_ticker(
    files: list[Path], expected_ticker: str, strict: bool = False
) -> tuple[MicroQuantOutput, Path] | None:
    try:
        return EquityVaultQueryAdapter()._latest_sidecar(files, expected_ticker, strict=strict)
    except Exception:
        return None


def _is_agent_generated(file_path: Path, content: str) -> bool:
    if file_path.suffix == ".json":
        return True
    val_entity = (extract_yaml_frontmatter_value(content, "entity_type") or "").lower().replace(" ", "_")
    val_agent = extract_yaml_frontmatter_value(content, "generated_by")
    if val_entity in ("company_news", "equity_analysis") or val_agent:
        return True
    return bool(re.search(r"(?i)latest[_\s]news|equity[_\s]analysis", file_path.name))


def _get_equity_news_from_vault(ticker: str) -> EquityNewsDTO | None:
    payload = EquityVaultQueryAdapter().get_news(ticker)
    return EquityNewsDTO.model_validate(payload) if payload else None


def _extract_note_datetime(filename: str, mtime: float, content: str = "") -> datetime:
    match = re.search(r"20\d{2}-\d{2}-\d{2}", filename)
    if not match and content:
        match = re.search(
            r"20\d{2}-\d{2}-\d{2}",
            extract_yaml_frontmatter_value(content, "date") or "",
        )
    if match:
        try:
            return datetime.strptime(match.group(0), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(mtime, tz=timezone.utc)


_ANALYST_LOCK = threading.Lock()
_ANALYST_KEY_LOCKS: dict[str, threading.Lock] = {}


def _get_analyst_lock(symbol: str) -> threading.Lock:
    with _ANALYST_LOCK:
        return _ANALYST_KEY_LOCKS.setdefault(symbol, threading.Lock())


def positive_int_or_none(value) -> int | None:
    try:
        value = int(value)
        return value if value >= 0 else None
    except (TypeError, ValueError):
        return None
