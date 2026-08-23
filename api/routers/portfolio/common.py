"""Shared helpers and exception handlers for portfolio sub-routers."""
import json
from contextlib import contextmanager
from filelock import Timeout
from fastapi import HTTPException
from pydantic import ValidationError

from tools.archivist.core import VAULT_PATH

_STRATEGY_SUBDIR = "30_Knowledge_Base/Strategies"


def _latest_strategy_json() -> dict:
    import sys
    routes_portfolio = sys.modules.get("api.routes_portfolio")
    vault_path = getattr(routes_portfolio, "VAULT_PATH", VAULT_PATH) if routes_portfolio else VAULT_PATH
    strategy_dir = vault_path / _STRATEGY_SUBDIR
    candidates = sorted(strategy_dir.glob("Macro_Strategy_Direction_*.json"))
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="ยังไม่มีรายงาน Macro Strategy ที่มี JSON sidecar — รอรายงานถัดไปหลัง Phase 0 อัปเดต",
        )
    latest = candidates[-1]
    return json.loads(latest.read_text(encoding="utf-8"))


@contextmanager
def handle_portfolio_exceptions(timeout_detail: str = "Portfolio lock timeout"):
    try:
        yield
    except ValidationError as exc:
        raise HTTPException(
            status_code=500, detail=f"Internal DTO validation error: {exc}"
        ) from exc
    except Timeout as exc:
        raise HTTPException(status_code=503, detail=timeout_detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
