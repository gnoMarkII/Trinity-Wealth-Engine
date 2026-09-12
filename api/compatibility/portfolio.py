"""Legacy Portfolio route helper namespace."""
import json
from fastapi import HTTPException
from tools.archivist.core import VAULT_PATH

_STRATEGY_SUBDIR = "30_Knowledge_Base/Strategies"


def _latest_strategy_json() -> dict:
    from tools.macro.adapters.strategy_vault_adapter import StrategyVaultAdapter
    try:
        return StrategyVaultAdapter().latest()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="ยังไม่มีรายงาน Macro Strategy ที่มี JSON sidecar — รอรายงานถัดไปหลัง Phase 0 อัปเดต",
        )
