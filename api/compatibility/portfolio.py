"""Legacy Portfolio route helper namespace."""
import json
from fastapi import HTTPException
from tools.archivist.core import VAULT_PATH

_STRATEGY_SUBDIR = "30_Knowledge_Base/Strategies"


def _latest_strategy_json() -> dict:
    strategy_dir = VAULT_PATH / _STRATEGY_SUBDIR
    candidates = sorted(strategy_dir.glob("Macro_Strategy_Direction_*.json"))
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="ยังไม่มีรายงาน Macro Strategy ที่มี JSON sidecar — รอรายงานถัดไปหลัง Phase 0 อัปเดต",
        )
    return json.loads(candidates[-1].read_text(encoding="utf-8"))
