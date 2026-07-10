"""GET /api/portfolio/latest, GET /api/macro/dashboard — อ่าน sidecar JSON ตรงๆ (Rev.3/5 ข้อ 1)

ไม่ parse markdown, ไม่ re-instantiate ผ่าน MacroStrategyDirection guardrail validator ซ้ำ —
ข้อมูลผ่าน validation ครบแล้วตอนที่ strategic_allocator_node เขียนไฟล์ (ดู tools/macro/report_formatter.py)
"""
import json

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_session
from api.schemas import (
    MacroDashboardDTO,
    PortfolioDTO,
    macro_dashboard_dto_from_raw,
    portfolio_dto_from_raw,
)
from tools.archivist.core import VAULT_PATH

router = APIRouter(dependencies=[Depends(require_session)])

_STRATEGY_SUBDIR = "30_Knowledge_Base/Strategies"


def _latest_strategy_json() -> dict:
    strategy_dir = VAULT_PATH / _STRATEGY_SUBDIR
    candidates = sorted(strategy_dir.glob("Macro_Strategy_Direction_*.json"))
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="ยังไม่มีรายงาน Macro Strategy ที่มี JSON sidecar — รอรายงานถัดไปหลัง Phase 0 อัปเดต",
        )
    latest = candidates[-1]
    return json.loads(latest.read_text(encoding="utf-8"))


@router.get("/api/portfolio/latest", response_model=PortfolioDTO)
def get_latest_portfolio() -> PortfolioDTO:
    return portfolio_dto_from_raw(_latest_strategy_json())


@router.get("/api/macro/dashboard", response_model=MacroDashboardDTO)
def get_macro_dashboard() -> MacroDashboardDTO:
    return macro_dashboard_dto_from_raw(_latest_strategy_json())
