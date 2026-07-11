"""GET /api/portfolio/latest, GET /api/macro/dashboard — อ่าน sidecar JSON ตรงๆ (Rev.3/5 ข้อ 1)

ไม่ parse markdown, ไม่ re-instantiate ผ่าน MacroStrategyDirection guardrail validator ซ้ำ —
ข้อมูลผ่าน validation ครบแล้วตอนที่ strategic_allocator_node เขียนไฟล์ (ดู tools/macro/report_formatter.py)
"""
import json

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_session
from api.schemas import (
    MacroIndicatorSeriesDTO,
    MacroDashboardDTO,
    PortfolioDTO,
    macro_dashboard_dto_from_raw,
    portfolio_dto_from_raw,
)
from tools.archivist.core import VAULT_PATH
from tools.macro.dashboard import load_indicator_series

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


@router.get("/api/macro/indicators/{indicator_id}/series", response_model=MacroIndicatorSeriesDTO)
def get_macro_indicator_series(indicator_id: str, range: str = "3m") -> MacroIndicatorSeriesDTO:
    raw = _latest_strategy_json()
    indicators = raw.get("dashboard_indicators", [])
    indicator = next(
        (
            item
            for item in indicators
            if isinstance(item, dict) and item.get("indicator_id") == indicator_id
        ),
        None,
    )
    if indicator is None:
        raise HTTPException(status_code=404, detail="Macro indicator not found in the latest report")
    if range not in {"1m", "3m", "1y"}:
        raise HTTPException(status_code=422, detail="range must be one of: 1m, 3m, 1y")

    try:
        points = load_indicator_series(VAULT_PATH, str(indicator.get("series_key", "")), range)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Macro indicator series is unavailable") from exc

    return MacroIndicatorSeriesDTO(
        indicator_id=indicator_id,
        series_key=str(indicator.get("series_key", "")),
        label=str(indicator.get("label", "")),
        unit=str(indicator.get("unit", "")),
        range=range,
        points=points,
    )
