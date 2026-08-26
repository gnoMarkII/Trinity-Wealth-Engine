"""Inbound HTTP adapter for Equity detail, news, and note queries."""
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_equity_query_service
from api.schemas import EquityDetailDTO, EquityNewsDTO, EquityNotesDTO
from application.equity.query_service import (
    EquityDataCorruptError,
    EquityResearchQueryService,
)

router = APIRouter()


@router.get("/{ticker}", response_model=EquityDetailDTO)
def get_equity_detail(
    ticker: str,
    service: EquityResearchQueryService = Depends(get_equity_query_service),
) -> EquityDetailDTO:
    try:
        payload = service.get_detail(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EquityDataCorruptError as exc:
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: Data corrupted or ticker mismatch",
        ) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="Equity not found")
    return EquityDetailDTO.model_validate(payload)


@router.get("/{ticker}/news", response_model=EquityNewsDTO)
def get_equity_news(
    ticker: str,
    service: EquityResearchQueryService = Depends(get_equity_query_service),
) -> EquityNewsDTO:
    try:
        payload = service.get_news(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="ยังไม่มีข้อมูลข่าวสำหรับหุ้นตัวนี้ในระบบ")
    return EquityNewsDTO.model_validate(payload)


@router.get("/{ticker}/notes", response_model=EquityNotesDTO)
def get_equity_notes(
    ticker: str,
    days: int = 3,
    service: EquityResearchQueryService = Depends(get_equity_query_service),
) -> EquityNotesDTO:
    try:
        payload = service.list_notes(ticker, days=days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return EquityNotesDTO.model_validate(payload)
