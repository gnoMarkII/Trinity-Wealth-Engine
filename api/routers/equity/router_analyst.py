"""Inbound HTTP adapter for analyst targets and earnings context."""
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import AnalystContextDTO
from api.dependencies import get_equity_analyst_service
from application.equity.service import EquityAnalystApplicationService
from application.equity.validation import validate_ticker

router = APIRouter()


@router.get("/{ticker}/analyst-context", response_model=AnalystContextDTO)
def get_equity_analyst_context(
    ticker: str,
    service: EquityAnalystApplicationService = Depends(get_equity_analyst_service),
) -> AnalystContextDTO:
    try:
        clean_ticker = validate_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AnalystContextDTO(**service.get_context(clean_ticker))


__all__ = ["router", "get_equity_analyst_context"]
