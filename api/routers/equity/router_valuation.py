"""Inbound HTTP adapter for Equity valuation targets."""
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import ValuationTargetsDTO
from api.dependencies import get_equity_valuation_service
from application.equity.service import EquityValuationApplicationService
from application.equity.validation import validate_ticker

router = APIRouter()


@router.get("/{ticker}/valuation-targets", response_model=ValuationTargetsDTO)
def get_equity_valuation_targets(
    ticker: str,
    service: EquityValuationApplicationService = Depends(get_equity_valuation_service),
) -> ValuationTargetsDTO:
    try:
        clean_ticker = validate_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ValuationTargetsDTO(**service.get_targets(clean_ticker))


__all__ = ["router", "get_equity_valuation_targets"]
