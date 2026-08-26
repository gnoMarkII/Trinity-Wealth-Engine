"""Inbound HTTP adapter for SEC Form 4 insider filings."""
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import InsiderFilingsResponseDTO, InsiderFilingDTO, InsiderTransactionDTO
from api.dependencies import get_equity_insider_service
from application.equity.service import EquityInsiderApplicationService
from application.equity.validation import validate_ticker

router = APIRouter()


@router.get("/{ticker}/insider-filings", response_model=InsiderFilingsResponseDTO)
def get_equity_insider_filings(
    ticker: str,
    range: str = "1y",
    interval: str = "1d",
    service: EquityInsiderApplicationService = Depends(get_equity_insider_service),
) -> InsiderFilingsResponseDTO:
    try:
        clean_ticker = validate_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = service.get_filings(clean_ticker, requested_range=range, interval=interval)
    payload["filings"] = [
        InsiderFilingDTO(
            **{
                **filing,
                "transactions": [InsiderTransactionDTO(**tx) for tx in filing.get("transactions", [])],
            }
        )
        for filing in payload.get("filings", [])
    ]
    return InsiderFilingsResponseDTO(**payload)


__all__ = ["router", "get_equity_insider_filings"]
