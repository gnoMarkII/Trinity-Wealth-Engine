"""FastAPI Router for OHLCV Market Data & Candlestick Charts."""
import logging
from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import require_session
from api.dependencies import get_ohlcv_service
from api.schemas import OHLCVResponseDTO
from tools.market.ohlcv.application.query_service import (
    OHLCVQueryService,
    OhlcvRequestValidationError,
)

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/equity",
    tags=["equity"],
    dependencies=[Depends(require_session)],
)


@router.get("/{ticker}/ohlcv", response_model=OHLCVResponseDTO)
def get_equity_ohlcv(
    ticker: str,
    range: str = Query("6mo", description="Historical range based on interval capability matrix"),
    interval: str = Query("1d", description="Bar interval (15m, 1h, 1d, 1wk, 1mo)"),
    service: OHLCVQueryService = Depends(get_ohlcv_service),
) -> OHLCVResponseDTO:
    try:
        return service.get_ohlcv(ticker, range_str=range, interval_str=interval)
    except OhlcvRequestValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        msg = str(exc)
        if "not found" in msg.lower() or "no ohlcv" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=502, detail=msg)
