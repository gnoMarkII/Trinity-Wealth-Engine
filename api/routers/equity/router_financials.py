"""FastAPI Router for Equity Financial Statements (Income Statement, Balance Sheet, Cash Flow)."""
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_session
from api.schemas import FinancialStatementsDTO
from api.dependencies import get_equity_financials_service
from application.equity.service import EquityFinancialsApplicationService

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{ticker}/financials", response_model=FinancialStatementsDTO)
def get_equity_financial_statements(
    ticker: str,
    market: Optional[str] = None,
    force_refresh: bool = False,
    session: dict = Depends(require_session),
    service: EquityFinancialsApplicationService = Depends(get_equity_financials_service),
) -> FinancialStatementsDTO:
    """ดึงข้อมูลงบการเงินย้อนหลัง (Income Statement, Balance Sheet, Cash Flow) พร้อมระบบ Dual-Provider (EDGAR/yfinance)"""
    try:
        return service.get_statements(
            ticker=ticker,
            market=market,
            force_refresh=force_refresh,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
