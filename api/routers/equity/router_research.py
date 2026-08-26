"""FastAPI Aggregating Router for Equity Research."""
from fastapi import APIRouter

from api.routers.equity.router_valuation import router as valuation_router, get_equity_valuation_targets
from api.routers.equity.router_insider import router as insider_router, get_equity_insider_filings
from api.routers.equity.router_analyst import router as analyst_router, get_equity_analyst_context
from api.routers.equity.router_financials import router as financials_router, get_equity_financial_statements
from api.routers.equity.router_earnings_call import (
    router as earnings_call_router,
    summarize_earnings_call,
    get_earnings_call_run,
    retry_earnings_call_run,
    get_earnings_calls,
)

router = APIRouter()
router.include_router(valuation_router)
router.include_router(insider_router)
router.include_router(analyst_router)
router.include_router(financials_router)
router.include_router(earnings_call_router)

__all__ = [
    "router",
    "get_equity_valuation_targets",
    "get_equity_insider_filings",
    "get_equity_analyst_context",
    "get_equity_financial_statements",
    "summarize_earnings_call",
    "get_earnings_call_run",
    "retry_earnings_call_run",
    "get_earnings_calls",
]
