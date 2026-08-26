"""FastAPI Equity Routers Aggregator.

Aggregates domain sub-routers with explicit route precedence:
  - router_meta: /latest, /notes/content (MUST be registered before /{ticker})
  - router_research: /{ticker}/valuation-targets, /{ticker}/insider-filings, /{ticker}/analyst-context, /{ticker}/financials
  - router_entity: /{ticker}, /{ticker}/news, /{ticker}/notes
"""
from fastapi import APIRouter, Depends

from api.auth import require_session
from .router_meta import (
    router as meta_router,
    get_latest_equities,
    get_equity_note_content,
)
from .router_research import (
    router as research_router,
    get_equity_valuation_targets,
    get_equity_insider_filings,
    get_equity_analyst_context,
    get_equity_financial_statements,
    summarize_earnings_call,
    get_earnings_call_run,
    retry_earnings_call_run,
    get_earnings_calls,
)
from .router_entity import (
    router as entity_router,
    get_equity_detail,
    get_equity_news,
    get_equity_notes,
)

router = APIRouter(
    prefix="/api/equity",
    tags=["equity"],
    dependencies=[Depends(require_session)],
)

# Crucial: Register meta_router (static paths) before /{ticker} catch-all
router.include_router(meta_router)
router.include_router(research_router)
router.include_router(entity_router)

__all__ = [
    "router",
    "get_latest_equities",
    "get_equity_note_content",
    "get_equity_valuation_targets",
    "get_equity_insider_filings",
    "get_equity_analyst_context",
    "get_equity_financial_statements",
    "summarize_earnings_call",
    "get_earnings_call_run",
    "retry_earnings_call_run",
    "get_earnings_calls",
    "get_equity_detail",
    "get_equity_news",
    "get_equity_notes",
]
