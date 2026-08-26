"""FastAPI Equity Routes Facade (Backward Compatibility Re-export).

All route handlers are now implemented under `api.routers.equity.*`.
This module maintains full backward compatibility for existing imports and test patches.
"""
import yfinance as yf
from tools.market.calendar import get_asset_calendar
from tools.market.earnings import fetch_earnings_dates
from tools.archivist.core import VAULT_PATH
from api.compatibility.equity import (
    _validate_ticker,
    _validate_schema,
    _get_equity_files,
    _extract_date_key,
    _get_latest_sidecar_for_ticker,
    _is_agent_generated,
    _get_equity_news_from_vault,
    _extract_note_datetime,
    _get_analyst_lock,
    positive_int_or_none,
)
from api.routers.equity import (
    router,
    get_latest_equities,
    get_equity_note_content,
    get_equity_valuation_targets,
    get_equity_insider_filings,
    get_equity_analyst_context,
    get_equity_financial_statements,
    get_equity_detail,
    get_equity_news,
    get_equity_notes,
)

__all__ = [
    "yf",
    "get_asset_calendar",
    "fetch_earnings_dates",
    "VAULT_PATH",
    "router",
    "_validate_ticker",
    "_validate_schema",
    "_get_equity_files",
    "_extract_date_key",
    "_get_latest_sidecar_for_ticker",
    "_is_agent_generated",
    "_get_equity_news_from_vault",
    "_extract_note_datetime",
    "_get_analyst_lock",
    "positive_int_or_none",
    "get_latest_equities",
    "get_equity_note_content",
    "get_equity_valuation_targets",
    "get_equity_insider_filings",
    "get_equity_analyst_context",
    "get_equity_financial_statements",
    "get_equity_detail",
    "get_equity_news",
    "get_equity_notes",
]
