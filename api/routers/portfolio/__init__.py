"""FastAPI Portfolio Routers Aggregator.

Aggregates domain sub-routers:
  - router_state: Portfolio CRUD, State, Buckets, Allocation Targets
  - router_trading: Trade execution, Cash Flow, Income, FX rate, Dividends/Prices sync
  - router_ledger: Transactions list, Note editing, Edit/Delete transactions
  - router_extensions: Watchlist, Goals, Journal, Performance history
  - router_macro: Macro Dashboard, Strategy, Indicators, News Funnel, Calendar
"""
from fastapi import APIRouter, Depends
from api.auth import require_session

from .common import handle_portfolio_exceptions
from .router_macro import (
    router as macro_router,
    get_latest_portfolio,
    get_macro_dashboard,
    get_macro_indicator_series,
    get_news_funnel_pending,
    get_news_funnel_filtered,
    delete_news_funnel_pending,
    get_portfolio_calendar,
)
from .router_state import (
    router as state_router,
    list_portfolios_endpoint,
    create_portfolio_endpoint,
    delete_portfolio_endpoint,
    rename_portfolio_endpoint,
    get_actual_portfolio_state,
    get_actual_bucket_allocations,
    upsert_allocation_targets,
    assign_holding_bucket,
    batch_assign_holding_buckets,
    batch_remove_holdings,
    reset_portfolio_clean_slate,
)
from .router_trading import (
    router as trading_router,
    execute_trade_endpoint,
    manage_cash_flow_endpoint,
    record_income_endpoint,
    edit_holding_endpoint,
    remove_holding_endpoint,
    get_fx_rate_endpoint,
    sync_dividends_endpoint,
)
from .router_ledger import (
    router as ledger_router,
    get_actual_transactions,
    update_transaction_note_endpoint,
    edit_transaction_endpoint,
    delete_transaction_endpoint,
)
from .router_extensions import (
    router as extensions_router,
    get_actual_watchlist,
    upsert_watchlist_item_endpoint,
    remove_watchlist_item_endpoint,
    get_actual_goals,
    upsert_goal_endpoint,
    remove_goal_endpoint,
    get_actual_journal,
    append_journal_endpoint,
    get_actual_performance,
    trigger_performance_snapshot,
)

router = APIRouter(dependencies=[Depends(require_session)])

# Include sub-routers in explicit order
router.include_router(state_router)
router.include_router(trading_router)
router.include_router(ledger_router)
router.include_router(extensions_router)
router.include_router(macro_router)

__all__ = [
    "router",
    "handle_portfolio_exceptions",
    "list_portfolios_endpoint",
    "create_portfolio_endpoint",
    "delete_portfolio_endpoint",
    "rename_portfolio_endpoint",
    "get_latest_portfolio",
    "get_macro_dashboard",
    "get_macro_indicator_series",
    "get_news_funnel_pending",
    "get_news_funnel_filtered",
    "delete_news_funnel_pending",
    "get_portfolio_calendar",
    "get_actual_portfolio_state",
    "get_actual_bucket_allocations",
    "get_actual_watchlist",
    "get_actual_goals",
    "get_actual_performance",
    "trigger_performance_snapshot",
    "get_actual_journal",
    "get_actual_transactions",
    "update_transaction_note_endpoint",
    "edit_transaction_endpoint",
    "delete_transaction_endpoint",
    "get_fx_rate_endpoint",
    "sync_dividends_endpoint",
    "upsert_allocation_targets",
    "assign_holding_bucket",
    "batch_assign_holding_buckets",
    "batch_remove_holdings",
    "reset_portfolio_clean_slate",
    "execute_trade_endpoint",
    "manage_cash_flow_endpoint",
    "record_income_endpoint",
    "edit_holding_endpoint",
    "remove_holding_endpoint",
    "upsert_watchlist_item_endpoint",
    "remove_watchlist_item_endpoint",
    "upsert_goal_endpoint",
    "remove_goal_endpoint",
    "append_journal_endpoint",
]
