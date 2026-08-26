"""FastAPI Portfolio Routes Facade (Backward Compatibility Re-export).

All route handlers are now implemented under `api.routers.portfolio.*`.
This module maintains full backward compatibility for existing imports and test patches.
"""
from tools.archivist.core import VAULT_PATH
from tools.portfolio import (
    core as portfolio_core,
    trading as portfolio_trading,
    watchlist as portfolio_watchlist,
    goals as portfolio_goals,
    performance as portfolio_perf,
    journal as portfolio_journal,
    prices as portfolio_prices,
    dividends as portfolio_dividends,
    ledger_replay as portfolio_ledger_replay,
)
from api.compatibility.portfolio import _latest_strategy_json, _STRATEGY_SUBDIR
from api.routers.portfolio import (
    router,
    handle_portfolio_exceptions,
    list_portfolios_endpoint,
    create_portfolio_endpoint,
    delete_portfolio_endpoint,
    rename_portfolio_endpoint,
    get_latest_portfolio,
    get_macro_dashboard,
    get_macro_indicator_series,
    get_news_funnel_pending,
    get_news_funnel_filtered,
    delete_news_funnel_pending,
    get_portfolio_calendar,
    get_actual_portfolio_state,
    get_actual_bucket_allocations,
    get_actual_watchlist,
    get_actual_goals,
    get_actual_performance,
    trigger_performance_snapshot,
    get_actual_journal,
    get_actual_transactions,
    update_transaction_note_endpoint,
    edit_transaction_endpoint,
    delete_transaction_endpoint,
    get_fx_rate_endpoint,
    sync_dividends_endpoint,
    upsert_allocation_targets,
    assign_holding_bucket,
    batch_assign_holding_buckets,
    batch_remove_holdings,
    reset_portfolio_clean_slate,
    execute_trade_endpoint,
    manage_cash_flow_endpoint,
    record_income_endpoint,
    edit_holding_endpoint,
    remove_holding_endpoint,
    upsert_watchlist_item_endpoint,
    remove_watchlist_item_endpoint,
    upsert_goal_endpoint,
    remove_goal_endpoint,
    append_journal_endpoint,
)

# Compatibility metadata is intentionally kept in this facade only.  New
# routers use ``Depends(get_portfolio_service)`` and never inspect these
# modules; the bridge below exists for integrations that still patch the old
# module-level tool objects during the migration.
_ORIGINAL_COMPAT_MODULES = {
    "core": portfolio_core,
    "trading": portfolio_trading,
}
_ORIGINAL_COMPAT_CALLABLES = {
    "core_state": getattr(portfolio_core, "get_structured_portfolio_state", None),
    "core_allocations": getattr(portfolio_core, "get_structured_bucket_allocation", None),
    "trading_execute": getattr(portfolio_trading, "structured_execute_trade", None),
}


def _invoke_legacy_tool(target, **kwargs):
    """Invoke either a legacy LangChain tool, function, or patched mock."""
    from unittest.mock import Mock

    if isinstance(target, Mock):
        return target(**kwargs)
    invoke = getattr(target, "invoke", None)
    if callable(invoke):
        return invoke(kwargs)
    return target(**kwargs)


class _LegacyPortfolioPatchProxy:
    """Forward normal calls to the application service and patched calls to legacy tools."""

    def __init__(self, service):
        self._service = service

    def structured_execute_trade(self, **kwargs):
        return _invoke_legacy_tool(portfolio_trading.structured_execute_trade, **kwargs)

    def get_structured_portfolio_state(self, **kwargs):
        return _invoke_legacy_tool(portfolio_core.get_structured_portfolio_state, **kwargs)

    def get_structured_bucket_allocation(self, **kwargs):
        return _invoke_legacy_tool(portfolio_core.get_structured_bucket_allocation, **kwargs)

    def __getattr__(self, name):
        return getattr(self._service, name)


def maybe_wrap_portfolio_service(service):
    """Apply the legacy patch bridge only when compatibility targets changed."""
    current_modules_changed = any(
        current is not _ORIGINAL_COMPAT_MODULES[key]
        for key, current in (("core", portfolio_core), ("trading", portfolio_trading))
    )
    current_callables_changed = (
        getattr(portfolio_core, "get_structured_portfolio_state", None)
        is not _ORIGINAL_COMPAT_CALLABLES["core_state"]
        or getattr(portfolio_core, "get_structured_bucket_allocation", None)
        is not _ORIGINAL_COMPAT_CALLABLES["core_allocations"]
        or getattr(portfolio_trading, "structured_execute_trade", None)
        is not _ORIGINAL_COMPAT_CALLABLES["trading_execute"]
    )
    if current_modules_changed or current_callables_changed:
        return _LegacyPortfolioPatchProxy(service)
    return service

__all__ = [
    "VAULT_PATH",
    "portfolio_core",
    "portfolio_trading",
    "portfolio_watchlist",
    "portfolio_goals",
    "portfolio_perf",
    "portfolio_journal",
    "portfolio_prices",
    "portfolio_dividends",
    "portfolio_ledger_replay",
    "router",
    "handle_portfolio_exceptions",
    "_latest_strategy_json",
    "_STRATEGY_SUBDIR",
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
    "maybe_wrap_portfolio_service",
]
