import inspect
import pytest
from typing import get_type_hints

from tools.portfolio import agent_tools
from tools.portfolio.service import PortfolioService
from tools.portfolio import (
    core as core_mod,
    trading as trading_mod,
    prices as prices_mod,
    journal as journal_mod,
    watchlist as watchlist_mod,
    goals as goals_mod,
    performance as perf_mod,
    dividends as div_mod,
    ledger_replay as ledger_mod,
)

ALL_AGENT_TOOL_NAMES = [
    "get_portfolio_state",
    "compute_allocation_breakdown",
    "tool_list_portfolios",
    "tool_create_portfolio",
    "tool_delete_portfolio",
    "tool_rename_portfolio",
    "execute_trade",
    "record_income",
    "batch_import_holdings",
    "manage_cash_flow",
    "update_fx_rate",
    "edit_holding",
    "sync_market_prices",
    "append_trading_journal",
    "read_trading_journal",
    "add_to_watchlist",
    "remove_from_watchlist",
    "read_watchlist",
    "set_goal",
    "remove_goal",
    "get_goals_progress",
    "record_performance_snapshot",
    "read_performance_history",
]

STRUCTURED_FACADE_AND_SERVICE_MAP = [
    ("create_portfolio", core_mod.create_portfolio, PortfolioService.create_portfolio),
    ("delete_portfolio", core_mod.delete_portfolio, PortfolioService.delete_portfolio),
    ("update_portfolio_name", core_mod.update_portfolio_name, PortfolioService.update_portfolio_name),
    ("list_portfolios", core_mod.list_portfolios, PortfolioService.list_portfolios),
    ("get_structured_portfolio_state", core_mod.get_structured_portfolio_state, PortfolioService.get_structured_portfolio_state),
    ("get_structured_bucket_allocation", core_mod.get_structured_bucket_allocation, PortfolioService.get_structured_bucket_allocation),
    ("structured_assign_holding_bucket", core_mod.structured_assign_holding_bucket, PortfolioService.structured_assign_holding_bucket),
    ("structured_batch_assign_holding_buckets", core_mod.structured_batch_assign_holding_buckets, PortfolioService.structured_batch_assign_holding_buckets),
    ("structured_batch_remove_holdings", core_mod.structured_batch_remove_holdings, PortfolioService.structured_batch_remove_holdings),
    ("structured_reset_clean_slate", core_mod.structured_reset_clean_slate, PortfolioService.structured_reset_clean_slate),
    ("structured_upsert_allocation_targets", core_mod.structured_upsert_allocation_targets, PortfolioService.structured_upsert_allocation_targets),
    ("structured_execute_trade", trading_mod.structured_execute_trade, PortfolioService.structured_execute_trade),
    ("structured_manage_cash_flow", trading_mod.structured_manage_cash_flow, PortfolioService.structured_manage_cash_flow),
    ("structured_record_income", trading_mod.structured_record_income, PortfolioService.structured_record_income),
    ("structured_edit_holding", trading_mod.structured_edit_holding, PortfolioService.structured_edit_holding),
    ("structured_remove_holding", trading_mod.structured_remove_holding, PortfolioService.structured_remove_holding),
    ("get_structured_trades_log", trading_mod.get_structured_trades_log, PortfolioService.get_structured_trades_log),
    ("update_trade_note", trading_mod.update_trade_note, PortfolioService.update_trade_note),
    ("fetch_fx_rate", prices_mod.fetch_fx_rate, PortfolioService.fetch_fx_rate),
    ("fetch_latest_price", prices_mod.fetch_latest_price, PortfolioService.fetch_latest_price),
    ("get_structured_watchlist", watchlist_mod.get_structured_watchlist, PortfolioService.get_structured_watchlist),
    ("structured_upsert_watchlist_item", watchlist_mod.structured_upsert_watchlist_item, PortfolioService.structured_upsert_watchlist_item),
    ("structured_remove_watchlist_item", watchlist_mod.structured_remove_watchlist_item, PortfolioService.structured_remove_watchlist_item),
    ("get_structured_goals", goals_mod.get_structured_goals, PortfolioService.get_structured_goals),
    ("structured_upsert_goal", goals_mod.structured_upsert_goal, PortfolioService.structured_upsert_goal),
    ("structured_remove_goal", goals_mod.structured_remove_goal, PortfolioService.structured_remove_goal),
    ("get_structured_journal", journal_mod.get_structured_journal, PortfolioService.get_structured_journal),
    ("structured_append_journal", journal_mod.structured_append_journal, PortfolioService.structured_append_journal),
    ("get_structured_performance_history", perf_mod.get_structured_performance_history, PortfolioService.get_structured_performance_history),
    ("sync_dividends_from_history", div_mod.sync_dividends_from_history, PortfolioService.sync_dividends_from_history),
    ("edit_transaction", ledger_mod.edit_transaction, PortfolioService.edit_transaction),
    ("delete_transaction", ledger_mod.delete_transaction, PortfolioService.delete_transaction),
]


@pytest.mark.parametrize("tool_name", ALL_AGENT_TOOL_NAMES)
def test_agent_tool_contract(tool_name):
    """Verify all 23 LangChain @tool objects exist and have proper callable attributes."""
    tool = getattr(agent_tools, tool_name, None)
    assert tool is not None, f"Missing tool {tool_name}"
    assert hasattr(tool, "name") and tool.name == tool_name
    assert hasattr(tool, "description") and len(tool.description) > 0
    assert hasattr(tool, "func") and callable(tool.func)


@pytest.mark.parametrize("fn_name, facade_fn, service_fn", STRUCTURED_FACADE_AND_SERVICE_MAP)
def test_structured_functions_signature_parity(fn_name, facade_fn, service_fn):
    """Verify all 32 structured API functions have identical parameter signatures with PortfolioService."""
    facade_sig = inspect.signature(facade_fn)
    service_sig = inspect.signature(service_fn)

    # Note: service_fn includes 'self' as first argument
    service_params = {k: v for k, v in service_sig.parameters.items() if k != "self"}
    facade_params = dict(facade_sig.parameters)

    assert list(facade_params.keys()) == list(service_params.keys())
    for p_name in facade_params:
        fp = facade_params[p_name]
        sp = service_params[p_name]
        assert fp.kind == sp.kind
        assert fp.default == sp.default
