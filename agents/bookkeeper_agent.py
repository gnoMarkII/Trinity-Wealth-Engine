import os

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain.agents import create_agent

from tools.portfolio.core import get_portfolio_state, compute_allocation_breakdown
from tools.portfolio.trading import (
    execute_trade, record_income, batch_import_holdings,
    manage_cash_flow, update_fx_rate, edit_holding,
)
from tools.portfolio.watchlist import add_to_watchlist, remove_from_watchlist, read_watchlist
from tools.portfolio.goals import set_goal, remove_goal, get_goals_progress
from tools.portfolio.journal import append_trading_journal, read_trading_journal
from tools.portfolio.prices import sync_market_prices
from tools.portfolio.performance import record_performance_snapshot, read_performance_history
from core.prompt_harness import get_harness

_FUND_NAME = os.getenv("FUND_NAME", "กองทุนส่วนตัว")

# BOOKKEEPER_SYSTEM_PROMPT ถูกย้ายไปที่ prompts/skills/bookkeeper/SKILL.md ผ่านระบบ PromptHarness


_bookkeeper_tools = [
    get_portfolio_state,
    execute_trade,
    record_income,
    append_trading_journal,
    add_to_watchlist,
    remove_from_watchlist,
    record_performance_snapshot,
    batch_import_holdings,
    sync_market_prices,
    manage_cash_flow,
    update_fx_rate,
    edit_holding,
    compute_allocation_breakdown,
    read_performance_history,
    read_watchlist,
    read_trading_journal,
    set_goal,
    remove_goal,
    get_goals_progress,
]


def create_bookkeeper(model: BaseChatModel | Runnable):
    """สร้าง Bookkeeper ReAct agent พร้อม Portfolio tools — caller ต้องส่ง model มาเสมอ"""
    harness = get_harness("bookkeeper")
    return create_agent(
        model=model,
        tools=_bookkeeper_tools,
        system_prompt=harness.get_system_prompt(fund_name=_FUND_NAME)
    )
