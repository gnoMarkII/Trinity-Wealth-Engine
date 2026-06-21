"""Pytest fixtures shared across the suite."""
import os
import sys
from pathlib import Path

# ทำให้ทุก test resolve absolute imports ได้ (agents/, tools/, core/, schemas/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import pytest  # noqa: E402


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    """แยก Vault per test — set OBSIDIAN_VAULT_PATH ให้ชี้ tmp_path"""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    yield tmp_path


@pytest.fixture
def isolated_portfolio(tmp_vault, monkeypatch):
    """โหลด portfolio submodules fresh ทุกครั้ง"""
    import importlib
    from types import SimpleNamespace

    for mod_name in list(sys.modules):
        if mod_name.startswith("tools.portfolio.") or mod_name.startswith("tools.portfolio_tools"):
            del sys.modules[mod_name]

    import tools.portfolio.core as core
    import tools.portfolio.trading as trading
    import tools.portfolio.watchlist as watchlist
    import tools.portfolio.goals as goals
    import tools.portfolio.journal as journal
    import tools.portfolio.prices as prices
    import tools.portfolio.performance as perf

    pt = SimpleNamespace()
    pt.Holding = core.Holding
    pt.PortfolioState = core.PortfolioState
    pt._recalc_holding = core._recalc_holding
    pt._recalc_all = core._recalc_all
    pt._load_or_init = core._load_or_init
    pt._find_holding = core._find_holding
    pt._save = core._save
    pt.compute_allocation_breakdown = core.compute_allocation_breakdown
    pt.get_portfolio_state = core.get_portfolio_state

    pt._compute_total_cost = trading._compute_total_cost
    pt._record_income_locked = trading._record_income_locked
    pt._execute_trade_locked = trading._execute_trade_locked
    pt._manage_cash_flow_locked = trading._manage_cash_flow_locked
    pt._update_fx_rate_locked = trading._update_fx_rate_locked
    pt._edit_holding_locked = trading._edit_holding_locked
    pt.record_income = trading.record_income
    pt.execute_trade = trading.execute_trade
    pt.manage_cash_flow = trading.manage_cash_flow
    pt.update_fx_rate = trading.update_fx_rate
    pt.edit_holding = trading.edit_holding
    pt.batch_import_holdings = trading.batch_import_holdings

    pt.sync_market_prices = prices.sync_market_prices
    pt._refresh_prices = prices._refresh_prices

    pt.add_to_watchlist = watchlist.add_to_watchlist
    pt.remove_from_watchlist = watchlist.remove_from_watchlist
    pt.read_watchlist = watchlist.read_watchlist

    pt.record_performance_snapshot = perf.record_performance_snapshot
    pt.read_performance_history = perf.read_performance_history

    pt.append_trading_journal = journal.append_trading_journal
    pt.read_trading_journal = journal.read_trading_journal

    pt.set_goal = goals.set_goal
    pt.remove_goal = goals.remove_goal
    pt.get_goals_progress = goals.get_goals_progress
    pt.GOALS_PATH = goals.GOALS_PATH
    pt.GOALS_ITEMS_DIR = goals.GOALS_ITEMS_DIR

    return pt


@pytest.fixture
def isolated_archivist(tmp_vault, monkeypatch):
    import importlib
    import unittest.mock
    from types import SimpleNamespace
    
    # Use existing modules


    import tools.archivist.core as core
    import tools.archivist.writer as writer
    import tools.archivist.indexer as indexer
    import tools.archivist.search as search
    import tools.archivist.linter as linter
    import tools.archivist.parser as parser
    
    for mod in [core, writer, indexer, search, linter, parser]:
        monkeypatch.setattr(mod, "VAULT_PATH", tmp_vault.resolve())

    monkeypatch.setattr(search, "CHROMA_PATH", tmp_vault.resolve() / ".chroma_index")
    monkeypatch.setattr(search, "_CHROMA_MTIME_FILE", tmp_vault.resolve() / ".chroma_mtime")
    
    at = SimpleNamespace()
    at.init_vault_structure = core.init_vault_structure
    at.read_file = core.read_file
    at.save_memory = writer.save_memory
    at.write_raw_markdown = writer.write_raw_markdown
    at.update_master_index = indexer.update_master_index
    at.search_all_memories = search.search_all_memories
    at.search_graph_context = search.search_graph_context
    at.lint_structural_health = linter.lint_structural_health
    at.lint_semantic_conflict = linter.lint_semantic_conflict
    
    monkeypatch.setattr(search, "get_embeddings", unittest.mock.MagicMock())
    
    return at
