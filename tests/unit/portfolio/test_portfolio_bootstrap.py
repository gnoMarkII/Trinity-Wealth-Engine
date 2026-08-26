"""Unit tests for Portfolio composition root, bootstrap, and PortfolioService facade."""
import pytest
from unittest.mock import MagicMock

from tools.portfolio.bootstrap import (
    PortfolioDependencies,
    build_default_portfolio_dependencies,
    build_portfolio_application,
)
from tools.portfolio.application import PortfolioApplication
from tools.portfolio.service import PortfolioService
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from tools.portfolio.ports.watchlist_port import WatchlistRepositoryPort
from tools.portfolio.ports.goals_port import GoalsRepositoryPort
from tools.portfolio.ports.performance_port import PerformanceRepositoryPort
from tools.portfolio.ports.journal_port import TradeJournalPort
from tools.portfolio.ports.price_port import MarketPricePort


def test_build_default_portfolio_dependencies():
    deps = build_default_portfolio_dependencies()
    assert isinstance(deps, PortfolioDependencies)
    assert isinstance(deps.repo, PortfolioRepositoryPort) or hasattr(deps.repo, "unit_of_work")
    assert isinstance(deps.watchlist_repo, WatchlistRepositoryPort) or hasattr(deps.watchlist_repo, "load_watchlist")
    assert isinstance(deps.goals_repo, GoalsRepositoryPort) or hasattr(deps.goals_repo, "load_goals")
    assert isinstance(deps.perf_repo, PerformanceRepositoryPort) or hasattr(deps.perf_repo, "read_history")
    assert isinstance(deps.journal_provider, TradeJournalPort) or hasattr(deps.journal_provider, "append_journal")
    assert isinstance(deps.price_provider, MarketPricePort) or hasattr(deps.price_provider, "fetch_price")


def test_build_portfolio_application_with_custom_deps():
    mock_repo = MagicMock(spec=PortfolioRepositoryPort)
    mock_watchlist = MagicMock(spec=WatchlistRepositoryPort)
    mock_goals = MagicMock(spec=GoalsRepositoryPort)
    mock_perf = MagicMock(spec=PerformanceRepositoryPort)
    mock_journal = MagicMock(spec=TradeJournalPort)
    mock_price = MagicMock(spec=MarketPricePort)

    deps = PortfolioDependencies(
        repo=mock_repo,
        watchlist_repo=mock_watchlist,
        goals_repo=mock_goals,
        perf_repo=mock_perf,
        journal_provider=mock_journal,
        price_provider=mock_price,
    )
    app = build_portfolio_application(deps=deps)
    assert isinstance(app, PortfolioApplication)
    assert app.state_service.repo == mock_repo
    assert app.trading_service.price_provider == mock_price
    assert app.goal_service.goals_repo == mock_goals


def test_portfolio_service_facade_delegation():
    mock_repo = MagicMock(spec=PortfolioRepositoryPort)
    mock_watchlist = MagicMock(spec=WatchlistRepositoryPort)
    mock_goals = MagicMock(spec=GoalsRepositoryPort)
    mock_perf = MagicMock(spec=PerformanceRepositoryPort)
    mock_journal = MagicMock(spec=TradeJournalPort)
    mock_price = MagicMock(spec=MarketPricePort)

    svc = PortfolioService(
        repo=mock_repo,
        watchlist_repo=mock_watchlist,
        goals_repo=mock_goals,
        perf_repo=mock_perf,
        journal_provider=mock_journal,
        price_provider=mock_price,
    )
    assert svc.repo == mock_repo
    assert svc.watchlist_repo == mock_watchlist
    assert svc.goals_repo == mock_goals
    assert svc.perf_repo == mock_perf
    assert svc.journal_provider == mock_journal
    assert svc.price_provider == mock_price
