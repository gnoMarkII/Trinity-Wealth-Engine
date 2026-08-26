"""PortfolioApplication — Core application aggregate holding all portfolio sub-services."""
from dataclasses import dataclass

from tools.portfolio.services.portfolio_state_service import PortfolioStateService
from tools.portfolio.services.trading_service import PortfolioTradingService
from tools.portfolio.services.cash_flow_service import PortfolioCashFlowService
from tools.portfolio.services.ledger_service import PortfolioLedgerService
from tools.portfolio.services.goal_service import PortfolioGoalService
from tools.portfolio.services.performance_service import PortfolioPerformanceService
from tools.portfolio.services.watchlist_service import PortfolioWatchlistService
from tools.portfolio.services.journal_service import PortfolioJournalService


class PortfolioApplication:
    """Orchestrator encapsulating all Domain Application Services for Portfolio."""

    def __init__(
        self,
        state_service: PortfolioStateService,
        trading_service: PortfolioTradingService,
        cash_flow_service: PortfolioCashFlowService,
        ledger_service: PortfolioLedgerService,
        goal_service: PortfolioGoalService,
        performance_service: PortfolioPerformanceService,
        watchlist_service: PortfolioWatchlistService,
        journal_service: PortfolioJournalService,
    ) -> None:
        self.state_service = state_service
        self.trading_service = trading_service
        self.cash_flow_service = cash_flow_service
        self.ledger_service = ledger_service
        self.goal_service = goal_service
        self.performance_service = performance_service
        self.watchlist_service = watchlist_service
        self.journal_service = journal_service
