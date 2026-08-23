"""Domain Sub-services for PortfolioService (Facade pattern).

Each sub-service owns a specific domain boundary:
  - PortfolioStateService       -- CRUD, Bucket Allocation, Reset Clean Slate
  - PortfolioTradingService     -- Buy/Sell, Batch Import, Edit Holding, FX/Prices
  - PortfolioCashFlowService    -- Deposit/Withdraw, Income, Dividends, FX
  - PortfolioLedgerService      -- Edit/Delete Transactions, Replay PnL
  - PortfolioGoalService        -- Goals CRUD & Tracking
  - PortfolioPerformanceService -- Snapshots, NAV History, Drawdown
  - PortfolioWatchlistService   -- Watchlist CRUD
  - PortfolioJournalService     -- Journal Notes, Formatting & Markdown
"""
from .portfolio_state_service import PortfolioStateService
from .trading_service import PortfolioTradingService
from .cash_flow_service import PortfolioCashFlowService
from .ledger_service import PortfolioLedgerService
from .goal_service import PortfolioGoalService
from .performance_service import PortfolioPerformanceService
from .watchlist_service import PortfolioWatchlistService
from .journal_service import PortfolioJournalService

__all__ = [
    "PortfolioStateService",
    "PortfolioTradingService",
    "PortfolioCashFlowService",
    "PortfolioLedgerService",
    "PortfolioGoalService",
    "PortfolioPerformanceService",
    "PortfolioWatchlistService",
    "PortfolioJournalService",
]
