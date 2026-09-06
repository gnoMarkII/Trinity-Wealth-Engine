"""PortfolioApplication — Core application aggregate holding all portfolio sub-services."""
from dataclasses import dataclass
from typing import Optional

from tools.portfolio.services.portfolio_state_service import PortfolioStateService
from tools.portfolio.services.trading_service import PortfolioTradingService
from tools.portfolio.services.cash_flow_service import PortfolioCashFlowService
from tools.portfolio.services.ledger_service import PortfolioLedgerService
from tools.portfolio.services.goal_service import PortfolioGoalService
from tools.portfolio.services.performance_service import PortfolioPerformanceService
from tools.portfolio.services.watchlist_service import PortfolioWatchlistService
from tools.portfolio.services.journal_service import PortfolioJournalService


from tools.portfolio.services.batch_trade_import_service import BatchTradeImportService
from tools.portfolio.services.dime_sync_service import DimeSyncService
from tools.portfolio.services.wealthx_sync_service import WealthXSyncService
from tools.portfolio.services.scbam_sync_service import SCBAMSyncService


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
        batch_import_service: Optional[BatchTradeImportService] = None,
        dime_sync_service: Optional[DimeSyncService] = None,
        wealthx_sync_service: Optional[WealthXSyncService] = None,
        scbam_sync_service: Optional[SCBAMSyncService] = None,
    ) -> None:
        self.state_service = state_service
        self.trading_service = trading_service
        self.cash_flow_service = cash_flow_service
        self.ledger_service = ledger_service
        self.goal_service = goal_service
        self.performance_service = performance_service
        self.watchlist_service = watchlist_service
        self.journal_service = journal_service
        self.batch_import_service = batch_import_service
        self.dime_sync_service = dime_sync_service
        self.wealthx_sync_service = wealthx_sync_service
        self.scbam_sync_service = scbam_sync_service
