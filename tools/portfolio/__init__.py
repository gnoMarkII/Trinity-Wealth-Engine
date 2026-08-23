import os
import threading
from typing import Optional

_SERVICE_LOCK = threading.Lock()
_service_instance: Optional["PortfolioService"] = None


def get_default_service() -> "PortfolioService":
    """Thread-safe Singleton Factory for PortfolioService orchestrator."""
    global _service_instance
    if _service_instance is None:
        with _SERVICE_LOCK:
            if _service_instance is None:
                from .service import PortfolioService
                from .adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
                from .adapters.markdown.watchlist_adapter import MarkdownWatchlistAdapter
                from .adapters.markdown.goals_adapter import MarkdownGoalsAdapter
                from .adapters.markdown.performance_adapter import MarkdownPerformanceAdapter
                from .adapters.markdown.journal_vault_adapter import JournalVaultAdapter
                from .adapters.sqlite_mirror_decorator import SqliteMirroredPortfolioRepository
                from .adapters.price_yfinance_adapter import PriceYFinanceAdapter

                md_repo = MarkdownVaultRepositoryAdapter()
                mirror_repo = SqliteMirroredPortfolioRepository(underlying_repo=md_repo)
                watchlist_repo = MarkdownWatchlistAdapter()
                goals_repo = MarkdownGoalsAdapter()
                perf_repo = MarkdownPerformanceAdapter()
                journal_provider = JournalVaultAdapter()
                price_provider = PriceYFinanceAdapter()

                _service_instance = PortfolioService(
                    repo=mirror_repo,
                    watchlist_repo=watchlist_repo,
                    goals_repo=goals_repo,
                    perf_repo=perf_repo,
                    journal_provider=journal_provider,
                    price_provider=price_provider,
                )
    return _service_instance


def set_default_service_for_testing(service: Optional["PortfolioService"]) -> None:
    """Test-only hook to inject mock service or reset singleton."""
    global _service_instance
    with _SERVICE_LOCK:
        _service_instance = service


# Top-level re-exports of domain models
from .domain.models import (
    AllocationTarget,
    DividendRound,
    Holding,
    Summary,
    PortfolioState,
    WatchlistItem,
    WatchlistState,
    PortfolioMeta,
    GoalItem,
    GoalsState,
)
from .domain.constants import CASH_THB_SYMBOL, CASH_USD_SYMBOL, CASH_SYMBOL
from .domain.ledger_change import LedgerChange
from .domain.errors import (
    PortfolioDomainError,
    InsufficientCashError,
    InvalidTradeError,
    HoldingNotFoundError,
    PortfolioNotFoundError,
    RecoveryConflictError,
)
