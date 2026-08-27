"""Composition Root & Dependency Injection Bootstrap for Portfolio Bounded Context."""
from dataclasses import dataclass
from typing import Optional

from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from tools.portfolio.ports.watchlist_port import WatchlistRepositoryPort
from tools.portfolio.ports.goals_port import GoalsRepositoryPort
from tools.portfolio.ports.performance_port import PerformanceRepositoryPort
from tools.portfolio.ports.journal_port import TradeJournalPort
from tools.portfolio.ports.price_port import MarketPricePort
from tools.portfolio.ports.dividend_port import DividendHistoryPort

from tools.portfolio.services.portfolio_state_service import PortfolioStateService
from tools.portfolio.services.trading_service import PortfolioTradingService
from tools.portfolio.services.cash_flow_service import PortfolioCashFlowService
from tools.portfolio.services.ledger_service import PortfolioLedgerService
from tools.portfolio.services.goal_service import PortfolioGoalService
from tools.portfolio.services.performance_service import PortfolioPerformanceService
from tools.portfolio.services.watchlist_service import PortfolioWatchlistService
from tools.portfolio.services.journal_service import PortfolioJournalService
from tools.portfolio.application import PortfolioApplication


@dataclass(frozen=True)
class PortfolioDependencies:
    """Dependency bundle for all ports required by the Portfolio Application."""
    repo: PortfolioRepositoryPort
    watchlist_repo: WatchlistRepositoryPort
    goals_repo: GoalsRepositoryPort
    perf_repo: PerformanceRepositoryPort
    journal_provider: TradeJournalPort
    price_provider: MarketPricePort
    dividend_provider: Optional[DividendHistoryPort] = None


def resolve_legacy_portfolio_dependencies(
    *,
    repo: Optional[PortfolioRepositoryPort] = None,
    watchlist_repo: Optional[WatchlistRepositoryPort] = None,
    goals_repo: Optional[GoalsRepositoryPort] = None,
    perf_repo: Optional[PerformanceRepositoryPort] = None,
    journal_provider: Optional[TradeJournalPort] = None,
    price_provider: Optional[MarketPricePort] = None,
    dividend_provider: Optional[DividendHistoryPort] = None,
    db_path: Optional[str] = None,
) -> PortfolioDependencies:
    """Resolve the legacy facade constructor into one dependency bundle.

    ``PortfolioService`` historically accepted partially populated dependency
    arguments and filled the remainder with defaults.  Keeping that behavior
    here leaves the facade as a compatibility surface while ensuring concrete
    adapters are constructed only in this composition root.
    """
    if all(
        dependency is not None
        for dependency in (
            repo,
            watchlist_repo,
            goals_repo,
            perf_repo,
            journal_provider,
            price_provider,
        )
    ):
        return PortfolioDependencies(
            repo=repo,
            watchlist_repo=watchlist_repo,
            goals_repo=goals_repo,
            perf_repo=perf_repo,
            journal_provider=journal_provider,
            price_provider=price_provider,
            dividend_provider=dividend_provider,
        )

    defaults = build_default_portfolio_dependencies(db_path=db_path)
    return PortfolioDependencies(
        repo=repo or defaults.repo,
        watchlist_repo=watchlist_repo or defaults.watchlist_repo,
        goals_repo=goals_repo or defaults.goals_repo,
        perf_repo=perf_repo or defaults.perf_repo,
        journal_provider=journal_provider or defaults.journal_provider,
        price_provider=price_provider or defaults.price_provider,
        dividend_provider=dividend_provider or defaults.dividend_provider,
    )


def build_default_portfolio_dependencies(db_path: Optional[str] = None) -> PortfolioDependencies:
    """Construct concrete adapters and bundle them as default dependencies."""
    from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
    from tools.portfolio.adapters.sqlite_mirror_decorator import SqliteMirroredPortfolioRepository
    from tools.portfolio.adapters.markdown.watchlist_adapter import MarkdownWatchlistAdapter
    from tools.portfolio.adapters.markdown.goals_adapter import MarkdownGoalsAdapter
    from tools.portfolio.adapters.markdown.performance_adapter import MarkdownPerformanceAdapter
    from tools.portfolio.adapters.markdown.journal_vault_adapter import JournalVaultAdapter
    from tools.portfolio.adapters.price_yfinance_adapter import PriceYFinanceAdapter
    from tools.portfolio.adapters.dividend_yfinance_adapter import DividendYFinanceAdapter

    md_repo = MarkdownVaultRepositoryAdapter()
    repo = SqliteMirroredPortfolioRepository(underlying_repo=md_repo, db_path=db_path)
    watchlist_repo = MarkdownWatchlistAdapter()
    goals_repo = MarkdownGoalsAdapter()
    perf_repo = MarkdownPerformanceAdapter()
    journal_provider = JournalVaultAdapter()
    price_provider = PriceYFinanceAdapter()
    dividend_provider = DividendYFinanceAdapter()

    return PortfolioDependencies(
        repo=repo,
        watchlist_repo=watchlist_repo,
        goals_repo=goals_repo,
        perf_repo=perf_repo,
        journal_provider=journal_provider,
        price_provider=price_provider,
        dividend_provider=dividend_provider,
    )


def build_portfolio_application(
    deps: Optional[PortfolioDependencies] = None,
    db_path: Optional[str] = None,
) -> PortfolioApplication:
    """Composition Root: Wire dependencies into individual sub-services and aggregate into PortfolioApplication."""
    resolved_deps = deps or build_default_portfolio_dependencies(db_path=db_path)

    state_service = PortfolioStateService(
        repo=resolved_deps.repo,
        price_provider=resolved_deps.price_provider,
    )
    trading_service = PortfolioTradingService(
        repo=resolved_deps.repo,
        price_provider=resolved_deps.price_provider,
        journal_provider=resolved_deps.journal_provider,
    )
    cash_flow_service = PortfolioCashFlowService(
        repo=resolved_deps.repo,
        price_provider=resolved_deps.price_provider,
        dividend_provider=resolved_deps.dividend_provider,
        journal_provider=resolved_deps.journal_provider,
    )
    ledger_service = PortfolioLedgerService(
        repo=resolved_deps.repo,
        journal_provider=resolved_deps.journal_provider,
    )
    goal_service = PortfolioGoalService(
        goals_repo=resolved_deps.goals_repo,
    )
    performance_service = PortfolioPerformanceService(
        repo=resolved_deps.repo,
        perf_repo=resolved_deps.perf_repo,
        price_provider=resolved_deps.price_provider,
    )
    watchlist_service = PortfolioWatchlistService(
        watchlist_repo=resolved_deps.watchlist_repo,
    )
    journal_service = PortfolioJournalService(
        journal_provider=resolved_deps.journal_provider,
    )

    return PortfolioApplication(
        state_service=state_service,
        trading_service=trading_service,
        cash_flow_service=cash_flow_service,
        ledger_service=ledger_service,
        goal_service=goal_service,
        performance_service=performance_service,
        watchlist_service=watchlist_service,
        journal_service=journal_service,
    )


def build_legacy_portfolio_service(deps: PortfolioDependencies):
    """Build the compatibility facade from a fully composed dependency set.

    This avoids the facade's legacy default-resolution path in production
    while keeping ``PortfolioService.__init__`` unchanged for external code.
    """
    from tools.portfolio.service import PortfolioService

    service = PortfolioService.__new__(PortfolioService)
    service._bind_dependencies(deps)
    return service
