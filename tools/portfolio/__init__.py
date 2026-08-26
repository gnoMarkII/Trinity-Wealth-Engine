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
                from .bootstrap import build_default_portfolio_dependencies
                from .adapters.legacy_price_compatibility import LegacyPriceCompatibilityAdapter

                deps = build_default_portfolio_dependencies()
                # Keep the legacy agent-tool patch points functional at the
                # composition boundary while services consume only ports.
                compat_price = LegacyPriceCompatibilityAdapter(deps.price_provider)
                _service_instance = PortfolioService(
                    repo=deps.repo,
                    watchlist_repo=deps.watchlist_repo,
                    goals_repo=deps.goals_repo,
                    perf_repo=deps.perf_repo,
                    journal_provider=deps.journal_provider,
                    price_provider=compat_price,
                )
                if deps.dividend_provider is not None:
                    _service_instance._cash_flow_service.dividend_provider = deps.dividend_provider
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
