from abc import ABC, abstractmethod
from typing import ContextManager, Optional, List, Dict, Union
from tools.portfolio.domain.models import PortfolioState, PortfolioMeta
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.mutation import PortfolioMutation


class PortfolioUnitOfWork(ContextManager["PortfolioUnitOfWork"], ABC):
    """Unit of Work interface guaranteeing exclusive lock and crash-consistent commit."""

    # Compatibility implementations may still expose the historical
    # ``commit(state, LedgerChange)`` contract. Concrete repositories that
    # understand the richer mutation envelope opt in with this flag.
    supports_staged_mutations: bool = False

    @abstractmethod
    def load_state(self) -> PortfolioState:
        """Load current authoritative PortfolioState within the transaction lock."""
        ...

    @abstractmethod
    def read_trade_log_locked(self) -> List[Dict]:
        """Read transaction ledger rows under the active transaction lock."""
        ...

    @abstractmethod
    def commit(
        self,
        state: PortfolioState,
        ledger_change: Optional[Union[LedgerChange, PortfolioMutation]] = None,
    ) -> None:
        """Crash-consistent recoverable commit writing Master + Ledger + System Journal + Sidecars."""
        ...

    def commit_mutation(self, state: PortfolioState, mutation: PortfolioMutation) -> bool:
        """Commit a mutation and report whether journal events were staged atomically.

        The default path deliberately targets the legacy ``commit`` shape so
        third-party/in-memory UoWs remain source-compatible. Concrete staged
        repositories opt into the richer envelope via ``supports_staged_mutations``.
        """
        if self.supports_staged_mutations:
            self.commit(state, mutation)
            return True
        self.commit(state, mutation.ledger_change)
        return False

    @abstractmethod
    def rollback(self) -> None:
        """Discard uncommitted modifications and cleanup staging."""
        ...


class PortfolioRepositoryPort(ABC):
    """Port interface for Portfolio master state and ledger storage."""

    @abstractmethod
    def unit_of_work(self, portfolio_id: str = "default") -> PortfolioUnitOfWork:
        """Acquire transaction lock and unit of work context."""
        ...

    @abstractmethod
    def load_state(self, portfolio_id: str = "default") -> PortfolioState:
        """Read-only query for portfolio state."""
        ...

    @abstractmethod
    def read_trade_log(self, portfolio_id: str = "default", symbol: Optional[str] = None) -> List[Dict]:
        """Read transaction ledger rows."""
        ...

    @abstractmethod
    def backup_and_reset_clean_slate(self, portfolio_id: str = "default") -> PortfolioState:
        """Backup existing markdown sidecars and reset portfolio state and trade ledger to a clean slate."""
        ...

    @abstractmethod
    def list_portfolios(self) -> List[PortfolioMeta]:
        """List all portfolios in the system."""
        ...

    @abstractmethod
    def create_portfolio(self, name: str, portfolio_id: Optional[str] = None) -> PortfolioMeta:
        """Create a new portfolio."""
        ...

    @abstractmethod
    def delete_portfolio(self, portfolio_id: str) -> None:
        """Delete an existing portfolio."""
        ...

    @abstractmethod
    def rename_portfolio(self, portfolio_id: str, new_name: str) -> PortfolioMeta:
        """Rename a portfolio."""
        ...

    @abstractmethod
    def portfolio_exists(self, portfolio_id: str) -> bool:
        """Check if a portfolio exists."""
        ...
