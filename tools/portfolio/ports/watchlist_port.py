from abc import ABC, abstractmethod
from tools.portfolio.domain.models import WatchlistState


class WatchlistRepositoryPort(ABC):
    """Port interface for Watchlist persistence."""

    @abstractmethod
    def load_watchlist(self, portfolio_id: str = "default") -> WatchlistState:
        """Load watchlist state for a portfolio."""
        ...

    @abstractmethod
    def save_watchlist(self, state: WatchlistState, portfolio_id: str = "default") -> None:
        """Atomically persist watchlist state and sidecars."""
        ...
