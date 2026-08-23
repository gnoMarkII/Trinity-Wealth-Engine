from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class TradeJournalPort(ABC):
    """Port interface for Trade Journal entries."""

    @abstractmethod
    def append_journal(self, entry: str, portfolio_id: str = "default") -> List[Dict]:
        """Append an entry to the trading journal and return updated entries."""
        ...

    @abstractmethod
    def read_journal(
        self, days: Optional[int] = 365, keyword: Optional[str] = None, limit: int = 100, portfolio_id: str = "default"
    ) -> List[Dict]:
        """Read journal entries matching query parameters."""
        ...
