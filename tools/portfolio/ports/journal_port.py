from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class TradeJournalPort(ABC):
    """Port interface for Trade Journal entries."""

    @abstractmethod
    def append_journal(
        self, entry: str, date_str: Optional[str] = None, portfolio_id: str = "default"
    ) -> List[Dict]:
        """Append an entry to the trading journal and return updated entries."""
        ...

    @abstractmethod
    def append_system_entry(
        self, entry: str, date_str: Optional[str] = None, portfolio_id: str = "default"
    ) -> None:
        """Append an automated system transaction entry to the trading journal."""
        ...

    @abstractmethod
    def read_journal(
        self, days: Optional[int] = 365, keyword: Optional[str] = None, limit: int = 100, portfolio_id: str = "default"
    ) -> List[Dict]:
        """Read journal entries matching query parameters."""
        ...
