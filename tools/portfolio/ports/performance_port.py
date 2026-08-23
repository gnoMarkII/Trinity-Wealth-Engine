from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class PerformanceRepositoryPort(ABC):
    """Port interface for Portfolio Performance time-series logging."""

    @abstractmethod
    def upsert_snapshot(self, portfolio_id: str, row: Dict) -> None:
        """Idempotently upsert daily performance snapshot (replaces existing row if same Date)."""
        ...

    @abstractmethod
    def read_history(self, portfolio_id: str = "default", days: Optional[int] = None) -> List[Dict]:
        """Read performance snapshot history rows."""
        ...
