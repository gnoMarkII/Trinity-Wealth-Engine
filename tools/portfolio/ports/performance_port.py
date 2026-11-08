from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class PerformanceRepositoryPort(ABC):
    """Port interface for immutable, versioned portfolio performance snapshots."""

    @abstractmethod
    def upsert_snapshot(self, portfolio_id: str, row: Dict) -> None:
        """Append a snapshot revision; do not overwrite a prior revision."""
        ...

    @abstractmethod
    def read_history(
        self, portfolio_id: str = "default", days: Optional[int] = None,
        as_of_utc: Optional[str] = None,
    ) -> List[Dict]:
        """Read versioned snapshots available no later than the requested cutoff."""
        ...
