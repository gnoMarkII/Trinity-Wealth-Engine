from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from tools.portfolio.domain.models import GoalsState


class GoalsRepositoryPort(ABC):
    """Port interface for Goals persistence."""

    @abstractmethod
    def load_goals(self, portfolio_id: Optional[str] = None) -> GoalsState:
        """Load goals state (optionally filtered by portfolio_id)."""
        ...

    @abstractmethod
    def save_goals(self, state: GoalsState) -> None:
        """Atomically persist goals state and items."""
        ...
