"""PortfolioMutation encapsulates atomic state mutation, ledger change, and system journal events."""
from dataclasses import dataclass, field
from typing import Optional, List, Union

from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.events import SystemJournalEvent


@dataclass
class PortfolioMutation:
    """Encapsulates all side-effecting mutations executed within a single Unit of Work."""
    ledger_change: Optional[LedgerChange] = None
    system_journal_events: List[SystemJournalEvent] = field(default_factory=list)

    @classmethod
    def from_change(cls, change: Optional[Union[LedgerChange, "PortfolioMutation"]] = None) -> "PortfolioMutation":
        if change is None:
            return cls(ledger_change=LedgerChange(kind="unchanged"))
        if type(change).__name__ == "PortfolioMutation" or isinstance(change, PortfolioMutation):
            return change
        if type(change).__name__ == "LedgerChange" or isinstance(change, LedgerChange) or hasattr(change, "kind"):
            return cls(ledger_change=change)
        raise TypeError(f"Expected LedgerChange or PortfolioMutation, got {type(change)}")
