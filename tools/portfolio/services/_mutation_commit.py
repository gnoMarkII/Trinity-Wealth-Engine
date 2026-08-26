"""Compatibility-aware commit helper for portfolio mutation envelopes."""

from typing import Optional

from tools.portfolio.domain.models import PortfolioState
from tools.portfolio.domain.mutation import PortfolioMutation
from tools.portfolio.ports.journal_port import TradeJournalPort
from tools.portfolio.ports.repository_port import PortfolioUnitOfWork


def commit_mutation(
    uow: PortfolioUnitOfWork,
    state: PortfolioState,
    mutation: PortfolioMutation,
    *,
    journal_provider: Optional[TradeJournalPort] = None,
    portfolio_id: str = "default",
) -> None:
    """Commit through the UoW capability boundary.

    Staged repositories persist state, ledger and system journal events as one
    recovery unit. Legacy/in-memory UoWs only understand ``LedgerChange``; for
    those implementations the state/ledger commit is retained and journal
    delivery follows the historical provider path.
    """
    # Keep the helper tolerant of duck-typed legacy UoWs that predate the
    # optional hook and therefore do not inherit PortfolioUnitOfWork.
    commit_hook = getattr(uow, "commit_mutation", None)
    if callable(commit_hook):
        staged = bool(commit_hook(state, mutation))
    else:
        uow.commit(state, mutation.ledger_change)
        staged = False
    if staged or journal_provider is None:
        return
    for event in mutation.system_journal_events:
        journal_provider.append_system_entry(
            event.message,
            date_str=event.timestamp,
            portfolio_id=portfolio_id,
        )
