"""External transaction source with Markdown-only human-readable projections.

This adapter keeps the existing Markdown repository as a projection writer and
compatibility reader during cutover. State and trade rows are first committed
to ``PortfolioTransactionStore``; Markdown is rebuilt from that source and is
never used as the transaction log after the stream has been bootstrapped.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.models import PortfolioMeta, PortfolioState
from tools.portfolio.domain.mutation import PortfolioMutation
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort, PortfolioUnitOfWork
from tools.portfolio.transaction_store import PortfolioTransactionStore


def _change(value: Optional[Union[LedgerChange, PortfolioMutation]]) -> Optional[LedgerChange]:
    if value is None:
        return None
    if isinstance(value, PortfolioMutation):
        return value.ledger_change
    return value


class TransactionalPortfolioUnitOfWork(PortfolioUnitOfWork):
    supports_staged_mutations = True

    def __init__(self, repo: "TransactionalPortfolioRepository", underlying: PortfolioUnitOfWork, portfolio_id: str) -> None:
        self.repo = repo
        self.underlying = underlying
        self.portfolio_id = portfolio_id
        self.sequence = 0
        self.ledger_rows: list[dict[str, Any]] = []
        self.acquired = False

    def __enter__(self) -> "TransactionalPortfolioUnitOfWork":
        self.underlying.__enter__()
        self.acquired = True
        self.sequence = self.repo.store.checkpoint(self.portfolio_id).sequence
        self.ledger_rows = self.repo._ledger_or_legacy(self.portfolio_id, self.underlying)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        return self.underlying.__exit__(exc_type, exc_val, exc_tb)

    def load_state(self) -> PortfolioState:
        state = self.repo._state_from_store(self.portfolio_id)
        if state is not None:
            return state
        state = self.underlying.load_state()
        self.repo.store.append_state(
            self.portfolio_id,
            state.model_dump(mode="json"),
            ledger_rows=self.ledger_rows,
            expected_sequence=self.sequence,
        )
        self.sequence += 1
        return state

    def read_trade_log_locked(self) -> List[Dict]:
        rows: List[Dict] = []
        for r in self.ledger_rows:
            item_dict = {k.lower(): v for k, v in r.items()}
            item_dict.update({k: v for k, v in r.items()})
            rows.append(item_dict)
        return rows

    def commit(
        self,
        state: PortfolioState,
        ledger_change: Optional[Union[LedgerChange, PortfolioMutation]] = None,
    ) -> None:
        change = _change(ledger_change)
        rows = [dict(row) for row in self.ledger_rows]
        if change is not None and change.kind == "replace_all":
            rows = [dict(row) for row in (change.rows or [])]
        elif change is not None and change.kind == "append" and change.row:
            rows.append(dict(change.row))
        self.repo.store.append_state(
            self.portfolio_id,
            state.model_dump(mode="json"),
            ledger_rows=rows,
            expected_sequence=self.sequence,
        )
        self.sequence += 1
        self.ledger_rows = rows
        # The legacy repository is now a projection writer/recovery adapter.
        # It may fail after the transaction source commits; the next rebuild
        # can safely regenerate the Markdown projection from the store.
        self.underlying.commit(state, ledger_change)

    def rollback(self) -> None:
        self.underlying.rollback()


class TransactionalPortfolioRepository(PortfolioRepositoryPort):
    """Repository whose durable source is the external transaction stream."""

    def __init__(self, underlying_repo: PortfolioRepositoryPort, *, store: Optional[PortfolioTransactionStore] = None) -> None:
        self.underlying_repo = underlying_repo
        self.store = store or PortfolioTransactionStore()

    def _state_from_store(self, portfolio_id: str) -> Optional[PortfolioState]:
        events = self.store.events(portfolio_id)
        if not events:
            return None
        return PortfolioState.model_validate(self.store.replay(portfolio_id))

    def _ledger_or_legacy(self, portfolio_id: str, underlying_uow: PortfolioUnitOfWork) -> list[dict[str, Any]]:
        events = self.store.events(portfolio_id)
        raw_rows = self.store.replay_ledger(portfolio_id) if events else underlying_uow.read_trade_log_locked()
        rows: list[dict[str, Any]] = []
        for r in raw_rows:
            item_dict = {k.lower(): v for k, v in r.items()}
            item_dict.update({k: v for k, v in r.items()})
            rows.append(item_dict)
        return rows

    def unit_of_work(self, portfolio_id: str = "default") -> PortfolioUnitOfWork:
        pid = validate_portfolio_id(portfolio_id)
        return TransactionalPortfolioUnitOfWork(self, self.underlying_repo.unit_of_work(pid), pid)

    def load_state(self, portfolio_id: str = "default") -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        state = self._state_from_store(pid)
        if state is not None:
            return state
        with self.unit_of_work(pid) as uow:
            return uow.load_state()

    def read_trade_log(self, portfolio_id: str = "default", symbol: Optional[str] = None) -> List[Dict]:
        pid = validate_portfolio_id(portfolio_id)
        raw_rows = self.store.replay_ledger(pid) if self.store.events(pid) else self.underlying_repo.read_trade_log(pid)
        rows: List[Dict] = []
        for r in raw_rows:
            item_dict = {k.lower(): v for k, v in r.items()}
            item_dict.update({k: v for k, v in r.items()})
            if symbol:
                wanted = symbol.strip().upper()
                if str(item_dict.get("symbol") or item_dict.get("Symbol") or "").upper() != wanted:
                    continue
            rows.append(item_dict)
        return rows

    def backup_and_reset_clean_slate(self, portfolio_id: str = "default") -> PortfolioState:
        state = self.underlying_repo.backup_and_reset_clean_slate(portfolio_id)
        pid = validate_portfolio_id(portfolio_id)
        self.store.append_state(pid, state.model_dump(mode="json"), ledger_rows=[], expected_sequence=self.store.checkpoint(pid).sequence)
        return state

    def list_portfolios(self) -> List[PortfolioMeta]:
        return self.underlying_repo.list_portfolios()

    def create_portfolio(self, name: str, portfolio_id: Optional[str] = None) -> PortfolioMeta:
        meta = self.underlying_repo.create_portfolio(name, portfolio_id=portfolio_id)
        self.load_state(meta.id)
        return meta

    def delete_portfolio(self, portfolio_id: str) -> None:
        self.underlying_repo.delete_portfolio(portfolio_id)
        # Keep a durable deletion marker; the Markdown directory is only a
        # projection and may be recreated during recovery.
        self.store.append_event(
            validate_portfolio_id(portfolio_id),
            "portfolio_deleted",
            {"portfolio_id": validate_portfolio_id(portfolio_id)},
            expected_sequence=self.store.checkpoint(portfolio_id).sequence,
        )

    def rename_portfolio(self, portfolio_id: str, new_name: str) -> PortfolioMeta:
        meta = self.underlying_repo.rename_portfolio(portfolio_id, new_name)
        self.load_state(meta.id)
        return meta

    def portfolio_exists(self, portfolio_id: str) -> bool:
        return self.underlying_repo.portfolio_exists(portfolio_id)
