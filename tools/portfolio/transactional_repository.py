"""External transaction source with Markdown-only human-readable projections.

This adapter keeps the existing Markdown repository as a projection writer and
compatibility reader during cutover. State and trade rows are first committed
to ``PortfolioTransactionStore``; Markdown is rebuilt from that source and is
never used as the transaction log after the stream has been bootstrapped.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.models import PortfolioMeta, PortfolioState
from tools.portfolio.domain.mutation import PortfolioMutation
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort, PortfolioUnitOfWork
from tools.portfolio.transaction_store import PortfolioTransactionStore


def _is_portfolio_mutation(value: Any) -> bool:
    """Recognize mutation envelopes across module reloads used by isolated callers."""
    return isinstance(value, PortfolioMutation) or (
        type(value).__name__ == "PortfolioMutation" and hasattr(value, "ledger_change")
    )


def _change(value: Optional[Union[LedgerChange, PortfolioMutation]]) -> Optional[LedgerChange]:
    if value is None:
        return None
    if _is_portfolio_mutation(value):
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
        command_receipt: Optional[Dict[str, Any]] = None,
    ) -> None:
        change = _change(ledger_change)
        rows = [dict(row) for row in self.ledger_rows]
        if change is not None and change.kind == "replace_all":
            rows = [dict(row) for row in (change.rows or [])]
        elif change is not None and change.kind == "append" and change.row:
            rows.append(dict(change.row))
        related_events: list[dict[str, Any]] = []
        if _is_portfolio_mutation(ledger_change):
            related_events.extend(ledger_change.durable_events)
            related_events.extend(
                {"event_type": "system_journal_event", "payload": asdict(event)}
                for event in ledger_change.system_journal_events
            )
        self.sequence = self.repo.store.append_state(
            self.portfolio_id,
            state.model_dump(mode="json"),
            ledger_rows=rows,
            expected_sequence=self.sequence,
            command_receipt=command_receipt,
            related_events=related_events,
        )
        self.ledger_rows = rows
        # The legacy repository is now a projection writer/recovery adapter.
        # It may fail after the transaction source commits; the next rebuild
        # can safely regenerate the Markdown projection from the store.
        # Preserve the full mutation envelope so the Markdown projection
        # atomically includes its journal events as well as state and ledger.
        self.underlying.commit(state, ledger_change)

    def commit_mutation(self, state: PortfolioState, mutation: PortfolioMutation) -> bool:
        self.commit(state, mutation)
        return True

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
        last_delete = max((i for i, event in enumerate(events) if event["event_type"] == "portfolio_deleted"), default=-1)
        last_snapshot = max((i for i, event in enumerate(events) if event["event_type"] == "state_snapshot"), default=-1)
        if last_delete > last_snapshot:
            return None
        state_dict = self.store.replay(portfolio_id)
        state = PortfolioState.model_validate(state_dict)
        if not state.name:
            try:
                underlying_state = self.underlying_repo.load_state(portfolio_id)
                if underlying_state and underlying_state.name:
                    state.name = underlying_state.name
            except Exception:
                pass
        return state

    def _ledger_or_legacy(self, portfolio_id: str, underlying_uow: PortfolioUnitOfWork) -> list[dict[str, Any]]:
        events = self.store.events(portfolio_id)
        raw_rows = self.store.replay_ledger(portfolio_id) if events else underlying_uow.read_trade_log_locked()
        rows: list[dict[str, Any]] = []
        for r in raw_rows:
            item_dict = {k.lower(): v for k, v in r.items()}
            item_dict.update({k: v for k, v in r.items()})
            rows.append(item_dict)
        return rows

    def get_cash_flow_event(self, portfolio_id: str, idempotency_key: str) -> Optional[dict[str, Any]]:
        return self.store.get_cash_flow_event(validate_portfolio_id(portfolio_id), idempotency_key)

    def list_cash_flow_events(
        self,
        portfolio_id: str,
        *,
        after_sequence: Optional[int] = None,
        through_sequence: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        return self.store.list_cash_flow_events(
            validate_portfolio_id(portfolio_id),
            after_sequence=after_sequence,
            through_sequence=through_sequence,
        )

    def performance_provenance(self, portfolio_id: str) -> dict[str, Any]:
        checkpoint = self.store.checkpoint(validate_portfolio_id(portfolio_id))
        return {
            "source_state_sequence": checkpoint.sequence,
            "source_state_sha256": checkpoint.state_hash,
            "event_log_sha256": checkpoint.event_log_hash,
        }

    def cash_flow_coverage_between(
        self, portfolio_id: str, *, after_sequence: int, through_sequence: int
    ) -> bool:
        return self.store.cash_flow_coverage_between(
            validate_portfolio_id(portfolio_id),
            after_sequence=after_sequence,
            through_sequence=through_sequence,
        )

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
        pid = validate_portfolio_id(portfolio_id)
        state = self.underlying_repo.backup_and_reset_clean_slate(pid)
        # A full reset invalidates every PVD scan staged against the previous epoch.
        state.pvd_import_epoch = uuid.uuid4().hex
        with self.underlying_repo.unit_of_work(pid) as uow:
            projected = uow.load_state()
            projected.pvd_import_epoch = state.pvd_import_epoch
            uow.commit(projected, LedgerChange(kind="unchanged"))
        state = projected
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
        pid = validate_portfolio_id(portfolio_id)
        meta = self.underlying_repo.rename_portfolio(pid, new_name)
        projected = self.underlying_repo.load_state(pid)
        projected.name = meta.name
        self.store.append_state(
            pid,
            projected.model_dump(mode="json"),
            ledger_rows=self.store.replay_ledger(pid) if self.store.events(pid) else self.underlying_repo.read_trade_log(pid),
            expected_sequence=self.store.checkpoint(pid).sequence,
        )
        return meta

    def portfolio_exists(self, portfolio_id: str) -> bool:
        pid = validate_portfolio_id(portfolio_id)
        if pid == "default":
            return True
        events = self.store.events(pid)
        if events:
            last_delete = max((i for i, event in enumerate(events) if event["event_type"] == "portfolio_deleted"), default=-1)
            last_snapshot = max((i for i, event in enumerate(events) if event["event_type"] == "state_snapshot"), default=-1)
            if last_delete > last_snapshot:
                return False
            if last_snapshot >= 0:
                return True
        return self.underlying_repo.portfolio_exists(pid)
