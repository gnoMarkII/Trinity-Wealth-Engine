from __future__ import annotations

import sqlite3
from dataclasses import asdict
from threading import RLock

import pytest

from tools.portfolio.domain.models import Holding, PortfolioState
from tools.portfolio.services.cash_flow_service import PortfolioCashFlowService
from tools.portfolio.transaction_store import PortfolioTransactionStore


class _PriceProvider:
    def __init__(self, source="historical"):
        self.source = source

    def fetch_fx_rate(self, date_str=None, fallback_rate=None):
        return 35.0, self.source


class _UnitOfWork:
    def __init__(self, repo):
        self.repo = repo

    def __enter__(self):
        self.repo.lock.acquire()
        return self

    def __exit__(self, *_):
        self.repo.lock.release()

    def load_state(self):
        return self.repo.state.model_copy(deep=True)

    def commit_mutation(self, state, mutation):
        related = list(mutation.durable_events)
        related.extend(
            {"event_type": "system_journal_event", "payload": asdict(event)}
            for event in mutation.system_journal_events
        )
        self.repo.sequence = self.repo.store.append_state(
            self.repo.portfolio_id,
            state.model_dump(mode="json"),
            expected_sequence=self.repo.sequence,
            related_events=related,
        )
        self.repo.state = state.model_copy(deep=True)
        return True


class _Repo:
    def __init__(self, runtime_root, portfolio_id="test"):
        self.portfolio_id = portfolio_id
        self.store = PortfolioTransactionStore(runtime_root=runtime_root)
        self.lock = RLock()
        self.state = PortfolioState(
            name="Test",
            last_updated="2026-11-07T00:00:00Z",
            fx_rates={"USDTHB": 35.0},
            holdings=[Holding(symbol="CASH_THB", asset_type="Cash", units=100.0)],
        )
        self.sequence = self.store.append_state(
            portfolio_id, self.state.model_dump(mode="json")
        )

    def unit_of_work(self, portfolio_id):
        assert portfolio_id == self.portfolio_id
        return _UnitOfWork(self)

    def get_cash_flow_event(self, portfolio_id, idempotency_key):
        assert portfolio_id == self.portfolio_id
        return self.store.get_cash_flow_event(portfolio_id, idempotency_key)

    def load_state(self, portfolio_id):
        assert portfolio_id == self.portfolio_id
        return self.state.model_copy(deep=True)


def test_cash_flow_is_atomic_signed_idempotent_and_immutable(tmp_path):
    repo = _Repo(tmp_path / "runtime")
    service = PortfolioCashFlowService(repo, _PriceProvider())

    state = service.structured_manage_cash_flow(
        amount="25.50",
        action="deposit",
        currency="THB",
        portfolio_id="test",
        idempotency_key="deposit-1",
    )
    assert next(h for h in state.holdings if h.symbol == "CASH_THB").units == 125.5
    flow = repo.get_cash_flow_event("test", "deposit-1")
    assert flow["amount_native_decimal"] == "25.50"
    assert flow["amount_thb_decimal"] == "25.50"
    assert flow["flow_coverage_status"] == "COMPLETE"
    assert flow["flow_kind"] == "EXTERNAL_CONTRIBUTION"

    duplicate = service.structured_manage_cash_flow(
        amount="25.50",
        action="deposit",
        currency="THB",
        portfolio_id="test",
        idempotency_key="deposit-1",
    )
    assert next(h for h in duplicate.holdings if h.symbol == "CASH_THB").units == 125.5
    assert len(repo.store.list_cash_flow_events("test")) == 1

    with pytest.raises(ValueError, match="different cash-flow request"):
        service.structured_manage_cash_flow(
            amount="26.00",
            action="deposit",
            currency="THB",
            portfolio_id="test",
            idempotency_key="deposit-1",
        )

    with repo.store._connect() as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE portfolio_cash_flow_events SET action='withdraw' WHERE idempotency_key='deposit-1'"
            )


def test_usd_flow_without_trusted_fx_is_retained_but_excluded_from_thb_pnl(tmp_path):
    repo = _Repo(tmp_path / "runtime", portfolio_id="usd-test")
    service = PortfolioCashFlowService(repo, _PriceProvider(source="fallback"))
    service.structured_manage_cash_flow(
        amount="10.00",
        action="deposit",
        currency="USD",
        portfolio_id="usd-test",
        idempotency_key="usd-deposit-1",
    )
    flow = repo.get_cash_flow_event("usd-test", "usd-deposit-1")
    assert flow["amount_native_decimal"] == "10.00"
    assert flow["amount_thb_decimal"] is None
    assert flow["fx_source_id"] is None
    assert flow["flow_coverage_status"] == "MISSING_FX"
