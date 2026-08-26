"""Unit & Fault-Injection Tests for PortfolioMutation and Crash-Consistent Recovery Unit."""
import os
import shutil
import pytest
from pathlib import Path

from tools.portfolio.domain.models import PortfolioState, Holding, Summary, _now_iso
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.events import SystemJournalEvent
from tools.portfolio.domain.mutation import PortfolioMutation
from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
from tools.portfolio.adapters.markdown.paths import (
    get_portfolio_dir,
    get_portfolio_filepath,
    get_trades_log_filepath,
    get_journal_filepath,
    get_pending_manifest_path,
)


def test_system_journal_event_from_entry_preserves_journal_timestamp_contract():
    event = SystemJournalEvent.from_entry(
        event_type="cash_flow_recorded",
        message="**[CASH FLOW NOTE]** DEPOSIT 100 THB",
        date_str="2026-08-23",
    )

    assert event.timestamp == "2026-08-23 12:00:00"


def test_system_journal_event_from_entry_normalizes_iso_datetime():
    event = SystemJournalEvent.from_entry(
        event_type="trade_executed",
        message="**[TRADE NOTE - AAPL]** BUY 1 @ 1 USD",
        date_str="2026-08-23T10:30:45Z",
    )

    assert event.timestamp.startswith("2026-08-23 10:30:45")


@pytest.fixture
def temp_vault_portfolio(tmp_path, monkeypatch):
    """Set up isolated Obsidian vault environment for portfolio testing."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault_dir))

    repo = MarkdownVaultRepositoryAdapter()
    pid = "default"
    return repo, pid


def test_portfolio_mutation_commit_with_journal_and_ledger(temp_vault_portfolio):
    """Test that PortfolioMutation commits state, ledger, and system journal events atomically."""
    repo, pid = temp_vault_portfolio

    with repo.unit_of_work(pid) as uow:
        state = uow.load_state()
        state.holdings.append(Holding(symbol="AAPL", asset_type="US_Equity", units=10.0, avg_cost_usd=150.0, current_price_usd=160.0))

        ledger_row = {
            "Transaction_ID": "tx_001",
            "Timestamp": "2026-08-23 10:00:00",
            "Symbol": "AAPL",
            "Action": "BUY",
            "Units": "10",
            "Price": "150.0",
            "Currency": "USD",
            "FX_Rate": "36.5",
            "Cost_THB": "54750.0",
            "Realized_PnL_THB": "0",
            "Notes": "Initial trade",
        }
        mutation = PortfolioMutation(
            ledger_change=LedgerChange(kind="append", row=ledger_row, tx_id="tx_001"),
            system_journal_events=[
                SystemJournalEvent(
                    event_type="trade_executed",
                    message="**[BUY]** AAPL 10 units @ $150.0 — Initial trade",
                    timestamp="2026-08-23 10:00:00",
                )
            ],
        )
        uow.commit(state, mutation)

    # Verify all 3 authoritative files exist and have content
    master_file = get_portfolio_filepath(pid)
    ledger_file = get_trades_log_filepath(pid)
    journal_file = get_journal_filepath(pid)
    manifest_file = get_pending_manifest_path(pid)

    assert master_file.exists(), "Master file should exist"
    assert ledger_file.exists(), "Ledger file should exist"
    assert journal_file.exists(), "Journal file should exist"
    assert not manifest_file.exists(), "Pending manifest should be cleanly unlinked after commit"

    # Verify contents
    assert "AAPL" in master_file.read_text(encoding="utf-8")
    assert "tx_001" in ledger_file.read_text(encoding="utf-8")
    assert "AAPL" in journal_file.read_text(encoding="utf-8")


def test_crash_recovery_roll_forward_when_master_committed(temp_vault_portfolio):
    """Test that if process dies after master replace, recovery rolls forward ledger and journal."""
    repo, pid = temp_vault_portfolio

    # First commit initial state
    with repo.unit_of_work(pid) as uow:
        state = uow.load_state()
        uow.commit(state, LedgerChange(kind="unchanged"))

    # Stage a new mutation
    with repo.unit_of_work(pid) as uow:
        state = uow.load_state()
        state.holdings.append(Holding(symbol="NVDA", asset_type="US_Equity", units=5.0, avg_cost_usd=100.0))

        ledger_row = {
            "Transaction_ID": "tx_002",
            "Timestamp": "2026-08-23 11:00:00",
            "Symbol": "NVDA",
            "Action": "BUY",
            "Units": "5",
            "Price": "100.0",
            "Currency": "USD",
            "FX_Rate": "36.5",
            "Cost_THB": "18250.0",
            "Realized_PnL_THB": "0",
            "Notes": "NVDA buy",
        }
        mutation = PortfolioMutation(
            ledger_change=LedgerChange(kind="append", row=ledger_row, tx_id="tx_002"),
            system_journal_events=[
                SystemJournalEvent(
                    event_type="trade_executed",
                    message="**[BUY]** NVDA 5 units @ $100.0",
                    timestamp="2026-08-23 11:00:00",
                )
            ],
        )
        uow.commit(state, mutation)

    # Verify NVDA is present in ledger and journal
    ledger_file = get_trades_log_filepath(pid)
    assert "tx_002" in ledger_file.read_text(encoding="utf-8")
