import hashlib
import json
import pytest
from pathlib import Path

from tools.portfolio.domain.constants import CASH_THB_SYMBOL
from tools.portfolio.domain.models import PortfolioState, Holding, Summary, _now_iso
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.errors import RecoveryConflictError
from tools.portfolio.adapters.markdown.repository_adapter import (
    MarkdownVaultRepositoryAdapter,
    _compute_sha256,
)
from tools.portfolio.adapters.markdown.paths import (
    get_portfolio_dir,
    get_portfolio_filepath,
    get_trades_log_filepath,
    get_pending_manifest_path,
)


@pytest.fixture
def temp_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    from tools.portfolio.adapters.markdown import paths
    paths.VAULT_PATH = tmp_path
    paths.PORTFOLIOS_DIR = tmp_path / "20_Portfolio_Management/Current_Holdings/Portfolios"
    return MarkdownVaultRepositoryAdapter()


def test_recovery_case_1_clean_rollback(temp_repo):
    """Case 1: Crash before master was replaced -> Rollback & cleanup."""
    pid = "test_case1"
    temp_repo.create_portfolio("Test Case 1", portfolio_id=pid)

    master_path = get_portfolio_filepath(pid)
    pre_master_sha = _compute_sha256(master_path)
    ledger_path = get_trades_log_filepath(pid)
    pre_ledger_sha = _compute_sha256(ledger_path)

    # Simulate fake pending manifest before master replace
    manifest_path = get_pending_manifest_path(pid)
    staged_master = master_path.parent / ".master_fake.staged"
    staged_master.write_text("fake staged", encoding="utf-8")

    manifest_data = {
        "schema_version": 1,
        "tx_id": "fake_tx_1",
        "portfolio_id": pid,
        "timestamp": 1234567.0,
        "ledger_kind": "append",
        "pre_master_sha256": pre_master_sha,
        "staged_master_file": str(staged_master),
        "staged_master_sha256": "different_hash",
        "pre_ledger_sha256": pre_ledger_sha,
        "staged_ledger_file": None,
        "staged_ledger_sha256": None,
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    # Opening UoW should trigger Case 1 rollback
    with temp_repo.unit_of_work(pid) as uow:
        recovered_state = uow.load_state()

    assert not manifest_path.exists()
    assert not staged_master.exists()


def test_recovery_case_2_complete_unchanged(temp_repo):
    """Case 2: Master committed with unchanged ledger -> Clean complete."""
    pid = "test_case2"
    temp_repo.create_portfolio("Test Case 2", portfolio_id=pid)

    master_path = get_portfolio_filepath(pid)
    master_sha = _compute_sha256(master_path)
    ledger_path = get_trades_log_filepath(pid)
    ledger_sha = _compute_sha256(ledger_path)

    manifest_path = get_pending_manifest_path(pid)
    manifest_data = {
        "schema_version": 1,
        "tx_id": "fake_tx_2",
        "portfolio_id": pid,
        "timestamp": 1234567.0,
        "ledger_kind": "unchanged",
        "pre_master_sha256": "old_pre_sha",
        "staged_master_file": str(master_path.parent / ".staged_none"),
        "staged_master_sha256": master_sha,
        "pre_ledger_sha256": ledger_sha,
        "staged_ledger_file": None,
        "staged_ledger_sha256": None,
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    # Opening UoW should clean manifest without conflict
    with temp_repo.unit_of_work(pid) as uow:
        st = uow.load_state()

    assert not manifest_path.exists()


def test_recovery_case_3_roll_forward_ledger(temp_repo):
    """Case 3: Master committed, Ledger at pre-state -> Roll-forward ledger."""
    pid = "test_case3"
    temp_repo.create_portfolio("Test Case 3", portfolio_id=pid)

    master_path = get_portfolio_filepath(pid)
    master_sha = _compute_sha256(master_path)
    ledger_path = get_trades_log_filepath(pid)
    pre_ledger_sha = _compute_sha256(ledger_path)

    # Create a staged ledger with new trade
    staged_ledger = ledger_path.parent / ".staged_ledger.csv"
    staged_ledger.write_text("Transaction_ID,Timestamp,Symbol\ntx_100,2026-08-23,AAPL\n", encoding="utf-8")
    staged_ledger_sha = _compute_sha256(staged_ledger)

    manifest_path = get_pending_manifest_path(pid)
    manifest_data = {
        "schema_version": 1,
        "tx_id": "tx_100",
        "portfolio_id": pid,
        "timestamp": 1234567.0,
        "ledger_kind": "append",
        "pre_master_sha256": "old_pre_sha",
        "staged_master_file": str(master_path.parent / ".staged_none"),
        "staged_master_sha256": master_sha,
        "pre_ledger_sha256": pre_ledger_sha,
        "staged_ledger_file": str(staged_ledger),
        "staged_ledger_sha256": staged_ledger_sha,
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    # Opening UoW should apply staged ledger and clean manifest
    with temp_repo.unit_of_work(pid) as uow:
        pass

    assert not manifest_path.exists()
    assert not staged_ledger.exists()
    assert "tx_100" in ledger_path.read_text(encoding="utf-8")


def test_recovery_case_4_conflict_on_unknown_hash(temp_repo):
    """Case 4: External modification -> RecoveryConflictError & preserve manifest."""
    pid = "test_case4"
    temp_repo.create_portfolio("Test Case 4", portfolio_id=pid)

    master_path = get_portfolio_filepath(pid)
    ledger_path = get_trades_log_filepath(pid)

    manifest_path = get_pending_manifest_path(pid)
    manifest_data = {
        "schema_version": 1,
        "tx_id": "tx_conflict",
        "portfolio_id": pid,
        "timestamp": 1234567.0,
        "ledger_kind": "append",
        "pre_master_sha256": "completely_unrelated_1",
        "staged_master_file": str(master_path.parent / ".staged_none"),
        "staged_master_sha256": "completely_unrelated_2",
        "pre_ledger_sha256": "completely_unrelated_3",
        "staged_ledger_file": None,
        "staged_ledger_sha256": None,
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    # Opening UoW must raise RecoveryConflictError and preserve manifest
    with pytest.raises(RecoveryConflictError):
        with temp_repo.unit_of_work(pid) as uow:
            pass

    assert manifest_path.exists()
