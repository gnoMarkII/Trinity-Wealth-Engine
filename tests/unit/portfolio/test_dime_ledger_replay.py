import csv
import tempfile
from pathlib import Path
import pytest

from tools.portfolio.domain.models import PortfolioState, Holding
from tools.portfolio.adapters.markdown.repository_adapter import (
    MarkdownVaultRepositoryAdapter,
    _read_and_migrate_trade_log_locked,
)
from tools.portfolio.adapters.markdown.paths import _TRADES_LOG_HEADER, get_trades_log_filepath
from tools.portfolio.services.ledger_service import PortfolioLedgerService


@pytest.fixture
def temp_portfolio_repo(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
        repo = MarkdownVaultRepositoryAdapter()
        # Initialize default portfolio
        state = PortfolioState(
            last_updated="2026-09-01T00:00:00",
            holdings=[
                Holding(symbol="CASH_USD", asset_type="Cash", units=10000.0, market_value_thb=365000.0),
                Holding(symbol="CASH_THB", asset_type="Cash", units=100000.0, market_value_thb=100000.0),
            ],
            fx_rates={"USDTHB": 36.5},
        )
        with repo.unit_of_work("default") as uow:
            uow.commit(state)
        yield repo


def test_legacy_11_column_migration_multi_currency_safe(tmp_path):
    # Create legacy 11-column CSV with USD trade and THB Cost_THB
    csv_file = tmp_path / "Trades_Log.csv"
    legacy_header = [
        "Transaction_ID", "Timestamp", "Symbol", "Action", "Units", "Price",
        "Currency", "FX_Rate", "Cost_THB", "Realized_PnL_THB", "Notes"
    ]
    legacy_rows = [
        ["tx_usd_1", "2026-08-01 12:00:00", "AAPL", "BUY", "10", "150.00", "USD", "36.00", "54000.00", "", "Legacy US Trade"],
        ["tx_thb_1", "2026-08-01 13:00:00", "PTT", "BUY", "100", "35.00", "THB", "", "3500.00", "", "Legacy TH Trade"],
    ]
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(legacy_header)
        writer.writerows(legacy_rows)

    # Run migration
    migrated = _read_and_migrate_trade_log_locked(csv_file)
    assert len(migrated) == 2

    # Check AAPL (USD): Gross_Amount must be 1500.00 USD (Units * Price in trade ccy), NOT 54000.00 THB!
    aapl_row = migrated[0]
    assert aapl_row["Gross_Amount"] == "1500.00"
    assert aapl_row["Net_Amount"] == "1500.00"
    assert aapl_row["Fee_Currency"] == "USD"
    assert aapl_row["Cost_THB"] == "54000.00"
    assert aapl_row["Cash_Adjusted"] == "YES"
    assert aapl_row["Source"] == "MANUAL"
    assert aapl_row["Related_Transaction_ID"] == ""

    # Check PTT (THB)
    ptt_row = migrated[1]
    assert ptt_row["Gross_Amount"] == "3500.00"
    assert ptt_row["Net_Amount"] == "3500.00"
    assert ptt_row["Fee_Currency"] == "THB"
    assert ptt_row["Cash_Adjusted"] == "YES"
    assert ptt_row["Source"] == "MANUAL"

    # Verify rewritten file matches _TRADES_LOG_HEADER
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert len(header) == len(_TRADES_LOG_HEADER)
        assert header == _TRADES_LOG_HEADER



def test_void_transaction_strict_mirror_yes_even_if_caller_requests_false(temp_portfolio_repo):
    ledger_svc = PortfolioLedgerService(repo=temp_portfolio_repo)
    # Seed a BUY trade with Cash_Adjusted="YES" and Net_Amount=1500.00
    tx_id = "tx_seed_1"
    initial_cash = 10000.0 - 1500.0  # Cash after buy
    with temp_portfolio_repo.unit_of_work("default") as uow:
        state = uow.load_state()
        cash = next(h for h in state.holdings if h.symbol == "CASH_USD")
        cash.units = initial_cash
        state.holdings.append(Holding(symbol="AAPL", asset_type="Stock", units=10.0, avg_cost_usd=150.0))
        trade_row = {
            "Transaction_ID": tx_id,
            "Timestamp": "2026-09-01 10:00:00",
            "Symbol": "AAPL",
            "Action": "BUY",
            "Units": "10",
            "Price": "150.00",
            "Currency": "USD",
            "Gross_Amount": "1500.00",
            "Net_Amount": "1500.00",
            "Cash_Adjusted": "YES",
            "Source": "DIME",
            "Related_Transaction_ID": "",
        }
        # Commit row
        from tools.portfolio.domain.ledger_change import LedgerChange
        uow.commit(state, LedgerChange(kind="replace_all", rows=[trade_row]))

    # Call void_transaction attempting adjust_cash=False
    # Strict mirror rule: original Cash_Adjusted is "YES" -> cash MUST be reversed (+1500.00)!
    new_state = ledger_svc.void_transaction(tx_id, "default", adjust_cash=False)
    cash_after = next(h for h in new_state.holdings if h.symbol == "CASH_USD").units
    assert cash_after == 10000.0  # Successfully reversed!

    # Holding units should drop to 0 (or removed)
    aapl = next((h for h in new_state.holdings if h.symbol == "AAPL"), None)
    assert aapl is None or aapl.units == 0.0

    # Check ledger rows
    with temp_portfolio_repo.unit_of_work("default") as uow:
        rows = uow.read_trade_log_locked()
        assert len(rows) == 2
        orig = rows[0]
        reversal = rows[1]
        assert orig["Transaction_ID"] == tx_id
        assert reversal["Action"] == "VOID_BUY"
        assert reversal["Related_Transaction_ID"] == tx_id
        assert reversal["Cash_Adjusted"] == "YES"


def test_void_transaction_strict_mirror_no_even_if_caller_requests_true(temp_portfolio_repo):
    ledger_svc = PortfolioLedgerService(repo=temp_portfolio_repo)
    # Seed a BUY trade with Cash_Adjusted="NO"
    tx_id = "tx_seed_2"
    initial_cash = 10000.0
    with temp_portfolio_repo.unit_of_work("default") as uow:
        state = uow.load_state()
        state.holdings.append(Holding(symbol="NVDA", asset_type="Stock", units=5.0, avg_cost_usd=100.0))
        trade_row = {
            "Transaction_ID": tx_id,
            "Timestamp": "2026-09-01 10:00:00",
            "Symbol": "NVDA",
            "Action": "BUY",
            "Units": "5",
            "Price": "100.00",
            "Currency": "USD",
            "Gross_Amount": "500.00",
            "Net_Amount": "500.00",
            "Cash_Adjusted": "NO",
            "Source": "DIME",
            "Related_Transaction_ID": "",
        }
        from tools.portfolio.domain.ledger_change import LedgerChange
        uow.commit(state, LedgerChange(kind="replace_all", rows=[trade_row]))

    # Call void attempting adjust_cash=True
    # Strict mirror: Cash_Adjusted="NO" -> cash MUST NOT be touched!
    new_state = ledger_svc.void_transaction(tx_id, "default", adjust_cash=True)
    cash_after = next(h for h in new_state.holdings if h.symbol == "CASH_USD").units
    assert cash_after == initial_cash

    with temp_portfolio_repo.unit_of_work("default") as uow:
        rows = uow.read_trade_log_locked()
        assert len(rows) == 2
        assert rows[1]["Cash_Adjusted"] == "NO"


def test_void_transaction_is_idempotent(temp_portfolio_repo):
    ledger_svc = PortfolioLedgerService(repo=temp_portfolio_repo)
    tx_id = "tx_idem_1"
    with temp_portfolio_repo.unit_of_work("default") as uow:
        state = uow.load_state()
        trade_row = {
            "Transaction_ID": tx_id,
            "Timestamp": "2026-09-01 10:00:00",
            "Symbol": "GOOG",
            "Action": "BUY",
            "Units": "2",
            "Price": "180.00",
            "Currency": "USD",
            "Gross_Amount": "360.00",
            "Net_Amount": "360.00",
            "Cash_Adjusted": "YES",
            "Source": "MANUAL",
            "Related_Transaction_ID": "",
        }
        from tools.portfolio.domain.ledger_change import LedgerChange
        uow.commit(state, LedgerChange(kind="replace_all", rows=[trade_row]))

    # First void
    s1 = ledger_svc.void_transaction(tx_id, "default")
    # Second void (repeated)
    s2 = ledger_svc.void_transaction(tx_id, "default")

    with temp_portfolio_repo.unit_of_work("default") as uow:
        rows = uow.read_trade_log_locked()
        # Must only have 2 rows (original + 1 reversal), NOT 3 rows!
        assert len(rows) == 2


def test_cannot_void_a_reversal_row(temp_portfolio_repo):
    ledger_svc = PortfolioLedgerService(repo=temp_portfolio_repo)
    tx_id = "tx_norm_1"
    with temp_portfolio_repo.unit_of_work("default") as uow:
        state = uow.load_state()
        trade_row = {
            "Transaction_ID": tx_id,
            "Timestamp": "2026-09-01 10:00:00",
            "Symbol": "TSLA",
            "Action": "BUY",
            "Units": "1",
            "Price": "200.00",
            "Currency": "USD",
            "Gross_Amount": "200.00",
            "Net_Amount": "200.00",
            "Cash_Adjusted": "YES",
            "Source": "MANUAL",
            "Related_Transaction_ID": "",
        }
        from tools.portfolio.domain.ledger_change import LedgerChange
        uow.commit(state, LedgerChange(kind="replace_all", rows=[trade_row]))

    # Void original
    ledger_svc.void_transaction(tx_id, "default")

    # Find the reversal row ID
    with temp_portfolio_repo.unit_of_work("default") as uow:
        rows = uow.read_trade_log_locked()
        rev_id = rows[1]["Transaction_ID"]

    # Attempt to void the reversal row -> MUST fail
    with pytest.raises(ValueError, match="ไม่สามารถยกเลิกรายการที่เป็น Reversal หรือ Void ได้"):
        ledger_svc.void_transaction(rev_id, "default")


def test_dime_trade_economic_fields_immutable(temp_portfolio_repo):
    ledger_svc = PortfolioLedgerService(repo=temp_portfolio_repo)
    tx_id = "tx_dime_immutable"
    with temp_portfolio_repo.unit_of_work("default") as uow:
        state = uow.load_state()
        trade_row = {
            "Transaction_ID": tx_id,
            "Timestamp": "2026-09-01 10:00:00",
            "Symbol": "META",
            "Action": "BUY",
            "Units": "5",
            "Price": "500.00",
            "Currency": "USD",
            "Gross_Amount": "2500.00",
            "Net_Amount": "2500.00",
            "Cash_Adjusted": "YES",
            "Source": "DIME",
            "Notes": "Initial note",
            "Related_Transaction_ID": "",
        }
        from tools.portfolio.domain.ledger_change import LedgerChange
        uow.commit(state, LedgerChange(kind="replace_all", rows=[trade_row]))

    # Attempt to edit units -> MUST fail
    with pytest.raises(ValueError, match="รายการที่นำเข้าจาก Dime ไม่สามารถแก้ไขตัวเลขได้โดยตรง"):
        ledger_svc.edit_transaction(tx_id, units=10.0, portfolio_id="default")

    # Editing note only -> MUST succeed
    updated_state = ledger_svc.edit_transaction(tx_id, notes="Updated note only", portfolio_id="default")
    with temp_portfolio_repo.unit_of_work("default") as uow:
        rows = uow.read_trade_log_locked()
        assert rows[0]["Notes"] == "Updated note only"
