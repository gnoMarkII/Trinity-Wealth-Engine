"""Unit tests for Ledger Replay Engine (tools/portfolio/ledger_replay.py)."""
import csv
import pytest
from pathlib import Path

from tools.portfolio.constants import _TRADES_LOG_HEADER
from tools.portfolio.ledger_replay import _replay_symbol_trades


def test_pure_replay_symbol_trades_basic():
    """Test pure chronological replay calculation."""
    rows = [
        {"Timestamp": "2026-01-01 10:00:00", "Action": "BUY", "Units": "10", "Price": "100.0", "Currency": "THB", "FX_Rate": ""},
        {"Timestamp": "2026-01-05 10:00:00", "Action": "BUY", "Units": "10", "Price": "200.0", "Currency": "THB", "FX_Rate": ""},
        {"Timestamp": "2026-01-10 10:00:00", "Action": "SELL", "Units": "10", "Price": "250.0", "Currency": "THB", "FX_Rate": ""},
    ]
    # Initial: avg cost before sell is (1000 + 2000)/20 = 150.
    # Sell 10 @ 250 -> realized PnL = (250 - 150)*10 = 1000. Final units = 10, final avg = 150.
    updated, units, avg_cost, pnl = _replay_symbol_trades(rows, "TEST", "THB")
    assert units == 10.0
    assert avg_cost == 150.0
    assert pnl == 1000.0
    assert updated[2]["Realized_PnL_THB"] == "1000.00"

    # Now change first buy to 10 @ 300 (unordered input)
    rows_mod = [
        {"Timestamp": "2026-01-10 10:00:00", "Action": "SELL", "Units": "10", "Price": "250.0", "Currency": "THB", "FX_Rate": ""},
        {"Timestamp": "2026-01-01 10:00:00", "Action": "BUY", "Units": "10", "Price": "300.0", "Currency": "THB", "FX_Rate": ""},
        {"Timestamp": "2026-01-05 10:00:00", "Action": "BUY", "Units": "10", "Price": "200.0", "Currency": "THB", "FX_Rate": ""},
    ]
    # Avg cost before sell is (3000 + 2000)/20 = 250.
    # Sell 10 @ 250 -> realized PnL = (250 - 250)*10 = 0. Final units = 10, final avg = 250.
    updated2, units2, avg_cost2, pnl2 = _replay_symbol_trades(rows_mod, "TEST", "THB")
    assert units2 == 10.0
    assert avg_cost2 == 250.0
    assert pnl2 == 0.0
    assert updated2[2]["Realized_PnL_THB"] == "0.00"


def test_pure_replay_rejects_negative_units():
    """Test that pure replay raises ValueError if units drop below zero."""
    rows = [
        {"Timestamp": "2026-01-01 10:00:00", "Action": "SELL", "Units": "10", "Price": "100.0", "Currency": "THB", "FX_Rate": ""},
        {"Timestamp": "2026-01-05 10:00:00", "Action": "BUY", "Units": "10", "Price": "100.0", "Currency": "THB", "FX_Rate": ""},
    ]
    with pytest.raises(ValueError, match="Insufficient units"):
        _replay_symbol_trades(rows, "TEST", "THB")


def test_edit_transaction_recalculates_downstream_realized_pnl(isolated_portfolio, tmp_vault):
    """Test editing a BUY trade recalculates subsequent SELL realized PnL and updates state."""
    pt = isolated_portfolio
    pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")

    # Trade 1: BUY 10 PTT @ 100 (ts: 2026-01-01)
    pt.structured_execute_trade(
        symbol="PTT", asset_type="Stock", action="buy", units=10.0, price=100.0,
        currency="THB", date="2026-01-01"
    )
    # Trade 2: BUY 10 PTT @ 200 (ts: 2026-01-02)
    pt.structured_execute_trade(
        symbol="PTT", asset_type="Stock", action="buy", units=10.0, price=200.0,
        currency="THB", date="2026-01-02"
    )
    # Trade 3: SELL 10 PTT @ 250 (ts: 2026-01-03) -> Avg cost = 150 -> Realized PnL = 1000
    state = pt.structured_execute_trade(
        symbol="PTT", asset_type="Stock", action="sell", units=10.0, price=250.0,
        currency="THB", date="2026-01-03"
    )
    assert state.summary.total_realized_profit_ytd == 1000.0

    csv_path = pt.trading._get_trades_log_filepath("default")
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    tx1_id = rows[0]["Transaction_ID"]

    # Edit Trade 1 from 100 to 300 (total buy cost becomes (3000 + 2000)/20 = 250 avg cost)
    # Realized PnL for Trade 3 becomes (250 - 250)*10 = 0
    new_state = pt.edit_transaction(
        tx_id=tx1_id,
        price=300.0,
        adjust_cash=True,
    )
    assert new_state.summary.total_realized_profit_ytd == 0.0
    ptt_holding = pt._find_holding(new_state, "PTT")
    assert ptt_holding is not None
    assert ptt_holding.units == 10.0
    assert ptt_holding.avg_cost_thb == 250.0

    # Cash was adjusted: delta_native = (10*300 - 10*100) = 2000 more spent
    cash = pt._find_holding(new_state, "CASH_THB")
    # Started 100,000 - 1000 - 2000 + 2500 - 2000(delta) = 97,500
    assert cash.units == 97500.0


def test_edit_transaction_rejects_negative_units_atomically(isolated_portfolio, tmp_vault):
    """Test that editing transaction timestamp/units to invalid sequence raises ValueError and keeps original state."""
    pt = isolated_portfolio
    pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")

    pt.structured_execute_trade(
        symbol="PTT", asset_type="Stock", action="buy", units=10.0, price=100.0,
        currency="THB", date="2026-01-01"
    )
    pt.structured_execute_trade(
        symbol="PTT", asset_type="Stock", action="sell", units=10.0, price=150.0,
        currency="THB", date="2026-01-05"
    )

    csv_path = pt.trading._get_trades_log_filepath("default")
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tx_buy_id = rows[0]["Transaction_ID"]

    # Try moving BUY date to 2026-01-10 (after SELL on 2026-01-05) -> should fail
    with pytest.raises(ValueError, match="Insufficient units"):
        pt.edit_transaction(
            tx_id=tx_buy_id,
            timestamp="2026-01-10 12:00:00",
        )

    # Try reducing BUY units to 5 (< 10 sold) -> should fail
    with pytest.raises(ValueError, match="Insufficient units"):
        pt.edit_transaction(
            tx_id=tx_buy_id,
            units=5.0,
        )

    # Verify original state unchanged
    post, state = pt._load_or_init(portfolio_id="default")
    assert state.summary.total_realized_profit_ytd == 500.0


def test_edit_transaction_lightweight_notes_and_fx(isolated_portfolio, tmp_vault):
    """Test lightweight edit path for notes and FX rate."""
    pt = isolated_portfolio
    pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")

    pt.structured_execute_trade(
        symbol="PTT", asset_type="Stock", action="buy", units=10.0, price=100.0,
        currency="THB", notes="Initial note"
    )

    csv_path = pt.trading._get_trades_log_filepath("default")
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tx_id = rows[0]["Transaction_ID"]

    pt.edit_transaction(
        tx_id=tx_id,
        notes="Updated note via edit",
    )

    with csv_path.open("r", encoding="utf-8") as f:
        updated_rows = list(csv.DictReader(f))
    assert updated_rows[0]["Notes"] == "Updated note via edit"


def test_delete_transaction_with_cash_adjustment(isolated_portfolio, tmp_vault):
    """Test deleting a BUY transaction refunds cash and removes/reduces holding."""
    pt = isolated_portfolio
    pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")

    pt.structured_execute_trade(
        symbol="PTT", asset_type="Stock", action="buy", units=10.0, price=100.0,
        currency="THB"
    )
    post, state = pt._load_or_init(portfolio_id="default")
    assert pt._find_holding(state, "PTT") is not None
    assert pt._find_holding(state, "CASH_THB").units == 99000.0

    csv_path = pt.trading._get_trades_log_filepath("default")
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tx_id = rows[0]["Transaction_ID"]

    new_state = pt.delete_transaction(tx_id=tx_id, adjust_cash=True)
    # Holding should be removed
    assert pt._find_holding(new_state, "PTT") is None
    # Cash refunded to 100,000
    assert pt._find_holding(new_state, "CASH_THB").units == 100000.0

    with csv_path.open("r", encoding="utf-8") as f:
        remaining_rows = list(csv.DictReader(f))
    voided_ids = {
        r.get("Related_Transaction_ID")
        for r in remaining_rows
        if (r.get("Action") or "").startswith("VOID_")
    }
    active_rows = [
        r for r in remaining_rows
        if not (r.get("Action") or "").startswith("VOID_")
        and r.get("Transaction_ID") not in voided_ids
    ]
    assert len(active_rows) == 0
    assert len(remaining_rows) == 2
    assert any((r.get("Action") or "").startswith("VOID_") for r in remaining_rows)


def test_delete_transaction_rejects_if_causes_negative_units(isolated_portfolio, tmp_vault):
    """Test deleting a BUY transaction when shares have already been sold raises ValueError."""
    pt = isolated_portfolio
    pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")

    pt.structured_execute_trade(
        symbol="PTT", asset_type="Stock", action="buy", units=10.0, price=100.0,
        currency="THB", date="2026-01-01"
    )
    pt.structured_execute_trade(
        symbol="PTT", asset_type="Stock", action="sell", units=10.0, price=150.0,
        currency="THB", date="2026-01-02"
    )

    csv_path = pt.trading._get_trades_log_filepath("default")
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tx_buy_id = rows[0]["Transaction_ID"]

    with pytest.raises(ValueError, match="Insufficient units"):
        pt.delete_transaction(tx_id=tx_buy_id)


def test_edit_resets_synced_dividend_source(isolated_portfolio, tmp_vault):
    """Test that editing trade units/timestamp for a holding with dividend_source='synced' resets it to None."""
    pt = isolated_portfolio
    pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")

    pt.structured_execute_trade(
        symbol="PTT", asset_type="Stock", action="buy", units=10.0, price=100.0,
        currency="THB"
    )
    post, state = pt._load_or_init(portfolio_id="default")
    ptt = pt._find_holding(state, "PTT")
    ptt.dividend_source = "synced"
    pt._save(post, state, portfolio_id="default")

    csv_path = pt.trading._get_trades_log_filepath("default")
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tx_id = rows[0]["Transaction_ID"]

    new_state = pt.edit_transaction(tx_id=tx_id, units=20.0)
    updated_ptt = pt._find_holding(new_state, "PTT")
    assert updated_ptt.units == 20.0
    assert updated_ptt.dividend_source is None
