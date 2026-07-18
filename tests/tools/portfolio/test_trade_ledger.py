"""Tests for Trade Ledger (Trades_Log.csv) append-only logging on trade execution."""
import csv
from pathlib import Path
import pytest


def test_buy_creates_ledger_row(isolated_portfolio, tmp_vault):
    pt = isolated_portfolio
    # Deposit cash to enable buy
    pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")

    result = pt.execute_trade.invoke({
        "symbol": "PTT",
        "asset_type": "Stock",
        "action": "buy",
        "units": 100.0,
        "price": 35.0,
        "currency": "THB",
    })
    assert "[BUY]" in result

    csv_path = tmp_vault / "20_Portfolio_Management" / "Journals_and_Reports" / "Trades_Log.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Header + 1 data row
    assert len(rows) == 2
    header, row = rows[0], rows[1]
    assert header == ["Timestamp", "Symbol", "Action", "Units", "Price", "Currency", "FX_Rate", "Cost_THB", "Realized_PnL_THB", "Notes"]
    assert row[1] == "PTT"
    assert row[2] == "BUY"
    assert row[3] == "100"
    assert row[4] == "35.00"
    assert row[5] == "THB"
    assert row[7] == "3500.00"
    assert row[8] == ""  # Realized PnL is empty for buy


def test_sell_appends_ledger_row_and_binary_check(isolated_portfolio, tmp_vault):
    pt = isolated_portfolio
    pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")

    # Buy 100 PTT @ 35
    pt.execute_trade.invoke({
        "symbol": "PTT",
        "asset_type": "Stock",
        "action": "buy",
        "units": 100.0,
        "price": 35.0,
        "currency": "THB",
    })

    # Sell 40 PTT @ 40
    pt.execute_trade.invoke({
        "symbol": "PTT",
        "asset_type": "Stock",
        "action": "sell",
        "units": 40.0,
        "price": 40.0,
        "currency": "THB",
    })

    csv_path = tmp_vault / "20_Portfolio_Management" / "Journals_and_Reports" / "Trades_Log.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Header + 1 buy row + 1 sell row
    assert len(rows) == 3
    buy_row, sell_row = rows[1], rows[2]

    assert buy_row[2] == "BUY"
    assert sell_row[1] == "PTT"
    assert sell_row[2] == "SELL"
    assert sell_row[3] == "40"
    assert sell_row[4] == "40.00"
    assert sell_row[7] == "1400.00"  # cost basis = 40 * 35 = 1400.00
    assert sell_row[8] == "200.00"   # realized profit = (40 - 35) * 40 = 200.00

    # Binary check: verify no \r\r\n (Windows CRCRLF bug)
    content = csv_path.read_bytes()
    assert b"\r\r\n" not in content


def test_trade_with_notes_recorded_in_ledger(isolated_portfolio, tmp_vault):
    pt = isolated_portfolio
    pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")

    pt.execute_trade.invoke({
        "symbol": "PTT",
        "asset_type": "Stock",
        "action": "buy",
        "units": 50.0,
        "price": 35.0,
        "currency": "THB",
        "notes": "Testing notes column",
    })

    csv_path = tmp_vault / "20_Portfolio_Management" / "Journals_and_Reports" / "Trades_Log.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[-1][9] == "Testing notes column"
