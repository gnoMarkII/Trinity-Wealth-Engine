"""Tests for Trade Ledger (Trades_Log.csv) append-only logging on trade execution, migration, and note updates."""
import csv
from pathlib import Path
import pytest
from tools.portfolio.constants import _TRADES_LOG_HEADER


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

    import tools.portfolio.trading as trading_mod
    csv_path = trading_mod._get_trades_log_filepath("default")
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Header + 1 data row
    assert len(rows) == 2
    header, row = rows[0], rows[1]
    assert header == _TRADES_LOG_HEADER
    assert row[0].startswith("tx_")
    assert row[2] == "PTT"
    assert row[3] == "BUY"
    assert float(row[5]) == 35.00
    assert row[6] == "THB"
    assert row[8] == "3500.00"
    assert row[9] == ""  # Realized PnL is empty for buy


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

    import tools.portfolio.trading as trading_mod
    csv_path = trading_mod._get_trades_log_filepath("default")
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Header + 1 buy row + 1 sell row
    assert len(rows) == 3
    buy_row, sell_row = rows[1], rows[2]

    assert buy_row[0].startswith("tx_")
    assert buy_row[3] == "BUY"
    assert sell_row[0].startswith("tx_")
    assert sell_row[2] == "PTT"
    assert sell_row[3] == "SELL"
    assert sell_row[4] == "40"
    assert float(sell_row[5]) == 40.00
    assert sell_row[8] == "1400.00"  # cost basis = 40 * 35 = 1400.00
    assert sell_row[9] == "200.00"   # realized profit = (40 - 35) * 40 = 200.00

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

    import tools.portfolio.trading as trading_mod
    csv_path = trading_mod._get_trades_log_filepath("default")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[-1][10] == "Testing notes column"


def test_pre_migration_on_read(isolated_portfolio, tmp_vault):
    """ทดสอบว่าการอ่าน CSV รูปแบบเดิม 10 คอลัมน์ จะถูก migrate เป็น 11 คอลัมน์อัตโนมัติ"""
    import tools.portfolio.trading as trading_mod
    csv_path = trading_mod._get_trades_log_filepath("default")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # เขียน CSV รูปแบบเก่า 10 คอลัมน์
    old_header = ["Timestamp", "Symbol", "Action", "Units", "Price", "Currency", "FX_Rate", "Cost_THB", "Realized_PnL_THB", "Notes"]
    old_row = ["2026-08-14T22:23:06", "UNH", "BUY", "0.287077", "348.34", "USD", "36.5000", "3650.00", "", "Old UNH buy"]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(old_header)
        writer.writerow(old_row)

    # อ่านผ่าน get_structured_trades_log
    items = trading_mod.get_structured_trades_log("default")
    assert len(items) == 1
    assert items[0]["symbol"] == "UNH"
    assert items[0]["action"] == "BUY"
    assert items[0]["transaction_id"].startswith("tx_")
    assert items[0]["notes"] == "Old UNH buy"

    # ตรวจสอบไฟล์บนดิสก์ว่าถูก migrate แล้ว
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert rows[0] == _TRADES_LOG_HEADER
    assert len(rows[1]) == len(_TRADES_LOG_HEADER)
    assert rows[1][0] == items[0]["transaction_id"]
    assert rows[1][2] == "UNH"


def test_pre_migration_on_write(isolated_portfolio, tmp_vault):
    """ทดสอบว่าการบันทึกเทรดใหม่ลงบนไฟล์ CSV 10 คอลัมน์เดิม จะ trigger migration ก่อน append เสมอ ไม่ทำให้คอลัมน์เลื่อน"""
    import tools.portfolio.trading as trading_mod
    pt = isolated_portfolio
    pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")

    csv_path = trading_mod._get_trades_log_filepath("default")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # เขียน CSV รูปแบบเก่า 10 คอลัมน์
    old_header = ["Timestamp", "Symbol", "Action", "Units", "Price", "Currency", "FX_Rate", "Cost_THB", "Realized_PnL_THB", "Notes"]
    old_row = ["2026-08-14T22:23:06", "UNH", "BUY", "0.287077", "348.34", "USD", "36.5000", "3650.00", "", "Legacy trade"]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(old_header)
        writer.writerow(old_row)

    # ดำเนินการเทรดใหม่โดยตรง (Write before Read)
    pt.execute_trade.invoke({
        "symbol": "PTT",
        "asset_type": "Stock",
        "action": "buy",
        "units": 10.0,
        "price": 35.0,
        "currency": "THB",
        "notes": "New trade",
    })

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) == 3
    assert rows[0] == _TRADES_LOG_HEADER

    # แถวเดิม 1 (Legacy trade) ต้องมี len(_TRADES_LOG_HEADER) คอลัมน์
    assert len(rows[1]) == len(_TRADES_LOG_HEADER)
    assert rows[1][0].startswith("tx_")
    assert rows[1][2] == "UNH"
    assert rows[1][10] == "Legacy trade"

    # แถวใหม่ 2 (New trade) ต้องมี len(_TRADES_LOG_HEADER) คอลัมน์
    assert len(rows[2]) == len(_TRADES_LOG_HEADER)
    assert rows[2][0].startswith("tx_")
    assert rows[2][2] == "PTT"
    assert rows[2][10] == "New trade"


def test_update_trade_note_and_injection_sanitization(isolated_portfolio, tmp_vault):
    """ทดสอบ update_trade_note พร้อมตรวจสอบ formula injection sanitization"""
    import tools.portfolio.trading as trading_mod
    pt = isolated_portfolio
    pt._manage_cash_flow_locked(100_000.0, "deposit", "THB")

    pt.execute_trade.invoke({
        "symbol": "PTT",
        "asset_type": "Stock",
        "action": "buy",
        "units": 10.0,
        "price": 35.0,
        "currency": "THB",
        "notes": "Initial note",
    })

    items = trading_mod.get_structured_trades_log("default")
    assert len(items) == 1
    tx_id = items[0]["transaction_id"]

    # แก้ไข Note ปกติ
    updated = trading_mod.update_trade_note(tx_id, "Updated note content", "default")
    assert updated["notes"] == "Updated note content"

    # ตรวจสอบอ่านกลับ
    items_after = trading_mod.get_structured_trades_log("default")
    assert items_after[0]["notes"] == "Updated note content"

    # แก้ไข Note ด้วย Formula Injection (ขึ้นต้นด้วย =)
    updated_injection = trading_mod.update_trade_note(tx_id, "=SUM(A1:A10)", "default")
    assert updated_injection["notes"] == "'=SUM(A1:A10)"

    csv_path = trading_mod._get_trades_log_filepath("default")
    content = csv_path.read_text(encoding="utf-8")
    assert "'=SUM(A1:A10)" in content
