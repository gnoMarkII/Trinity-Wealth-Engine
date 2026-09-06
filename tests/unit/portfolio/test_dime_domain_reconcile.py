from decimal import Decimal
import pytest

from tools.portfolio.domain.calculations import (
    validate_reconciliation_invariant,
    allocate_document_fees_pro_rata,
    _replay_symbol_trades,
)
from tools.portfolio.domain.models import TradeFeeBreakdown


def test_reconciliation_happy_path_buy():
    # Buy 10.5 shares @ 150.25 = 1577.625 -> rounded gross = 1577.63
    # Commission: 1.50, VAT: 0.11, Other: 0.05 -> Total fees: 1.66
    # Net: 1577.63 + 1.66 = 1579.29
    fees = TradeFeeBreakdown(
        commission=Decimal("1.50"),
        vat=Decimal("0.11"),
        other_fees=Decimal("0.05"),
        fee_currency="USD",
    )
    ok, msg = validate_reconciliation_invariant(
        units=Decimal("10.5"),
        price=Decimal("150.25"),
        gross_amount=Decimal("1577.63"),
        fees=fees,
        net_amount=Decimal("1579.29"),
        action="BUY",
    )
    assert ok is True, msg


def test_reconciliation_happy_path_sell():
    # Sell 5 shares @ 200.00 = 1000.00
    # Fees: 2.00
    # Net: 1000.00 - 2.00 = 998.00
    fees = TradeFeeBreakdown(
        commission=Decimal("2.00"),
        vat=Decimal("0.00"),
        other_fees=Decimal("0.00"),
        fee_currency="THB",
    )
    ok, msg = validate_reconciliation_invariant(
        units=Decimal("5"),
        price=Decimal("200.00"),
        gross_amount=Decimal("1000.00"),
        fees=fees,
        net_amount=Decimal("998.00"),
        action="SELL",
    )
    assert ok is True, msg


def test_reconciliation_line_item_mismatch_fails():
    # Units * price = 100, but gross reported is 110
    ok, msg = validate_reconciliation_invariant(
        units=Decimal("10"),
        price=Decimal("10.00"),
        gross_amount=Decimal("110.00"),
        fees=Decimal("0.00"),
        net_amount=Decimal("110.00"),
        action="BUY",
    )
    assert ok is False
    assert "Line-item mismatch" in msg


def test_reconciliation_statement_fee_mismatch_fails():
    # Gross = 100, fees = 5, but net reported is 108 instead of 105
    ok, msg = validate_reconciliation_invariant(
        units=Decimal("10"),
        price=Decimal("10.00"),
        gross_amount=Decimal("100.00"),
        fees=Decimal("5.00"),
        net_amount=Decimal("108.00"),
        action="BUY",
    )
    assert ok is False
    assert "Statement mismatch" in msg


def test_pro_rata_fee_allocation_with_tie_breaker_penny():
    # 3 lines with gross amounts: 100.00, 200.00, 300.00 (Total gross = 600.00)
    # Total fee to allocate: 1.00
    # Proportions:
    # 100/600 * 1.00 = 0.1666... -> 0.17
    # 200/600 * 1.00 = 0.3333... -> 0.33
    # 300/600 * 1.00 = 0.5000... -> 0.50
    # Sum of rounded = 0.17 + 0.33 + 0.50 = 1.00 (exact)
    grosses = [Decimal("100.00"), Decimal("200.00"), Decimal("300.00")]
    allocated = allocate_document_fees_pro_rata(grosses, Decimal("1.00"))
    assert sum(allocated) == Decimal("1.00")

    # Now with a fee that produces a residual: total fee = 10.00, grosses = [100, 100, 100]
    # 10 / 3 = 3.33 each, sum = 9.99, residual = 0.01
    # Residual should go to the first line with largest gross (here all equal -> line 0)
    grosses_equal = [Decimal("100.00"), Decimal("100.00"), Decimal("100.00")]
    allocated_res = allocate_document_fees_pro_rata(grosses_equal, Decimal("10.00"))
    assert sum(allocated_res) == Decimal("10.00")
    assert allocated_res[0] == Decimal("3.34")
    assert allocated_res[1] == Decimal("3.33")
    assert allocated_res[2] == Decimal("3.33")

    # If line 2 has the largest gross: [50, 50, 200], total = 300, fee = 1.00
    # 50/300 * 1 = 0.17, 50/300 * 1 = 0.17, 200/300 * 1 = 0.67 -> sum = 1.01
    # residual = 1.00 - 1.01 = -0.01 -> line with largest gross gets -0.01 (0.66)
    grosses_skewed = [Decimal("50.00"), Decimal("50.00"), Decimal("200.00")]
    allocated_skewed = allocate_document_fees_pro_rata(grosses_skewed, Decimal("1.00"))
    assert sum(allocated_skewed) == Decimal("1.00")


def test_replay_with_net_amount_includes_fees():
    # BUY AAPL: 10 units @ 150.00, Gross = 1500.00, Net_Amount = 1515.00 (with $15 fees)
    trades = [
        {
            "Transaction_ID": "tx_1",
            "Timestamp": "2026-09-01 10:00:00",
            "Symbol": "AAPL",
            "Action": "BUY",
            "Units": "10",
            "Price": "150.00",
            "Currency": "USD",
            "Gross_Amount": "1500.00",
            "Net_Amount": "1515.00",
        }
    ]
    updated, units_held, avg_cost, pnl = _replay_symbol_trades(trades, "AAPL", "USD")
    assert units_held == 10.0
    # Avg cost should reflect 1515.00 / 10 = 151.50
    assert avg_cost == 151.50


def test_replay_non_destructive_void_nets_to_zero():
    # BUY 10 units @ 100, then VOID_BUY referencing tx_1
    trades = [
        {
            "Transaction_ID": "tx_1",
            "Timestamp": "2026-09-01 10:00:00",
            "Symbol": "MSFT",
            "Action": "BUY",
            "Units": "10",
            "Price": "100.00",
            "Net_Amount": "1005.00",
        },
        {
            "Transaction_ID": "tx_2",
            "Timestamp": "2026-09-01 11:00:00",
            "Symbol": "MSFT",
            "Action": "VOID_BUY",
            "Units": "10",
            "Price": "100.00",
            "Net_Amount": "1005.00",
            "Related_Transaction_ID": "tx_1",
        },
    ]
    updated, units_held, avg_cost, pnl = _replay_symbol_trades(trades, "MSFT", "USD")
    assert units_held == 0.0
    assert avg_cost == 0.0
    assert len(updated) == 2
    assert updated[0]["Cost_THB"] == "0.00"
    assert updated[1]["Cost_THB"] == "0.00"
