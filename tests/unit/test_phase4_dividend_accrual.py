from decimal import Decimal
import pytest

from core.portfolio.dividend_accrual import (
    DividendScheduleEntry,
    DividendEntitlementSnapshot,
    DividendSettlementRecord,
    AccruedDividendSummary,
    compute_accrued_dividend_receivable,
)


def test_thai_equity_dividend_accrual_lifecycle():
    """Verify ex-dividend accrual: 0 before ex_date, net locked during window, 0 on/after pay_date"""
    holdings = [
        {"symbol": "PTT", "units": Decimal("10000")}
    ]
    schedules = [
        DividendScheduleEntry(
            ticker="PTT",
            ex_date="2026-03-05",
            pay_date="2026-03-25",
            dps_native=Decimal("1.20"),
            market="TH",
            currency="THB",
            withholding_tax_pct=Decimal("10.0"),  # 10% Thai statutory withholding
        )
    ]

    # 1. Before ex-date: no accrual
    res_before = compute_accrued_dividend_receivable(holdings, schedules, as_of_date="2026-03-04")
    assert res_before.accrued_receivable_thb == Decimal("0.00")
    assert len(res_before.items) == 0

    # 2. Exactly on ex-date: locked in True Economic NAV
    res_on_ex = compute_accrued_dividend_receivable(holdings, schedules, as_of_date="2026-03-05")
    # Gross = 10,000 * 1.20 = 12,000 THB; WHT = 1,200 THB; Net = 10,800 THB
    assert res_on_ex.accrued_receivable_thb == Decimal("10800.00")
    assert len(res_on_ex.items) == 1
    assert res_on_ex.items[0].gross_amount_native == Decimal("12000.00")
    assert res_on_ex.items[0].withholding_tax_native == Decimal("1200.00")
    assert res_on_ex.items[0].net_amount_thb == Decimal("10800.00")
    assert res_on_ex.items[0].days_to_payment == 20

    # 3. Midway between ex-date and pay-date: remains accrued
    res_mid = compute_accrued_dividend_receivable(holdings, schedules, as_of_date="2026-03-15")
    assert res_mid.accrued_receivable_thb == Decimal("10800.00")
    assert res_mid.items[0].days_to_payment == 10

    # 4. On pay-date: payment arrives as settled cash, receivable drops to 0
    res_paid = compute_accrued_dividend_receivable(holdings, schedules, as_of_date="2026-03-25")
    assert res_paid.accrued_receivable_thb == Decimal("0.00")
    assert len(res_paid.items) == 0


def test_us_equity_dividend_accrual_with_fx():
    """Verify US equity dividend accrual with 15% W-8BEN tax and USD/THB FX conversion"""
    holdings = [
        {"symbol": "AAPL", "units": Decimal("100")}
    ]
    schedules = [
        DividendScheduleEntry(
            ticker="AAPL",
            ex_date="2026-02-10",
            pay_date="2026-02-25",
            dps_native=Decimal("0.25"),
            market="US",
            currency="USD",
            withholding_tax_pct=Decimal("15.0"),  # 15% US W-8BEN tax
        )
    ]
    fx_rate = Decimal("35.50")
    res = compute_accrued_dividend_receivable(
        holdings, schedules, as_of_date="2026-02-15", usd_thb_fx_rate=fx_rate
    )
    # Gross = 100 * 0.25 = 25.00 USD
    # Tax = 25 * 0.15 = 3.75 USD
    # Net USD = 21.25 USD
    # Net THB = 21.25 * 35.50 = 754.38 THB (rounded)
    assert res.accrued_receivable_usd == Decimal("21.25")
    assert res.accrued_receivable_thb == Decimal("754.38")
    assert len(res.items) == 1
    assert res.items[0].ticker == "AAPL"
    assert res.items[0].currency == "USD"
    assert res.items[0].days_to_payment == 10


def test_portfolio_with_multiple_dividends_and_unheld_stocks():
    """Verify aggregation across holdings and ignoring unannounced or unheld tickers"""
    holdings = [
        {"symbol": "PTT", "units": Decimal("5000")},
        {"symbol": "SCB", "units": Decimal("2000")},
    ]
    schedules = [
        DividendScheduleEntry(
            ticker="PTT",
            ex_date="2026-04-01",
            pay_date="2026-04-20",
            dps_native=Decimal("1.00"),
            market="TH",
            currency="THB",
        ),
        DividendScheduleEntry(
            ticker="BDMS",  # Not held in portfolio
            ex_date="2026-04-01",
            pay_date="2026-04-20",
            dps_native=Decimal("0.50"),
            market="TH",
            currency="THB",
        ),
    ]
    res = compute_accrued_dividend_receivable(holdings, schedules, as_of_date="2026-04-10")
    # Only PTT should accrue: 5000 * 1.00 * 0.9 = 4500.00 THB
    assert res.accrued_receivable_thb == Decimal("4500.00")
    assert len(res.items) == 1
    assert res.items[0].ticker == "PTT"
    assert res.entitlement_verified is False
    assert res.entitlement_basis == "holdings_fallback_unverified"
    assert res.warnings


def test_r02_entitlement_derived_from_trade_history_blocks_post_xd_buy_and_retains_post_xd_sell():
    """R02 regression: ex-date holdings (trade_date < ex_date) govern dividend entitlement."""
    sched = [
        DividendScheduleEntry(
            ticker="PTT",
            ex_date="2026-03-05",
            pay_date="2026-03-25",
            dps_native=Decimal("1.20"),
            market="TH",
            currency="THB",
            withholding_tax_pct=Decimal("10.0"),
        )
    ]

    # Case 1: Buy 100 shares on 2026-03-06 (after ex_date) -> 0 entitlement despite current holding
    post_xd_holdings = [{"symbol": "PTT", "units": Decimal("100")}]
    post_xd_trades = [
        {"Symbol": "PTT", "Action": "BUY", "Units": "100", "Timestamp": "2026-03-06T10:00:00Z"}
    ]
    res_post_buy = compute_accrued_dividend_receivable(
        post_xd_holdings, sched, as_of_date="2026-03-10", trade_history=post_xd_trades
    )
    assert res_post_buy.accrued_receivable_thb == Decimal("0.00")
    assert res_post_buy.items == []
    assert res_post_buy.entitlement_verified is True
    assert res_post_buy.entitlement_basis == "trade_history"

    # Case 2: Buy 100 on 2026-03-01 (before XD), buy 50 on XD (03-05, excluded), sell 150 on 03-06 -> current position 0
    # Receivable MUST still retain the 100 pre-XD entitled shares = 100 * 1.20 * 0.9 = 108.00 THB!
    sold_out_holdings = [{"symbol": "PTT", "units": Decimal("0")}]
    mixed_trades = [
        {"Symbol": "PTT.BK", "Action": "BUY", "Units": "100", "Timestamp": "2026-03-01T10:00:00Z"},
        {"Symbol": "PTT", "Action": "BUY", "Units": "50", "Timestamp": "2026-03-05T10:00:00Z"},
        {"Symbol": "PTT", "Action": "SELL", "Units": "150", "Timestamp": "2026-03-06T14:00:00Z"},
    ]
    res_sold = compute_accrued_dividend_receivable(
        sold_out_holdings, sched, as_of_date="2026-03-10", trade_history=mixed_trades
    )
    assert res_sold.accrued_receivable_thb == Decimal("108.00")
    assert len(res_sold.items) == 1
    assert res_sold.items[0].units == Decimal("100")
    assert res_sold.items[0].entitlement_source == "trade_history"


def test_r02_aggregates_duplicate_rows_and_dedups_schedules_and_clears_settlements():
    """R02 regression: aggregate duplicate holdings, dedup schedule events, use snapshots, clear settlements."""
    # Duplicate holding rows (100 + 200 = 300 shares) when using fallback
    dup_holdings = [
        {"symbol": "PTT", "units": Decimal("100")},
        {"symbol": "PTT.BK", "units": Decimal("200")},
    ]
    dup_schedules = [
        DividendScheduleEntry(
            ticker="PTT",
            ex_date="2026-03-05",
            pay_date="2026-03-25",
            dps_native=Decimal("1.00"),
            market="TH",
            currency="THB",
            withholding_tax_pct=Decimal("10.0"),
        ),
        DividendScheduleEntry(
            ticker="PTT.BK",
            ex_date="2026-03-05",
            pay_date="2026-03-25",
            dps_native=Decimal("1.00"),
            market="TH",
            currency="THB",
            withholding_tax_pct=Decimal("10.0"),
        ),
    ]
    res_agg = compute_accrued_dividend_receivable(
        dup_holdings, dup_schedules, as_of_date="2026-03-10"
    )
    assert res_agg.accrued_receivable_thb == Decimal("270.00")
    assert len(res_agg.items) == 1

    # Early settlement before nominal pay_date clears receivable so cash + receivable aren't double-counted
    snaps = [DividendEntitlementSnapshot(ticker="PTT", ex_date="2026-03-05", entitled_units=Decimal("300"))]
    settlements = [
        DividendSettlementRecord(
            ticker="PTT",
            ex_date="2026-03-05",
            settled_date="2026-03-18",
            settled_amount_thb=Decimal("270.00"),
            currency="THB",
            status="settled",
        )
    ]
    res_early_settled = compute_accrued_dividend_receivable(
        [], dup_schedules, as_of_date="2026-03-20", entitlement_snapshots=snaps, settlement_records=settlements
    )
    assert res_early_settled.accrued_receivable_thb == Decimal("0.00")
    assert res_early_settled.items == []

    # Fail-closed when history is required but unavailable
    res_strict = compute_accrued_dividend_receivable(
        dup_holdings, dup_schedules, as_of_date="2026-03-10", require_verified_entitlement=True
    )
    assert res_strict.accrued_receivable_thb == Decimal("0.00")
    assert res_strict.entitlement_verified is False
    assert res_strict.entitlement_basis == "unavailable"


def test_r03_float_fx_and_invalid_fx_handling():
    """R03 regression: float/string FX rates convert cleanly; non-finite/invalid FX raises ValueError."""
    holdings = [{"symbol": "AAPL", "units": Decimal("100")}]
    trades = [{"Symbol": "AAPL", "Action": "BUY", "Units": "100", "Timestamp": "2026-02-01T10:00:00Z"}]
    sched = [
        DividendScheduleEntry(
            ticker="AAPL",
            ex_date="2026-02-10",
            pay_date="2026-02-25",
            dps_native=Decimal("0.25"),
            market="US",
            currency="USD",
        )
    ]
    # Float FX (as returned by PortfolioState.fx_rates["USDTHB"]) must not raise TypeError
    res = compute_accrued_dividend_receivable(
        holdings, sched, as_of_date="2026-02-15", usd_thb_fx_rate=35.5, trade_history=trades
    )
    assert res.accrued_receivable_usd == Decimal("21.25")
    assert res.accrued_receivable_thb == Decimal("754.38")

    with pytest.raises(ValueError):
        compute_accrued_dividend_receivable(holdings, sched, as_of_date="2026-02-15", usd_thb_fx_rate=0)
    with pytest.raises(ValueError):
        compute_accrued_dividend_receivable(holdings, sched, as_of_date="2026-02-15", usd_thb_fx_rate=True)
