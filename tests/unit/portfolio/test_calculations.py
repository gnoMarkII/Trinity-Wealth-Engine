import pytest
from tools.portfolio.domain.constants import CASH_THB_SYMBOL, CASH_USD_SYMBOL
from tools.portfolio.domain.models import Holding, PortfolioState, Summary, AllocationTarget
from tools.portfolio.domain.calculations import (
    calc_weighted_avg_cost,
    calc_realized_pnl,
    calc_holding_currency,
    recalc_holding,
    compute_total_cost,
    recalc_summary,
    recalc_fundamentals_derived,
    recalc_all,
    compute_allocation_breakdown,
    compute_target_allocation_variance,
)


def test_calc_weighted_avg_cost():
    # Buy 10 @ 100, then Buy 10 @ 200 -> Avg cost 150
    cost = calc_weighted_avg_cost(10.0, 100.0, 10.0, 200.0)
    assert cost == 150.0

    # Buy 0 initial, buy 5 @ 50 -> Avg cost 50
    cost = calc_weighted_avg_cost(0.0, 0.0, 5.0, 50.0)
    assert cost == 50.0


def test_calc_realized_pnl():
    # Buy @ 100, Sell 5 @ 120 (THB) -> Profit 100
    pnl = calc_realized_pnl(100.0, 5.0, 120.0, fx_rate=1.0)
    assert pnl == 100.0

    # USD trade with FX 36.0: Buy @ 10, Sell 2 @ 15 -> Native profit 10 USD -> 360 THB
    pnl_usd = calc_realized_pnl(10.0, 2.0, 15.0, fx_rate=36.0)
    assert pnl_usd == 360.0


def test_recalc_holding_thb():
    h = Holding(
        symbol="PTT",
        asset_type="Stock",
        units=100.0,
        avg_cost_thb=30.0,
        current_price_thb=35.0,
    )
    recalc_holding(h, current_fx=36.0)
    assert h.market_value_thb == 3500.0
    assert h.unrealized_pnl_percent == pytest.approx(16.67, rel=1e-2)


def test_recalc_holding_usd():
    h = Holding(
        symbol="AAPL",
        asset_type="Stock",
        units=10.0,
        avg_cost_usd=200.0,
        current_price_usd=220.0,
    )
    recalc_holding(h, current_fx=35.0)
    # Market value in THB: 10 * 220 * 35 = 77,000 THB
    assert h.market_value_thb == 77000.0
    # Unrealized PnL %: (220 - 200) / 200 * 100 = 10%
    assert h.unrealized_pnl_percent == 10.0


def test_recalc_holding_cash():
    cash_thb = Holding(symbol=CASH_THB_SYMBOL, asset_type="Cash", units=5000.0)
    recalc_holding(cash_thb, current_fx=35.0)
    assert cash_thb.market_value_thb == 5000.0
    assert cash_thb.unrealized_pnl_percent is None

    cash_usd = Holding(symbol=CASH_USD_SYMBOL, asset_type="Cash", units=100.0)
    recalc_holding(cash_usd, current_fx=35.0)
    assert cash_usd.market_value_thb == 3500.0


def test_recalc_all_and_summary():
    state = PortfolioState(
        last_updated="2026-08-23T12:00:00",
        fx_rates={"USDTHB": 35.0},
        holdings=[
            Holding(symbol=CASH_THB_SYMBOL, asset_type="Cash", units=10000.0),
            Holding(symbol="PTT", asset_type="Stock", units=100.0, avg_cost_thb=30.0, current_price_thb=35.0),
            Holding(symbol="AAPL", asset_type="Stock", units=10.0, avg_cost_usd=200.0, current_price_usd=220.0),
        ],
    )
    recalc_all(state)

    # Cost basis: 10000 (cash) + 3000 (PTT) + (2000 * 35) (AAPL) = 10000 + 3000 + 70000 = 83000
    assert state.summary.total_cost_basis_thb == 83000.0

    # Total value: 10000 + 3500 + 77000 = 90500
    assert state.summary.total_value_thb == 90500.0

    # Unrealized profit: (35-30)*100 + (220-200)*10*35 = 500 + 7000 = 7500
    assert state.summary.total_unrealized_profit == 7500.0


def test_compute_allocation_breakdown():
    state = PortfolioState(
        last_updated="2026-08-23T12:00:00",
        fx_rates={"USDTHB": 35.0},
        holdings=[
            Holding(symbol=CASH_THB_SYMBOL, asset_type="Cash", units=50000.0),
            Holding(symbol="PTT", asset_type="Stock", units=1000.0, avg_cost_thb=50.0, current_price_thb=50.0),
        ],
    )
    recalc_all(state)
    # Total NAV: 100,000 THB (Cash 50k = 50%, Stock 50k = 50%)
    breakdown = compute_allocation_breakdown(state, group_by="asset_type")
    assert len(breakdown) == 2
    assert breakdown[0]["pct"] == 50.0
    assert breakdown[1]["pct"] == 50.0
