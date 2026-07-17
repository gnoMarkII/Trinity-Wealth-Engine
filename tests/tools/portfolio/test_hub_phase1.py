"""Test Phase 1 Portfolio Hub backend extensions: fundamentals cache, derived metrics, structured accessors, and atomic upsert."""
import time
from unittest.mock import MagicMock, patch
import pytest


def test_fetch_fundamentals_ttl_and_caching(isolated_portfolio):
    pt = isolated_portfolio
    post, state = pt._load_or_init()
    # Add a stock holding
    h = pt.Holding(
        symbol="AAPL",
        asset_type="Stock",
        units=10,
        avg_cost_usd=150.0,
        current_price_usd=200.0,
    )
    state.holdings.append(h)

    mock_info = {
        "trailingPE": 25.5,
        "trailingEps": 4.12,
        "payoutRatio": 0.4,
        "marketCap": 250_000_000_000,
        "dividendRate": 1.5,
        "dividendYield": 0.015,
        "longName": "Apple Inc.",
    }

    with patch("yfinance.Ticker") as mock_ticker:
        mock_instance = MagicMock()
        mock_instance.info = mock_info
        mock_ticker.return_value = mock_instance

        results = pt._fetch_fundamentals(state, force=False)
        assert results.get("AAPL") == "ok"
        assert h.pe_ratio == 25.5
        assert h.eps == 4.12
        assert h.payout_ratio == 40.0
        assert h.market_cap_value == 250_000_000_000
        assert h.dividend_per_share == 1.5
        assert h.dividend_yield == 1.5
        assert h.company_name == "Apple Inc."
        assert h.fundamentals_updated_at is not None
        assert mock_ticker.call_count == 1

        # Second call within TTL should NOT trigger yfinance call when force=False
        results2 = pt._fetch_fundamentals(state, force=False)
        assert mock_ticker.call_count == 1

        # Force call should trigger yfinance call again
        results3 = pt._fetch_fundamentals(state, force=True)
        assert mock_ticker.call_count == 2


def test_recalc_fundamentals_derived(isolated_portfolio):
    pt = isolated_portfolio
    post, state = pt._load_or_init()
    state.fx_rates["USDTHB"] = 35.0

    # Stock 1: Mega Cap + Dividend + USD cost
    h1 = pt.Holding(
        symbol="MEGA",
        asset_type="Stock",
        units=100,
        avg_cost_usd=100.0,
        current_price_usd=120.0,
        market_cap_value=250_000_000_000,  # >= $200B -> Mega
        dividend_per_share=5.0,
    )
    # Stock 2: Small Cap + THB cost
    h2 = pt.Holding(
        symbol="SMALL",
        asset_type="Stock",
        units=1000,
        avg_cost_thb=10.0,
        current_price_thb=12.0,
        market_cap_value=1_000_000_000,  # < $2B -> Small
    )
    state.holdings.extend([h1, h2])

    pt._recalc_all(state)

    # Check H1
    assert h1.market_cap_tier == "Mega"
    # yield_on_cost: (5.0 / 100.0) * 100 = 5.0%
    assert h1.yield_on_cost == 5.0
    # unrealized_pnl_value: market_value_thb - cost_thb
    # market_value_thb = 100 * 120 * 35 = 420000; cost_thb = 100 * 100 * 35 = 350000 -> 70000
    assert h1.unrealized_pnl_value == 70000.0

    # Check H2
    assert h2.market_cap_tier == "Small"
    assert h2.yield_on_cost is None
    # market_value_thb = 1000 * 12 = 12000; cost_thb = 1000 * 10 = 10000 -> 2000
    assert h2.unrealized_pnl_value == 2000.0


def test_get_structured_bucket_allocation(isolated_portfolio):
    pt = isolated_portfolio
    post, state = pt._load_or_init()
    state.fx_rates["USDTHB"] = 35.0

    # Define targets: 60% Core, 40% Growth
    state.allocation_targets = [
        pt.AllocationTarget(bucket_id="core", name="Core Portfolio", target_percent=60.0),
        pt.AllocationTarget(bucket_id="growth", name="Growth Portfolio", target_percent=40.0),
    ]

    h1 = pt.Holding(
        symbol="CORE_ETF",
        asset_type="Stock",
        units=100,
        avg_cost_thb=600.0,
        current_price_thb=600.0,
        bucket_id="core",
    )
    h2 = pt.Holding(
        symbol="GROWTH_STK",
        asset_type="Stock",
        units=100,
        avg_cost_thb=400.0,
        current_price_thb=400.0,
        bucket_id="growth",
    )
    state.holdings.extend([h1, h2])
    pt._save(post, state)

    summaries, warning = pt.get_structured_bucket_allocation(state)
    assert warning is None
    assert len(summaries) == 2
    core_sum = next(s for s in summaries if s["bucket_id"] == "core")
    assert core_sum["actual_value_thb"] == 60000.0
    assert core_sum["actual_percent"] == 60.0
    assert core_sum["variance"] == 0.0

    # Check warning when targets don't sum to 100%
    state.allocation_targets[1].target_percent = 30.0  # Sum is now 90%
    _, warning = pt.get_structured_bucket_allocation(state)
    assert warning is not None
    assert "ไม่เท่ากับ 100%" in warning


def test_record_performance_snapshot_atomic_upsert(isolated_portfolio):
    pt = isolated_portfolio
    post, state = pt._load_or_init()
    state.fx_rates["USDTHB"] = 35.0
    h = pt.Holding(symbol="CASH_THB", asset_type="Cash", units=100_000.0)
    state.holdings.append(h)
    pt._save(post, state)

    # First snapshot record
    res1 = pt.record_performance_snapshot.invoke({"refresh_prices": False})
    assert "recorded" in res1
    history1 = pt.get_structured_performance_history()
    assert len(history1) == 1
    assert history1[0]["Total_NAV"] == 100_000.0

    # Modify NAV and record snapshot again on the same day
    h.units = 150_000.0
    pt._save(post, state)
    res2 = pt.record_performance_snapshot.invoke({"refresh_prices": False})
    assert "updated" in res2

    # Should have replaced the same-day row instead of appending
    history2 = pt.get_structured_performance_history()
    assert len(history2) == 1
    assert history2[0]["Total_NAV"] == 150_000.0


def test_structured_read_accessors(isolated_portfolio):
    pt = isolated_portfolio
    # Portfolio state accessor
    state = pt.get_structured_portfolio_state(refresh_prices=False, fetch_fundamentals=False)
    assert state.summary is not None

    # Watchlist accessor
    pt.add_to_watchlist.invoke({"symbol": "TSLA", "asset_type": "Stock", "target_price": 200.0})
    wl = pt.get_structured_watchlist()
    assert len(wl.items) == 1
    assert wl.items[0].symbol == "TSLA"

    # Journal accessor
    pt.append_trading_journal.invoke({"entry": "Initial setup note."})
    journal = pt.get_structured_journal()
    assert len(journal) == 1
    assert "Initial setup note." in journal[0]["content"]

    # Goals accessor
    pt.set_goal.invoke({"name": "Retirement", "target_amount_thb": 10_000_000.0, "goal_type": "nav_target"})
    goals = pt.get_structured_goals()
    assert len(goals) == 1
    assert goals[0]["name"] == "Retirement"

