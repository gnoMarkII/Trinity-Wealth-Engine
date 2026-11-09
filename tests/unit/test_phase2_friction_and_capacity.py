from decimal import Decimal
import pandas as pd
import pytest

from core.portfolio_advisory.friction import (
    estimate_order_friction,
    compute_rebalance_turnover_and_cost,
    OrderFrictionEstimate,
)
from core.portfolio_advisory.lot_rounding import (
    calculate_fee_inclusive_lot_sizing,
    calculate_market_lot_sizing,
)
from core.opportunity_finder.capacity import (
    calculate_candidate_capacity,
)
from core.opportunity_finder.models import OpportunityCandidate
from tools.market.quant_engine import (
    compute_amihud_illiquidity_20d,
    _select_return_price_series,
)


def test_estimate_order_friction_th():
    """Test Thai equity friction: 0.168% base fee (min 50 THB), square-root impact"""
    est = estimate_order_friction(
        market="TH",
        action_side="BUY",
        order_amount_native=Decimal("100000.00"),
        adtv_native=Decimal("10000000.00"),  # Q/ADTV = 100k / 10M = 0.01 -> sqrt = 0.1
        annualized_volatility_pct=Decimal("31.75"),  # daily vol = 31.75 / (sqrt(252)*100) = ~0.02
    )
    assert isinstance(est, OrderFrictionEstimate)
    # Explicit fee: 100,000 * 0.00168 = 168.00 THB (fee_bps = 16.8)
    assert est.explicit_fee_native == Decimal("168.00")
    assert est.explicit_fee_bps == Decimal("16.8")
    # Impact: Y=0.60, sqrt(0.01) = 0.1, daily_vol = 0.02 -> impact_rate = 0.6 * 0.02 * 0.1 = 0.0012 -> 12.0 bps
    assert est.market_impact_bps == Decimal("12.0")
    assert est.total_friction_bps == Decimal("28.8")
    assert est.total_friction_native == Decimal("288.00")


def test_estimate_order_friction_us():
    """Test US equity friction: 10 bps on BUY, 12 bps on SELL, square-root impact"""
    est_buy = estimate_order_friction(
        market="US",
        action_side="BUY",
        order_amount_native=Decimal("10000.00"),
        adtv_native=Decimal("1000000.00"),
        annualized_volatility_pct=Decimal("25.0"),
    )
    # Buy in US has 10.0 bps base fee
    assert est_buy.explicit_fee_bps == Decimal("10.0")
    assert est_buy.explicit_fee_native == Decimal("10.00")
    assert est_buy.market_impact_bps > Decimal("0")

    est_sell = estimate_order_friction(
        market="US",
        action_side="SELL",
        order_amount_native=Decimal("10000.00"),
        adtv_native=Decimal("1000000.00"),
        annualized_volatility_pct=Decimal("25.0"),
    )
    # Sell has SEC/TAF additional 2 bps: 12.0 bps total
    assert est_sell.explicit_fee_bps == Decimal("12.0")
    assert est_sell.explicit_fee_native == Decimal("12.00")


def test_compute_rebalance_turnover_and_cost():
    current_weights = {
        "PTT": Decimal("40.0"),
        "AAPL": Decimal("60.0"),
    }
    target_weights = {
        "PTT": Decimal("50.0"),
        "AAPL": Decimal("50.0"),
    }
    # abs diff: |50-40| + |50-60| = 10 + 10 = 20. Turnover = 20 / 2 = 10.00%
    turnover_pct, cost_bps = compute_rebalance_turnover_and_cost(
        current_weights_pct=current_weights,
        target_weights_pct=target_weights,
        one_way_cost_bps=Decimal("20.0"),
    )

    assert turnover_pct == Decimal("10.00")
    # Cost = 2 * (10 / 100) * 20.0 = 4.0 bps
    assert cost_bps == Decimal("4.0")


def test_calculate_fee_inclusive_lot_sizing():
    """Verify that fee-inclusive sizing does not exceed target budget once friction is added"""
    budget = Decimal("100000.00")
    price = Decimal("50.00")

    # Regular lot sizing without fees
    reg_sizing = calculate_market_lot_sizing(
        market="TH",
        native_currency="THB",
        target_amount_native=budget,
        current_price_native=price,
        fx_rate=Decimal("1.0"),
    )
    # 100,000 / 50 = 2000 shares
    assert reg_sizing.board_lot_shares == 2000
    assert reg_sizing.amount_native == Decimal("100000.00")

    # Fee-inclusive lot sizing: with ~28.8 bps friction, 2000 shares would cost 100,288 THB (> 100k)
    # It must round down to 1900 shares (board lot of 100) so 1900 * 50 = 95,000 + friction <= 100,000
    fee_inclusive_sizing = calculate_fee_inclusive_lot_sizing(
        market="TH",
        native_currency="THB",
        max_cash_budget_native=budget,
        current_price_native=price,
        fx_rate=Decimal("1.0"),
        adtv_native=Decimal("10000000.00"),
        annualized_volatility_pct=Decimal("30.0"),
    )
    assert fee_inclusive_sizing.board_lot_shares == 1900
    assert fee_inclusive_sizing.whole_shares == 1994
    assert fee_inclusive_sizing.odd_lot_shares == 94
    assert fee_inclusive_sizing.amount_native == Decimal("99700.00")
    # Verify friction on this sizing plus amount does not exceed budget
    friction = estimate_order_friction(
        market="TH",
        action_side="BUY",
        order_amount_native=fee_inclusive_sizing.amount_native,
        adtv_native=Decimal("10000000.00"),
        annualized_volatility_pct=Decimal("30.0"),
    )
    assert fee_inclusive_sizing.amount_native + friction.total_friction_native <= budget


def test_assess_opportunity_capacity_with_adtv():
    """Verify that ADTV restriction caps allocation room"""
    candidate = OpportunityCandidate(
        security_id="TH:PTT",
        issuer_id="PTT",
        symbol="PTT",
        market="TH",
        currency="THB",
        asset_type="EQUITY",
        company_name="PTT PCL",
        sector_name="Energy",
        sector_name_th="พลังงาน",
    )
    # 1. Without ADTV
    status, usable_cap, issuer_room, sector_room, reasons = calculate_candidate_capacity(
        candidate=candidate,
        total_nav_thb=Decimal("1000000"),
        available_cash_thb=Decimal("200000"),
        current_sector_exposure_thb=Decimal("0"),
        single_stock_cap_pct=Decimal("15.0"),
        sector_cap_pct=Decimal("30.0"),
    )
    assert status == "available"
    # Min(issuer_cap=150k, sector_cap=300k, cash=200k) = 150k
    assert usable_cap == Decimal("150000.00")

    # 2. With tight ADTV: 5% of 1,000,000 = 50,000 THB room cap
    status2, usable_cap2, issuer_room2, sector_room2, reasons2 = calculate_candidate_capacity(
        candidate=candidate,
        total_nav_thb=Decimal("1000000"),
        available_cash_thb=Decimal("200000"),
        current_sector_exposure_thb=Decimal("0"),
        single_stock_cap_pct=Decimal("15.0"),
        sector_cap_pct=Decimal("30.0"),
        adtv_thb=Decimal("1000000"),
        max_adv_participation_pct=Decimal("5.0"),
    )
    assert status2 == "available"
    # Min(issuer_cap=150k, sector_cap=300k, cash=200k, liq=50k) = 50k
    assert usable_cap2 == Decimal("50000.00")
    assert any("Liquidity capacity limit reached" in r for r in reasons2)


def test_compute_amihud_illiquidity_20d():
    """Test Amihud calculation on synthetic price/volume data"""
    dates = pd.date_range("2026-01-01", periods=25, freq="B")
    prices = [100.0 * (1.01 ** i) for i in range(25)]
    volumes = [10000.0] * 25

    df = pd.DataFrame({
        "Close": prices,
        "Volume": volumes,
    }, index=dates)

    amihud = compute_amihud_illiquidity_20d(df)
    assert amihud is not None
    assert amihud > 0.0

    # Insufficient data
    assert compute_amihud_illiquidity_20d(df.head(3)) is None

    # Zero volume
    df_zero_vol = df.copy()
    df_zero_vol["Volume"] = 0.0
    assert compute_amihud_illiquidity_20d(df_zero_vol) is None


def test_select_return_price_series():
    """Test that Adj Close is preferred over Close when valid"""
    df = pd.DataFrame({
        "Close": [100.0, 102.0, 105.0],
        "Adj Close": [50.0, 51.0, 52.5],
    })
    series = _select_return_price_series(df)
    assert series.iloc[0] == 50.0

    # Fallback to Close when Adj Close is missing
    df_no_adj = pd.DataFrame({
        "Close": [100.0, 102.0, 105.0],
    })
    series_fallback = _select_return_price_series(df_no_adj)
    assert series_fallback.iloc[0] == 100.0


def test_r05_friction_authoritative_native_fee_and_participation_without_vol():
    """R05: Exact native fees are authoritative even when rounded bps would understate fee;
    participation rate and liquidity warning still compute if volatility is missing."""
    # Exact R05 reproduction: 28,500 TH order has min fee 50.00 THB (17.5438... -> 17.5 bps)
    est = estimate_order_friction(
        market="TH",
        action_side="BUY",
        order_amount_native=Decimal("28500.00"),
    )
    assert est.explicit_fee_native == Decimal("50.00")
    assert est.explicit_fee_bps == Decimal("17.5")
    assert est.total_friction_native == Decimal("50.00")
    assert est.market_impact_status == "UNAVAILABLE_MISSING_BOTH"

    # Participation rate & warning still compute when ADTV > 0 and vol is None
    est_no_vol = estimate_order_friction(
        market="US",
        action_side="BUY",
        order_amount_native=Decimal("150000.00"),
        adtv_native=Decimal("1000000.00"),
        annualized_volatility_pct=None,
    )
    assert est_no_vol.participation_rate_pct == Decimal("15.00")
    assert est_no_vol.liquidity_warning is not None
    assert est_no_vol.market_impact_status == "UNAVAILABLE_MISSING_VOLATILITY"
    assert est_no_vol.total_friction_native == Decimal("150.00")

    with pytest.raises(ValueError):
        estimate_order_friction("TH", "BUY", Decimal("NaN"))


def test_r05_discrete_fee_inclusive_lot_sizing_boundaries_and_large_budgets():
    """R05: Fee-inclusive discrete search enforces budget >= order + exact_friction AND next unit > budget."""
    # 1. Exact reproduction: 1 share @ 28,500 + 50.00 min fee = 28,550.00 > 28,549.88
    under_budget = calculate_fee_inclusive_lot_sizing(
        market="TH",
        native_currency="THB",
        max_cash_budget_native=Decimal("28549.88"),
        current_price_native=Decimal("28500.00"),
        fx_rate=Decimal("35.50"),
    )
    assert under_budget.whole_shares == 0
    assert under_budget.amount_native == Decimal("0.00")

    # At exact 28,550.00 budget, 1 share is admitted cleanly
    exact_budget = calculate_fee_inclusive_lot_sizing(
        market="TH",
        native_currency="THB",
        max_cash_budget_native=Decimal("28550.00"),
        current_price_native=Decimal("28500.00"),
        fx_rate=Decimal("35.50"),
    )
    assert exact_budget.whole_shares == 1
    assert exact_budget.odd_lot_shares == 1
    assert exact_budget.amount_native == Decimal("28500.00")

    # 2. Large TH budget (25,000,000 THB >> 2^16 pennies): board + odd lot + exact next-share bound
    large_budget = Decimal("25000000.00")
    large_price = Decimal("123.50")
    large_sizing = calculate_fee_inclusive_lot_sizing(
        market="TH",
        native_currency="THB",
        max_cash_budget_native=large_budget,
        current_price_native=large_price,
        fx_rate=Decimal("35.50"),
        adtv_native=Decimal("500000000.00"),
        annualized_volatility_pct=Decimal("28.0"),
    )
    f_large = estimate_order_friction(
        market="TH",
        action_side="BUY",
        order_amount_native=large_sizing.amount_native,
        adtv_native=Decimal("500000000.00"),
        annualized_volatility_pct=Decimal("28.0"),
    )
    assert large_sizing.amount_native + f_large.total_friction_native <= large_budget
    next_share_amt = Decimal(large_sizing.whole_shares + 1) * large_price
    f_next = estimate_order_friction(
        market="TH",
        action_side="BUY",
        order_amount_native=next_share_amt,
        adtv_native=Decimal("500000000.00"),
        annualized_volatility_pct=Decimal("28.0"),
    )
    assert next_share_amt + f_next.total_friction_native > large_budget

    # 3. US 0.0001-share fractional discrete bound
    us_budget = Decimal("1000.00")
    us_price = Decimal("185.50")
    us_sizing = calculate_fee_inclusive_lot_sizing(
        market="US",
        native_currency="USD",
        max_cash_budget_native=us_budget,
        current_price_native=us_price,
        fx_rate=Decimal("35.50"),
        adtv_native=Decimal("5000000.00"),
        annualized_volatility_pct=Decimal("25.0"),
    )
    f_us = estimate_order_friction(
        market="US",
        action_side="BUY",
        order_amount_native=us_sizing.amount_native,
        adtv_native=Decimal("5000000.00"),
        annualized_volatility_pct=Decimal("25.0"),
    )
    assert us_sizing.amount_native + f_us.total_friction_native <= us_budget
    next_frac = us_sizing.fractional_shares + Decimal("0.0001")
    next_us_amt = (next_frac * us_price).quantize(Decimal("0.01"))
    f_us_next = estimate_order_friction(
        market="US",
        action_side="BUY",
        order_amount_native=next_us_amt,
        adtv_native=Decimal("5000000.00"),
        annualized_volatility_pct=Decimal("25.0"),
    )
    assert next_us_amt + f_us_next.total_friction_native > us_budget


def test_w02_capacity_fail_closed_on_zero_or_non_finite_adtv():
    """W02: Zero or non-finite ADTV must never report ample liquidity."""
    cand = OpportunityCandidate(
        security_id="TH:ILLIQ",
        issuer_id="ILLIQ",
        symbol="ILLIQ",
        market="TH",
        currency="THB",
        asset_type="EQUITY",
        company_name="Illiquid Co",
        sector_name="Energy",
        sector_name_th="พลังงาน",
    )
    st_zero, cap_zero, _, _, _ = calculate_candidate_capacity(
        candidate=cand,
        total_nav_thb=Decimal("1000000"),
        available_cash_thb=Decimal("200000"),
        current_sector_exposure_thb=Decimal("0"),
        adtv_thb=Decimal("0"),
    )
    assert st_zero == "constrained"
    assert cap_zero == Decimal("0.00")

    st_nan, cap_nan, _, _, _ = calculate_candidate_capacity(
        candidate=cand,
        total_nav_thb=Decimal("1000000"),
        available_cash_thb=Decimal("200000"),
        current_sector_exposure_thb=Decimal("0"),
        adtv_thb=Decimal("NaN"),
    )
    assert st_nan == "unknown"
    assert cap_nan is None


def test_w05_amihud_inf_negative_guard_and_metadata_provenance():
    """W05: Non-finite Adj Close/Inf/double-negative Close*Volume excluded; metadata exposes basis and status."""
    from tools.market.quant_engine import compute_amihud_illiquidity_20d_with_metadata

    dates = pd.date_range("2026-01-01", periods=10, freq="B")
    # Adj Close contains inf -> _select_return_price_series must fall back to Close
    df_inf_adj = pd.DataFrame(
        {
            "Close": [100.0 + i for i in range(10)],
            "Adj Close": [100.0] * 9 + [float("inf")],
            "Volume": [10_000.0] * 10,
        },
        index=dates,
    )
    ret_series = _select_return_price_series(df_inf_adj)
    assert ret_series.iloc[-1] == 109.0  # Used Close rather than inf Adj Close

    meta = compute_amihud_illiquidity_20d_with_metadata(df_inf_adj, currency="THB")
    assert meta["return_price_basis"] == "close"
    assert meta["status"] == "partial"  # < 20 valid days
    assert meta["currency"] == "THB"
    assert meta["value"] is not None

    # Double negative Close and Volume must NOT slip past close * volume > 0
    df_double_neg = pd.DataFrame(
        {
            "Close": [-100.0] * 10,
            "Volume": [-5000.0] * 10,
        },
        index=dates,
    )
    assert compute_amihud_illiquidity_20d(df_double_neg) is None
