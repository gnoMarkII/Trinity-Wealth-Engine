"""Comprehensive Unit Tests for Terminal V2 Domain Calculations.

Validates pure mathematical algorithms across all capabilities:
1. Max Pain argmin calculation, non-standard contract filtering, and tie-breakers.
2. Put/Call volume and Open Interest ratios with zero-division handling.
3. Basis points rate spreads (NY Fed & BIS policy rates).
4. CFTC COT percentile rank with boundary clamping.
5. OFR Financial Stress Index systemic risk regime classification.
6. Commodity volatility 52-week percentile & regime classification.
7. US Treasury auction moving average (prior 8) & demand delta (NO WI tail).
8. Financial ratios (FCF, FCF margin, Debt-to-OCF, division by zero).
9. Form 4 90-day net buying ratio (P/S only, award/gift exclusion, 90d window filter).
"""
import pytest
from tools.market.terminal_v2.domain.calculations import (
    calculate_auction_demand_summary,
    calculate_commodity_vol_percentile,
    calculate_cot_percentile,
    calculate_financial_ratios,
    calculate_insider_net_buying_90d,
    calculate_max_pain,
    calculate_policy_rate_spreads,
    calculate_put_call_ratios,
    calculate_rate_spread_bps,
    classify_commodity_vol_regime,
    classify_fsi_regime,
)
from tools.market.terminal_v2.domain.models import (
    InsiderTransaction,
    OptionContract,
    PolicyRateItem,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_contract(
    underlying: str = "AAPL",
    strike: float = 100.0,
    side: str = "call",
    oi: int = 10,
    vol: int = 5,
    expiry: str = "2026-10-16",
    multiplier: int = 100,
    is_standard: bool = True,
) -> OptionContract:
    occ = f"{underlying}261016{'C' if side == 'call' else 'P'}{int(strike * 1000):08d}"
    return OptionContract(
        occ_symbol=occ,
        underlying=underlying,
        expiry=expiry,
        strike=strike,
        side=side,
        open_interest=oi,
        volume=vol,
        multiplier=multiplier,
        is_standard=is_standard,
    )


# ============================================================================
# 1. Options & Derivatives Calculations
# ============================================================================

def test_calculate_max_pain_deterministic():
    """Verify Max Pain picks the strike with minimum total theoretical payout."""
    expiry = "2026-10-16"
    contracts = [
        _make_contract(strike=90.0, side="call", oi=100, expiry=expiry),
        _make_contract(strike=100.0, side="call", oi=500, expiry=expiry),
        _make_contract(strike=110.0, side="call", oi=200, expiry=expiry),
        _make_contract(strike=90.0, side="put", oi=200, expiry=expiry),
        _make_contract(strike=100.0, side="put", oi=400, expiry=expiry),
        _make_contract(strike=110.0, side="put", oi=100, expiry=expiry),
    ]

    result = calculate_max_pain(contracts, underlying="AAPL", expiry=expiry, spot_price=101.5)
    assert result.underlying == "AAPL"
    assert result.expiry == expiry
    assert result.strike == 100.0
    assert result.minimum_theoretical_payout == 200_000.0
    assert result.candidate_count == 3
    assert result.excluded_contract_count == 0
    assert result.spot_price == 101.5
    assert result.distance_from_spot == pytest.approx(1.5, 0.001)


def test_calculate_max_pain_excludes_non_standard_and_multipliers():
    """Verify contracts with multiplier != 100 or non-standard flag are excluded."""
    expiry = "2026-10-16"
    contracts = [
        _make_contract(strike=100.0, side="call", oi=100, expiry=expiry, multiplier=100),
        _make_contract(strike=100.0, side="put", oi=100, expiry=expiry, multiplier=100),
        _make_contract(strike=95.0, side="call", oi=1000, expiry=expiry, multiplier=33),  # excluded
        _make_contract(strike=95.0, side="put", oi=1000, expiry=expiry, is_standard=False),  # excluded
        _make_contract(strike=95.0, side="call", oi=-5, expiry=expiry),  # excluded negative OI
        _make_contract(strike=-10.0, side="put", oi=100, expiry=expiry),  # excluded negative strike
    ]

    result = calculate_max_pain(contracts, underlying="AAPL", expiry=expiry)
    assert result.strike == 100.0
    assert result.excluded_contract_count == 4


def test_calculate_max_pain_no_valid_contracts_raises():
    """Verify ValueError is raised if no standard contracts match expiry."""
    contracts = [
        _make_contract(strike=100.0, side="call", oi=10, expiry="2026-11-20"),
    ]
    with pytest.raises(ValueError, match="No valid standard option contracts"):
        calculate_max_pain(contracts, underlying="AAPL", expiry="2026-10-16")


def test_calculate_max_pain_tie_breaker_spot_price():
    """Verify tie-breaker chooses the strike closest to spot price."""
    expiry = "2026-10-16"
    contracts = [
        _make_contract(strike=100.0, side="call", oi=10, expiry=expiry),
        _make_contract(strike=100.0, side="put", oi=10, expiry=expiry),
        _make_contract(strike=120.0, side="call", oi=10, expiry=expiry),
        _make_contract(strike=120.0, side="put", oi=10, expiry=expiry),
    ]
    result_closer_to_100 = calculate_max_pain(contracts, "AAPL", expiry, spot_price=105.0)
    assert result_closer_to_100.strike == 100.0

    result_closer_to_120 = calculate_max_pain(contracts, "AAPL", expiry, spot_price=118.0)
    assert result_closer_to_120.strike == 120.0


def test_calculate_put_call_ratios_standard():
    """Verify Put/Call volume and OI ratios calculate correctly."""
    expiry = "2026-10-16"
    contracts = [
        _make_contract(strike=100.0, side="call", vol=200, oi=1000, expiry=expiry),
        _make_contract(strike=105.0, side="call", vol=300, oi=1500, expiry=expiry),
        _make_contract(strike=95.0, side="put", vol=250, oi=2000, expiry=expiry),
        _make_contract(strike=90.0, side="put", vol=250, oi=500, expiry=expiry),
    ]

    pcr = calculate_put_call_ratios(contracts, underlying="AAPL", expiry=expiry)
    assert pcr.put_volume == 500
    assert pcr.call_volume == 500
    assert pcr.volume_ratio == 1.0
    assert pcr.put_open_interest == 2500
    assert pcr.call_open_interest == 2500
    assert pcr.oi_ratio == 1.0


def test_calculate_put_call_ratios_zero_call_handles_division():
    """Verify None is returned when call volume or call OI is zero."""
    expiry = "2026-10-16"
    contracts = [
        _make_contract(strike=95.0, side="put", vol=100, oi=500, expiry=expiry),
    ]
    pcr = calculate_put_call_ratios(contracts, underlying="AAPL", expiry=expiry)
    assert pcr.call_volume == 0
    assert pcr.call_open_interest == 0
    assert pcr.volume_ratio is None
    assert pcr.oi_ratio is None


# ============================================================================
# 2. Macro Spreads, COT, and Regimes
# ============================================================================

def test_rate_spread_bps_calculation():
    """Verify rate spread calculation in basis points."""
    assert calculate_rate_spread_bps(5.05, 5.00) == 5.0
    assert calculate_rate_spread_bps(4.82, 4.90) == -8.0
    assert calculate_rate_spread_bps(5.33, 5.33) == 0.0


def test_cot_percentile_calculations():
    assert calculate_cot_percentile(100, []) == 50.0
    assert calculate_cot_percentile(100, [100, 100]) == 50.0

    history = [100, 200, 300, 400, 500]
    assert calculate_cot_percentile(100, history) == 0.0
    assert calculate_cot_percentile(500, history) == 100.0
    assert calculate_cot_percentile(300, history) == 50.0

    assert calculate_cot_percentile(50, history) == 0.0
    assert calculate_cot_percentile(600, history) == 100.0


def test_policy_rate_spreads_calculation():
    rates = [
        PolicyRateItem(
            country="TH",
            rate_value=2.50,
            rate_type="1-Day Repo",
            effective_date="2026-09",
            currency="THB",
            central_bank="BOT",
        ),
        PolicyRateItem(
            country="US",
            rate_value=5.00,
            rate_type="Fed Funds",
            effective_date="2026-09",
            currency="USD",
            central_bank="Fed",
        ),
        PolicyRateItem(
            country="JP",
            rate_value=0.25,
            rate_type="Call Rate",
            effective_date="2026-09",
            currency="JPY",
            central_bank="BOJ",
        ),
    ]

    spreads = calculate_policy_rate_spreads(rates, benchmark_country="TH")
    assert spreads["TH"] == 0.0
    assert spreads["US"] == 250.0
    assert spreads["JP"] == -225.0
    assert calculate_policy_rate_spreads(rates, benchmark_country="ZZ") == {}


def test_fsi_regime_classification():
    assert classify_fsi_regime(-1.2) == "calm"
    assert classify_fsi_regime(-0.51) == "calm"
    assert classify_fsi_regime(-0.5) == "normal"
    assert classify_fsi_regime(0.0) == "normal"
    assert classify_fsi_regime(0.5) == "normal"
    assert classify_fsi_regime(0.51) == "elevated"
    assert classify_fsi_regime(1.5) == "elevated"
    assert classify_fsi_regime(1.51) == "severe"
    assert classify_fsi_regime(3.0) == "severe"


# ============================================================================
# 3. Commodity Volatility & Treasury Demand
# ============================================================================

def test_commodity_vol_percentile_under_min_samples():
    history = [20.0 + i for i in range(50)]
    percentile, count = calculate_commodity_vol_percentile(35.0, history, min_samples=100)
    assert percentile is None
    assert count == 50


def test_commodity_vol_percentile_hand_verified():
    history = [10.0 + i for i in range(100)]
    percentile, count = calculate_commodity_vol_percentile(59.0, history, min_samples=100)
    assert count == 100
    assert percentile == 50.0

    p_top, _ = calculate_commodity_vol_percentile(200.0, history, min_samples=100)
    assert p_top == 100.0

    p_bot, _ = calculate_commodity_vol_percentile(5.0, history, min_samples=100)
    assert p_bot == 0.0


def test_commodity_vol_regime_classification():
    assert classify_commodity_vol_regime(None) is None
    assert classify_commodity_vol_regime(85.0) == "extreme_panic"
    assert classify_commodity_vol_regime(92.4) == "extreme_panic"
    assert classify_commodity_vol_regime(65.0) == "elevated"
    assert classify_commodity_vol_regime(75.0) == "elevated"
    assert classify_commodity_vol_regime(35.0) == "normal"
    assert classify_commodity_vol_regime(50.0) == "normal"
    assert classify_commodity_vol_regime(34.9) == "complacent"
    assert classify_commodity_vol_regime(12.0) == "complacent"


def test_auction_demand_summary_missing_or_under_threshold():
    mean, delta, count = calculate_auction_demand_summary(None, [2.5, 2.6, 2.7])
    assert mean is None
    assert delta is None

    mean, delta, count = calculate_auction_demand_summary(2.5, [2.4, 2.3], min_samples=3)
    assert mean is None
    assert delta is None
    assert count == 2


def test_auction_demand_summary_hand_verified():
    prior_btcs = [2.5, 2.6, 2.4, 2.5, 2.7, 2.3, 2.5, 2.5]
    latest_btc = 2.75

    mean, delta, count = calculate_auction_demand_summary(latest_btc, prior_btcs, min_samples=3, max_prior=8)
    assert count == 8
    assert mean == 2.5
    assert delta == 0.25

    more_prior = [2.5, 2.6, 2.4, 2.5, 2.7, 2.3, 2.5, 2.5, 100.0, 200.0]
    mean_clamped, delta_clamped, count_clamped = calculate_auction_demand_summary(latest_btc, more_prior, max_prior=8)
    assert count_clamped == 8
    assert mean_clamped == 2.5
    assert delta_clamped == 0.25


# ============================================================================
# 4. Financial Ratios & Form 4 Insider Trades
# ============================================================================

def test_financial_ratios_hand_verified():
    revenue = 1000.0
    ocf = 300.0
    capex = 100.0
    debt = 600.0

    fcf, margin, debt_to_ocf = calculate_financial_ratios(revenue, ocf, capex, debt)
    assert fcf == 200.0
    assert margin == 0.20
    assert debt_to_ocf == 2.0


def test_financial_ratios_edge_cases():
    fcf, margin, debt_to_ocf = calculate_financial_ratios(None, None, None, None)
    assert fcf is None
    assert margin is None
    assert debt_to_ocf is None

    # Negative revenue/OCF division by zero guards
    fcf, margin, debt_to_ocf = calculate_financial_ratios(0.0, -100.0, 50.0, 200.0)
    assert fcf == -150.0
    assert margin is None
    assert debt_to_ocf is None


def test_insider_net_buying_90d_p_and_s_only():
    as_of = "2026-09-20"
    txs = [
        InsiderTransaction(
            transaction_date="2026-09-10",
            reporting_owner="CEO",
            officer_title="Chief Executive Officer",
            is_officer=True,
            is_director=True,
            is_ten_percent_owner=False,
            transaction_code="P",
            shares=1000.0,
            price_per_share=150.0,
            notional_usd=150000.0,
            direct_or_indirect="D",
            accession_number="0001",
            is_amendment=False,
        ),
        InsiderTransaction(
            transaction_date="2026-08-01",
            reporting_owner="CFO",
            officer_title="Chief Financial Officer",
            is_officer=True,
            is_director=False,
            is_ten_percent_owner=False,
            transaction_code="S",
            shares=500.0,
            price_per_share=100.0,
            notional_usd=50000.0,
            direct_or_indirect="D",
            accession_number="0002",
            is_amendment=False,
        ),
        # Excluded codes: A (Award), M (Option exercise), G (Gift)
        InsiderTransaction(
            transaction_date="2026-09-05",
            reporting_owner="Director",
            officer_title=None,
            is_officer=False,
            is_director=True,
            is_ten_percent_owner=False,
            transaction_code="A",
            shares=10000.0,
            price_per_share=0.0,
            notional_usd=0.0,
            direct_or_indirect="D",
            accession_number="0003",
            is_amendment=False,
        ),
        # Excluded older than 90 days
        InsiderTransaction(
            transaction_date="2026-01-01",
            reporting_owner="CEO",
            officer_title="Chief Executive Officer",
            is_officer=True,
            is_director=True,
            is_ten_percent_owner=False,
            transaction_code="P",
            shares=5000.0,
            price_per_share=100.0,
            notional_usd=500000.0,
            direct_or_indirect="D",
            accession_number="0004",
            is_amendment=False,
        ),
    ]

    ratio, p_sum, s_sum, count = calculate_insider_net_buying_90d(txs, as_of)
    assert count == 2
    assert p_sum == 150000.0
    assert s_sum == 50000.0
    # Net buy ratio = (150000 - 50000) / (150000 + 50000) = 100000 / 200000 = 0.5
    assert ratio == 0.5
