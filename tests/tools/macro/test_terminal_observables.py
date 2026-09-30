import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from schemas.macro_schemas import MarketObservable
from tools.macro.terminal_observables import (
    build_thai_market_observables,
    build_rates_observables,
    build_thai_market_stance,
)
from tools.market.terminal_v2.domain.models import (
    ThaiFundFlowSnapshot,
    InvestorTypeRow,
    MarketBreadth,
    MarketValuation,
    ThaiRetailGoldQuote,
    GoldPriceDetail,
    TreasuryYieldCurveSnapshot,
    TreasuryYieldPoint,
    GlobalPolicyRateSnapshot,
    PolicyRateItem,
    FinancialStressSnapshot,
)


def test_build_thai_market_observables_from_mock_service():
    mock_terminal = MagicMock()

    flow_snapshot = ThaiFundFlowSnapshot(
        market="SET",
        as_of="2026-09-25",
        total_value=45210340000.0,
        investors=(
            InvestorTypeRow("Local Institutions", "Local Institutions", 4.2e9, 3.8e9, 400000000.0),
            InvestorTypeRow("Proprietary Trading", "Proprietary Trading", 3.1e9, 3.2e9, -100000000.0),
            InvestorTypeRow("Foreign Investors", "Foreign Investors", 24.5e9, 23.0e9, 1500000000.0),
            InvestorTypeRow("Retail Investors", "Retail Investors", 13.41e9, 15.21e9, -1800000000.0),
        ),
    )
    mock_terminal.get_investor_flow.return_value = flow_snapshot

    breadth_snapshot = MarketBreadth(
        market="SET",
        as_of="2026-09-25",
        gainers=285,
        losers=190,
        unchanged=160,
    )
    mock_terminal.get_market_breadth.return_value = breadth_snapshot

    valuation_snapshot = MarketValuation(
        market="SET",
        as_of="2026-09-25",
        market_cap=1.85e13,
        pe_ratio=15.2,
        pbv_ratio=1.35,
        dividend_yield=3.42,
        turnover_ratio=45.2,
    )
    mock_terminal.get_market_valuation.return_value = valuation_snapshot

    gold_quote = ThaiRetailGoldQuote(
        source="Gold Traders Association",
        unit="THB/Baht-weight",
        bar=GoldPriceDetail(buy=40500.0, sell=40600.0),
        ornament=GoldPriceDetail(buy=39800.0, sell=41100.0),
        announced_at="2026-09-26 09:20:00",
    )
    mock_terminal.get_retail_gold.return_value = gold_quote

    observables = build_thai_market_observables(terminal_service=mock_terminal)

    assert len(observables) >= 8

    # Foreign flow check
    foreign_obs = next(o for o in observables if o.observable_id == "obs_set_flow_foreign")
    assert foreign_obs.value == "1500.00"
    assert foreign_obs.unit == "THB Mil"
    assert foreign_obs.status == "verified"

    # Breadth check
    ad_obs = next(o for o in observables if o.observable_id == "obs_set_advance_decline_ratio")
    assert ad_obs.value == "1.50"
    assert ad_obs.unit == "ratio"

    # Valuation check
    pe_obs = next(o for o in observables if o.observable_id == "obs_set_valuation_pe")
    assert pe_obs.value == "15.20"

    # Gold check
    gold_obs = next(o for o in observables if o.observable_id == "obs_gta_gold_bar_sell")
    assert gold_obs.value == "40600.00"
    assert gold_obs.unit == "THB/Baht-weight"


def test_build_rates_observables_from_mock_service():
    mock_data_service = MagicMock()

    curve_snapshot = TreasuryYieldCurveSnapshot(
        observation_date="2026-09-25",
        yields=(
            TreasuryYieldPoint("3 Mo", 5.35),
            TreasuryYieldPoint("2 Yr", 4.90),
            TreasuryYieldPoint("10 Yr", 4.55),
        ),
        spread_10y_2y_bps=-35.0,
        spread_10y_3m_bps=-80.0,
    )
    mock_data_service.get_treasury_yield_curve.return_value = curve_snapshot

    policy_snapshot = GlobalPolicyRateSnapshot(
        as_of_date="2026-09-20",
        rates=(
            PolicyRateItem("United States", 5.25, "Target Rate", "2024-07-31", "USD", "Fed"),
            PolicyRateItem("Thailand", 2.50, "Repo Rate", "2023-09-27", "THB", "BOT"),
        ),
    )
    mock_data_service.get_global_policy_rates.return_value = policy_snapshot

    fsi_snapshot = FinancialStressSnapshot(
        as_of_date="2026-09-25",
        published_at="2026-09-25",
        fsi_value=-0.42,
        categories=(),
        trend_90d=(),
    )
    mock_data_service.get_financial_stress.return_value = fsi_snapshot

    observables = build_rates_observables(terminal_data_service=mock_data_service)

    assert len(observables) >= 6

    # 10Y Yield
    y10_obs = next(o for o in observables if o.observable_id == "obs_ust_10y_yield")
    assert y10_obs.value == "4.55"
    assert y10_obs.unit == "%"

    # 10Y-2Y Spread
    spread_obs = next(o for o in observables if o.observable_id == "obs_spread_us_10y_2y_bps")
    assert spread_obs.value == "-35.0"
    assert spread_obs.unit == "bps"

    # Differential US-TH
    diff_obs = next(o for o in observables if o.observable_id == "obs_diff_us_th_policy_rate_bis")
    assert diff_obs.value == "275.0"
    assert diff_obs.unit == "bps"

    # OFR Stress
    stress_obs = next(o for o in observables if o.observable_id == "obs_ofr_financial_stress")
    assert stress_obs.value == "-0.42"
    assert stress_obs.unit == "pts"


def test_build_thai_market_stance_aggregation():
    obs_list = [
        MarketObservable(
            observable_id="obs_set_flow_foreign",
            asset_bucket="equities",
            region="Thailand",
            indicator="SET Foreign Flow",
            value="1500.00",
            unit="THB Mil",
            observed_at="2026-09-25",
            source_file="test.md",
        ),
        MarketObservable(
            observable_id="obs_set_advance_decline_ratio",
            asset_bucket="equities",
            region="Thailand",
            indicator="SET AD Ratio",
            value="1.50",
            unit="ratio",
            observed_at="2026-09-25",
            source_file="test.md",
        ),
        MarketObservable(
            observable_id="obs_set_valuation_pe",
            asset_bucket="equities",
            region="Thailand",
            indicator="SET PE",
            value="15.20",
            unit="ratio",
            observed_at="2026-09-25",
            source_file="test.md",
        ),
        MarketObservable(
            observable_id="obs_gta_gold_bar_sell",
            asset_bucket="commodities",
            region="Thailand",
            indicator="GTA Gold",
            value="40600.00",
            unit="THB/Baht-weight",
            observed_at="2026-09-25",
            source_file="test.md",
        ),
        MarketObservable(
            observable_id="obs_diff_us_th_policy_rate_bis",
            asset_bucket="cash",
            region="Global",
            indicator="Rate Diff",
            value="275.0",
            unit="bps",
            observed_at="2026-09-25",
            source_file="test.md",
        ),
    ]

    stance = build_thai_market_stance(obs_list)

    assert stance["investor_flow"]["foreign_net_mb"] == 1500.00
    assert stance["market_breadth"]["advance_decline_ratio"] == 1.50
    assert stance["market_breadth"]["sentiment"] == "bullish"
    assert stance["valuation"]["pe_ratio"] == 15.20
    assert stance["physical_gold"]["bar_sell_thb"] == 40600.00
    assert stance["policy_spread_bps"] == 275.0


def test_terminal_observables_graceful_on_failure():
    bad_service = MagicMock()
    bad_service.get_investor_flow.side_effect = RuntimeError("Settrade network down")
    bad_service.get_market_breadth.side_effect = RuntimeError("Breadth parse error")
    bad_service.get_market_valuation.side_effect = RuntimeError("Valuation unavailable")
    bad_service.get_retail_gold.side_effect = RuntimeError("GTA site unreachable")

    # Must return empty list rather than exploding
    observables = build_thai_market_observables(terminal_service=bad_service)
    assert observables == []


def test_bis_policy_rate_standard_codes_and_spread_calculation():
    """Case 1: Both US and TH rates present with standard 'US'/'TH' codes (from bis_adapter)."""
    mock_service = MagicMock()
    mock_service.get_treasury_yield_curve.return_value = TreasuryYieldCurveSnapshot(
        observation_date="2026-09-25", yields=(), spread_10y_2y_bps=None, spread_10y_3m_bps=None
    )
    mock_service.get_financial_stress.return_value = FinancialStressSnapshot(
        as_of_date="2026-09-25", published_at="2026-09-25", fsi_value=0.0, categories=(), trend_90d=()
    )
    # Production bis_adapter emits country='US' and country='TH'
    policy_snapshot = GlobalPolicyRateSnapshot(
        as_of_date="2026-09-20",
        rates=(
            PolicyRateItem("US", 5.25, "Target Rate", "2024-07-31", "USD", "Fed"),
            PolicyRateItem("TH", 2.50, "Repo Rate", "2023-09-27", "THB", "BOT"),
        ),
    )
    mock_service.get_global_policy_rates.return_value = policy_snapshot

    observables = build_rates_observables(terminal_data_service=mock_service)

    us_obs = next(o for o in observables if o.observable_id == "obs_us_policy_rate_bis")
    assert us_obs.value == "5.25"
    assert us_obs.region == "United States"
    assert us_obs.is_valid is True

    th_obs = next(o for o in observables if o.observable_id == "obs_thai_policy_rate_bis")
    assert th_obs.value == "2.50"
    assert th_obs.region == "Thailand"
    assert th_obs.is_valid is True

    diff_obs = next(o for o in observables if o.observable_id == "obs_diff_us_th_policy_rate_bis")
    assert diff_obs.value == "275.0"
    assert diff_obs.unit == "bps"

    stance = build_thai_market_stance(observables)
    assert stance["policy_spread_bps"] == 275.0


def test_bis_policy_rate_missing_th_rate():
    """Case 2: Missing TH rate -> diff observable must NOT be created, policy_spread_bps must be None."""
    mock_service = MagicMock()
    mock_service.get_treasury_yield_curve.return_value = TreasuryYieldCurveSnapshot(
        observation_date="2026-09-25", yields=(), spread_10y_2y_bps=None, spread_10y_3m_bps=None
    )
    mock_service.get_financial_stress.return_value = FinancialStressSnapshot(
        as_of_date="2026-09-25", published_at="2026-09-25", fsi_value=0.0, categories=(), trend_90d=()
    )
    # Only US is present
    policy_snapshot = GlobalPolicyRateSnapshot(
        as_of_date="2026-09-20",
        rates=(
            PolicyRateItem("US", 5.25, "Target Rate", "2024-07-31", "USD", "Fed"),
        ),
    )
    mock_service.get_global_policy_rates.return_value = policy_snapshot

    observables = build_rates_observables(terminal_data_service=mock_service)

    # US rate is created
    assert any(o.observable_id == "obs_us_policy_rate_bis" for o in observables)
    # TH rate and diff are NOT created
    assert not any(o.observable_id == "obs_thai_policy_rate_bis" for o in observables)
    assert not any(o.observable_id == "obs_diff_us_th_policy_rate_bis" for o in observables)

    stance = build_thai_market_stance(observables)
    assert stance["policy_spread_bps"] is None


def test_bis_policy_rate_stale_rate_no_spread():
    """Case 3: TH rate is stale -> diff observable must NOT be created, policy_spread_bps must be None."""
    mock_service = MagicMock()
    mock_service.get_treasury_yield_curve.return_value = TreasuryYieldCurveSnapshot(
        observation_date="2026-09-25", yields=(), spread_10y_2y_bps=None, spread_10y_3m_bps=None
    )
    mock_service.get_financial_stress.return_value = FinancialStressSnapshot(
        as_of_date="2026-09-25", published_at="2026-09-25", fsi_value=0.0, categories=(), trend_90d=()
    )
    stale_th_item = PolicyRateItem("TH", 2.50, "Repo Rate", "2023-09-27", "THB", "BOT", is_stale=True)

    policy_snapshot = GlobalPolicyRateSnapshot(
        as_of_date="2026-09-20",
        rates=(
            PolicyRateItem("US", 5.25, "Target Rate", "2024-07-31", "USD", "Fed"),
            stale_th_item,
        ),
    )
    mock_service.get_global_policy_rates.return_value = policy_snapshot

    observables = build_rates_observables(terminal_data_service=mock_service)

    th_obs = next(o for o in observables if o.observable_id == "obs_thai_policy_rate_bis")
    assert th_obs.is_valid is False
    assert th_obs.status == "stale"

    # diff observable must not be created when one side is stale
    assert not any(o.observable_id == "obs_diff_us_th_policy_rate_bis" for o in observables)

    stance = build_thai_market_stance(observables)
    assert stance["policy_spread_bps"] is None


def test_bis_policy_rate_negative_and_zero_spread():
    """Case 4: Negative spread (US 2.00 vs TH 2.50 -> -50 bps) and Zero spread (2.50 vs 2.50 -> 0 bps)."""
    mock_service = MagicMock()
    mock_service.get_treasury_yield_curve.return_value = TreasuryYieldCurveSnapshot(
        observation_date="2026-09-25", yields=(), spread_10y_2y_bps=None, spread_10y_3m_bps=None
    )
    mock_service.get_financial_stress.return_value = FinancialStressSnapshot(
        as_of_date="2026-09-25", published_at="2026-09-25", fsi_value=0.0, categories=(), trend_90d=()
    )

    # Negative spread: US 2.00, TH 2.50 -> -50.0 bps
    policy_neg = GlobalPolicyRateSnapshot(
        as_of_date="2026-09-20",
        rates=(
            PolicyRateItem("US", 2.00, "Target Rate", "2024-07-31", "USD", "Fed"),
            PolicyRateItem("TH", 2.50, "Repo Rate", "2023-09-27", "THB", "BOT"),
        ),
    )
    mock_service.get_global_policy_rates.return_value = policy_neg
    obs_neg = build_rates_observables(terminal_data_service=mock_service)
    diff_neg = next(o for o in obs_neg if o.observable_id == "obs_diff_us_th_policy_rate_bis")
    assert diff_neg.value == "-50.0"
    stance_neg = build_thai_market_stance(obs_neg)
    assert stance_neg["policy_spread_bps"] == -50.0

    # Zero spread: US 2.50, TH 2.50 -> 0.0 bps
    policy_zero = GlobalPolicyRateSnapshot(
        as_of_date="2026-09-20",
        rates=(
            PolicyRateItem("US", 2.50, "Target Rate", "2024-07-31", "USD", "Fed"),
            PolicyRateItem("TH", 2.50, "Repo Rate", "2023-09-27", "THB", "BOT"),
        ),
    )
    mock_service.get_global_policy_rates.return_value = policy_zero
    obs_zero = build_rates_observables(terminal_data_service=mock_service)
    diff_zero = next(o for o in obs_zero if o.observable_id == "obs_diff_us_th_policy_rate_bis")
    assert diff_zero.value == "0.0"
    stance_zero = build_thai_market_stance(obs_zero)
    assert stance_zero["policy_spread_bps"] == 0.0


def test_ag215_regression_baseline_fixture_execution():
    """Verify that ag215_regression_baseline.json reproduces the behavior without live network calls."""
    fixture_path = Path(__file__).resolve().parent.parent.parent / "fixtures" / "macro" / "ag215_regression_baseline.json"
    assert fixture_path.exists()
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data["scenarios"]
    assert "ag215_complete" in scenarios
    assert "missing_th_rate" in scenarios
    assert "stale_rate" in scenarios
    assert "negative_spread" in scenarios
    assert "zero_spread" in scenarios

