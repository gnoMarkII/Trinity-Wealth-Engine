import pytest
from schemas.macro_schemas import MarketObservable
from tools.macro.derived_ratios import build_derived_pair_observables
from tools.macro.risk_analytics import build_risk_correlation_observables


def test_derived_pair_ratios_with_metadata():
    obs_qqq = MarketObservable(
        observable_id="obs_qqq",
        asset_bucket="equities",
        region="US",
        indicator="Invesco QQQ Trust",
        value="480.00",
        unit="USD",
        observed_at="2026-07-05",
        source_file="test.py",
        is_valid=True
    )
    obs_spy = MarketObservable(
        observable_id="obs_spy",
        asset_bucket="equities",
        region="US",
        indicator="SPDR S&P 500 ETF Trust",
        value="550.00",
        unit="USD",
        observed_at="2026-07-05",
        source_file="test.py",
        is_valid=True
    )
    
    results = build_derived_pair_observables(existing_observables=[obs_qqq, obs_spy], use_mock_fallback=True)
    assert len(results) > 0
    qqq_spy_pair = next((o for o in results if o.observable_id == "obs_pair_qqq_spy"), None)
    assert qqq_spy_pair is not None
    assert qqq_spy_pair.is_valid is True
    assert qqq_spy_pair.metadata is not None
    assert qqq_spy_pair.metadata.get("long_ticker") == "QQQ"
    assert qqq_spy_pair.metadata.get("short_ticker") == "SPY"
    assert "ratio" in qqq_spy_pair.metadata
    assert "z_score_1y" in qqq_spy_pair.metadata

    # In production default (use_mock_fallback=False), missing historical series should mark as invalid
    def fail_calc(long_sym, short_sym): return None
    results_prod = build_derived_pair_observables(existing_observables=[obs_qqq, obs_spy], use_mock_fallback=False, historical_ratio_calculator=fail_calc)
    qqq_spy_prod = next((o for o in results_prod if o.observable_id == "obs_pair_qqq_spy"), None)
    assert qqq_spy_prod is not None
    assert qqq_spy_prod.is_valid is False
    assert "Missing real market historical price series" in qqq_spy_prod.stale_reason


def test_risk_correlation_guardrail_insufficient_days():
    # Test calculator returning insufficient overlapping days (< 45)
    def fake_calc(a1, a2, window=60):
        return {"correlation": 0.45, "overlapping_days": 30}

    results = build_risk_correlation_observables(correlation_calculator=fake_calc)
    spy_tlt = next((o for o in results if o.observable_id == "obs_corr_spy_tlt_60d"), None)
    assert spy_tlt is not None
    assert spy_tlt.is_valid is False
    assert "45" in spy_tlt.stale_reason
    assert spy_tlt.metadata.get("is_valid") is False


def test_risk_correlation_valid_and_breakdown():
    # Test calculator returning valid overlapping days (>= 45) and breakdown (> 0.30)
    def fake_calc(a1, a2, window=60):
        return {"correlation": 0.35, "overlapping_days": 50}

    results = build_risk_correlation_observables(correlation_calculator=fake_calc)
    spy_tlt = next((o for o in results if o.observable_id == "obs_corr_spy_tlt_60d"), None)
    assert spy_tlt is not None
    assert spy_tlt.is_valid is True
    assert spy_tlt.value == "0.35"
    assert spy_tlt.metadata.get("is_breakdown") is True


def test_etf_price_mapping_does_not_use_index_levels():
    # Test that ^GSPC index level (7483.24) is NOT mistakenly mapped to SPY ETF price
    obs_gspc = MarketObservable(
        observable_id="^GSPC",
        asset_bucket="equities",
        region="US",
        indicator="S&P 500 Index Level",
        value="7483.24",
        unit="Index",
        observed_at="2026-07-05",
        source_file="test.py",
        is_valid=True
    )
    obs_qqq = MarketObservable(
        observable_id="obs_qqq",
        asset_bucket="equities",
        region="US",
        indicator="Invesco QQQ Trust",
        value="508.99",
        unit="USD",
        observed_at="2026-07-05",
        source_file="test.py",
        is_valid=True
    )
    
    # If price getter returns true ETF price for SPY (550.0)
    def mock_price_getter(sym):
        if sym == "SPY":
            return 550.00
        return None

    results = build_derived_pair_observables(
        existing_observables=[obs_gspc, obs_qqq],
        price_getter=mock_price_getter,
        use_mock_fallback=True
    )
    qqq_spy = next((o for o in results if o.observable_id == "obs_pair_qqq_spy"), None)
    assert qqq_spy is not None
    # If 7483.24 was used, ratio would be 508.99 / 7483.24 = 0.068
    # With correct 550.00 ETF price, ratio is 508.99 / 550.00 = 0.925
    val_float = float(qqq_spy.value)
    assert 0.8 < val_float < 1.1, f"Ratio {val_float} should be realistic ETF price ratio around ~0.92, not index level ratio!"

