import pytest
from schemas.macro_schemas import AssetAllocationView, AssetStance, MarketObservable, _validate_supporting_data_against_registry
from schemas.warning_registry import WarningMessage, SUPPORTING_DATA_MISMATCH


def test_mismatch_warning_appears_once_per_asset():
    obs1 = MarketObservable(
        observable_id="obs_1", asset_bucket="equities", region="US", indicator="US Equities Score",
        value="10.0", unit="Score", observed_at="2026-07-05", source_file="test.py", is_valid=True
    )
    obs2 = MarketObservable(
        observable_id="obs_2", asset_bucket="equities", region="US", indicator="US Equities P/E",
        value="20.0", unit="Ratio", observed_at="2026-07-05", source_file="test.py", is_valid=True
    )
    obs3 = MarketObservable(
        observable_id="obs_3", asset_bucket="equities", region="US", indicator="US Equities Yield",
        value="5.0", unit="%", observed_at="2026-07-05", source_file="test.py", is_valid=True
    )
    registry = {"obs_1": obs1, "obs_2": obs2, "obs_3": obs3}
    
    # 3 mismatched numbers (>10x difference) for 3 indicators
    supporting_data = [
        "US Equities Score = 500.0",
        "US Equities P/E = 5000.0",
        "US Equities Yield = 500.0"
    ]
    warnings_list = []
    _validate_supporting_data_against_registry(supporting_data, ["obs_1", "obs_2", "obs_3"], registry, warnings_list)
    
    mismatch_warnings = [w for w in warnings_list if SUPPORTING_DATA_MISMATCH in w]
    assert len(mismatch_warnings) == 1, f"Expected exactly 1 mismatch warning, got {len(mismatch_warnings)}: {mismatch_warnings}"


def test_no_false_positive_for_ratio_and_zscore_same_item():
    obs = MarketObservable(
        observable_id="obs_pair_spy_vgk", asset_bucket="equities", region="Global", indicator="SPY vs VGK Ratio",
        value="8.3355", unit="Ratio", observed_at="2026-07-05", source_file="test.py", is_valid=True
    )
    registry = {"obs_pair_spy_vgk": obs}
    
    # Clause splitting should separate the ratio and Z-score so 0.13 is not compared to 8.3355
    supporting_data = ["SPY/VGK Ratio 8.3355, Z-score 0.13"]
    warnings_list = []
    _validate_supporting_data_against_registry(supporting_data, ["obs_pair_spy_vgk"], registry, warnings_list)
    
    mismatch_warnings = [w for w in warnings_list if SUPPORTING_DATA_MISMATCH in w]
    assert len(mismatch_warnings) == 0, f"Expected 0 mismatch warnings for auxiliary z-score, got {mismatch_warnings}"


def test_real_mismatch_still_detected():
    obs = MarketObservable(
        observable_id="obs_cpi", asset_bucket="fixed_income", region="US", indicator="US CPI YoY",
        value="4.17", unit="%", observed_at="2026-07-05", source_file="test.py", is_valid=True
    )
    registry = {"obs_cpi": obs}
    
    # Real mismatch: 45.5 vs 4.17 (>10x difference and not a small integer <= 100)
    supporting_data = ["US CPI YoY = 45.5%"]
    warnings_list = []
    _validate_supporting_data_against_registry(supporting_data, ["obs_cpi"], registry, warnings_list)
    
    mismatch_warnings = [w for w in warnings_list if SUPPORTING_DATA_MISMATCH in w]
    assert len(mismatch_warnings) == 1, "Expected real mismatch (>10x difference) to be detected"


def test_no_false_positive_generic_token_collision():
    obs1 = MarketObservable(
        observable_id="obs_treasury_yield", asset_bucket="fixed_income", region="US", indicator="10-Year Treasury Yield",
        value="4.485", unit="%", observed_at="2026-07-05", source_file="test.py", is_valid=True
    )
    obs2 = MarketObservable(
        observable_id="obs_yield_spread", asset_bucket="fixed_income", region="US", indicator="10Y-2Y Yield Spread",
        value="0.35", unit="%", observed_at="2026-07-05", source_file="test.py", is_valid=True
    )
    registry = {"obs_treasury_yield": obs1, "obs_yield_spread": obs2}

    # Clause has token 'yield' matching both obs1 (4.485) and obs2 (0.35).
    # Since 4.485 matches obs1 perfectly, it should NOT flag a mismatch even though 4.485 is >10x of 0.35.
    supporting_data = ["10Y Treasury Yield 4.485%, 10Y-2Y Yield Spread 0.35%"]
    warnings_list = []
    _validate_supporting_data_against_registry(supporting_data, ["obs_treasury_yield", "obs_yield_spread"], registry, warnings_list)

    mismatch_warnings = [w for w in warnings_list if SUPPORTING_DATA_MISMATCH in w]
    assert len(mismatch_warnings) == 0, f"Expected 0 mismatch warnings when at least one matched indicator is close, got {mismatch_warnings}"


def test_real_mismatch_when_no_indicator_matches():
    obs1 = MarketObservable(
        observable_id="obs_treasury_yield", asset_bucket="fixed_income", region="US", indicator="10-Year Treasury Yield",
        value="4.485", unit="%", observed_at="2026-07-05", source_file="test.py", is_valid=True
    )
    obs2 = MarketObservable(
        observable_id="obs_yield_spread", asset_bucket="fixed_income", region="US", indicator="10Y-2Y Yield Spread",
        value="0.35", unit="%", observed_at="2026-07-05", source_file="test.py", is_valid=True
    )
    registry = {"obs_treasury_yield": obs1, "obs_yield_spread": obs2}

    # Number is 500.0%, which is >10x away from both 4.485 and 0.35.
    supporting_data = ["10Y Treasury Yield 500.0%"]
    warnings_list = []
    _validate_supporting_data_against_registry(supporting_data, ["obs_treasury_yield"], registry, warnings_list)

    mismatch_warnings = [w for w in warnings_list if SUPPORTING_DATA_MISMATCH in w]
    assert len(mismatch_warnings) == 1, "Expected mismatch warning when no matched indicator is close"


def test_missing_observable_refs_retryable_and_source_refs_ordering():
    from schemas.warning_registry import MISSING_OBSERVABLE_REFS, RETRYABLE_CRITICAL_IDS, SOURCE_REF_PENALTY
    from schemas.macro_schemas import MacroStrategyDirection, AssetAllocationView, AssetStance, EconomicState, MarketObservable

    assert MISSING_OBSERVABLE_REFS in RETRYABLE_CRITICAL_IDS, "MISSING_OBSERVABLE_REFS must be a retryable critical ID"

    obs_valid = MarketObservable(
        observable_id="obs_valid",
        asset_bucket="equities",
        region="US",
        indicator="Forward PE",
        value="20",
        unit="x",
        observed_at="2026-07-06",
        source_file="us_sector.md",
        is_valid=True
    )

    obs_valid_2 = MarketObservable(
        observable_id="obs_valid_2",
        asset_bucket="equities",
        region="US",
        indicator="US Stock Index",
        value="6000",
        unit="USD",
        observed_at="2026-07-06",
        source_file="global_macro.md",
        is_valid=True
    )

    asset_missing_obs = AssetAllocationView(
        asset_class="US Equities",
        asset_bucket="equities",
        stance=AssetStance.OVERWEIGHT,
        confidence="high",
        rationale="Strong tech earnings",
        supporting_data=["US Stock Index Forward Earnings Yield 4.63%"],
        observable_refs=[],  # Empty! Should trigger MISSING_OBSERVABLE_REFS
        source_refs=["us_sector.md", "global_macro.md"]
    )

    asset_ordering = AssetAllocationView(
        asset_class="Global Equities",
        asset_bucket="equities",
        stance=AssetStance.OVERWEIGHT,
        confidence="high",
        rationale="Global growth",
        supporting_data=["Forward P/E Ratio = 20x, US Stock Index = 6000 USD"],
        observable_refs=["obs_valid", "obs_valid_2"],
        source_refs=[]       # Empty! Should be backfilled before SOURCE_REF_PENALTY check
    )

    strategy = MacroStrategyDirection(
        evaluated_at="2026-07-06T12:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[asset_missing_obs, asset_ordering],
        focus_themes=["Tech Growth"],
        conviction_level="high",
        conviction_rationale="Strong data",
        quant_narrative_alignment="aligned",
        source_files=["global_macro.md", "us_sector.md", "fred_macro.md"]
    )

    revalidated = strategy.revalidate_with_registry({"obs_valid": obs_valid, "obs_valid_2": obs_valid_2})
    rev_missing_obs = revalidated.asset_allocation[0]
    rev_ordering = revalidated.asset_allocation[1]

    # 1. Check that MISSING_OBSERVABLE_REFS warning is raised for asset without observables
    missing_warnings = [w for w in rev_missing_obs.validation_warnings if MISSING_OBSERVABLE_REFS in w]
    assert len(missing_warnings) == 1, f"Expected MISSING_OBSERVABLE_REFS warning, got {rev_missing_obs.validation_warnings}"

    # 2. Check source_refs backfill ordering fix for asset with valid observables but omitted source_refs
    assert len(rev_ordering.source_refs) > 0, "source_refs should be backfilled"
    penalty_warnings = [w for w in rev_ordering.validation_warnings if SOURCE_REF_PENALTY in w]
    assert len(penalty_warnings) == 0, f"Expected 0 SOURCE_REF_PENALTY warnings due to ordering fix, got {penalty_warnings}"
    assert rev_ordering.confidence == "high", f"Confidence should remain high when valid observable is present and source_refs is backfilled cleanly, got {rev_ordering.confidence}"


