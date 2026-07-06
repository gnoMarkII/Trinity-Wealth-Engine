"""Unit tests for structured_output_retry layer."""
from dataclasses import dataclass, field
from unittest.mock import MagicMock
from schemas.macro_schemas import (
    AssetAllocationView,
    AssetStance,
    EconomicState,
    MacroStrategyDirection,
)
from schemas.warning_registry import (
    COVERAGE_WARNING_INCOMPLETE,
    DEFENSIVE_LOW_SUPPORTING_DATA,
    WarningMessage,
)
from validators.structured_output_retry import (
    explicit_fallback_placeholders,
    invoke_with_retry,
)


def test_explicit_fallback_placeholders_completes_5_buckets_and_clears_coverage_warning():
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-05T00:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[
            AssetAllocationView(
                asset_class="US Equities",
                asset_bucket="equities",
                stance=AssetStance.OVERWEIGHT,
                rationale="S&P 500",
                supporting_data=["S&P 500 = 6000"],
                observable_refs=["obs_1"],
            ),
            AssetAllocationView(
                asset_class="Gold",
                asset_bucket="commodities",
                stance=AssetStance.OVERWEIGHT,
                rationale="Gold",
                supporting_data=["Gold = 3000 USD/oz"],
                observable_refs=["obs_1"],
            ),
        ],
        focus_themes=["Test"],
        conviction_level="high",
        conviction_rationale="Test",
        quant_narrative_alignment="aligned",
        validation_warnings=[str(WarningMessage(COVERAGE_WARNING_INCOMPLETE, {"count": "2"}))],
    )

    result = explicit_fallback_placeholders(direction, observable_registry={})
    assert len(result.asset_allocation) >= 5
    buckets = {a.asset_bucket for a in result.asset_allocation if a.asset_bucket}
    assert {"equities", "fixed_income", "commodities", "fx", "cash"}.issubset(buckets)
    assert not any("COVERAGE_WARNING_INCOMPLETE" in str(w) for w in result.validation_warnings)
    assert any("SYSTEM_PLACEHOLDER" in str(w) for a in result.asset_allocation for w in a.validation_warnings)


def test_invoke_with_retry_does_not_retry_on_non_retryable_critical():
    w = str(WarningMessage(DEFENSIVE_LOW_SUPPORTING_DATA))
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-05T00:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[
            AssetAllocationView(asset_class="US Equities", asset_bucket="equities", stance=AssetStance.OVERWEIGHT, rationale="S&P 500", supporting_data=["S&P 500"], observable_refs=["obs_1"]),
            AssetAllocationView(asset_class="Fixed Income", asset_bucket="fixed_income", stance=AssetStance.NEUTRAL, rationale="10Y Yield", supporting_data=["10Y Yield"], observable_refs=["obs_1"]),
            AssetAllocationView(asset_class="Gold", asset_bucket="commodities", stance=AssetStance.OVERWEIGHT, rationale="Gold", supporting_data=["Gold"], observable_refs=["obs_1"]),
            AssetAllocationView(asset_class="FX / Currencies", asset_bucket="fx", stance=AssetStance.NEUTRAL, rationale="USD/THB", supporting_data=["USD/THB"], observable_refs=["obs_1"]),
            AssetAllocationView(asset_class="Cash", asset_bucket="cash", stance=AssetStance.NEUTRAL, rationale="Cash", supporting_data=["Cash"], observable_refs=["obs_1"]),
        ],
        focus_themes=["Test"],
        conviction_level="high",
        conviction_rationale="Test",
        quant_narrative_alignment="aligned",
        validation_warnings=[w],
    )

    mock_structured = MagicMock()
    mock_structured.invoke.return_value = direction
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    res_dir, res_quality = invoke_with_retry(mock_model, [], MacroStrategyDirection, {}, max_retries=2)
    
    assert mock_structured.invoke.call_count == 1
    assert res_quality.should_retry is False
    assert DEFENSIVE_LOW_SUPPORTING_DATA in res_quality.critical_ids
    assert DEFENSIVE_LOW_SUPPORTING_DATA not in res_quality.retryable_ids


def test_invoke_with_retry_retries_on_retryable_critical():
    w_bad = str(WarningMessage(COVERAGE_WARNING_INCOMPLETE, {"count": "3"}))
    bad_dir = MacroStrategyDirection(
        evaluated_at="2026-07-05T00:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[],
        focus_themes=["Test"],
        conviction_level="high",
        conviction_rationale="Test",
        quant_narrative_alignment="aligned",
        validation_warnings=[w_bad],
    )
    good_dir = MacroStrategyDirection(
        evaluated_at="2026-07-05T00:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[
            AssetAllocationView(asset_class="US Equities", asset_bucket="equities", stance=AssetStance.OVERWEIGHT, rationale="S&P 500", supporting_data=["S&P 500"], observable_refs=["obs_1"]),
            AssetAllocationView(asset_class="Fixed Income", asset_bucket="fixed_income", stance=AssetStance.NEUTRAL, rationale="10Y Yield", supporting_data=["10Y Yield"], observable_refs=["obs_1"]),
            AssetAllocationView(asset_class="Gold", asset_bucket="commodities", stance=AssetStance.OVERWEIGHT, rationale="Gold", supporting_data=["Gold"], observable_refs=["obs_1"]),
            AssetAllocationView(asset_class="FX / Currencies", asset_bucket="fx", stance=AssetStance.NEUTRAL, rationale="USD/THB", supporting_data=["USD/THB"], observable_refs=["obs_1"]),
            AssetAllocationView(asset_class="Cash", asset_bucket="cash", stance=AssetStance.NEUTRAL, rationale="Cash", supporting_data=["Cash"], observable_refs=["obs_1"]),
        ],
        focus_themes=["Test"],
        conviction_level="high",
        conviction_rationale="Test",
        quant_narrative_alignment="aligned",
        validation_warnings=[],
    )

    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = [bad_dir, good_dir]
    mock_model = MagicMock()
    mock_model.with_structured_output.return_value = mock_structured

    res_dir, res_quality = invoke_with_retry(mock_model, [], MacroStrategyDirection, {}, max_retries=1)
    
    assert mock_structured.invoke.call_count == 2
    assert res_quality.should_retry is False
    assert len(res_dir.asset_allocation) == 5
