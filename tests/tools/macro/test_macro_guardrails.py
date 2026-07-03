import pytest
from pydantic import ValidationError
from schemas.macro_schemas import (
    MarketObservable,
    AssetAllocationView,
    PairTradeStrategy,
    MacroStrategyDirection,
    AssetStance,
    EconomicState,
)


def test_market_observable_date_validation():
    # Valid date format YYYY-MM-DD
    obs = MarketObservable(
        observable_id="us_10y_yield",
        asset_bucket="fixed_income",
        region="US",
        indicator="10Y Treasury Yield",
        value="4.40",
        unit="%",
        observed_at="2026-07-03",
        source_file="fred_data.md",
    )
    assert obs.observed_at == "2026-07-03"

    # Invalid date format should raise ValidationError
    with pytest.raises(ValidationError):
        MarketObservable(
            observable_id="us_10y_yield",
            asset_bucket="fixed_income",
            region="US",
            indicator="10Y Treasury Yield",
            value="4.40",
            unit="%",
            observed_at="03-07-2026",  # DD-MM-YYYY is invalid
            source_file="fred_data.md",
        )


def test_macro_strategy_direction_registry_revalidation():
    obs_valid = MarketObservable(
        observable_id="obs_us_growth_1",
        asset_bucket="equities",
        region="US",
        indicator="Quant Score US",
        value="+0.85",
        unit="score",
        observed_at="2026-07-03",
        source_file="global_macro.md",
        is_valid=True,
    )

    obs_invalid = MarketObservable(
        observable_id="obs_us_growth_1",
        asset_bucket="equities",
        region="US",
        indicator="Quant Score US",
        value="+0.85",
        unit="score",
        observed_at="2026-07-03",
        source_file="global_macro.md",
        is_valid=False,
        stale_reason="Proxy data without direct hard data verification",
    )

    def _get_assets():
        return [
            AssetAllocationView(
                asset_class="Equities US Tech Growth",
                stance=AssetStance.OVERWEIGHT,
                rationale="Strong AI growth fundamentals",
                confidence="high",
                data_confidence="high",
                implementation_confidence="high",
                supporting_data=["Quant Score US = +0.85"],
                observable_refs=["obs_us_growth_1"],
            ),
            AssetAllocationView(
                asset_class="Fixed Income Short Duration",
                stance=AssetStance.NEUTRAL,
                rationale="Stable yield environment",
                confidence="high",
                data_confidence="high",
                implementation_confidence="high",
                supporting_data=["Yield 2Y = 4.20%"],
                source_refs=["fred_data.md", "global_macro.md"],
            ),
            AssetAllocationView(
                asset_class="Commodities Gold",
                stance=AssetStance.NEUTRAL,
                rationale="Hedge against inflation",
                confidence="high",
                data_confidence="high",
                implementation_confidence="high",
                supporting_data=["Real yield = 1.5%"],
                source_refs=["fred_data.md", "global_macro.md"],
            ),
            AssetAllocationView(
                asset_class="FX USD",
                stance=AssetStance.NEUTRAL,
                rationale="Stable currency",
                confidence="high",
                data_confidence="high",
                implementation_confidence="high",
                supporting_data=["DXY = 104.5"],
                source_refs=["fred_data.md", "global_macro.md"],
            ),
            AssetAllocationView(
                asset_class="Cash USD",
                stance=AssetStance.NEUTRAL,
                rationale="Liquidity buffer",
                confidence="high",
                data_confidence="high",
                implementation_confidence="high",
                supporting_data=["Rate = 5.0%"],
                source_refs=["fred_data.md", "global_macro.md"],
            ),
        ]

    # 1. With invalid/proxy registry observable, source_refs inferred from report -> penalty applied
    direction_invalid = MacroStrategyDirection(
        evaluated_at="2026-07-03T00:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=_get_assets(),
        focus_themes=["AI", "Tech"],
        conviction_level="high",
        conviction_rationale="Solid hard data support across all core classes",
        source_files=["fred_data.md", "global_macro.md"],
        quant_narrative_alignment="aligned",
        observable_registry={"obs_us_growth_1": obs_invalid},
    )
    assert direction_invalid.asset_allocation[0].confidence == "medium"
    assert any("Source Reference Penalty" in w for w in direction_invalid.asset_allocation[0].validation_warnings)

    # 2. When created or validated with valid registry, no penalty is applied -> high confidence preserved
    direction_valid = MacroStrategyDirection(
        evaluated_at="2026-07-03T00:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=_get_assets(),
        focus_themes=["AI", "Tech"],
        conviction_level="high",
        conviction_rationale="Solid hard data support across all core classes",
        source_files=["fred_data.md", "global_macro.md"],
        quant_narrative_alignment="aligned",
        observable_registry={"obs_us_growth_1": obs_valid},
    )
    assert direction_valid.asset_allocation[0].confidence == "high"
    assert not any("Source Reference Penalty" in w for w in direction_valid.asset_allocation[0].validation_warnings)

    # 3. Test revalidate_with_registry idempotency without duplicate warnings
    revalidated = direction_valid.revalidate_with_registry({"obs_us_growth_1": obs_valid})
    assert revalidated.asset_allocation[0].confidence == "high"
    assert len(revalidated.validation_warnings) == len(direction_valid.validation_warnings)


def test_contradiction_guardrail_equities_and_bonds():
    eq_asset = AssetAllocationView(
        asset_class="Equities Growth",
        stance=AssetStance.OVERWEIGHT,
        rationale="bullish on equities",
        confidence="high",
        supporting_data=["GDP = 3.0%"],
        source_refs=["doc1.md", "doc2.md"],
    )
    bond_asset = AssetAllocationView(
        asset_class="Long Duration Treasuries",
        stance=AssetStance.OVERWEIGHT,
        rationale="bullish on bonds",
        confidence="high",
        supporting_data=["Yield = 4.0%"],
        source_refs=["doc1.md", "doc2.md"],
    )

    # Without hedge/reconciliation keywords, both Overweight Equities and Bonds should downgrade
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-03T00:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[eq_asset, bond_asset],
        focus_themes=["Growth", "Duration"],
        conviction_level="high",
        conviction_rationale="Dual overweight without hedge",
        quant_narrative_alignment="aligned",
    )
    assert direction.asset_allocation[0].confidence == "medium"
    assert direction.asset_allocation[1].confidence == "medium"
    assert any("Contradiction Warning" in w for w in direction.validation_warnings)


def test_pair_trade_requires_mandatory_execution_fields_and_valid_observable_refs():
    obs_ratio = MarketObservable(
        observable_id="obs_ratio_hyg_lqd",
        asset_bucket="risk",
        region="US",
        indicator="HYG/LQD Relative Price Ratio",
        value="0.80",
        unit="ratio",
        observed_at="2026-07-03",
        source_file="Global_Macro_Snapshot_2026-07-03.md",
        provider="Derived",
        is_valid=True,
    )
    valid_trade = PairTradeStrategy(
        long_leg="High Yield Credit",
        short_leg="Investment Grade Credit",
        thesis="ใช้ beta เชิงเครดิตจาก ratio HYG/LQD 0.80 เพื่อเล่น risk-on spread",
        catalyst="Credit spread ratio 0.80 เริ่มฟื้น",
        risk="หาก spread กลับทิศ 3% ให้ลดความเสี่ยง",
        time_horizon="1-3 Months",
        confidence="high",
        supporting_data=["HYG/LQD ratio = 0.80"],
        observable_refs=["obs_ratio_hyg_lqd"],
        instrument_proxy="Long HYG / Short LQD via ETFs",
        hedge_ratio="0.80x HYG per 1.00x LQD",
        stop_loss_trigger="Stop loss if ratio falls below 0.76 (-5.0%)",
        target_gain_or_rebalance="Take profit if ratio reaches 0.86 (+7.5%)",
        max_drawdown_limit="Max drawdown limit 4.0%",
    )
    missing_exec_trade = PairTradeStrategy(
        long_leg="High Yield Credit",
        short_leg="Investment Grade Credit",
        thesis="ratio HYG/LQD 0.80",
        catalyst="ratio 0.80",
        risk="drawdown 4.0%",
        time_horizon="1-3 Months",
        confidence="high",
        supporting_data=["HYG/LQD ratio = 0.80"],
        observable_refs=["obs_ratio_hyg_lqd"],
        instrument_proxy="Long HYG / Short LQD via ETFs",
        hedge_ratio="0.80x HYG per 1.00x LQD",
        stop_loss_trigger="Stop loss if ratio falls below 0.76 (-5.0%)",
        target_gain_or_rebalance="Take profit if ratio reaches 0.86 (+7.5%)",
    )
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-03T00:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[],
        focus_themes=["Credit relative value"],
        conviction_level="medium",
        conviction_rationale="Relative spread evidence is available",
        quant_narrative_alignment="aligned",
        pair_trades=[valid_trade, missing_exec_trade],
        observable_registry={"obs_ratio_hyg_lqd": obs_ratio},
    )

    assert len(direction.pair_trades) == 1
    assert direction.pair_trades[0].confidence == "medium"
    assert any("Pair Trade Graceful Downgrade" in w for w in direction.pair_trades[0].validation_warnings)
    assert any("Graceful Drop: 1 pair trade(s)" in w for w in direction.validation_warnings)


def test_why_not_high_is_overridden_when_asset_is_downgraded():
    asset = AssetAllocationView(
        asset_class="Gold",
        stance=AssetStance.OVERWEIGHT,
        rationale="ทองคำได้แรงหนุนจาก geopolitical tension",
        confidence="high",
        supporting_data=["Gold = 4186 USD/oz"],
        source_refs=["Global_Macro_Snapshot_2026-07-03.md"],
        why_not_high="ไม่มีเหตุผลที่ความมั่นใจไม่ถึงระดับสูง",
    )
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-03T00:00:00Z",
        overall_regime=EconomicState.REFLATION,
        asset_allocation=[asset],
        focus_themes=["Gold"],
        conviction_level="high",
        conviction_rationale="Gold hedge",
        quant_narrative_alignment="aligned",
    )

    gold = next(a for a in direction.asset_allocation if a.asset_class == "Gold")
    assert gold.confidence == "medium"
    assert "ไม่มีเหตุผล" not in gold.why_not_high
