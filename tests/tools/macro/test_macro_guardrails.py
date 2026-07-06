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
                asset_bucket="equities",
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
                asset_bucket="fixed_income",
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
                asset_bucket="commodities",
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
                asset_bucket="fx",
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
                asset_bucket="cash",
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
    assert any("SOURCE_REF_PENALTY" in w or "Source Reference Penalty" in w for w in direction_invalid.asset_allocation[0].validation_warnings)

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
    assert not any("SOURCE_REF_PENALTY" in w or "Source Reference Penalty" in w for w in direction_valid.asset_allocation[0].validation_warnings)

    # 3. Test revalidate_with_registry idempotency without duplicate warnings.
    # One valid observable source remains a single-source view, so MEDIUM is expected.
    revalidated = direction_valid.revalidate_with_registry({"obs_us_growth_1": obs_valid})
    assert revalidated.asset_allocation[0].confidence == "medium"
    assert len(revalidated.validation_warnings) == len(direction_valid.validation_warnings)


def test_contradiction_guardrail_equities_and_bonds():
    eq_asset = AssetAllocationView(
        asset_class="Equities Growth",
        asset_bucket="equities",
        stance=AssetStance.OVERWEIGHT,
        rationale="bullish on equities",
        confidence="high",
        supporting_data=["GDP = 3.0%"],
        source_refs=["doc1.md", "doc2.md"],
    )
    bond_asset = AssetAllocationView(
        asset_class="Long Duration Treasuries",
        asset_bucket="fixed_income",
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
    assert any("CONVICTION_CONTRADICTION" in w or "Contradiction Warning" in w for w in direction.validation_warnings)


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
    obs_spread = MarketObservable(
        observable_id="obs_spread_hyg_lqd",
        asset_bucket="risk",
        region="US",
        indicator="HYG/LQD Relative Spread",
        value="1.20",
        unit="spread",
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
        observable_refs=["obs_ratio_hyg_lqd", "obs_spread_hyg_lqd"],
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
        observable_refs=["obs_ratio_hyg_lqd", "obs_spread_hyg_lqd"],
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
        observable_registry={"obs_ratio_hyg_lqd": obs_ratio, "obs_spread_hyg_lqd": obs_spread},
    )

    assert len(direction.pair_trades) == 1
    assert direction.pair_trades[0].confidence == "medium"
    assert any("PT_GRACEFUL_DOWNGRADE" in w or "Pair Trade Graceful Downgrade" in w for w in direction.pair_trades[0].validation_warnings)
    assert any("GRACEFUL_DROP_PAIR_TRADES" in w or "Graceful Drop:" in w for w in direction.validation_warnings)


def test_why_not_high_is_overridden_when_asset_is_downgraded():
    asset = AssetAllocationView(
        asset_class="Gold",
        asset_bucket="commodities",
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


def test_staticproxy_invalidity():
    from tools.macro.evaluation import _apply_validity
    from schemas.macro_schemas import MarketObservable
    obs = MarketObservable(
        observable_id="TH10Y",
        asset_bucket="fixed_income",
        region="TH",
        indicator="TH10Y",
        name="Thailand 10Y [StaticProxy]",
        value="2.65",
        unit="%",
        observed_at="2026-07-03",
        source_file="Country_Macro_Snapshot_2026-07-03.md",
        provider="StaticProxy",
    )
    valid_obs = _apply_validity(obs, "2026-07-03")
    assert valid_obs.is_valid is False
    assert valid_obs.confidence == "low"
    assert "Mock/static proxy" in str(valid_obs.stale_reason)


def test_gold_allowlist_cross_bucket():
    obs_dfii = MarketObservable(
        observable_id="obs_dfii10",
        indicator="DFII10",
        name="10-Year Treasury Inflation-Indexed Security",
        value="2.05",
        unit="%",
        observed_at="2026-07-03",
        source_file="Global_Macro_Snapshot_2026-07-03.md",
        source_section="US Real Yields",
        asset_bucket="fixed_income",
        region="US",
        confidence="high",
        provider="FRED",
        is_valid=True,
    )
    asset = AssetAllocationView(
        asset_class="Gold",
        asset_bucket="commodities",
        stance=AssetStance.OVERWEIGHT,
        rationale="Gold benefits from real yield DFII10 trends at 2.05%",
        confidence="high",
        supporting_data=["DFII10 = 2.05%"],
        observable_refs=["obs_dfii10"],
        source_refs=["Global_Macro_Snapshot_2026-07-03.md", "Regional_Macro_Snapshot_2026-07-03.md"],
    )
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-03T00:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[asset],
        focus_themes=["Gold"],
        conviction_level="high",
        conviction_rationale="Real yield anchoring",
        quant_narrative_alignment="aligned",
        observable_registry={"obs_dfii10": obs_dfii},
    )
    gold = next(a for a in direction.asset_allocation if a.asset_class == "Gold")
    assert gold.confidence == "high"
    assert "obs_dfii10" in gold.observable_refs


def test_defensive_low_when_supporting_data_empty():
    asset = AssetAllocationView(
        asset_class="US Equities",
        asset_bucket="equities",
        stance=AssetStance.OVERWEIGHT,
        rationale="S&P 500 index target around 5600 points with strong earnings growth.",
        confidence="high",
        supporting_data=[],
        observable_refs=[],
        source_refs=["Global_Macro_Snapshot_2026-07-03.md", "Regional_Macro_Snapshot_2026-07-03.md"],
    )
    assert len(asset.supporting_data) == 0
    assert asset.confidence == "low"
    assert any("DEFENSIVE_LOW_SUPPORTING_DATA" in str(w) for w in asset.validation_warnings)


def test_pair_trade_sizing_consistency():
    obs_ratio = MarketObservable(
        observable_id="obs_ratio_hyg_lqd",
        indicator="HYG/LQD",
        name="Credit Spread Ratio",
        value="0.80",
        unit="",
        observed_at="2026-07-03",
        source_file="Global_Macro_Snapshot_2026-07-03.md",
        source_section="Credit Spread",
        asset_bucket="fixed_income",
        region="US",
        confidence="high",
        provider="Derived",
        is_valid=True,
    )
    obs_spread = MarketObservable(
        observable_id="obs_spread_hyg_lqd",
        indicator="HYG/LQD relative spread",
        name="Credit Relative Spread",
        value="1.20",
        unit="spread",
        observed_at="2026-07-03",
        source_file="Global_Macro_Snapshot_2026-07-03.md",
        source_section="Credit Spread",
        asset_bucket="fixed_income",
        region="US",
        confidence="high",
        provider="Derived",
        is_valid=True,
    )
    trade = PairTradeStrategy(
        long_leg="High Yield Credit",
        short_leg="Investment Grade Credit",
        thesis="ใช้ beta เชิงเครดิตจาก ratio HYG/LQD 0.80 เพื่อเล่น risk-on spread",
        catalyst="Credit spread ratio 0.80 เริ่มฟื้น",
        risk="หาก spread กลับทิศ 3% ให้ลดความเสี่ยง",
        time_horizon="1-3 Months",
        confidence="high",
        sizing_guidance="high_risk_budget",
        supporting_data=["HYG/LQD ratio = 0.80"],
        observable_refs=["obs_ratio_hyg_lqd", "obs_spread_hyg_lqd"],
        instrument_proxy="Long HYG / Short LQD via ETFs",
        hedge_ratio="0.80x HYG per 1.00x LQD",
        stop_loss_trigger="Stop loss if ratio falls below 0.76 (-5.0%)",
        target_gain_or_rebalance="Take profit if ratio reaches 0.86 (+7.5%)",
        max_drawdown_limit="Max drawdown limit 4.0%",
    )
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-03T00:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[],
        focus_themes=["Credit relative value"],
        conviction_level="medium",
        conviction_rationale="Relative spread evidence is available",
        quant_narrative_alignment="aligned",
        pair_trades=[trade],
        observable_registry={"obs_ratio_hyg_lqd": obs_ratio, "obs_spread_hyg_lqd": obs_spread},
    )
    assert direction.pair_trades[0].confidence == "medium"
    assert direction.pair_trades[0].sizing_guidance == "medium_risk_budget"


def test_source_refs_are_not_backfilled_from_report_sources_without_valid_observable_support():
    obs = MarketObservable(
        observable_id="obs_gold",
        indicator="Gold Futures",
        value="4186",
        unit="USD/oz",
        observed_at="2026-07-03",
        source_file="Global_Macro_Snapshot_2026-07-03.md",
        asset_bucket="commodities",
        region="Global",
        provider="Yahoo",
        is_valid=True,
    )
    asset = AssetAllocationView(
        asset_class="Gold",
        asset_bucket="commodities",
        stance=AssetStance.OVERWEIGHT,
        rationale="Gold at 4186 USD/oz",
        confidence="high",
        supporting_data=["Gold = 4186 USD/oz"],
        observable_refs=["obs_gold"],
        source_refs=["Global_Macro_Snapshot_2026-07-03.md"],
    )
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-03T00:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[asset],
        focus_themes=["Gold"],
        conviction_level="high",
        conviction_rationale="Gold supported by price data",
        quant_narrative_alignment="aligned",
        regime_probabilities={"Goldilocks": "60%", "Reflation": "40%"},
        source_files=[
            "Global_Macro_Snapshot_2026-07-03.md",
            "Country_Macro_Snapshot_2026-07-03.md",
        ],
        observable_registry={"obs_gold": obs},
    )

    gold = next(a for a in direction.asset_allocation if a.asset_class == "Gold")
    assert gold.source_refs == ["Global_Macro_Snapshot_2026-07-03.md"]
    assert any("SINGLE_SOURCE_PENALTY" in w or "Single-Source Penalty" in w for w in gold.validation_warnings)


def test_gold_allowlist_requires_relevant_valid_observables_and_rejects_credit_spread():
    gold_obs = MarketObservable(
        observable_id="obs_gold",
        indicator="Gold Futures",
        value="4186",
        unit="USD/oz",
        observed_at="2026-07-03",
        source_file="Global_Macro_Snapshot_2026-07-03.md",
        asset_bucket="commodities",
        region="Global",
        provider="Yahoo",
        is_valid=True,
    )
    real_yield = MarketObservable(
        observable_id="obs_dfii10",
        indicator="DFII10 TIPS Real Yield",
        value="2.05",
        unit="%",
        observed_at="2026-07-03",
        source_file="Country_Macro_Snapshot_2026-07-03.md",
        asset_bucket="fixed_income",
        region="US",
        provider="FRED",
        is_valid=True,
    )
    credit_spread = MarketObservable(
        observable_id="obs_high_yield",
        indicator="High Yield Credit Spread",
        value="3.20",
        unit="%",
        observed_at="2026-07-03",
        source_file="Global_Macro_Snapshot_2026-07-03.md",
        asset_bucket="fixed_income",
        region="US",
        provider="FRED",
        is_valid=True,
    )
    asset = AssetAllocationView(
        asset_class="Gold",
        asset_bucket="commodities",
        stance=AssetStance.OVERWEIGHT,
        rationale="Gold relates to DFII10 real yield at 2.05%",
        confidence="high",
        supporting_data=["Gold = 4186 USD/oz", "DFII10 = 2.05%"],
        observable_refs=["obs_gold", "obs_dfii10", "obs_high_yield"],
    )
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-03T00:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[asset],
        focus_themes=["Gold"],
        conviction_level="high",
        conviction_rationale="Real yield anchoring",
        quant_narrative_alignment="aligned",
        observable_registry={
            "obs_gold": gold_obs,
            "obs_dfii10": real_yield,
            "obs_high_yield": credit_spread,
        },
    )

    gold = next(a for a in direction.asset_allocation if a.asset_class == "Gold")
    assert "obs_gold" in gold.observable_refs
    assert "obs_dfii10" in gold.observable_refs
    assert "obs_high_yield" not in gold.observable_refs
    assert len([r for r in gold.observable_refs if r in {"obs_gold", "obs_dfii10"}]) >= 2


def test_pair_trade_is_dropped_without_two_valid_observables_even_with_regime_probabilities():
    obs = MarketObservable(
        observable_id="obs_ratio_hyg_lqd",
        indicator="HYG/LQD ratio",
        value="0.80",
        unit="ratio",
        observed_at="2026-07-03",
        source_file="Global_Macro_Snapshot_2026-07-03.md",
        asset_bucket="fixed_income",
        region="US",
        provider="Derived",
        is_valid=True,
    )
    from schemas.macro_schemas import PairTradeStrategy
    trade = PairTradeStrategy(
        long_leg="Gold",
        short_leg="Nasdaq 100",
        thesis="Relative spread 1.5x supports the trade",
        catalyst="Spread above 1.5x",
        risk="Loss if spread falls 3%",
        time_horizon="1-3 Months",
        confidence="medium",
        sizing_guidance="medium_risk_budget",
        supporting_data=["Spread = 1.5x"],
        observable_refs=["obs_ratio_hyg_lqd"],
        instrument_proxy="Long GC=F / Short QQQ",
        hedge_ratio="1.0 : 1.0",
        stop_loss_trigger="-3.0%",
        target_gain_or_rebalance="+6.0%",
        max_drawdown_limit="-4.5%",
    )
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-03T00:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[
            AssetAllocationView(asset_class="US Equities", asset_bucket="equities", stance=AssetStance.OVERWEIGHT, rationale="S&P 500", supporting_data=["S&P 500"]),
            AssetAllocationView(asset_class="Fixed Income", asset_bucket="fixed_income", stance=AssetStance.NEUTRAL, rationale="10Y Yield", supporting_data=["10Y Yield"]),
            AssetAllocationView(asset_class="Gold", asset_bucket="commodities", stance=AssetStance.OVERWEIGHT, rationale="Gold", supporting_data=["Gold"]),
            AssetAllocationView(asset_class="FX / Currencies", asset_bucket="fx", stance=AssetStance.NEUTRAL, rationale="USD/THB", supporting_data=["USD/THB"]),
            AssetAllocationView(asset_class="Cash", asset_bucket="cash", stance=AssetStance.NEUTRAL, rationale="Cash", supporting_data=["Cash"]),
        ],
        pair_trades=[trade],
        focus_themes=["Pair"],
        conviction_level="medium",
        conviction_rationale="Test pair trade drop",
        quant_narrative_alignment="aligned",
        regime_probabilities={"Goldilocks": 0.8},
        observable_registry={"obs_ratio_hyg_lqd": obs},
    )
    assert len(direction.pair_trades) == 0
    assert any("GRACEFUL_DROP_PAIR_TRADES" in str(w) or "Graceful Drop:" in str(w) for w in direction.validation_warnings)


def test_empty_supporting_data_without_valid_observable_refs_is_low_confidence():
    asset = AssetAllocationView(
        asset_class="US Equities",
        asset_bucket="equities",
        stance=AssetStance.OVERWEIGHT,
        rationale="S&P 500 target 5600 points from earnings growth",
        confidence="medium",
        supporting_data=[],
        observable_refs=[],
        source_refs=[],
    )

    assert asset.confidence == "low"
    assert len(asset.supporting_data) == 0
    assert any("DEFENSIVE_LOW_SUPPORTING_DATA" in w for w in asset.validation_warnings)


def test_missing_asset_warns_and_registry_populates_provided_assets():
    fed_funds = MarketObservable(
        observable_id="obs_fedfunds",
        indicator="Fed Funds Rate",
        value="3.63",
        unit="%",
        observed_at="2026-07-04",
        source_file="Country_Macro_Snapshot_2026-07-04.md",
        asset_bucket="cash",
        region="US",
        provider="FRED",
        is_valid=True,
    )
    core_pce = MarketObservable(
        observable_id="obs_core_pce",
        indicator="Core PCE Inflation",
        value="3.41",
        unit="% YoY",
        observed_at="2026-07-04",
        source_file="Country_Macro_Snapshot_2026-07-04.md",
        asset_bucket="cash",
        region="US",
        provider="FRED",
        is_valid=True,
    )
    t_bill = MarketObservable(
        observable_id="obs_tbill",
        indicator="13-Week T-Bill Yield",
        value="3.67",
        unit="%",
        observed_at="2026-07-04",
        source_file="Global_Macro_Snapshot_2026-07-04.md",
        asset_bucket="cash",
        region="US",
        provider="Yahoo",
        is_valid=True,
    )

    direction = MacroStrategyDirection(
        evaluated_at="2026-07-04T10:05:12",
        overall_regime=EconomicState.REFLATION,
        asset_allocation=[
            AssetAllocationView(asset_class="US Equities", asset_bucket="equities", stance=AssetStance.OVERWEIGHT, rationale="S&P 500 = 7483.24", supporting_data=["S&P 500 = 7483.24"]),
            AssetAllocationView(asset_class="Fixed Income", asset_bucket="fixed_income", stance=AssetStance.NEUTRAL, rationale="10Y Yield = 4.485%", supporting_data=["10Y Yield = 4.485%"]),
            AssetAllocationView(asset_class="Gold", asset_bucket="commodities", stance=AssetStance.OVERWEIGHT, rationale="Gold = 4187.30", supporting_data=["Gold = 4187.30 USD/oz"]),
            AssetAllocationView(asset_class="FX / Currencies", asset_bucket="fx", stance=AssetStance.NEUTRAL, rationale="USD/THB = 33.14", supporting_data=["USD/THB = 33.14"]),
        ],
        focus_themes=["Cash coverage"],
        conviction_level="medium",
        conviction_rationale="Test coverage warning",
        quant_narrative_alignment="aligned",
        observable_registry={
            "obs_fedfunds": fed_funds,
            "obs_core_pce": core_pce,
            "obs_tbill": t_bill,
        },
    )

    assert len(direction.asset_allocation) == 4
    assert any("COVERAGE_WARNING_INCOMPLETE" in str(w) for w in direction.validation_warnings)

    direction_with_cash = MacroStrategyDirection(
        evaluated_at="2026-07-04T10:05:12",
        overall_regime=EconomicState.REFLATION,
        asset_allocation=[
            AssetAllocationView(asset_class="US Equities", asset_bucket="equities", stance=AssetStance.OVERWEIGHT, rationale="S&P 500 = 7483.24", supporting_data=["S&P 500 = 7483.24"]),
            AssetAllocationView(asset_class="Fixed Income", asset_bucket="fixed_income", stance=AssetStance.NEUTRAL, rationale="10Y Yield = 4.485%", supporting_data=["10Y Yield = 4.485%"]),
            AssetAllocationView(asset_class="Gold", asset_bucket="commodities", stance=AssetStance.OVERWEIGHT, rationale="Gold = 4187.30", supporting_data=["Gold = 4187.30 USD/oz"]),
            AssetAllocationView(asset_class="FX / Currencies", asset_bucket="fx", stance=AssetStance.NEUTRAL, rationale="USD/THB = 33.14", supporting_data=["USD/THB = 33.14"]),
            AssetAllocationView(asset_class="Cash / T-Bills", asset_bucket="cash", stance=AssetStance.NEUTRAL, rationale="Cash assessment", confidence="low", supporting_data=[]),
        ],
        focus_themes=["Cash coverage"],
        conviction_level="medium",
        conviction_rationale="Test cash registry population",
        quant_narrative_alignment="aligned",
        observable_registry={
            "obs_fedfunds": fed_funds,
            "obs_core_pce": core_pce,
            "obs_tbill": t_bill,
        },
    )
    cash = next(a for a in direction_with_cash.asset_allocation if a.asset_class == "Cash / T-Bills")
    assert cash.confidence == "low"
    assert len(cash.supporting_data) == 0


def test_fx_stance_mismatch_warning_for_baht_weakening():
    from schemas.macro_schemas import AssetAllocationView, AssetStance
    view = AssetAllocationView(
        asset_class="Currencies (USD/THB)",
        asset_bucket="fx",
        stance=AssetStance.UNDERWEIGHT,
        rationale="เงินบาทมีแนวโน้มอ่อนค่าจากส่วนต่างดอกเบี้ย 1.13% pts และดอลลาร์แข็งแกร่ง",
        supporting_data=["Policy differential = 1.13% pts"],
        confidence="medium",
        observable_refs=["obs_001"],
        source_refs=["Country_Macro_Snapshot.md", "Global_Macro_Snapshot.md"]
    )
    assert view.stance == AssetStance.UNDERWEIGHT
    assert any("FX_STANCE_MISMATCH" in str(w) or "การระบุมุมมองค่าเงิน" in str(w) for w in view.validation_warnings)


def test_pair_trade_statistical_overclaim_warning():
    from schemas.macro_schemas import PairTradeStrategy
    trade = PairTradeStrategy(
        long_leg="US Tech Equities",
        short_leg="European Industrials",
        thesis="ราคาหุ้นเทคแข็งแกร่ง Nasdaq = 29329",
        catalyst="Q2 Big Tech earnings",
        risk="Valuation risk",
        time_horizon="3 เดือน",
        confidence="medium",
        supporting_data=["Nasdaq 100 = 29329.21", "VGK ETF = 89.35"],
        observable_refs=["obs_nasdaq", "obs_vgk"],
        instrument_proxy="Long QQQ / Short VGK ETF",
        hedge_ratio="1.0 : 1.0 Beta-adjusted",
        entry_trigger="Spread widening > 1.5 SD",
        stop_loss_trigger="-3.0% relative spread divergence",
        target_gain_or_rebalance="+6.0% relative spread convergence",
        max_drawdown_limit="-4.5% of risk budget",
        review_frequency="Weekly",
        sizing_guidance="medium_risk_budget",
        source_refs=["Global_Macro_Snapshot.md", "Regional_Macro_Snapshot.md"]
    )
    assert trade.hedge_ratio == "1.0 : 1.0 Beta-adjusted"
    assert trade.entry_trigger == "Spread widening > 1.5 SD"
    assert any("STATISTICAL_OVERCLAIM" in str(w) or "การกล่าวอ้างสถิติขั้นสูง" in str(w) for w in trade.validation_warnings)


def test_institutional_review_guardrails():
    from schemas.macro_schemas import PairTradeStrategy, AssetAllocationView, AssetStance, MacroStrategyDirection, EconomicState
    # Test Fix 3: Pair trade relative spread without baseline triggers downgrade
    trade = PairTradeStrategy(
        long_leg="US Tech Equities",
        short_leg="Thai Equities",
        thesis="US AI growth vs Thai structure",
        catalyst="Q3 earnings",
        risk="Bubble risk",
        time_horizon="3 เดือน",
        confidence="high",
        supporting_data=["Nasdaq 100 = 29,329.21", "SET Index = 1,611.28"],
        observable_refs=["obs_nasdaq", "obs_set"],
        instrument_proxy="Long QQQ / Short THB Proxy",
        hedge_ratio="1.0 : 1.0 Notional",
        entry_trigger="Relative spread divergence > 2.0%",
        stop_loss_trigger="-3.0% relative spread divergence",
        target_gain_or_rebalance="+6.0% spread convergence",
        max_drawdown_limit="-4.5% of risk budget",
        review_frequency="Weekly",
        sizing_guidance="high_risk_budget",
        source_refs=["Global_Macro_Snapshot.md", "Country_Macro_Snapshot.md"]
    )
    assert not any("Implied Relative Ratio" in str(d) for d in trade.supporting_data)
    assert trade.confidence == "medium"
    assert trade.sizing_guidance == "medium_risk_budget"
    assert any("PT_GRACEFUL_DOWNGRADE" in str(w) or "ลดความมั่นใจและ Risk Budget" in str(w) for w in trade.validation_warnings)

    # Test Fix 2: Single source sizing warning
    asset = AssetAllocationView(
        asset_class="US Equities",
        asset_bucket="equities",
        stance=AssetStance.OVERWEIGHT,
        allocation_delta="+5% vs benchmark",
        time_horizon="3-6 เดือน",
        confidence="high",
        rationale="AI Tech Growth S&P 500 = 7483",
        supporting_data=["S&P 500 = 7483.24"],
        observable_refs=["obs_sp500"],
        source_refs=["Global_Macro_Snapshot.md"],
        why_not_high="-"
    )
    assert asset.confidence == "medium"
    assert asset.allocation_delta == "+5% vs benchmark"
    assert any("ALLOCATION_DELTA_INVALID" in str(w) or "SINGLE_SOURCE_PENALTY" in str(w) for w in asset.validation_warnings)

    # Test Fix 1: Conviction rationale contradiction warning
    direction = MacroStrategyDirection(
        overall_regime=EconomicState.REFLATION,
        time_horizon="3-6 เดือน",
        conviction_level="medium",
        quant_narrative_alignment="aligned",
        evaluated_at="2026-07-05T10:00:00",
        focus_themes=["AI Tech Growth"],
        asset_allocation=[asset],
        pair_trades=[trade],
        conviction_rationale="ความเชื่อมั่นระดับสูงเกิดจากสภาพคล่องทั่วโลก",
        divergence_note="-",
        source_files=["Global_Macro_Snapshot.md", "Country_Macro_Snapshot.md"],
        observable_registry={}
    )
    assert direction.conviction_rationale == "ความเชื่อมั่นระดับสูงเกิดจากสภาพคล่องทั่วโลก"
    assert any("CONVICTION_CONTRADICTION" in str(w) or "ความขัดแย้งด้านระดับความเชื่อมั่น" in str(w) for w in direction.validation_warnings)
