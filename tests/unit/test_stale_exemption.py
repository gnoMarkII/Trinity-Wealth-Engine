import pytest
from schemas.macro_schemas import MarketObservable, MacroStrategyDirection, AssetAllocationView, RegimeEvidenceComponent


def test_stale_data_no_exemption():
    obs_ip = MarketObservable(
        observable_id="obs_ip",
        asset_bucket="equities",
        region="US",
        indicator="Industrial Production",
        value="0.2",
        unit="%",
        observed_at="2026-05-01",
        source_file="test1.py",
        is_valid=True
    )
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-05T00:00:00",
        overall_regime="Reflation",
        stale_data_warnings=["IP is stale"],
        asset_allocation=[
            AssetAllocationView(
                asset_class="US Equities",
                asset_bucket="equities",
                stance="Overweight",
                confidence="high",
                rationale="Growth is strong",
                supporting_data=["GDP +3%"],
                allocation_delta="+2% vs benchmark",
                source_refs=["test1.py", "test2.py"],
                observable_refs=["obs_ip"]
            )
        ],
        focus_themes=["Growth"],
        conviction_level="high",
        conviction_rationale="Strong economy",
        quant_narrative_alignment="aligned",
        observable_registry={"obs_ip": obs_ip}
    )

    assert direction.conviction_level == "medium"
    assert any("STALE_DATA_DEGRADATION" in str(w) for w in direction.validation_warnings)


def test_stale_data_with_exemption():
    obs_nfci = MarketObservable(
        observable_id="obs_nfci",
        asset_bucket="equities",
        region="US",
        indicator="National Financial Conditions Index (NFCI)",
        value="-0.50",
        unit="Index",
        observed_at="2026-07-01",
        source_file="test1.py",
        is_valid=True
    )
    obs_icsa = MarketObservable(
        observable_id="obs_icsa",
        asset_bucket="equities",
        region="US",
        indicator="Initial Claims ICSA",
        value="220",
        unit="Thousands",
        observed_at="2026-07-01",
        source_file="test2.py",
        is_valid=True
    )
    obs_ip = MarketObservable(
        observable_id="obs_ip",
        asset_bucket="equities",
        region="US",
        indicator="Industrial Production",
        value="0.2",
        unit="%",
        observed_at="2026-05-01",
        source_file="test1.py",
        is_valid=True
    )
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-05T00:00:00",
        overall_regime="Reflation",
        stale_data_warnings=["IP is stale"],
        regime_evidence=[
            RegimeEvidenceComponent(
                dimension="growth",
                signal="Positive",
                summary="Financial conditions loose and claims low",
                evidence="NFCI loose and ICSA low",
                observable_refs=["obs_nfci", "obs_icsa"]
            )
        ],
        asset_allocation=[
            AssetAllocationView(
                asset_class="US Equities",
                asset_bucket="equities",
                stance="Overweight",
                confidence="high",
                rationale="Growth is strong",
                supporting_data=["GDP +3%"],
                allocation_delta="+2% vs benchmark",
                source_refs=["test1.py", "test2.py"],
                observable_refs=["obs_ip"]
            )
        ],
        focus_themes=["Growth"],
        conviction_level="high",
        conviction_rationale="Strong economy",
        quant_narrative_alignment="aligned",
        observable_registry={
            "obs_ip": obs_ip,
            "obs_nfci": obs_nfci,
            "obs_icsa": obs_icsa
        }
    )

    assert direction.conviction_level == "high"
    assert not any("STALE_DATA_DEGRADATION" in str(w) for w in direction.validation_warnings)
