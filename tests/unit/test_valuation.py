import pytest
from tools.macro.valuation import build_valuation_observables, build_credit_spread_observable
from validators.valuation_guardrails import check_valuation_contradiction, check_credit_spread_warning
from schemas.macro_schemas import MarketObservable, MacroStrategyDirection, AssetAllocationView


def test_valuation_normal_erp():
    # Mock forward PE = 20 (EY = 5.00%), DGS10 = 4.25%. ERP = 0.75% (< 1.5% rich threshold)
    def mock_getter(sym):
        if sym == "^GSPC":
            return {"forwardPE": 20.0, "trailingPE": 22.0}
        return {}

    obs = build_valuation_observables(ticker_info_getter=mock_getter, dgs10_value=4.25)
    assert len(obs) == 2
    ey_obs = next(o for o in obs if o.observable_id == "obs_ey_gspc")
    erp_obs = next(o for o in obs if o.observable_id == "obs_erp_gspc")

    assert ey_obs.is_valid is True
    assert ey_obs.value == "5.00"
    assert erp_obs.is_valid is True
    assert erp_obs.value == "0.75"
    assert erp_obs.metadata["is_rich"] is True
    assert erp_obs.metadata["symbol_used"] == "^GSPC"


def test_valuation_missing_forward_pe_fallback():
    # All symbols missing forwardPE, trailingPE available on SPY
    def mock_getter(sym):
        if sym == "SPY":
            return {"trailingPE": 25.5}
        return {}

    obs = build_valuation_observables(ticker_info_getter=mock_getter, dgs10_value=4.25)
    assert len(obs) == 2
    erp_obs = next(o for o in obs if o.observable_id == "obs_erp_gspc")
    tpe_obs = next(o for o in obs if o.observable_id == "obs_trailing_pe_gspc")

    assert erp_obs.is_valid is False
    assert "Missing forwardPE" in erp_obs.stale_reason
    assert erp_obs.value == "N/A"

    assert tpe_obs.is_valid is True
    assert tpe_obs.value == "25.50"
    assert tpe_obs.unit == "ratio"
    assert tpe_obs.metadata["symbol_used"] == "SPY"


def test_valuation_guardrail_rich_erp_downgrade():
    erp_obs = MarketObservable(
        observable_id="obs_erp_gspc",
        asset_bucket="equities",
        region="US",
        indicator="S&P 500 Equity Risk Premium",
        value="0.75",
        unit="%",
        observed_at="2026-07-05",
        source_file="test.py",
        is_valid=True,
        metadata={"erp_decimal": 0.0075}
    )
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-05T00:00:00",
        overall_regime="Reflation",
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
                observable_refs=["obs_erp_gspc"]
            )
        ],
        focus_themes=["Growth"],
        conviction_level="high",
        conviction_rationale="Strong economy",
        quant_narrative_alignment="aligned",
        observable_registry={"obs_erp_gspc": erp_obs}
    )

    # Guardrail runs automatically during Pydantic initialization via validate_all_contradictions
    assert direction.asset_allocation[0].confidence == "medium"
    assert any("VALUATION_RICH_WARNING" in str(w) for w in direction.asset_allocation[0].validation_warnings)


def test_valuation_guardrail_with_hedge_keywords():
    erp_obs = MarketObservable(
        observable_id="obs_erp_gspc",
        asset_bucket="equities",
        region="US",
        indicator="S&P 500 Equity Risk Premium",
        value="0.75",
        unit="%",
        observed_at="2026-07-05",
        source_file="test.py",
        is_valid=True,
        metadata={"erp_decimal": 0.0075}
    )
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-05T00:00:00",
        overall_regime="Reflation",
        asset_allocation=[
            AssetAllocationView(
                asset_class="US Equities",
                asset_bucket="equities",
                stance="Overweight",
                confidence="high",
                rationale="Growth is strong with put option hedge to protect downside",
                supporting_data=["GDP +3%"],
                allocation_delta="+2% vs benchmark",
                source_refs=["test1.py", "test2.py"],
                observable_refs=["obs_erp_gspc"]
            )
        ],
        focus_themes=["Growth"],
        conviction_level="high",
        conviction_rationale="Strong economy",
        quant_narrative_alignment="aligned",
        observable_registry={"obs_erp_gspc": erp_obs}
    )

    findings = check_valuation_contradiction(direction)
    assert len(findings) == 0


def test_credit_spread_warning():
    hy_obs = MarketObservable(
        observable_id="obs_hy_spread",
        asset_bucket="fixed_income",
        region="US",
        indicator="High Yield Bond Spread",
        value="5.50",
        unit="% pts",
        observed_at="2026-07-05",
        source_file="test.py",
        is_valid=True,
        metadata={"hy_spread_pct": 5.50}
    )
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-05T00:00:00",
        overall_regime="Reflation",
        asset_allocation=[],
        focus_themes=["Defensive"],
        conviction_level="medium",
        conviction_rationale="Cautious",
        quant_narrative_alignment="aligned",
        observable_registry={"obs_hy_spread": hy_obs}
    )

    findings = check_credit_spread_warning(direction)
    assert len(findings) == 1
    assert "CREDIT_SPREAD_WARNING" in str(findings[0].warning)
    assert findings[0].downgrade_conviction is False
