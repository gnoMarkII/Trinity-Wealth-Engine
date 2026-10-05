import json
import pytest
from pathlib import Path
from schemas.macro_schemas import (
    EconomicState,
    QuantScore,
    RegionQuantMetrics,
    MarketObservable,
    MacroStrategyDirection,
    AssetAllocationView,
    RegimeEvidenceComponent,
)

from tools.macro.scoring import (
    _determine_economic_state,
    _calculate_us_recession_risk,
    _calculate_recession_probability,
    _get_global_risk_sentiment,
    _calculate_matrix_scores_from_markdown,
    _calculate_matrix_scores_from_observables,
)
from tools.macro.evaluation import _apply_validity, _add_relative_observables


def test_economic_state_fail_closed_when_data_missing():
    # If growth is None, state MUST be Unknown
    assert _determine_economic_state(None, 0.5) == EconomicState.UNKNOWN.value
    # If inflation is None, state MUST be Unknown
    assert _determine_economic_state(0.5, None) == EconomicState.UNKNOWN.value
    # If both None, state MUST be Unknown
    assert _determine_economic_state(None, None) == EconomicState.UNKNOWN.value

    # When both are valid numbers, standard quadrants apply
    assert _determine_economic_state(0.5, 0.2) == EconomicState.GOLDILOCKS.value
    assert _determine_economic_state(0.5, -0.2) == EconomicState.REFLATION.value
    assert _determine_economic_state(-0.5, -0.2) == EconomicState.STAGFLATION.value
    assert _determine_economic_state(-0.5, 0.2) == EconomicState.RECESSION.value


def test_us_recession_risk_does_not_blend_other_countries():
    # If US growth or monetary is missing, returns None
    assert _calculate_us_recession_risk(None, 0.5, 1.0) is None
    assert _calculate_us_recession_risk(0.5, None, 1.0) is None

    # Calculate deterministic US recession risk
    risk = _calculate_us_recession_risk(0.5, 0.5, 1.0)
    assert risk is not None
    assert 0.0 <= risk <= 1.0

    # matrices without United States must NOT blend other countries (e.g. Thailand, Euro Area)
    matrices = {
        "Thailand": {"growth": -1.0, "monetary": -1.0},
        "Euro Area": {"growth": -0.8, "monetary": -0.8},
    }
    prob = _calculate_recession_probability(matrices, 1.0)
    assert prob == 0.5  # Neutral default rather than blending non-US countries


def test_real_rate_calculation_rejects_raw_index():
    # Snapshot where Core PCE is raw index level (122.5) instead of % YoY
    md_raw_index = """# United States
| ดัชนี | ค่าล่าสุด | ก่อนหน้า | MA ย้อนหลัง |
|-------|----------|----------|------------|
| **Fed Funds Rate** | 5.25% | 5.25% | 5.25% |
| **Core PCE** | 122.5 | 121.8 | 120.0 |
"""
    scores = _calculate_matrix_scores_from_markdown(md_raw_index)
    us = scores.get("United States", {})
    # Since PCE is > 30.0, real rate was skipped, and with no 10Y-2Y, monetary score is None
    assert us.get("monetary") is None

    # Snapshot where Core PCE is valid YoY % (2.6%)
    md_yoy = """# United States
| ดัชนี | ค่าล่าสุด | ก่อนหน้า | MA ย้อนหลัง |
|-------|----------|----------|------------|
| **Fed Funds Rate** | 5.25% | 5.25% | 5.25% |
| **Core PCE** | 2.6% | 2.7% | 2.8% |
"""
    scores_yoy = _calculate_matrix_scores_from_markdown(md_yoy)
    us_yoy = scores_yoy.get("United States", {})
    # 5.25 - 2.6 = 2.65 > 1.0 => monetary score is -1.0 (restrictive)
    assert us_yoy.get("monetary") == -1.0


def test_thai_mock_rejection_marks_thai_state_unknown():
    fixture_path = Path("tests/fixtures/macro/thai_mock_rejection.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        fixture = json.load(f)

    thai_md = fixture["mock_markdown_snapshot"]
    scores = _calculate_matrix_scores_from_markdown(thai_md)
    thai = scores.get("Thailand", {})

    # Since all growth/inflation metrics were mock, final growth & inflation are None
    assert thai.get("growth") is None
    assert thai.get("inflation") is None
    assert thai.get("state") == EconomicState.UNKNOWN.value
    assert thai.get("confidence") == 0.0
    assert len(thai.get("data_gaps", [])) >= 2


def test_apply_validity_rejects_mock_and_stale():
    today_str = "2026-09-27"
    mock_obs = MarketObservable(
        observable_id="obs_th_mock_gdp",
        asset_bucket="equities",
        region="Thailand",
        indicator="Real GDP [Mock]",
        value="2.5",
        unit="%",
        observed_at="2026-09-27",
        source_file="test.md",
        provider="StaticProxy",
    )
    validated_mock = _apply_validity(mock_obs, today_str)
    assert validated_mock.is_valid is False
    assert validated_mock.status == "mock"
    assert validated_mock.confidence == "low"

    # Stale daily asset (> 7 days)
    stale_asset = MarketObservable(
        observable_id="obs_oil",
        asset_bucket="commodities",
        region="Global",
        indicator="WTI Crude Oil (`CL=F`)",
        value="70.0",
        unit="USD",
        observed_at="2026-08-01",
        source_file="test.md",
        provider="Yahoo",
    )
    validated_stale = _apply_validity(stale_asset, today_str)
    assert validated_stale.is_valid is False
    assert validated_stale.status == "stale"


def test_relative_observables_policy_rate_differential():
    today_str = "2026-09-27"
    us_rate = MarketObservable(
        observable_id="obs_fedfunds",
        asset_bucket="cash",
        region="United States",
        indicator="Fed Funds Rate (`FEDFUNDS`)",
        value="5.25",
        unit="%",
        observed_at=today_str,
        source_file="Country_Macro_Snapshot_2026-09-27.md",
        is_valid=True,
    )
    thai_rate = MarketObservable(
        observable_id="obs_thai_policy",
        asset_bucket="cash",
        region="Thailand",
        indicator="Policy Rate",
        value="2.50",
        unit="%",
        observed_at=today_str,
        source_file="Country_Macro_Snapshot_2026-09-27.md",
        is_valid=True,
    )
    observables = [us_rate, thai_rate]
    _add_relative_observables(observables, today_str)

    diff_obs = next((o for o in observables if o.observable_id == "obs_diff_us_th_policy_rate"), None)
    assert diff_obs is not None
    assert diff_obs.is_valid is True
    assert diff_obs.status == "verified"
    assert diff_obs.metadata.get("diff_bps") == 275.0
    assert float(diff_obs.value) == 2.75


def test_quant_score_backward_and_forward_compatibility():
    qs = QuantScore(
        evaluated_at="2026-09-27T12:00:00Z",
        regions={
            "United States": RegionQuantMetrics(
                growth_score=0.5,
                inflation_score=0.2,
                monetary_score=-0.5,
                economic_state=EconomicState.GOLDILOCKS,
                confidence=0.85,
                coverage=1.0,
            ),
            "Thailand": RegionQuantMetrics(
                growth_score=None,
                inflation_score=None,
                monetary_score=None,
                economic_state=EconomicState.UNKNOWN,
                confidence=0.0,
                coverage=0.0,
                data_gaps=["Thailand Growth", "Thailand Inflation"],
            ),
        },
        us_recession_risk_score=0.35,
        global_risk_sentiment_score=1.0,
        data_freshness_note="Snapshot: 2026-09-27",
    )
    # Check that model_validator synced legacy fields seamlessly
    assert qs.recession_probability == 0.35
    assert qs.global_geopolitics_score == 1.0
    assert qs.formula_version == "2.0.0"
    assert qs.regions["Thailand"].economic_state == EconomicState.UNKNOWN


def test_thai_hard_data_scenarios_from_fixture():
    fixture_path = Path("tests/fixtures/macro/thai_hard_data_scenarios.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from tools.macro.adapters.thai_hard_data_adapter import ThaiHardDataAdapter, ThaiHardDataRecord

    scenarios = data["scenarios"]

    # Scenario 1: Complete
    s_comp = scenarios["complete"]
    records_comp = {
        k: ThaiHardDataRecord(**v) for k, v in s_comp["hard_data_records"].items()
    }
    adapter_comp = ThaiHardDataAdapter(override_records=records_comp)
    obs_comp = adapter_comp.as_observables("2026-09-27")
    scores_comp = _calculate_matrix_scores_from_observables(obs_comp)
    th_comp = scores_comp.get("Thailand", {})
    assert th_comp.get("growth") is not None
    assert th_comp.get("inflation") is not None
    assert th_comp.get("state") == "Reflation"
    assert th_comp.get("confidence") >= 0.5

    # Scenario 2: Missing GDP
    s_no_gdp = scenarios["missing_gdp"]
    records_no_gdp = {
        k: ThaiHardDataRecord(**v) for k, v in s_no_gdp["hard_data_records"].items()
    }
    adapter_no_gdp = ThaiHardDataAdapter(override_records=records_no_gdp)
    obs_no_gdp = adapter_no_gdp.as_observables("2026-09-27")
    scores_no_gdp = _calculate_matrix_scores_from_observables(obs_no_gdp)
    th_no_gdp = scores_no_gdp.get("Thailand", {})
    assert th_no_gdp.get("growth") is None
    assert th_no_gdp.get("inflation") is not None
    assert th_no_gdp.get("state") == "Unknown"
    assert th_no_gdp.get("confidence") == 0.0
    assert any("Thailand Growth" in gap for gap in th_no_gdp.get("data_gaps", []))

    # Scenario 3: Missing CPI
    s_no_cpi = scenarios["missing_cpi"]
    records_no_cpi = {
        k: ThaiHardDataRecord(**v) for k, v in s_no_cpi["hard_data_records"].items()
    }
    adapter_no_cpi = ThaiHardDataAdapter(override_records=records_no_cpi)
    obs_no_cpi = adapter_no_cpi.as_observables("2026-09-27")
    scores_no_cpi = _calculate_matrix_scores_from_observables(obs_no_cpi)
    th_no_cpi = scores_no_cpi.get("Thailand", {})
    assert th_no_cpi.get("growth") is not None
    assert th_no_cpi.get("inflation") is None
    assert th_no_cpi.get("state") == "Unknown"
    assert th_no_cpi.get("confidence") == 0.0
    assert any("Thailand Inflation" in gap for gap in th_no_cpi.get("data_gaps", []))

    # Scenario 4: Market Only (Foreign flow + Breadth alone cannot alter macro regime)
    s_market = scenarios["market_only"]
    records_market = {
        k: ThaiHardDataRecord(**v) for k, v in s_market["hard_data_records"].items()
    }
    adapter_market = ThaiHardDataAdapter(override_records=records_market)
    obs_market = adapter_market.as_observables("2026-09-27")
    # Add market stance observables (SET flow and breadth)
    for m in s_market["market_stance_observables"]:
        obs_market.append(MarketObservable(
            observable_id=m["observable_id"],
            asset_bucket="equities",
            region=m["region"],
            indicator=m["indicator"],
            value=m["value"],
            unit=m["unit"],
            observed_at="2026-09-27",
            source_file="SET_Feed",
            is_valid=m["is_valid"],
            status="verified",
        ))
    scores_market = _calculate_matrix_scores_from_observables(obs_market)
    th_market = scores_market.get("Thailand", {})
    assert th_market.get("growth") is None
    assert th_market.get("inflation") is None
    assert th_market.get("state") == "Unknown"
    assert th_market.get("confidence") == 0.0


def test_invalid_observables_excluded_from_quant_scores():
    # Observables with is_valid=False (e.g. stale RSAFS, mock GDP) must be strictly ignored
    stale_rsafs = MarketObservable(
        observable_id="obs_us_rsafs_stale",
        asset_bucket="equities",
        region="United States",
        indicator="Retail Sales (`RSAFS`)",
        value="715000",
        unit="USD",
        observed_at="2026-06-01",
        source_file="Country_Macro_Snapshot_2026-09-27.md",
        is_valid=False,
        status="stale",
        stale_reason="Exceeded 90 days cutoff",
    )
    valid_indpro = MarketObservable(
        observable_id="obs_us_indpro_valid",
        asset_bucket="equities",
        region="United States",
        indicator="Industrial Production (`INDPRO`)",
        value="103.5",
        unit="index",
        observed_at="2026-09-25",
        source_file="Country_Macro_Snapshot_2026-09-27.md",
        is_valid=True,
        status="verified",
        metadata={"val": 103.5, "prev": 102.5, "ma": 102.0},
    )
    scores = _calculate_matrix_scores_from_observables([stale_rsafs, valid_indpro])
    us = scores.get("United States", {})
    # Only valid_indpro should be scored (+1.0)
    assert us.get("growth") == 1.0


def test_stale_rsafs_excluded_from_growth_evidence():
    stale_rsafs = MarketObservable(
        observable_id="obs_rsafs",
        asset_bucket="equities",
        region="United States",
        indicator="Retail Sales (`RSAFS`)",
        value="715000",
        unit="USD",
        observed_at="2026-06-01",
        source_file="Country_Macro_Snapshot_2026-09-27.md",
        is_valid=False,
        status="stale",
    )
    registry = {"obs_rsafs": stale_rsafs}

    direction = MacroStrategyDirection(
        evaluated_at="2026-09-27T12:00:00",
        overall_regime=EconomicState.REFLATION,
        asset_allocation=[],
        focus_themes=[],
        conviction_level="medium",
        conviction_rationale="Growth supported by retail sales.",
        quant_narrative_alignment="aligned",
        observable_registry=registry,
        regime_evidence=[
            RegimeEvidenceComponent(
                dimension="Growth",
                signal="Positive",
                evidence="RSAFS Retail Sales reached $715B",
                observable_refs=["obs_rsafs"],
            )
        ],
    )

    # Re-validate with registry
    direction = direction.revalidate_with_registry(registry)

    growth_ev = next(ev for ev in direction.regime_evidence if ev.dimension == "Growth")
    # obs_rsafs must be dropped because it is invalid
    assert "obs_rsafs" not in growth_ev.observable_refs
    # Stale warning must be recorded
    assert any("GROWTH_EVIDENCE_STALE" in w for w in direction.validation_warnings)


def test_fx_fed_bot_spread_guardrail_when_unavailable():
    # Registry has no policy rate spread observable
    registry = {
        "obs_us10y": MarketObservable(
            observable_id="obs_us10y",
            asset_bucket="fixed_income",
            region="United States",
            indicator="10Y Treasury",
            value="4.25",
            unit="%",
            observed_at="2026-09-27",
            source_file="Global_Macro_Snapshot.md",
            is_valid=True,
        )
    }

    fx_asset = AssetAllocationView(
        asset_class="USD vs THB",
        asset_bucket="fx",
        region="Thailand",
        stance="Overweight",
        allocation_delta="+3% vs benchmark",
        confidence="high",
        rationale="ดอลลาร์ได้เปรียบจากส่วนต่างอัตราดอกเบี้ยนโยบาย Fed-BoT ที่กว้าง +275 bps",
        supporting_data=["ส่วนต่างอัตราดอกเบี้ยนโยบาย +275 bps"],
        source_refs=["Global_Macro_Snapshot.md"],
        observable_refs=["obs_us10y"],
    )

    direction = MacroStrategyDirection(
        evaluated_at="2026-09-27T12:00:00",
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[fx_asset],
        focus_themes=[],
        conviction_level="medium",
        conviction_rationale="US-TH macro divergence.",
        quant_narrative_alignment="aligned",
        observable_registry=registry,
    )

    direction = direction.revalidate_with_registry(registry)

    alloc = direction.asset_allocation[0]
    # Rationale must be sanitized
    assert "+275 bps" not in alloc.rationale
    assert "ปัจจุบันยังไม่มีข้อมูลส่วนต่างอัตราดอกเบี้ยนโยบายที่ยืนยันได้" in alloc.rationale
    # Confidence must be downgraded from high
    assert alloc.confidence in ("medium", "low")
    # Warning must be emitted
    assert any("FX_SPREAD_DATA_UNAVAILABLE" in w for w in alloc.validation_warnings)
    assert any("FX_SPREAD_DATA_UNAVAILABLE" in w for w in direction.validation_warnings)


def test_observable_region_canonicalization_merges_us_and_united_states():
    obs_list = [
        MarketObservable(
            observable_id="obs_gdp",
            asset_bucket="equities",
            region="United States",
            indicator="Real GDP",
            value="2.8%",
            unit="%",
            observed_at="2026-09-01",
            source_file="us_gdp.csv",
            is_valid=True,
            metadata={"prev": 2.5, "ma": 2.4},
        ),
        MarketObservable(
            observable_id="obs_cpi",
            asset_bucket="fixed_income",
            region="US",
            indicator="CPI",
            value="2.5%",
            unit="%",
            observed_at="2026-09-01",
            source_file="us_cpi.csv",
            is_valid=True,
            metadata={"prev": 2.7, "ma": 2.8},
        ),
    ]
    scores = _calculate_matrix_scores_from_observables(obs_list)
    assert "US" not in scores
    assert "United States" in scores
    us_data = scores["United States"]
    assert us_data["growth"] is not None
    assert us_data["inflation"] is not None


