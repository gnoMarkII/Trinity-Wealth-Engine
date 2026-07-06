"""
Dedicated unit test suites for the 8-Point Review Fixes (Rev.6).
Verifies:
1. revalidate_with_registry() idempotency
2. Mock data removal from Pillar 3/4 in production
3. 5 core buckets check for COVERAGE_WARNING_INCOMPLETE
4. FX_STANCE_MISMATCH guardrail covering USD vs THB and THB vs USD
5. GRACEFUL_DROP_PAIR_TRADES included in RETRYABLE_CRITICAL_IDS
6. ALLOCATION_DELTA_INVALID percentage substring removal
7. Stale data conviction restoration (was_stale_low check)
8. Robust YAML frontmatter extraction ignoring quotes and body text
"""

import pytest
from datetime import datetime
from schemas.macro_schemas import (
    MacroStrategyDirection,
    AssetAllocationView,
    AssetStance,
    MarketObservable,
    RegimeEvidenceComponent,
    PairTradeStrategy
)
from schemas.warning_registry import (
    RETRYABLE_CRITICAL_IDS,
    GRACEFUL_DROP_PAIR_TRADES,
    COVERAGE_WARNING_INCOMPLETE,
    FX_STANCE_MISMATCH,
    ALLOCATION_DELTA_INVALID,
    STALE_DATA_DEGRADATION
)
from tools.macro.derived_ratios import build_derived_pair_observables
from tools.macro.risk_analytics import build_risk_correlation_observables
from tools.archivist.parser import extract_yaml_frontmatter_value


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: revalidate_with_registry() Idempotency
# ──────────────────────────────────────────────────────────────────────────────
def test_fix_1_revalidate_idempotency():
    obs = MarketObservable(
        observable_id="obs_gdp",
        asset_bucket="equities",
        region="US",
        indicator="US GDP Growth",
        value="2.5",
        unit="%",
        observed_at="2026-07-05",
        source_file="test_source.py",
        is_valid=True
    )
    registry = {"obs_gdp": obs}
    
    asset = AssetAllocationView(
        asset_class="US Equities",
        asset_bucket="equities",
        stance=AssetStance.OVERWEIGHT,
        allocation_delta="+5% vs benchmark",
        time_horizon="6 เดือน",
        conviction_level="high",
        rationale="GDP Growth is strong",
        supporting_data=["US GDP Growth at 2.5%"],
        observable_refs=["obs_gdp"],
        source_refs=[],
        confidence="high"
    )
    
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-05T00:00:00Z",
        overall_regime="Goldilocks",
        focus_themes=["US Growth"],
        quant_narrative_alignment="aligned",
        conviction_level="high",
        conviction_rationale="Strong macroeconomic fundamentals across regions",
        time_horizon="6 เดือน",
        regime_evidence=[
            RegimeEvidenceComponent(
                dimension="Economic Growth",
                signal="Strong",
                evidence="GDP growth robust at 2.5%",
                observable_refs=["obs_gdp"],
                source_refs=[]
            )
        ],
        asset_allocation=[asset],
        pair_trades=[],
        risk_scenarios=[],
        source_files=["test_source.py"]
    )
    
    # Pass 1
    dir1 = direction.revalidate_with_registry(registry)
    # Pass 2
    dir2 = dir1.revalidate_with_registry(registry)
    
    assert dir1.model_dump() == dir2.model_dump(), "revalidate_with_registry must be strictly idempotent"
    assert dir1.conviction_level == dir2.conviction_level
    assert dir1.asset_allocation[0].confidence == dir2.asset_allocation[0].confidence


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Mock Data Removal from Pillar 3/4 in Production
# ──────────────────────────────────────────────────────────────────────────────
def test_fix_2_no_mock_fallback_in_production():
    obs_qqq = MarketObservable(
        observable_id="obs_qqq",
        asset_bucket="equities",
        region="US",
        indicator="QQQ Trust",
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
        indicator="SPY Trust",
        value="550.00",
        unit="USD",
        observed_at="2026-07-05",
        source_file="test.py",
        is_valid=True
    )
    
    # In production (use_mock_fallback=False), missing historical calculator returns invalid without 0.4 mock z-score
    def fail_calc(long_sym, short_sym): return None
    res_pair = build_derived_pair_observables(
        existing_observables=[obs_qqq, obs_spy],
        use_mock_fallback=False,
        historical_ratio_calculator=fail_calc
    )
    pair = next((o for o in res_pair if o.observable_id == "obs_pair_qqq_spy"), None)
    assert pair is not None
    assert pair.is_valid is False
    assert "Missing real market historical price series" in pair.stale_reason
    
    # For risk correlation, missing calculator and use_mock_fallback=False returns invalid without -0.15/0.05/-0.30 mock
    def fail_corr(a1, a2, window=60): return None
    res_corr = build_risk_correlation_observables(use_mock_fallback=False, correlation_calculator=fail_corr)
    corr = next((o for o in res_corr if o.observable_id == "obs_corr_spy_tlt_60d"), None)
    assert corr is not None
    assert corr.is_valid is False
    assert corr.value == "N/A"


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: 5 Core Buckets Check for COVERAGE_WARNING_INCOMPLETE
# ──────────────────────────────────────────────────────────────────────────────
def test_fix_3_five_core_buckets_coverage():
    # Create 5 assets, but all in equities bucket (missing fixed_income, commodities, fx, cash)
    assets = [
        AssetAllocationView(
            asset_class=f"US Equities {i}",
            asset_bucket="equities",
            stance=AssetStance.OVERWEIGHT,
            allocation_delta="+5% vs benchmark",
            time_horizon="6 เดือน",
            conviction_level="high",
            rationale="Growth strong",
            supporting_data=["GDP Growth 2.5%"],
            confidence="high"
        )
        for i in range(5)
    ]
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-05T00:00:00Z",
        overall_regime="Goldilocks",
        focus_themes=["US Growth"],
        quant_narrative_alignment="aligned",
        conviction_level="high",
        conviction_rationale="Strong macroeconomic fundamentals",
        time_horizon="6 เดือน",
        regime_evidence=[],
        asset_allocation=assets,
        pair_trades=[],
        risk_scenarios=[]
    )
    
    dir_val = direction.revalidate_with_registry({})
    has_cov_warning = any("COVERAGE_WARNING_INCOMPLETE" in str(w) or "missing" in str(w).lower() for w in dir_val.validation_warnings)
    assert has_cov_warning, "Must warn when any of the 5 core buckets is missing even if asset count >= 5"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: FX_STANCE_MISMATCH Guardrail Covering USD vs THB and THB vs USD
# ──────────────────────────────────────────────────────────────────────────────
def test_fix_4_fx_stance_mismatch_guardrail():
    # Case 1: USD vs THB (base USD) with rationale saying baht weakening -> stance UNDERWEIGHT is mismatch
    asset_usd_thb = AssetAllocationView(
        asset_class="USD vs THB (สกุลเงินดอลลาร์เมื่อเทียบกับบาท)",
        asset_bucket="fx",
        stance=AssetStance.UNDERWEIGHT,
        allocation_delta="-3% vs benchmark",
        time_horizon="6 เดือน",
        conviction_level="medium",
        rationale="เงินบาทมีแนวโน้มอ่อนค่าลงจากเงินทุนไหลออก และดอลลาร์แข็งค่า",
        supporting_data=["THB depreciated 2.5%"],
        confidence="medium"
    )
    assert any("FX_STANCE_MISMATCH" in str(w) or "มุมมองค่าเงินขัดแย้ง" in str(w) for w in asset_usd_thb.validation_warnings)
    
    # Case 2: THB vs USD (base THB) with rationale saying baht weakening -> stance UNDERWEIGHT is correct!
    asset_thb_usd = AssetAllocationView(
        asset_class="THB vs USD (ค่าเงินบาทเมื่อเทียบกับดอลลาร์)",
        asset_bucket="fx",
        stance=AssetStance.UNDERWEIGHT,
        allocation_delta="-3% vs benchmark",
        time_horizon="6 เดือน",
        conviction_level="medium",
        rationale="เงินบาทมีแนวโน้มอ่อนค่าลงจากเงินทุนไหลออก และดอลลาร์แข็งค่า",
        supporting_data=["THB depreciated 2.5%"],
        confidence="medium"
    )
    assert not any("FX_STANCE_MISMATCH" in str(w) for w in asset_thb_usd.validation_warnings)


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: GRACEFUL_DROP_PAIR_TRADES Included in RETRYABLE_CRITICAL_IDS
# ──────────────────────────────────────────────────────────────────────────────
def test_fix_5_graceful_drop_pair_trades_retryable():
    assert GRACEFUL_DROP_PAIR_TRADES in RETRYABLE_CRITICAL_IDS, (
        "GRACEFUL_DROP_PAIR_TRADES must be in RETRYABLE_CRITICAL_IDS to allow LLM retry when pair trades are dropped"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: ALLOCATION_DELTA_INVALID Percentage Substring Removal
# ──────────────────────────────────────────────────────────────────────────────
def test_fix_6_allocation_delta_percentage_allowed():
    # Explicit percentage difference "+5% vs benchmark" must NOT trigger ALLOCATION_DELTA_INVALID
    asset_valid_delta = AssetAllocationView(
        asset_class="US Equities",
        asset_bucket="equities",
        stance=AssetStance.OVERWEIGHT,
        allocation_delta="+5% vs benchmark",
        time_horizon="6 เดือน",
        conviction_level="high",
        rationale="Strong earnings growth",
        supporting_data=["EPS growth +12% YoY"],
        confidence="high",
        source_refs=["file1.py", "file2.py"]
    )
    assert not any("ALLOCATION_DELTA_INVALID" in str(w) for w in asset_valid_delta.validation_warnings)
    
    # But writing only "overweight" without numbers MUST trigger ALLOCATION_DELTA_INVALID
    asset_invalid_delta = AssetAllocationView(
        asset_class="US Equities",
        asset_bucket="equities",
        stance=AssetStance.OVERWEIGHT,
        allocation_delta="overweight",
        time_horizon="6 เดือน",
        conviction_level="high",
        rationale="Strong earnings growth",
        supporting_data=["EPS growth +12% YoY"],
        confidence="high",
        source_refs=["file1.py", "file2.py"]
    )
    assert any("ALLOCATION_DELTA_INVALID" in str(w) for w in asset_invalid_delta.validation_warnings)


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: Stale Data Conviction Restoration (was_stale_low Check)
# ──────────────────────────────────────────────────────────────────────────────
def test_fix_7_was_stale_conviction_restoration():
    # Simulate a direction that was previously downgraded by STALE_DATA_DEGRADATION
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-05T00:00:00Z",
        overall_regime="Goldilocks",
        focus_themes=["US Growth"],
        quant_narrative_alignment="aligned",
        conviction_level="low",
        conviction_rationale="Downgraded due to stale data",
        time_horizon="6 เดือน",
        validation_warnings=[str(STALE_DATA_DEGRADATION) + " ข้อมูลล่าช้าเกินเกณฑ์"],
        regime_evidence=[],
        asset_allocation=[],
        pair_trades=[],
        risk_scenarios=[]
    )
    
    # When revalidated with registry containing valid leading indicator exemption
    obs_lead1 = MarketObservable(
        observable_id="obs_pmi1",
        asset_bucket="equities",
        region="US",
        indicator="US ISM Manufacturing PMI",
        value="52.5",
        unit="index",
        observed_at="2026-07-05",
        source_file="test.py",
        is_valid=True
    )
    obs_lead2 = MarketObservable(
        observable_id="obs_pmi2",
        asset_bucket="equities",
        region="US",
        indicator="Global Manufacturing PMI",
        value="51.0",
        unit="index",
        observed_at="2026-07-05",
        source_file="test.py",
        is_valid=True
    )
    registry = {"obs_pmi1": obs_lead1, "obs_pmi2": obs_lead2}
    
    direction.regime_evidence = [
        RegimeEvidenceComponent(
            dimension="Leading Indicators",
            signal="Strong",
            evidence="PMI expanding at 52.5",
            observable_refs=["obs_pmi1", "obs_pmi2"],
            source_refs=["test.py"]
        )
    ]
    
    res = direction.revalidate_with_registry(registry)
    assert res.conviction_level == "medium", "Conviction previously downgraded to LOW by stale data should be restored when exempt"
    assert not any("STALE_DATA_DEGRADATION" in str(w) for w in res.validation_warnings)


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: Robust YAML Frontmatter Extraction (Quotes & Body Text)
# ──────────────────────────────────────────────────────────────────────────────
def test_fix_8_extract_yaml_frontmatter_value():
    content_quoted = '''---
entity_type: "macro_strategy"
title: "Weekly Macro Direction"
---
# Body Title
Some body text mentioning entity_type: goal here.
'''
    assert extract_yaml_frontmatter_value(content_quoted, "entity_type") == "macro_strategy"
    assert extract_yaml_frontmatter_value(content_quoted, "title") == "Weekly Macro Direction"
    
    content_single_quote = '''---
entity_type: 'youtube_insight'
---
# Body
'''
    assert extract_yaml_frontmatter_value(content_single_quote, "entity_type") == "youtube_insight"
    
    content_no_frontmatter_but_in_body = '''# Title Only
This document contains entity_type: macro_strategy in body text only.
'''
    assert extract_yaml_frontmatter_value(content_no_frontmatter_but_in_body, "entity_type") is None
