import pytest
from schemas.micro_quant_schemas import (
    QuantSignals,
    PiotroskiFScoreBreakdown,
    ReverseDCFResult,
    EarningsGuidanceContext,
    TacticalSetup,
)
from tools.market.equity_rules_engine import compute_deterministic_scorecard


def test_scorecard_dual_conviction_and_expectations_separation():
    signals = QuantSignals(
        ticker="FTNT",
        market="US",
        currency="USD",
        company_name="Fortinet, Inc.",
        as_of_date="2026-08-27",
        evaluated_at="2026-08-27T10:00:00Z",
        quality_score=76.0,
        roic_pct=15.0,
        ocf_to_net_income=0.85,
        eps_revision_net_30d=40,  # Analyst expectations
    )
    piotroski = PiotroskiFScoreBreakdown(status="available", f_score=8)
    
    # Non-actionable DCF (ERP anomaly)
    dcf_non_actionable = ReverseDCFResult(
        status="available",
        is_eligible=True,
        is_actionable=False,
        actionability_reason="Macro parameter anomaly: ERP (-0.22%) <= 0",
        target_price_12m=157.64,
        upside_12m_pct=-8.8,
    )
    
    tactical = TacticalSetup(
        status="available",
        price_stage="STAGE_2_MARKUP",
        current_price=172.78,
        atr_14=6.14,
        key_support_level=154.46,
        key_resistance_level=169.91,
        buy_zone_min=154.46,
        buy_zone_max=157.53,
        invalidation_stop_loss=149.855,
        tactical_target_price=169.91,
        current_rr_ratio=None,
        pullback_entry_status="at_or_above_target",
        is_in_buy_zone=False,
        breakout_trigger_price=170.524,
        breakout_target_price=184.032,
        breakout_stop_loss=163.770,
        breakout_planned_rr=2.00,
        breakout_current_rr=1.25,
        max_breakout_chase_price=173.594,
        breakout_entry_status="chased",
        breakout_entry_eligible=False,
    )

    scorecard, _, _ = compute_deterministic_scorecard(
        ticker="FTNT",
        market="US",
        quant_signals=signals,
        piotroski=piotroski,
        dcf=dcf_non_actionable,
        tactical=tactical,
    )

    # 1. Expectations separated from guidance
    assert scorecard.analyst_expectations_score == 90.0
    assert scorecard.management_guidance_score is None
    assert scorecard.guidance_expectation_score == 90.0

    # 2. Dual conviction scores
    assert scorecard.business_conviction_score == 8.1
    assert scorecard.investment_conviction_score is None
    assert scorecard.valuation_margin_score is None

    # 3. Execution Readiness penalized for chased setup (5.2 instead of 6.8)
    assert scorecard.setup_readiness_score == 20.0
    assert scorecard.execution_readiness_score == 5.2
    assert scorecard.execution_score_breakdown == {
        "stage": 85.0,
        "setup": 20.0,
        "insider": 50.0,
        "setup_reason": "breakout_chased_and_pullback_above_target",
    }

    # 4. Reweighting metadata explicitly recorded
    assert scorecard.reweighting_metadata is not None
    assert scorecard.reweighting_metadata["excluded_pillars"] == ["valuation"]
    assert scorecard.reweighting_metadata["weights_used"] == {"fundamental": 0.55, "expectations": 0.45}
    assert "Macro parameter anomaly" in scorecard.reweighting_metadata["reason"]

    # 5. Action stance locked to conditional plan
    assert scorecard.action_stance == "ACCUMULATE_ON_DIP"
    assert scorecard.stance_mode == "conditional"


def test_action_stance_truth_table_gating():
    # 1. Weak business conviction < 5.0 -> REDUCE
    weak_signals = QuantSignals(
        ticker="WEAK",
        market="US",
        currency="USD",
        as_of_date="2026-08-27",
        evaluated_at="2026-08-27T10:00:00Z",
        quality_score=20.0,
        roic_pct=-5.0,
        ocf_to_net_income=0.2,
        eps_revision_net_30d=-20,
        value_score=25.0,
        momentum_score=20.0,
        growth_score=15.0,
        dividend_score=10.0,
        solvency_score=20.0,
        fcf_quality_score=20.0,
        debt_quality_score=20.0,
    )
    piotroski_weak = PiotroskiFScoreBreakdown(status="available", f_score=2)
    tactical_stage2 = TacticalSetup(
        status="available",
        price_stage="STAGE_2_MARKUP",
        current_price=50.0,
        breakout_entry_eligible=True,
        breakout_current_rr=2.5,
    )
    scorecard, _, _ = compute_deterministic_scorecard(
        ticker="WEAK",
        market="US",
        quant_signals=weak_signals,
        piotroski=piotroski_weak,
        tactical=tactical_stage2,
    )
    assert scorecard.action_stance == "REDUCE"
    assert scorecard.stance_mode == "reduce"

    # 2. Stale multiple sessions -> HOLD_WAIT
    from schemas.micro_quant_schemas import AtomicMarketSnapshot
    stale_snapshot = AtomicMarketSnapshot(
        analysis_price=100.0,
        analysis_price_as_of="2026-08-20",
        price_source="ohlcv_close",
        latest_ohlcv_close=100.0,
        latest_ohlcv_date="2026-08-20",
        freshness_status="stale",
        data_freshness_status="stale_multiple_sessions",
        missing_trading_sessions=3,
        retrieved_at="2026-08-27T10:00:00Z",
    )
    strong_signals = QuantSignals(
        ticker="STRONG",
        market="US",
        currency="USD",
        as_of_date="2026-08-27",
        evaluated_at="2026-08-27T10:00:00Z",
        quality_score=90.0,
        roic_pct=25.0,
        ocf_to_net_income=1.1,
        eps_revision_net_30d=20,
        value_score=85.0,
        momentum_score=80.0,
        growth_score=85.0,
        dividend_score=60.0,
        solvency_score=90.0,
        fcf_quality_score=90.0,
        debt_quality_score=90.0,
        atomic_market_snapshot=stale_snapshot,
    )
    scorecard_stale, _, _ = compute_deterministic_scorecard(
        ticker="STRONG",
        market="US",
        quant_signals=strong_signals,
        piotroski=PiotroskiFScoreBreakdown(status="available", f_score=9),
        tactical=tactical_stage2,
    )
    assert scorecard_stale.action_stance == "HOLD_WAIT"
    assert scorecard_stale.stance_mode == "wait"
