"""Unit tests for deterministic scorecard action stance gating and rules engine."""
import pytest
from schemas.micro_quant_schemas import (
    DCFResult,
    DCFScenario,
    DeterministicScorecard,
    QuantSignals,
    ReverseDCFResult,
    TacticalSetup,
    EarningsGuidanceContext,
)
from tools.market.financial_autopsy import PiotroskiFScoreBreakdown
from tools.market.equity_rules_engine import compute_deterministic_scorecard


def _mock_quant_signals(ticker="FTNT", market="US"):
    return QuantSignals(
        ticker=ticker,
        market=market,
        quality_score=90.0,
        value_score=52.5,
        growth_score=80.0,
        momentum_score=75.0,
        dividend_score=50.0,
        roic_pct=25.0,
        eps_revision_net_30d=3,
        beta=1.1,
        volatility_pct=28.0,
        mdd_pct=-15.0,
        evaluated_at="2026-08-30T12:00:00Z",
        data_quality_flags=[],
    )


def test_ftnt_at_166_is_accumulate_on_dip_not_accumulate_now():
    """FTNT at $166.00 has R:R = 0.24:1 and is outside buy zone ($154.46-$157.53). Must be ACCUMULATE_ON_DIP."""
    signals = _mock_quant_signals()
    piotroski = PiotroskiFScoreBreakdown(status="available", f_score=8)
    reverse_dcf = ReverseDCFResult(
        status="available",
        is_eligible=True,
        target_price_12m=157.7,
        upside_12m_pct=-5.0,
        market_implied_growth_pct=16.48,
        valuation_verdict="fairly_valued",
        reported_ebit_margin_pct=33.86,
    )
    guidance = EarningsGuidanceContext(
        status="available",
        guidance_status="provided",
        operating_margin_trajectory="expanding",
        management_stance="confident",
    )
    tactical = TacticalSetup(
        status="available",
        price_stage="STAGE_2_MARKUP",
        current_price=166.0,
        atr_14=6.14,
        key_support_level=154.46,
        key_resistance_level=169.91,
        buy_zone_min=154.46,
        buy_zone_max=157.53,
        invalidation_stop_loss=149.85,
        tactical_target_price=169.91,
        current_rr_ratio=0.24,
        is_in_buy_zone=False,
        breakout_trigger_price=170.52,
        breakout_target_price=184.03,
        breakout_stop_loss=163.77,
        breakout_planned_rr=2.00,
        breakout_current_rr=None,
    )

    scorecard, falsifiers, flags = compute_deterministic_scorecard(
        ticker="FTNT",
        market="US",
        quant_signals=signals,
        piotroski=piotroski,
        dcf=reverse_dcf,
        guidance=guidance,
        tactical=tactical,
    )

    assert scorecard.action_stance == "ACCUMULATE_ON_DIP"
    assert scorecard.action_stance_reason is not None
    assert "0.24:1" in scorecard.action_stance_reason or "Buy Zone" in scorecard.action_stance_reason


def test_ftnt_at_172_is_accumulate_on_dip_due_to_low_breakout_rr():
    """FTNT at $172.78 is above trigger $170.52, but Breakout Current R:R is 1.25:1 (< 1.50). Must be ACCUMULATE_ON_DIP."""
    signals = _mock_quant_signals()
    piotroski = PiotroskiFScoreBreakdown(status="available", f_score=8)
    reverse_dcf = ReverseDCFResult(
        status="available",
        is_eligible=True,
        target_price_12m=157.7,
        upside_12m_pct=-5.0,
        market_implied_growth_pct=16.48,
        valuation_verdict="fairly_valued",
        reported_ebit_margin_pct=33.86,
    )
    guidance = EarningsGuidanceContext(
        status="available",
        guidance_status="provided",
        operating_margin_trajectory="expanding",
        management_stance="confident",
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
        invalidation_stop_loss=149.85,
        tactical_target_price=169.91,
        current_rr_ratio=None,
        is_in_buy_zone=False,
        breakout_trigger_price=170.524,
        breakout_target_price=184.032,
        breakout_stop_loss=163.770,
        breakout_planned_rr=2.00,
        breakout_current_rr=1.25,
    )

    scorecard, falsifiers, flags = compute_deterministic_scorecard(
        ticker="FTNT",
        market="US",
        quant_signals=signals,
        piotroski=piotroski,
        dcf=reverse_dcf,
        guidance=guidance,
        tactical=tactical,
    )

    assert scorecard.action_stance == "ACCUMULATE_ON_DIP"
    assert scorecard.action_stance != "BREAKOUT_BUY"


def test_valid_breakout_buy_passes_gating():
    """Stock entering breakout trigger with R:R >= 1.50 gets BREAKOUT_BUY."""
    signals = _mock_quant_signals()
    piotroski = PiotroskiFScoreBreakdown(status="available", f_score=8)
    reverse_dcf = ReverseDCFResult(
        status="available",
        target_price_12m=200.0,
        upside_12m_pct=25.0,
        valuation_verdict="undervalued",
        reported_ebit_margin_pct=30.0,
    )
    guidance = EarningsGuidanceContext(
        status="available",
        guidance_status="provided",
        management_stance="confident",
    )
    tactical = TacticalSetup(
        status="available",
        price_stage="STAGE_2_MARKUP",
        current_price=170.60,
        atr_14=6.14,
        key_support_level=154.46,
        key_resistance_level=169.91,
        buy_zone_min=154.46,
        buy_zone_max=157.53,
        invalidation_stop_loss=149.85,
        tactical_target_price=169.91,
        breakout_trigger_price=170.524,
        breakout_target_price=184.032,
        breakout_stop_loss=163.770,
        breakout_planned_rr=2.00,
        breakout_current_rr=1.97,
        breakout_entry_eligible=True,
    )

    scorecard, falsifiers, flags = compute_deterministic_scorecard(
        ticker="FTNT",
        market="US",
        quant_signals=signals,
        piotroski=piotroski,
        dcf=reverse_dcf,
        guidance=guidance,
        tactical=tactical,
    )

    assert scorecard.action_stance == "BREAKOUT_BUY"
    assert "Breakout R:R 1.97:1" in (scorecard.action_stance_reason or "")


def test_insufficient_data_when_tactical_unavailable():
    """When tactical setup is unavailable or coverage is low, stance must be INSUFFICIENT_DATA."""
    signals = _mock_quant_signals()
    scorecard, _, _ = compute_deterministic_scorecard(
        ticker="UNK",
        market="US",
        quant_signals=signals,
        tactical=None,
    )
    assert scorecard.action_stance == "INSUFFICIENT_DATA"
