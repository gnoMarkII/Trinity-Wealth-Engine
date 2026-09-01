"""Unit tests for Atomic Market Snapshot, FTNT $172.78 baseline, and Negative Regression Fixtures."""
from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest

from schemas.micro_quant_schemas import (
    AtomicMarketSnapshot,
    QuantSignals,
    ReverseDCFResult,
    TacticalSetup,
    EarningsGuidanceContext,
)
from tools.market.quant_engine import create_atomic_market_snapshot
from tools.market.technical import compute_tactical_setup
from tools.market.dcf_valuation import compute_institutional_reverse_dcf
from tools.market.equity_rules_engine import compute_deterministic_scorecard
from tools.market.financial_autopsy import PiotroskiFScoreBreakdown
from tools.market.evidence_manifest_builder import build_analysis_evidence_snapshot


def test_atomic_market_snapshot_synced():
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=10, freq="B")
    df = pd.DataFrame({"Close": [172.78] * 10}, index=dates)
    info = {"currentPrice": 172.78, "sharesOutstanding": 733713653}
    
    snapshot, flags = create_atomic_market_snapshot("FTNT", df, info, market="US")
    
    assert snapshot.analysis_price == 172.78
    assert snapshot.price_source == "ohlcv_close"
    assert snapshot.price_sync_status == "synced"
    assert snapshot.market_cap == pytest.approx(172.78 * 733713653, abs=1.0)
    assert flags == []


def test_atomic_market_snapshot_quote_ohlcv_mismatch_negative_regression():
    """Negative regression fixture: Live quote $166.00 vs OHLCV close $172.78.
    Must flag quote_ohlcv_mismatch:market_data and strictly bind analysis_price to $172.78.
    """
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=10, freq="B")
    df = pd.DataFrame({"Close": [172.78] * 10}, index=dates)
    info = {"currentPrice": 166.00, "sharesOutstanding": 733713653}
    
    snapshot, flags = create_atomic_market_snapshot("FTNT", df, info, market="US")
    
    assert snapshot.analysis_price == 172.78
    assert snapshot.latest_ohlcv_close == 172.78
    assert snapshot.price_sync_status == "quote_ohlcv_mismatch"
    assert "quote_ohlcv_mismatch:market_data" in flags
    assert snapshot.market_cap == pytest.approx(172.78 * 733713653, abs=1.0)


def test_atomic_market_snapshot_stale_ohlcv():
    old_date = datetime.now(timezone.utc) - timedelta(days=10)
    df = pd.DataFrame({"Close": [172.78]}, index=[old_date])
    info = {"currentPrice": 172.78, "sharesOutstanding": 733713653}
    
    snapshot, flags = create_atomic_market_snapshot("FTNT", df, info, market="US")
    
    assert snapshot.price_sync_status == "stale"
    assert "stale_ohlcv:market_data" in flags


def test_atomic_market_snapshot_fallback_when_ohlcv_missing():
    info = {"currentPrice": 172.78, "sharesOutstanding": 733713653}
    
    snapshot, flags = create_atomic_market_snapshot("FTNT", pd.DataFrame(), info, market="US")
    
    assert snapshot.analysis_price == 172.78
    assert snapshot.price_source == "verified_live_quote"
    assert "ohlcv_unavailable_fallback:market_data" in flags


def test_ftnt_active_baseline_at_172_78():
    """Verify FTNT baseline numbers at active $172.78 close:
    R = 169.91, ATR = 6.14
    Trigger = 170.524, Target = 184.032, Stop = 163.770, Max Chase = 173.594
    At P = $172.78:
    - Pullback R:R is None (P > R)
    - Breakout Planned R:R = 2.00:1
    - Breakout Current R:R = 1.25:1 (< 1.50)
    - Breakout Entry Status = 'chased'
    - Breakout Entry Eligible = False
    - Action Stance = ACCUMULATE_ON_DIP
    """
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=250, freq="B")
    df = pd.DataFrame(index=dates)
    # Construct series with High max = 169.91 and current close = 172.78
    df["Close"] = [160.0] * 249 + [172.78]
    df["High"] = [169.91] * 249 + [172.78]
    df["Low"] = [150.0] * 250
    df["Open"] = [160.0] * 250
    df["Volume"] = 1_000_000

    tactical, _ = compute_tactical_setup("FTNT", market="US", current_price=172.78, price_history_df=df)
    assert tactical is not None
    assert tactical.breakout_planned_rr == 2.00
    assert tactical.max_breakout_chase_price is not None

    tactical_exact = TacticalSetup(
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
        max_breakout_chase_price=173.594,
        breakout_entry_status="chased",
        breakout_entry_eligible=False,
    )

    signals = QuantSignals(
        ticker="FTNT",
        market="US",
        quality_score=90.0,
        value_score=52.5,
        growth_score=80.0,
        momentum_score=75.0,
        dividend_score=50.0,
        roic_pct=25.0,
        beta=1.1,
        volatility_pct=28.0,
        mdd_pct=-15.0,
        evaluated_at="2026-08-30T12:00:00Z",
        data_quality_flags=[],
    )
    piotroski = PiotroskiFScoreBreakdown(status="available", f_score=8)
    reverse_dcf = ReverseDCFResult(
        status="available",
        is_eligible=True,
        target_price_12m=157.66,
        upside_12m_pct=-8.75,
        market_implied_growth_pct=16.48,
        valuation_verdict="fairly_valued",
        reported_ebit_margin_pct=33.86,
        ebit_margin_fiscal_period="2025-12-31",
        ebit_margin_period_type="annual",
        ebit_margin_source_tier="filing_authoritative",
    )
    guidance = EarningsGuidanceContext(
        status="available",
        guidance_status="provided",
        operating_margin_trajectory="expanding",
        management_stance="confident",
    )

    scorecard, _, _ = compute_deterministic_scorecard(
        ticker="FTNT",
        market="US",
        quant_signals=signals,
        piotroski=piotroski,
        dcf=reverse_dcf,
        guidance=guidance,
        tactical=tactical_exact,
    )

    assert scorecard.action_stance == "ACCUMULATE_ON_DIP"
    assert scorecard.action_stance != "BREAKOUT_BUY"


def test_evidence_coverage_dynamic_clamping():
    """Evidence coverage must be dynamically computed from total applicable payloads and clamped <= 100%."""
    raw_payloads = {
        "market_data": {"price": 172.78},
        "technical_ohlcv_1y": {"bars": 250},
        "price_context_5y": {"bars": 1250},
        "mdd_3y": {"mdd": -15.0},
        "beta_2y_stock": {"beta": 1.1},
        "beta_2y_benchmark": {"index": "^GSPC"},
        "analyst_targets": {"mean": 180.0},
        "ownership": {"insiders": 0.05},
        "macro_valuation": {"rf": 4.25},
        "valuation_parameters": {"wacc": 8.5},
        "guidance": {"summary": "confident"},
    }
    manifest = build_analysis_evidence_snapshot(
        ticker="FTNT",
        market="US",
        as_of_date="2026-08-30",
        raw_payloads=raw_payloads,
    )

    assert manifest.metadata.coverage_pct <= 100.0
    assert manifest.metadata.coverage_pct == 100.0
    assert len(manifest.manifest_items) == 11
    assert "technical_ohlcv_1y" in manifest.manifest_items
    assert "guidance" in manifest.manifest_items


def test_frozen_ftnt_fixture_reconcile_map():
    """Verify Frozen FTNT fixture reconcile map and scorecard decontamination."""
    import json
    import math
    from pathlib import Path
    fixture_path = Path(__file__).resolve().parent.parent.parent / "fixtures" / "frozen_ftnt_fixture.json"
    assert fixture_path.exists(), f"Frozen fixture not found: {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Verify Price & Market Cap reconciliation
    fresh = data["fresh_snapshot"]
    computed_mcap = round(fresh["analysis_price"] * fresh["shares_outstanding"], 2)
    assert math.isclose(computed_mcap, fresh["market_cap"], rel_tol=1e-3)

    # 2. Verify Financial Ratios
    fin = data["raw_financials"]
    ebit_margin = round((fin["ebit"] / fin["totalRevenue"]) * 100.0, 2)
    fcf_margin = round((fin["freeCashFlow"] / fin["totalRevenue"]) * 100.0, 2)
    fcf_yield = round((fin["freeCashFlow"] / fresh["market_cap"]) * 100.0, 2)

    assert ebit_margin == data["expected_metrics"]["ebit_margin_pct"]
    assert fcf_margin == data["expected_metrics"]["fcf_margin_pct"]
    assert fcf_yield == data["expected_metrics"]["fcf_yield_pct"]

    # 3. Verify Scorecard Decontamination when ERP <= 0
    reverse_dcf_anomaly = ReverseDCFResult(
        status="available",
        is_eligible=True,
        is_actionable=False,
        actionability_reason="Macro parameter anomaly: ERP (-0.22%) <= 0",
        target_price_12m=157.66,
        upside_12m_pct=-8.75,
    )
    signals = QuantSignals(
        ticker="FTNT",
        market="US",
        currency="USD",
        company_name="Fortinet, Inc.",
        as_of_date="2026-08-27",
        evaluated_at="2026-08-27T10:00:00Z",
        quality_score=85.0,
        value_score=52.5,
    )
    piotroski = PiotroskiFScoreBreakdown(status="available", f_score=8)
    tactical_ftnt = TacticalSetup(
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
        dcf=reverse_dcf_anomaly,
        tactical=tactical_ftnt,
    )

    # Scorecard must not contain valuation_margin_score and must tag valuation_not_actionable
    assert scorecard.valuation_margin_score is None
    assert "valuation_not_actionable:scorecard" in scorecard.data_quality_flags
    assert scorecard.action_stance == "ACCUMULATE_ON_DIP"
