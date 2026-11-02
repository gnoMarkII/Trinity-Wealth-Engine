"""Unit tests for Tier 1 Deterministic Portfolio Advisory Engine and Synthesizer.

Verifies:
- S4: 3-Mode Graceful Degradation (Full, Micro-Only, Bucket-Only)
- D2: Dual-Currency Cash Ledger and Archived Holdings exclusion
- D5: Anti-Falling-Knife enforcement and Two-Basket Execution Budget
- I5: Cash Overdraft & FX conversion calculation
- I6: Post-LLM Narrative Consistency & Contradiction Detection
"""
from decimal import Decimal
from unittest.mock import MagicMock
import pytest

from core.portfolio_advisory.engine import evaluate_portfolio_advisory
from core.portfolio_advisory.models import (
    AdvisoryActionCode,
    AdvisoryDataMode,
    PortfolioAdvisoryNarrative,
)
from core.portfolio_advisory.narrative_validator import validate_advisory_narrative_consistency


def _build_mock_holding(
    symbol: str,
    units: float,
    current_price_thb: float,
    asset_type: str = "Stock",
    status: str = "active",
    avg_cost_thb: float = None,
    avg_cost_usd: float = None,
    current_price_usd: float = None,
    bucket_id: str = "core_equities",
    unrealized_pnl_percent: float = 5.0,
):
    h = MagicMock()
    h.symbol = symbol
    h.name = f"{symbol} Company"
    h.units = units
    h.market_value_thb = units * current_price_thb
    h.current_price_thb = current_price_thb
    h.current_price_usd = current_price_usd
    h.asset_type = asset_type
    h.status = status
    h.avg_cost_thb = avg_cost_thb
    h.avg_cost_usd = avg_cost_usd
    h.bucket_id = bucket_id
    h.unrealized_pnl_percent = unrealized_pnl_percent
    return h


def test_engine_graceful_degradation_modes():
    """S4: Graceful degradation from FULL_CROSS_SCALE to PORTFOLIO_PLUS_MICRO and BUCKET_ONLY."""
    mock_state = MagicMock()
    mock_state.holdings = [_build_mock_holding("ADVANC.BK", 100, 240.0, avg_cost_thb=220.0)]
    mock_state.buckets = [MagicMock(id="core_equities", target_percent=100.0)]
    mock_state.summary = MagicMock(usd_thb_rate=35.50, cash_balance_thb=50000.0)

    # 1. Bucket only (no macro, no micro)
    rep_bucket = evaluate_portfolio_advisory(portfolio_state=mock_state)
    assert rep_bucket.data_mode == AdvisoryDataMode.PORTFOLIO_BUCKET_ONLY
    assert rep_bucket.total_nav_thb > Decimal("0")

    # 2. Portfolio + Micro
    fake_sidecars = {
        "ADVANC": {
            "quant_signals": {
                "raw_analysis_price": 240.0,
                "peer_sector": "Communication Services",
                "deterministic_scorecard": {
                    "core_conviction_score": 8.0,
                    "action_stance": "ACCUMULATE_NOW",
                },
            }
        }
    }
    rep_micro = evaluate_portfolio_advisory(
        portfolio_state=mock_state,
        equity_sidecars=fake_sidecars,
    )
    assert rep_micro.data_mode == AdvisoryDataMode.PORTFOLIO_PLUS_MICRO

    # 3. Full Cross Scale
    mock_macro = MagicMock()
    mock_macro.evaluated_at = "2026-10-18T10:00:00Z"
    mock_macro.sector_analysis = {"leading_sectors": ["communication services"], "lagging_sectors": []}
    mock_macro.model_dump.return_value = {"overall_regime": "Expansion"}

    rep_full = evaluate_portfolio_advisory(
        portfolio_state=mock_state,
        macro_direction=mock_macro,
        equity_sidecars=fake_sidecars,
    )
    assert rep_full.data_mode == AdvisoryDataMode.FULL_CROSS_SCALE


def test_engine_anti_falling_knife_and_two_basket_budget():
    """D5: Anti-falling-knife blocks buying Stage 4 markdown stocks; separates immediate vs conditional dip."""
    h_winner = _build_mock_holding("NVDA", 10, 130.0 * 35.5, avg_cost_usd=100.0, current_price_usd=130.0)
    h_knife = _build_mock_holding("FALLEN", 100, 50.0, avg_cost_thb=100.0)
    h_dip = _build_mock_holding("DIPPER", 200, 100.0, avg_cost_thb=90.0)

    mock_state = MagicMock()
    mock_state.holdings = [h_winner, h_knife, h_dip]
    mock_state.buckets = [MagicMock(id="core_equities", target_percent=100.0)]
    mock_state.summary = MagicMock(usd_thb_rate=35.50, cash_balance_thb=500000.0)

    sidecars = {
        "NVDA": {
            "quant_signals": {
                "raw_analysis_price": 130.0,
                "peer_sector": "Technology",
                "deterministic_scorecard": {"core_conviction_score": 8.5, "business_conviction_score": 8.5, "action_stance": "ACCUMULATE_NOW"},
                "tactical_setup": {"price_stage": "STAGE_2_UPTREND", "buy_zone_max": 135.0, "buy_zone_min": 120.0},
            }
        },
        "FALLEN": {
            "quant_signals": {
                "raw_analysis_price": 50.0,
                "peer_sector": "Consumer Discretionary",
                "deterministic_scorecard": {"core_conviction_score": 4.0, "action_stance": "REDUCE"},
                "tactical_setup": {"price_stage": "STAGE_4_MARKDOWN", "invalidation_stop_loss": 55.0},
            }
        },
        "DIPPER": {
            "quant_signals": {
                "raw_analysis_price": 100.0,
                "peer_sector": "Financial Services",
                "deterministic_scorecard": {"core_conviction_score": 7.5, "action_stance": "ACCUMULATE_ON_DIP"},
                "tactical_setup": {"price_stage": "STAGE_1_ACCUMULATION", "buy_zone_max": 95.0, "buy_zone_min": 90.0},
            }
        },
    }

    report = evaluate_portfolio_advisory(portfolio_state=mock_state, equity_sidecars=sidecars)

    # 1. Check FALLEN stock: Must be URGENT_REDUCE and NOT buyable
    fallen_advisory = next(a for a in report.holdings_advisory if a.symbol == "FALLEN")
    assert fallen_advisory.action_code == AdvisoryActionCode.URGENT_REDUCE
    assert "Stop Loss" in fallen_advisory.action_rationale or "Stage 4" in fallen_advisory.action_rationale

    # 2. Check NVDA: In buy zone -> ACCUMULATE_NOW with immediate execution
    nvda_advisory = next(a for a in report.holdings_advisory if a.symbol == "NVDA")
    assert nvda_advisory.action_code == AdvisoryActionCode.ACCUMULATE_NOW
    assert nvda_advisory.is_immediate_execution is True

    # 3. Check DIPPER: Current price 100 > buy_zone_max 95 -> ACCUMULATE_ON_DIP with conditional limit
    dipper_advisory = next(a for a in report.holdings_advisory if a.symbol == "DIPPER")
    assert dipper_advisory.action_code == AdvisoryActionCode.ACCUMULATE_ON_DIP
    assert dipper_advisory.is_immediate_execution is False

    # 4. Check Cash Plan separation
    assert report.cash_plan.immediate_deploy_usd > Decimal("0")
    assert report.cash_plan.conditional_limit_order_thb > Decimal("0")


def test_dual_currency_cash_overdraft_guard():
    """I5: Buying US stocks exceeding USD cash pool calculates fx_conversion_needed_thb."""
    # US stock buy that requires USD cash
    h_us = _build_mock_holding("AAPL", 10, 180.0 * 35.5, avg_cost_usd=160.0, current_price_usd=180.0)

    # Cash has 500,000 THB but only $50 USD
    h_cash_thb = _build_mock_holding("CASH_THB", 500000, 1.0, asset_type="cash_thb")
    h_cash_usd = _build_mock_holding("CASH_USD", 50, 35.5, asset_type="cash_usd", current_price_usd=1.0)

    mock_state = MagicMock()
    mock_state.holdings = [h_us, h_cash_thb, h_cash_usd]
    mock_state.buckets = [MagicMock(id="core_equities", target_percent=100.0)]
    mock_state.summary = MagicMock(usd_thb_rate=35.50, cash_balance_thb=500000.0)

    sidecars = {
        "AAPL": {
            "quant_signals": {
                "raw_analysis_price": 180.0,
                "peer_sector": "Technology",
                "deterministic_scorecard": {"core_conviction_score": 8.0, "action_stance": "ACCUMULATE_NOW"},
                "tactical_setup": {"buy_zone_max": 185.0},
            }
        }
    }

    report = evaluate_portfolio_advisory(portfolio_state=mock_state, equity_sidecars=sidecars)
    assert report.cash_plan.cash_usd_available == Decimal("50")
    if report.cash_plan.immediate_deploy_usd > Decimal("50"):
        assert report.cash_plan.fx_conversion_needed_thb > Decimal("0")
        assert any(w.guardrail_code == "CASH_OVERDRAFT_AND_FX_POOL_GUARD" for w in report.anomaly_warnings)


def test_narrative_consistency_and_contradiction_validation():
    """I6: Post-LLM Validator detects directional contradictions and hallucinated tickers."""
    mock_state = MagicMock()
    h1 = _build_mock_holding("AAPL", 10, 180.0 * 35.5, avg_cost_usd=160.0, current_price_usd=180.0)
    h2 = _build_mock_holding("BADCO", 100, 50.0, avg_cost_thb=100.0)
    mock_state.holdings = [h1, h2]
    mock_state.buckets = [MagicMock(id="core_equities", target_percent=100.0)]
    mock_state.summary = MagicMock(usd_thb_rate=35.50, cash_balance_thb=50000.0)

    sidecars = {
        "AAPL": {
            "quant_signals": {
                "raw_analysis_price": 180.0,
                "deterministic_scorecard": {"core_conviction_score": 8.0, "action_stance": "ACCUMULATE_NOW"},
            }
        },
        "BADCO": {
            "quant_signals": {
                "raw_analysis_price": 50.0,
                "deterministic_scorecard": {"core_conviction_score": 3.0, "action_stance": "REDUCE"},
                "tactical_setup": {"invalidation_stop_loss": 55.0},
            }
        },
    }

    report = evaluate_portfolio_advisory(portfolio_state=mock_state, equity_sidecars=sidecars)

    # 1. Valid narrative
    valid_narrative = {
        "macro_executive_summary": "ภาพรวมพอร์ตมีความสมดุล แนะนำเคาะซื้อ AAPL และตัดขาดทุน BADCO",
        "immediate_action_playbook": [
            "เคาะซื้อ AAPL ตามสัดส่วนคำสั่งซื้อวันนี้",
            "ขายตัดขาดทุน BADCO เนื่องจากหลุดระดับ Stop Loss",
        ],
        "conditional_dip_plan": [],
        "risk_mitigation_and_hedging": ["ลดความเสี่ยงโดยขาย BADCO"],
    }
    is_ok, errs = validate_advisory_narrative_consistency(valid_narrative, report)
    assert is_ok is True
    assert len(errs) == 0

    # 2. Contradiction: Advising to buy BADCO which is URGENT_REDUCE
    contradictory_narrative = {
        "macro_executive_summary": "ภาพรวมพอร์ตมีความสมดุล",
        "immediate_action_playbook": [
            "แนะนำให้เคาะซื้อ BADCO เพิ่มเติมเพื่อถัวเฉลี่ยต้นทุน",
        ],
        "conditional_dip_plan": [],
        "risk_mitigation_and_hedging": [],
    }
    is_ok_c, errs_c = validate_advisory_narrative_consistency(contradictory_narrative, report)
    assert is_ok_c is False
    assert any("Directional Contradiction on BADCO" in e for e in errs_c)

    # 3. Hallucinated Ticker: Mentioning TSLA which is not in the portfolio
    hallucinated_narrative = {
        "macro_executive_summary": "แนะนำให้เปิดสถานะใน TSLA ทันที",
        "immediate_action_playbook": [
            "ซื้อ TSLA จำนวน 50 หุ้น",
        ],
        "conditional_dip_plan": [],
        "risk_mitigation_and_hedging": [],
    }
    is_ok_h, errs_h = validate_advisory_narrative_consistency(hallucinated_narrative, report)
    assert is_ok_h is False
    assert any("Hallucinated Ticker Detected: 'TSLA'" in e for e in errs_h)

    # 4. Thai-adjacent Hallucinated Ticker without spaces
    hallucinated_thai_adjacent = {
        "macro_executive_summary": "แนะนำให้เปิดสถานะในNVDAทันทีเพื่อรับคลื่นAI",
        "immediate_action_playbook": [],
        "conditional_dip_plan": [],
        "risk_mitigation_and_hedging": [],
    }
    is_ok_th, errs_th = validate_advisory_narrative_consistency(hallucinated_thai_adjacent, report)
    assert is_ok_th is False
    assert any("Hallucinated Ticker Detected: 'NVDA'" in e for e in errs_th)


def test_real_portfolio_state_schema_integration():
    """Verifies real PortfolioState fields: fx_rates['USDTHB'], allocation_targets, company_name, and QuantSignals.beta."""
    from tools.portfolio.domain.models import AllocationTarget, Holding, PortfolioState, Summary

    real_state = PortfolioState(
        portfolio_id="default",
        name="Main Portfolio",
        last_updated="2026-10-18T10:00:00Z",
        fx_rates={"USDTHB": 36.25},
        allocation_targets=[
            AllocationTarget(bucket_id="core_equities", name="Core Equities", target_percent=60.0),
            AllocationTarget(bucket_id="cash", name="Cash Buffer", target_percent=40.0),
        ],
        holdings=[
            Holding(
                symbol="ADVANC.BK",
                asset_type="Stock",
                bucket_id="core_equities",
                units=200.0,
                avg_cost_thb=220.0,
                current_price_thb=250.0,
                market_value_thb=50000.0,
                company_name="Advanced Info Service PCL",
            ),
            Holding(
                symbol="MSFT",
                asset_type="Stock",
                bucket_id="core_equities",
                units=10.0,
                avg_cost_usd=400.0,
                current_price_usd=420.0,
                current_price_thb=420.0 * 36.25,
                market_value_thb=10.0 * 420.0 * 36.25,
                company_name="Microsoft Corp.",
            ),
            Holding(
                symbol="CASH_THB",
                asset_type="cash_thb",
                bucket_id="cash",
                units=100000.0,
                current_price_thb=1.0,
                market_value_thb=100000.0,
            ),
        ],
        summary=Summary(
            total_value_thb=50000.0 + (4200.0 * 36.25) + 100000.0,
            cash_balance_thb=100000.0,
        ),
    )

    sidecars = {
        "ADVANC": {
            "quant_signals": {
                "raw_analysis_price": 250.0,
                "beta": 0.85,
                "peer_sector": "Communication Services",
                "deterministic_scorecard": {"core_conviction_score": 8.0, "action_stance": "HOLD"},
            }
        },
        "MSFT": {
            "quant_signals": {
                "raw_analysis_price": 420.0,
                "beta": 1.15,
                "peer_sector": "Technology",
                "deterministic_scorecard": {"core_conviction_score": 8.5, "action_stance": "HOLD"},
            }
        },
    }

    report = evaluate_portfolio_advisory(portfolio_state=real_state, equity_sidecars=sidecars)
    assert report.cash_plan.usd_thb_fx_rate == Decimal("36.2500")
    assert report.risk_radar.weighted_beta is not None
    assert report.risk_radar.weighted_beta > Decimal("1.00")
    assert "core_equities" in report.risk_radar.purpose_bucket_drift
    assert "cash" in report.risk_radar.purpose_bucket_drift

    advanc_item = next(i for i in report.holdings_advisory if i.symbol == "ADVANC.BK")
    assert advanc_item.asset_name == "Advanced Info Service PCL"
    # 60% bucket target divided across 2 holdings = 30.0% per holding
    assert advanc_item.target_weight_pct == Decimal("30.0")


def test_strategic_rm_narrative_and_consistency():
    """Verifies Strategic RM Lens (What's Working, What Concerns Me, Discussion Points)."""
    from agents.portfolio_synthesizer import build_deterministic_fallback_narrative

    mock_state = MagicMock()
    h1 = _build_mock_holding("NVDA", 10, 130.0 * 35.5, avg_cost_usd=100.0, current_price_usd=130.0)
    h2 = _build_mock_holding("BADCO", 100, 50.0, avg_cost_thb=100.0)
    h_cash = _build_mock_holding("CASH_THB", 50000, 1.0, asset_type="cash_thb")
    mock_state.holdings = [h1, h2, h_cash]
    mock_state.buckets = [MagicMock(id="core_equities", target_percent=100.0)]
    mock_state.summary = MagicMock(usd_thb_rate=35.50, cash_balance_thb=50000.0)

    sidecars = {
        "NVDA": {
            "quant_signals": {
                "raw_analysis_price": 130.0,
                "deterministic_scorecard": {"core_conviction_score": 9.0, "action_stance": "ACCUMULATE_NOW"},
                "tactical_setup": {"buy_zone_max": 135.0, "buy_zone_min": 120.0},
            }
        },
        "BADCO": {
            "quant_signals": {
                "raw_analysis_price": 50.0,
                "deterministic_scorecard": {"core_conviction_score": 3.0, "action_stance": "REDUCE"},
                "tactical_setup": {"invalidation_stop_loss": 55.0},
            }
        },
    }

    report = evaluate_portfolio_advisory(portfolio_state=mock_state, equity_sidecars=sidecars)

    # 1. Test Deterministic Fallback builds Strategic RM fields
    fallback_dto = build_deterministic_fallback_narrative(report)
    assert len(fallback_dto.whats_working) > 0
    assert any("NVDA" in item for item in fallback_dto.whats_working)
    assert len(fallback_dto.what_concerns_me) > 0
    assert any("BADCO" in item for item in fallback_dto.what_concerns_me)
    assert len(fallback_dto.strategic_discussion_points) > 0

    # 2. Test validator accepts valid strategic narrative
    valid_narrative = {
        "macro_executive_summary": "ภาพรวมพอร์ตมุ่งเน้นการเติบโตแบบ Selective Growth",
        "whats_working": ["NVDA ได้ประโยชน์จากความต้องการโครงสร้างพื้นฐาน AI"],
        "what_concerns_me": ["BADCO หลุดกรอบแนวรับสำคัญและต้องตัดขาดทุน"],
        "immediate_action_playbook": ["เคาะซื้อ NVDA ตามโซนราคา", "ขายลดน้ำหนัก BADCO"],
        "conditional_dip_plan": [],
        "risk_mitigation_and_hedging": [],
        "strategic_discussion_points": ["ทบทวนสัดส่วน Hedge ค่าเงิน USD"],
    }
    is_ok, errs = validate_advisory_narrative_consistency(valid_narrative, report)
    assert is_ok is True
    assert len(errs) == 0

    # 3. Test validator catches hallucinated ticker in whats_working
    hallucinated_narrative = {
        **valid_narrative,
        "whats_working": ["TSLA ได้รับแรงหนุนจากยอดส่งมอบรถยนต์พุ่งสูง"],  # TSLA not in portfolio
    }
    is_ok_h, errs_h = validate_advisory_narrative_consistency(hallucinated_narrative, report)
    assert is_ok_h is False
    assert any("Hallucinated Ticker Detected: 'TSLA'" in e for e in errs_h)


def test_hhi_and_effective_holdings_math():
    """Verifies deterministic HHI, effective holdings count, and concentration risk grading."""
    # 1. Single-stock portfolio (100% in one stock) -> HHI = 1.0, Neff = 1.0
    mock_single = MagicMock()
    h_single = _build_mock_holding("ONE", 100, 100.0)  # 10,000 THB
    mock_single.holdings = [h_single]
    mock_single.buckets = [MagicMock(id="core", target_percent=100.0)]
    mock_single.summary = MagicMock(usd_thb_rate=35.0, cash_balance_thb=0.0)

    report_single = evaluate_portfolio_advisory(portfolio_state=mock_single, equity_sidecars={})
    assert report_single.risk_radar.hhi == Decimal("1.0000")
    assert report_single.risk_radar.effective_holdings_count == Decimal("1.0")
    assert report_single.risk_radar.top_1_weight_pct == Decimal("100.0")
    assert report_single.risk_radar.concentration_risk_level == "HIGH"

    # 2. Skewed portfolio (False Diversification): 5 stocks (1 stock 60%, 4 stocks 10% each)
    # HHI = 0.6^2 + 4 * 0.1^2 = 0.36 + 0.04 = 0.4000
    # Neff = 1 / 0.40 = 2.5
    mock_skewed = MagicMock()
    h_big = _build_mock_holding("BIG", 60, 100.0)  # 6,000 THB (60%)
    h_s1 = _build_mock_holding("S1", 10, 100.0)   # 1,000 THB (10%)
    h_s2 = _build_mock_holding("S2", 10, 100.0)   # 1,000 THB (10%)
    h_s3 = _build_mock_holding("S3", 10, 100.0)   # 1,000 THB (10%)
    h_s4 = _build_mock_holding("S4", 10, 100.0)   # 1,000 THB (10%)
    mock_skewed.holdings = [h_big, h_s1, h_s2, h_s3, h_s4]
    mock_skewed.buckets = [MagicMock(id="core", target_percent=100.0)]
    mock_skewed.summary = MagicMock(usd_thb_rate=35.0, cash_balance_thb=0.0)

    report_skewed = evaluate_portfolio_advisory(portfolio_state=mock_skewed, equity_sidecars={})
    assert report_skewed.risk_radar.hhi == Decimal("0.4000")
    assert report_skewed.risk_radar.effective_holdings_count == Decimal("2.5")
    assert report_skewed.risk_radar.top_1_weight_pct == Decimal("60.0")
    assert report_skewed.risk_radar.top_3_weight_pct == Decimal("80.0")
    assert report_skewed.risk_radar.concentration_risk_level == "HIGH"

    # 3. Well-diversified 10-stock portfolio (10 stocks, 10% each)
    # HHI = 10 * 0.01 = 0.1000
    # Neff = 10.0
    mock_div = MagicMock()
    mock_div.holdings = [_build_mock_holding(f"D{i}", 10, 100.0) for i in range(10)]
    mock_div.buckets = [MagicMock(id="core", target_percent=100.0)]
    mock_div.summary = MagicMock(usd_thb_rate=35.0, cash_balance_thb=0.0)

    report_div = evaluate_portfolio_advisory(portfolio_state=mock_div, equity_sidecars={})
    assert report_div.risk_radar.hhi == Decimal("0.1000")
    assert report_div.risk_radar.effective_holdings_count == Decimal("10.0")
    assert report_div.risk_radar.top_1_weight_pct == Decimal("10.0")
    assert report_div.risk_radar.top_3_weight_pct == Decimal("30.0")
    assert report_div.risk_radar.concentration_risk_level == "LOW"

