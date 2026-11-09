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


def test_r08_new_radar_and_holding_fields_survive_dto_mapping_and_llm_payload():
    """R08: Newly computed risk/friction/active-share fields survive DTO mapping and synthesis payload."""
    from application.portfolio.advisory_service import _convert_report_to_dto
    from core.portfolio_advisory.llm_payload_builder import build_advisory_llm_payload

    mock_state = MagicMock()
    h1 = _build_mock_holding("PTT.BK", 500, 100.0, bucket_id="core")
    h2 = _build_mock_holding("SCB.BK", 500, 100.0, bucket_id="core")
    mock_state.holdings = [h1, h2]
    mock_state.buckets = [MagicMock(id="core", target_percent=100.0)]
    mock_state.summary = MagicMock(usd_thb_rate=35.50, cash_balance_thb=100000.0)

    sidecars = {
        "PTT.BK": {
            "ticker": "PTT.BK",
            "market": "TH",
            "quant_signals": {
                "raw_analysis_price": 100.0,
                "volatility_pct": 10.0,
                "adtv_local_currency": 2_000_000.0,
                "deterministic_scorecard": {"core_conviction_score": 0.0, "action_stance": "HOLD"},
            },
        },
        "SCB.BK": {
            "ticker": "SCB.BK",
            "market": "TH",
            "quant_signals": {
                "raw_analysis_price": 100.0,
                "volatility_pct": 40.0,
                "adtv_local_currency": 1_000_000.0,
                "deterministic_scorecard": {"core_conviction_score": 8.5, "action_stance": "ACCUMULATE_NOW"},
                "tactical_setup": {"buy_zone_max": 105.0, "buy_zone_min": 95.0},
            },
        },
    }

    report = evaluate_portfolio_advisory(portfolio_state=mock_state, equity_sidecars=sidecars)
    dto = _convert_report_to_dto(report)
    payload = build_advisory_llm_payload(report)

    # Radar fields preserved in DTO & LLM payload
    assert dto.risk_radar.bucket_active_share_pct == float(report.risk_radar.bucket_active_share_pct)
    assert dto.risk_radar.bucket_active_share_status == "AVAILABLE"
    assert dto.risk_radar.risk_contribution_status == "AVAILABLE"
    assert dto.risk_radar.total_portfolio_volatility_pct == float(report.risk_radar.total_portfolio_volatility_pct)
    assert dto.risk_radar.risk_contributions_pct["PTT.BK"] == 11.34
    assert dto.risk_radar.risk_contributions_pct["SCB.BK"] == 88.66
    assert dto.risk_radar.inverse_vol_weights_pct["PTT.BK"] == 80.0
    assert dto.risk_radar.inverse_vol_weights_pct["SCB.BK"] == 20.0
    assert dto.synthesis_input_hash == report.synthesis_input_hash != ""

    assert payload["risk_radar"]["bucket_active_share_pct"] == float(report.risk_radar.bucket_active_share_pct)
    assert payload["risk_radar"]["risk_contributions_pct"] == {"PTT.BK": 11.34, "SCB.BK": 88.66}
    assert payload["risk_radar"]["inverse_vol_weights_pct"] == {"PTT.BK": 80.0, "SCB.BK": 20.0}
    assert payload["risk_radar"]["total_portfolio_volatility_pct"] == float(report.risk_radar.total_portfolio_volatility_pct)
    assert payload["synthesis_input_hash"] == report.synthesis_input_hash

    # Holding fields preserved (including 0.0 core_conviction_score not dropped!)
    ptt_dto = next(h for h in dto.holdings_advisory if h.symbol == "PTT.BK")
    assert ptt_dto.core_conviction_score == 0.0
    assert ptt_dto.risk_contribution_pct == 11.34
    assert ptt_dto.volatility_pct == 10.0
    assert ptt_dto.volatility_source == "observed"
    assert ptt_dto.is_volatility_estimated is False

    scb_dto = next(h for h in dto.holdings_advisory if h.symbol == "SCB.BK")
    scb_payload = next(h for h in payload["holdings_advisory"] if h["symbol"] == "SCB.BK")
    assert scb_dto.risk_contribution_pct == 88.66
    assert scb_dto.estimated_friction_bps == float(next(h for h in report.holdings_advisory if h.symbol == "SCB.BK").estimated_friction_bps)
    assert scb_payload["risk_contribution_pct"] == 88.66
    assert scb_payload["estimated_friction_bps"] == scb_dto.estimated_friction_bps
    assert scb_payload["volatility_pct"] == 40.0
    assert scb_payload["volatility_source"] == "observed"


def test_w04_synthesis_cache_invalidates_on_macro_sector_or_sidecar_changes():
    """W04: Cached playbook is invalidated if macro, sector rotation, or sidecars change despite identical holdings hash."""
    from application.portfolio.advisory_service import PortfolioAdvisoryService
    from schemas.portfolio_advisory_schemas import AdvisorySidecarRecord, PortfolioAdvisoryNarrativeDTO

    mock_state = MagicMock()
    mock_state.holdings = [_build_mock_holding("ADVANC.BK", 100, 240.0)]
    mock_state.allocation_targets = []
    mock_state.buckets = [MagicMock(id="core_equities", target_percent=100.0)]
    mock_state.summary = MagicMock(usd_thb_rate=35.50, cash_balance_thb=50000.0)

    state_reader = MagicMock()
    state_reader.get_current_portfolio_state.return_value = mock_state

    macro_state = {"val": {"overall_regime": "Goldilocks", "evaluated_at": "2026-11-01"}}
    sector_state = {"val": {"as_of_date": "2026-11-01", "rows": [{"ticker": "XLC", "quadrant": "Leading"}]}}
    sidecar_state = {
        "val": {
            "ADVANC": {
                "ticker": "ADVANC.BK",
                "market": "TH",
                "quant_signals": {
                    "raw_analysis_price": 240.0,
                    "volatility_pct": 15.0,
                    "deterministic_scorecard": {"core_conviction_score": 8.0, "action_stance": "ACCUMULATE_NOW"},
                },
            }
        }
    }

    macro_reader = MagicMock()
    macro_reader.get_latest_macro_strategy.side_effect = lambda: macro_state["val"]
    macro_reader.get_latest_sector_rotation.side_effect = lambda: sector_state["val"]
    sidecar_reader = MagicMock()
    sidecar_reader.get_all_equity_sidecars.side_effect = lambda: sidecar_state["val"]
    sidecar_reader.get_watchlist_sidecars.return_value = {}

    saved_records: list[AdvisorySidecarRecord] = []
    vault_port = MagicMock()
    vault_port.load_latest_advisory_sidecar.side_effect = lambda portfolio_id="default": (
        saved_records[-1] if saved_records else None
    )
    vault_port.save_advisory_sidecar.side_effect = lambda rec, portfolio_id="default": saved_records.append(rec)

    synth_calls = {"count": 0}

    def fake_synth(payload_json, report):
        synth_calls["count"] += 1
        return PortfolioAdvisoryNarrativeDTO(
            synthesized_at=f"2026-11-09T12:00:0{synth_calls['count']}Z",
            portfolio_state_hash=report.portfolio_state_hash,
            macro_executive_summary=f"Regime={macro_state['val']['overall_regime']} run={synth_calls['count']}",
            immediate_action_playbook=["Hold ADVANC.BK"],
            conditional_dip_plan=[],
            risk_mitigation_and_hedging=[],
            consistency_verified=True,
        )

    synthesizer = MagicMock()
    synthesizer.synthesize_playbook.side_effect = fake_synth

    svc = PortfolioAdvisoryService(
        portfolio_reader=state_reader,
        macro_reader=macro_reader,
        sidecar_reader=sidecar_reader,
        vault_port=vault_port,
        synthesizer_port=synthesizer,
    )

    # 1. Initial synthesis
    n1 = svc.synthesize_playbook()
    assert synth_calls["count"] == 1
    assert n1.synthesis_input_hash != ""

    # 2. Same inputs -> cache hit in both synthesize_playbook and get_portfolio_advisory
    n2 = svc.synthesize_playbook()
    assert synth_calls["count"] == 1
    assert n2.macro_executive_summary == n1.macro_executive_summary
    dto_hit = svc.get_portfolio_advisory()
    assert dto_hit.cached_narrative is not None

    # 3. Change macro regime -> portfolio_state_hash identical, MUST invalidate cache!
    macro_state["val"] = {"overall_regime": "Stagflation", "evaluated_at": "2026-11-08"}
    dto_stale = svc.get_portfolio_advisory()
    assert dto_stale.cached_narrative is None
    n3 = svc.synthesize_playbook()
    assert synth_calls["count"] == 2
    assert "Stagflation" in n3.macro_executive_summary

    # 4. Change sidecar conviction/stance -> MUST invalidate cache!
    sidecar_state["val"]["ADVANC"]["quant_signals"]["deterministic_scorecard"] = {
        "core_conviction_score": 4.0,
        "action_stance": "REDUCE",
    }
    assert svc.get_portfolio_advisory().cached_narrative is None
    n4 = svc.synthesize_playbook()
    assert synth_calls["count"] == 3
    assert n4.synthesis_input_hash != n3.synthesis_input_hash

    # 5. Legacy cached narrative missing synthesis_input_hash must NOT be treated as fresh
    saved_records[-1].synthesis_input_hash = ""
    saved_records[-1].narrative.synthesis_input_hash = ""
    assert svc.get_portfolio_advisory().cached_narrative is None


def test_w01_engine_fee_inclusive_buy_sizing_and_mutual_fund_global_schedule():
    """W01: BUY recommendations use fee-inclusive lot sizing (order + friction <= target budget)
    and cash surplus accounts for friction; Track 2 mutual funds use 0-bps GLOBAL friction schedule."""
    from core.portfolio_advisory.friction import estimate_order_friction

    # Setup: 1 TH equity with high minimum commission boundary + 1 Mutual Fund
    mock_state = MagicMock()
    h_th = _build_mock_holding("SCB.BK", 50, 100.0, bucket_id="core")  # 5,000 THB
    h_mf = _build_mock_holding("K-USA-A", 1000, 10.0, asset_type="Mutual_Fund", bucket_id="core")  # 10,000 THB
    # Cash = 4,000 THB -> NAV = 19,000 THB
    # For SCB.BK: target weight = 50%, current = 5000/19000 = 26.3% -> drift = -23.7% -> add_pct = 4.0% -> target budget = 760 THB
    # Without fee-inclusive sizing: 7 shares * 100 = 700 + 50 fee = 750 THB <= 760, or if budget is 749:
    # Let's craft exact boundary: 1,000,000 NAV -> 4% add = 40,000 THB, or custom exact boundary:
    # Let's make holding=28,500 at 285/share, or make total NAV so 4% = 28,549.88? Easy:
    # NAV = 28,549.88 / 0.04 = 713,747.00 THB!
    h_edge = _build_mock_holding("BIGP.BK", 0.5, 28500.0, bucket_id="core")  # wait units=1 -> 28,500
    mock_state.holdings = [
        _build_mock_holding("BIGP.BK", 1, 28500.0, bucket_id="core"),  # 28,500 (weight ~4.0%, target 50%)
        _build_mock_holding("K-USA-A", 1000, 10.0, asset_type="Mutual_Fund", bucket_id="core"),  # 10,000
    ]
    # Cash THB = 713,747 - 38,500 = 675,247.00 THB -> total NAV = 713,747.00 THB -> 4% target budget = 28,549.88 THB!
    mock_state.buckets = [MagicMock(id="core", target_percent=100.0)]
    mock_state.summary = MagicMock(usd_thb_rate=35.50, cash_balance_thb=675247.00)

    sidecars = {
        "BIGP.BK": {
            "ticker": "BIGP.BK",
            "market": "TH",
            "quant_signals": {
                "raw_analysis_price": 28500.0,
                "volatility_pct": 15.0,
                "adtv_local_currency": 50000000.0,
                "deterministic_scorecard": {"core_conviction_score": 8.0, "action_stance": "ACCUMULATE_NOW"},
                "tactical_setup": {"buy_zone_max": 30000.0},
            },
        },
        "K-USA-A": {
            "ticker": "K-USA-A",
            "market": "TH",
            "quant_signals": {
                "raw_analysis_price": 10.0,
                "deterministic_scorecard": {"core_conviction_score": 8.0, "action_stance": "ACCUMULATE_NOW"},
                "tactical_setup": {"buy_zone_max": 12.0},
            },
        },
    }

    report = evaluate_portfolio_advisory(portfolio_state=mock_state, equity_sidecars=sidecars)
    bigp = next(h for h in report.holdings_advisory if h.symbol == "BIGP.BK")
    assert bigp.action_code == AdvisoryActionCode.ACCUMULATE_NOW
    # Target budget was 28,549.88 THB, 1 share costs 28,500 + 50.00 fee = 28,550.00 > 28,549.88!
    # Fee-inclusive lot sizing must size 0 shares rather than breaching budget with 50 THB fee!
    assert bigp.lot_sizing is not None
    assert bigp.lot_sizing.amount_native == Decimal("0.00")

    # Mutual fund uses GLOBAL schedule (0 bps explicit brokerage fee instead of 50 THB equity min fee)
    mf = next(h for h in report.holdings_advisory if h.symbol == "K-USA-A")
    assert mf.action_code == AdvisoryActionCode.ACCUMULATE_NOW
    assert mf.lot_sizing is not None
    assert mf.lot_sizing.market == "GLOBAL"
    assert mf.lot_sizing.amount_native > Decimal("0")
    assert mf.estimated_friction_bps == Decimal("0.0")

    # Verify cash surplus deducts friction when equity order actually executes with fees
    mock_state2 = MagicMock()
    mock_state2.holdings = [_build_mock_holding("PTT.BK", 100, 30.0, bucket_id="core")]  # 3,000 (3.0% weight < 15% cap -> BUY!)
    # NAV = 100,000 -> 4% budget = 4,000 THB -> 1 board lot (100 sh @ 30 = 3,000) + 50 fee = 3,050 <= 4,000
    mock_state2.buckets = [MagicMock(id="core", target_percent=100.0)]
    mock_state2.summary = MagicMock(usd_thb_rate=35.50, cash_balance_thb=97000.0)
    sidecars2 = {
        "PTT.BK": {
            "ticker": "PTT.BK",
            "market": "TH",
            "quant_signals": {
                "raw_analysis_price": 30.0,
                "deterministic_scorecard": {"core_conviction_score": 8.0, "action_stance": "ACCUMULATE_NOW"},
                "tactical_setup": {"buy_zone_max": 35.0},
            },
        }
    }
    rep2 = evaluate_portfolio_advisory(portfolio_state=mock_state2, equity_sidecars=sidecars2)
    ptt = rep2.holdings_advisory[0]
    assert ptt.action_code == AdvisoryActionCode.ACCUMULATE_NOW
    assert ptt.lot_sizing is not None
    assert ptt.lot_sizing.amount_native == Decimal("3930.00")
    f_ptt = estimate_order_friction("TH", "BUY", ptt.lot_sizing.amount_native)
    assert f_ptt.total_friction_native == Decimal("50.00")
    assert ptt.lot_sizing.amount_native + f_ptt.total_friction_native <= Decimal("4000.00")
    assert rep2.cash_plan.thb_surplus_after_immediate == (
        Decimal("97000.00") - rep2.cash_plan.immediate_deploy_thb - f_ptt.total_friction_native
    )

