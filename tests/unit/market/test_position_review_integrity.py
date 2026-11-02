import pytest
from core.equity_analytics.decomposition_engine import PriceMoveDecompositionEngine
from application.equity.refresh_workflow import sanitize_quant_payload_for_llm
from schemas.micro_quant_schemas import QuantSignals


def test_price_move_decomposition_mathematical_identity():
    """Verify that Total Price Return correctly decomposes into EPS Growth and P/E Expansion."""
    decomp = PriceMoveDecompositionEngine.evaluate(
        horizon_label="1Y",
        start_date="2025-10-30",
        end_date="2026-10-30",
        start_price=100.0,
        end_price=150.0,
        start_eps=5.0,
        end_eps=6.0,
    )

    # 1. Price grew by 50%
    assert decomp.total_return_pct == 50.0
    # 2. EPS grew from 5.0 to 6.0 (+20%)
    assert decomp.eps_growth_pct == 20.0
    # 3. P/E expanded from 20.0 (100/5) to 25.0 (150/6) (+25%)
    assert decomp.start_pe == 20.0
    assert decomp.end_pe == 25.0
    assert decomp.pe_expansion_pct == 25.0
    # 4. Identity: (1 + 0.20) * (1 + 0.25) = 1.20 * 1.25 = 1.50 (+50% Total Return)
    reconstructed_return = ((1 + decomp.eps_growth_pct / 100.0) * (1 + decomp.pe_expansion_pct / 100.0) - 1.0) * 100.0
    assert pytest.approx(reconstructed_return, 0.01) == decomp.total_return_pct
    assert decomp.diagnosis_tier == "SPECULATIVE_EXPANSION" or decomp.diagnosis_tier == "FUNDAMENTAL_DRIVEN"


def test_price_move_decomposition_boundary_conditions_no_nan():
    """Verify that zero/negative EPS and zero prices do not crash or produce NaN/Infinity."""
    # Loss-making start
    decomp_loss = PriceMoveDecompositionEngine.evaluate(
        horizon_label="1Y",
        start_date="2025-10-30",
        end_date="2026-10-30",
        start_price=50.0,
        end_price=60.0,
        start_eps=-1.5,
        end_eps=2.0,
    )
    assert not (decomp_loss.total_return_pct != decomp_loss.total_return_pct)  # No NaN
    assert decomp_loss.start_pe == 0.0
    assert decomp_loss.end_pe == 0.0
    assert "Negative/Zero EPS" in decomp_loss.diagnosis_summary

    # Zero price guard
    decomp_zero_p = PriceMoveDecompositionEngine.evaluate(
        horizon_label="1Y",
        start_date="2025-10-30",
        end_date="2026-10-30",
        start_price=0.0,
        end_price=10.0,
        start_eps=1.0,
        end_eps=1.2,
    )
    assert decomp_zero_p.total_return_pct == 0.0
    assert not (decomp_zero_p.total_return_pct != decomp_zero_p.total_return_pct)


def test_ai_quant_payload_sanitization_integrity():
    """Verify that sanitize_quant_payload_for_llm preserves all critical analytical signals."""
    quant_dict = {
        "ticker": "NVDA",
        "market": "US",
        "evaluated_at": "2026-10-30T00:00:00Z",
        "company_name": "NVIDIA Corporation",
        "deterministic_scorecard": {
            "action_stance": "ACCUMULATE_ON_DIP",
            "action_stance_reason": "High business conviction but waiting for tactical pullback",
            "fundamental_quality_score": 85.0,
            "guidance_expectation_score": 90.0,
            "valuation_margin_score": 35.0,
            "core_conviction_score": 7.8,
        },
        "reverse_dcf_result": {
            "market_implied_growth_pct": 18.5,
            "reported_ebit_margin_pct": 32.0,
            "solver_status": "converged",
        },
        "tactical_setup": {
            "buy_zone_min": 120.0,
            "buy_zone_max": 128.0,
            "invalidation_stop_loss": 115.0,
            "tactical_target_price": 145.0,
            "price_stage": "STAGE_2_MARKUP",
        },
        "thesis_falsifiers": [
            {
                "falsifier_id": "MARGIN_FLOOR",
                "metric_name": "Operating Margin Floor",
                "condition": "< 25.0%",
                "threshold_value": 25.0,
                "narrative_explanation": "Break below 25% margin falsifies thesis",
            }
        ],
        "smart_money_flags": {
            "overall_smart_money_flag": "bullish_signal",
            "institutional_ownership_pct": 74.5,
            "short_interest_pct": 1.8,
        },
        "insider_conviction": {
            "open_market_p_count_90d": 2,
            "open_market_p_value_usd": 1500000,
        },
    }

    payload = sanitize_quant_payload_for_llm(quant_dict)

    # Verify that payload contains all analytical signals required by AI
    assert "deterministic_scorecard" in payload
    assert payload["deterministic_scorecard"]["action_stance"] == "ACCUMULATE_ON_DIP"
    assert "reverse_dcf_result" in payload
    assert payload["reverse_dcf_result"]["market_implied_growth_pct"] == 18.5
    assert "tactical_setup" in payload
    assert payload["tactical_setup"]["invalidation_stop_loss"] == 115.0
    assert "thesis_falsifiers" in payload
    assert len(payload["thesis_falsifiers"]) == 1
    assert "smart_money_flags" in payload
    assert "insider_conviction" in payload

    # Also verify with QuantSignals Pydantic instance
    pydantic_signals = QuantSignals(
        ticker="NVDA",
        market="US",
        evaluated_at="2026-10-30T00:00:00Z",
        company_name="NVIDIA Corporation",
        value_score=65.0,
        quality_score=92.0,
    )
    pydantic_payload = sanitize_quant_payload_for_llm(pydantic_signals)
    assert pydantic_payload["ticker"] == "NVDA"
    assert pydantic_payload["quality_score"] == 92.0

