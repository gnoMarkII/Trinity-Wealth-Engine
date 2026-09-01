"""Unit tests สำหรับ agents/equity_synthesizer.py และ prompt harness rendering"""
import json
from unittest.mock import MagicMock, patch

from agents.equity_synthesizer import invoke_equity_synthesizer
from core.prompt_harness import get_harness
from schemas.micro_quant_schemas import EquityNarrativeOutput, QuantSignals, EquitySentimentContext, DCFResult, DCFScenario


def test_equity_synthesizer_prompt_harness_renders_properly():
    """ตรวจสอบว่า prompt harness โหลด SKILL.md และ HUMAN.md ได้สมบูรณ์ โดยไม่เกิด Mustache format error"""
    harness = get_harness("equity_synthesizer")
    system_prompt = harness.get_system_prompt()
    assert "Synthesizer" in system_prompt
    assert "Data Quality & Confidence Analysis" in system_prompt

    human_content = harness.get_skill_text(
        "HUMAN.md",
        quant_json='{"ticker": "AAPL", "data_quality_flags": ["hardcoded_us_risk_free:dcf"]}',
        narrative_json='{"market_sentiment": "bullish"}',
    )
    assert "Quant Signals" in human_content
    assert "AAPL" in human_content
    assert "hardcoded_us_risk_free:dcf" in human_content
    assert "bullish" in human_content


@patch("validators.structured_output_retry.invoke_with_retry")
def test_invoke_equity_synthesizer_calls_retry_layer_correctly(mock_invoke_with_retry):
    """ตรวจสอบว่า invoke_equity_synthesizer ส่ง parameters และ schema ไปยัง invoke_with_retry ตรงตามสัญญา"""
    expected_output = EquityNarrativeOutput(
        narrative_analysis=(
            "1. Apple Inc. มีคุณภาพกระแสเงินสดระดับสูง...\n"
            "2. DCF Engine ประเมินมูลค่าเหมาะสม...\n"
            "3. Data Quality Flags มี hardcoded_us_risk_free:dcf...\n"
            "4. Smart Money สะท้อนความเชื่อมั่น...\n"
            "5. Catalysts และ Risks..."
        ),
        base_case_summary="ราคาประเมินเหมาะสม $524.4 สถานะ UNDERVALUED",
    )
    mock_invoke_with_retry.return_value = (expected_output, [])

    quant = QuantSignals(
        ticker="AAPL",
        market="US",
        evaluated_at="2026-08-08T00:00:00Z",
        data_quality_flags=["hardcoded_us_risk_free:dcf", "rich_market_valuation_low_erp:dcf"],
    )
    narrative = EquitySentimentContext(
        evaluated_at="2026-08-08T00:00:00Z",
        market_sentiment="bullish",
        key_themes=["Strong FCF Yield"],
        tail_risks=["Low ERP"],
        sources_summary="Vault + News",
    )

    mock_model = MagicMock()
    res = invoke_equity_synthesizer(
        model=mock_model,
        quant_json=quant.model_dump_json(),
        narrative_json=narrative.model_dump_json(),
    )

    assert res.narrative_analysis == expected_output.narrative_analysis
    assert res.base_case_summary == expected_output.base_case_summary
    mock_invoke_with_retry.assert_called_once()
    kwargs = mock_invoke_with_retry.call_args.kwargs
    assert kwargs["output_schema"] == EquityNarrativeOutput
    assert kwargs["agent_name"] == "equity_synthesizer"
