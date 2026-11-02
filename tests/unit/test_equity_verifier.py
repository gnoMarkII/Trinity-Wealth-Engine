"""Unit tests for Equity Narrative Verifier in validators/equity_verifier.py."""
from schemas.micro_quant_schemas import EquityNarrativeOutput
from validators.equity_verifier import EquityVerificationResult, invoke_equity_with_verifier
from unittest.mock import MagicMock


def test_equity_verifier_detects_empty_content():
    narrative = EquityNarrativeOutput(
        narrative_analysis="Short",
        base_case_summary="",
    )
    result = EquityVerificationResult.from_narrative(narrative, {})
    assert result.should_retry is True
    assert any("สั้นเกินไป" in issue for issue in result.issues)


def test_equity_verifier_detects_dcf_contradiction_when_dcf_is_unavailable():
    narrative = EquityNarrativeOutput(
        narrative_analysis="บริษัทมีคุณภาพดีมาก เป้าหมายราคาตาม DCF อยู่ที่ 150 บาทต่อหุ้น",
        base_case_summary="ประเมินมูลค่าด้วย DCF ได้ 150 บาท",
    )
    quant_data = {
        "quant_signals": {
            "data_quality_flags": ["negative_fcf_dcf_unavailable:dcf"]
        }
    }
    result = EquityVerificationResult.from_narrative(narrative, quant_data)
    assert result.should_retry is True
    assert any("พบความขัดแย้งเชิงตรรกะ" in issue for issue in result.issues)


def test_equity_verifier_detects_action_stance_contradiction():
    narrative = EquityNarrativeOutput(
        narrative_analysis="แนะนำซื้ออย่างยิ่งและเข้าสะสมเต็มที่สำหรับหุ้นตัวนี้",
        base_case_summary="Strong Buy ซื้อเต็มพอร์ต",
    )
    quant_data = {
        "deterministic_scorecard": {
            "action_stance": "WAIT_DO_NOT_ENTER"
        }
    }
    result = EquityVerificationResult.from_narrative(narrative, quant_data)
    assert result.should_retry is True
    assert any("พบความขัดแย้งกับ Scorecard" in issue for issue in result.issues)


def test_equity_verifier_passes_valid_narrative():
    narrative = EquityNarrativeOutput(
        narrative_analysis="Apple Inc. มีกระแสเงินสดที่แข็งแกร่งอย่างต่อเนื่องจากกลุ่ม Services แม้ Valuation อยู่ในระดับพรีเมียม",
        base_case_summary="ราคาเหมาะสมอยู่ที่ $220 ประเมินด้วย Multiple P/E",
    )
    quant_data = {
        "quant_signals": {
            "data_quality_flags": ["hardcoded_us_risk_free:dcf"]
        },
        "deterministic_scorecard": {
            "action_stance": "HOLD_FULL_SIZED"
        }
    }
    result = EquityVerificationResult.from_narrative(narrative, quant_data)
    assert result.should_retry is False
    assert len(result.issues) == 0


def test_invoke_equity_with_verifier_records_clean_telemetry():
    narrative = EquityNarrativeOutput(
        narrative_analysis="Microsoft Corporation มีการเติบโตอย่างแข็งแกร่งของ Azure Cloud และ AI Copilot",
        base_case_summary="ราคาเหมาะสมอยู่ที่ $450 ประเมินด้วย Multiple EV/EBITDA",
    )
    mock_model = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = narrative
    mock_model.with_structured_output.return_value = mock_structured

    out, quality = invoke_equity_with_verifier(
        model=mock_model,
        messages=[{"role": "user", "content": "Analyze MSFT"}],
        quant_data={"ticker": "MSFT"},
        max_retries=1,
    )
    assert quality.should_retry is False
    assert len(quality.issues) == 0

