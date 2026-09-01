"""Unit tests for Legacy Sidecar Fallback & Backward Compatibility (Phase 0 & v3.1)."""
import json
import pytest
from schemas.micro_quant_schemas import MicroQuantOutput, QuantSignals


def test_legacy_v1_sidecar_json_deserializes_without_evidence_snapshot():
    legacy_json_payload = {
        "ticker": "PTT",
        "market": "TH",
        "analysis_date": "2026-08-03",
        "quant_signals": {
            "ticker": "PTT",
            "market": "TH",
            "evaluated_at": "2026-08-03T09:30:00Z",
            "composite_score": 55.0,
            "value_score": 60.0,
            "growth_score": 40.0,
            "quality_score": 50.0,
            "momentum_score": 45.0,
            "data_quality_flags": ["missing_eps_q3"]
        },
        "sentiment_context": {
            "evaluated_at": "2026-08-03T09:30:00Z",
            "market_sentiment": "neutral",
            "key_themes": ["Oil volatility"],
            "tail_risks": ["Oversupply"],
            "sources_summary": "Daily energy brief",
            "report_references": []
        },
        "narrative_analysis": "PTT is navigating a challenging commodity cycle.",
        "base_case_summary": "Stable dividend play with limited multiple expansion.",
        "generated_by": "equity_intel"
    }

    # Deserializing v1 payload into MicroQuantOutput must succeed cleanly with evidence_snapshot=None
    output = MicroQuantOutput.model_validate(legacy_json_payload)
    assert output.ticker == "PTT"
    assert output.quant_signals.composite_score == 55.0
    assert output.quant_signals.evidence_snapshot is None
    assert output.evidence_snapshot is None


def test_legacy_quant_signals_deserialization():
    legacy_signals_payload = {
        "ticker": "AAPL",
        "market": "US",
        "evaluated_at": "2026-08-03T10:00:00Z",
        "composite_score": 82.5,
        "value_score": 65.0,
        "growth_score": 90.0,
        "quality_score": 95.0,
        "momentum_score": 80.0
    }

    signals = QuantSignals.model_validate(legacy_signals_payload)
    assert signals.ticker == "AAPL"
    assert signals.composite_score == 82.5
    assert signals.evidence_snapshot is None
    assert signals.piotroski_breakdown is None
