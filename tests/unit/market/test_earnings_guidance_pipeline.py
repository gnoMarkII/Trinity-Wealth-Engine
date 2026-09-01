"""Unit tests for Semantic Guidance & Quote Verification Pipeline (Phase 1 & v3.1)."""
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.market.guidance_engine import (
    extract_verified_earnings_guidance,
    verify_quote_presence,
    compute_quote_hash,
)
from tools.market.equity_rules_engine import compute_deterministic_scorecard
from schemas.micro_quant_schemas import QuantSignals, EarningsGuidanceContext


def test_quote_verification_helpers():
    content = "Apple delivered record June quarter revenue of $85.8 billion, up 5 percent from a year ago."
    quote = "record June quarter revenue of $85.8 billion"
    
    is_verified, start, end = verify_quote_presence(quote, content)
    assert is_verified is True
    assert start > 0
    assert content[start:end] == quote
    
    q_hash = compute_quote_hash(quote)
    assert len(q_hash) == 16


def test_guidance_extraction_with_verbatim_quote_verification():
    with TemporaryDirectory() as tmp_dir:
        vault_base = Path(tmp_dir)
        note_dir = vault_base / "30_Knowledge_Base" / "Earnings_Calls" / "NVDA"
        note_dir.mkdir(parents=True, exist_ok=True)
        
        note_content = """---
title: NVDA Earnings Call 2025-Q2
ticker: NVDA
period: 2025-Q2
guidance:
  revenue_growth_guidance_pct: 125.0
  operating_margin_trajectory: expanding
  management_tone: bullish
  key_executive_quotes:
    - "Demand for Blackwell platforms is incredible and continues to exceed supply."
    - "This quote does not exist anywhere in transcript"
---
# NVDA Earnings Call — 2025-Q2

## 🤖 AI Highlights
Management expects revenue growth of 125% next quarter.

## 📄 Full Transcript
Demand for Blackwell platforms is incredible and continues to exceed supply. We are seeing tremendous enterprise momentum.
"""
        note_file = note_dir / "2025-Q2_NVDA_Earnings_Call.md"
        note_file.write_text(note_content, encoding="utf-8")

        guidance, flags = extract_verified_earnings_guidance(
            ticker="NVDA",
            target_fiscal_period="2025-Q2",
            vault_base=vault_base,
        )

        assert guidance is not None
        assert guidance.fiscal_quarter == "2025-Q2"
        assert guidance.revenue_growth_guidance_pct == 125.0
        assert guidance.operating_margin_trajectory == "expanding"
        assert guidance.management_tone == "bullish"
        assert len(guidance.key_executive_quotes) == 1
        assert "Demand for Blackwell platforms" in guidance.key_executive_quotes[0]
        assert "unverified_candidate_quote_dropped" in flags


def test_guidance_missing_transcript_fallback():
    with TemporaryDirectory() as tmp_dir:
        vault_base = Path(tmp_dir)
        guidance, flags = extract_verified_earnings_guidance(
            ticker="UNKNOWN_TICKER",
            target_fiscal_period="2025-Q2",
            vault_base=vault_base,
        )

        assert guidance is None
        assert "missing_earnings_call_transcript" in flags

        # Verify that deterministic scorecard handles None guidance gracefully
        signals = QuantSignals(
            ticker="UNKNOWN_TICKER",
            market="US",
            evaluated_at="2026-08-28T12:00:00Z",
            quality_score=75.0,
            value_score=60.0,
            momentum_score=80.0,
        )
        scorecard, falsifiers, sc_flags = compute_deterministic_scorecard(
            ticker="UNKNOWN_TICKER",
            market="US",
            quant_signals=signals,
            guidance=None,
        )
        assert scorecard.core_conviction_score >= 1.0
        assert scorecard.coverage_pct >= 50.0
