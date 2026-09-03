"""Unit tests for Guidance Sourced Facts vs Inferences, FY2025 audited baselines (19.1% and 25.1%), and thesis falsifiers."""
import pytest
from pathlib import Path
from tools.market.guidance_engine import extract_verified_earnings_guidance
from schemas.micro_quant_schemas import ThesisFalsifier


def test_guidance_facts_extraction_and_baseline_growth(tmp_path):
    # Setup mock earnings note in vault structure
    note_dir = tmp_path / "30_Knowledge_Base" / "Earnings_Calls" / "FTNT"
    note_dir.mkdir(parents=True, exist_ok=True)
    note_file = note_dir / "2026-Q2_FTNT_Earnings.md"

    body_text = """# Fortinet Q2 2026 Earnings Call
> "For the full year 2026, we expect revenue to be in the range of $8.020 billion to $8.180 billion, representing strong execution across our unified SASE and SecOps portfolios."
> "Non-GAAP operating margin is expected to be in the range of 35.0% to 37.0%."
"""
    frontmatter = """---
period: "2026-Q2"
guidance:
  management_tone: "bullish"
  key_executive_quotes:
    - "For the full year 2026, we expect revenue to be in the range of $8.020 billion to $8.180 billion, representing strong execution across our unified SASE and SecOps portfolios."
    - "Non-GAAP operating margin is expected to be in the range of 35.0% to 37.0%."
  revenue_guidance_range_usd_b: [8.020, 8.180]
  billings_guidance_range_usd_b: [9.350, 9.550]
  non_gaap_operating_margin_range_pct: [35.0, 37.0]
  diluted_non_gaap_eps_range_usd: [3.41, 3.47]
  fy2025_audited_revenue_usd_b: 6.7996
  fy2025_audited_billings_usd_b: 7.5537
  operating_margin_trajectory: "expanding"
---
"""
    note_file.write_text(frontmatter + body_text, encoding="utf-8")

    ctx, flags = extract_verified_earnings_guidance(ticker="FTNT", target_fiscal_period="2026-Q2", vault_base=tmp_path)
    assert ctx is not None
    assert ctx.status == "available"
    assert ctx.management_tone == "bullish"
    assert ctx.operating_margin_trajectory == "expanding"
    assert len(ctx.verified_claims) >= 3

    # Check Sourced Claims & Baseline Growth
    # Rev growth: (8.100 - 6.7996) / 6.7996 = +19.1246% -> 19.1%
    rev_claim = next(c for c in ctx.verified_claims if c.metric_name == "revenue_growth_guidance_pct")
    assert rev_claim.numeric_value == 19.1

    # Billings growth: (9.450 - 7.5537) / 7.5537 = +25.1042% -> 25.1%
    billings_claim = next(c for c in ctx.verified_claims if c.metric_name == "billings_growth_guidance_pct")
    assert billings_claim.numeric_value == 25.1

    # Margin midpoint: (35.0 + 37.0) / 2 = 36.0%
    margin_claim = next(c for c in ctx.verified_claims if c.metric_name == "non_gaap_operating_margin_midpoint_pct")
    assert margin_claim.numeric_value == 36.0


def test_thesis_falsifier_schema_validation():
    falsifier = ThesisFalsifier(
        falsifier_id="falsifier_rev_growth_below_15",
        metric_name="revenue_growth_guidance_pct",
        condition="<",
        threshold_value=15.0,
        source_basis="guidance_quote",
        source_ref="vault:///30_Knowledge_Base/Earnings_Calls/FTNT/2026-Q2_FTNT_Earnings.md#char_0_50",
        source_quote="revenue growth expected to be 19.1%",
        narrative_explanation="Revenue growth falling below 15% violates positive forward outlook.",
    )
    assert falsifier.falsifier_id == "falsifier_rev_growth_below_15"
    assert falsifier.threshold_value == 15.0
