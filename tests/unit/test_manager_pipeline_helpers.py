"""Unit tests for agents/pipelines helper functions extracted from manager_agent.py."""
from datetime import datetime, timezone
import pytest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from agents.pipelines.archivist_helpers import (
    build_archivist_task_instruction,
    resolve_archivist_verification_spec,
)
from agents.pipelines.macro_helpers import (
    build_economist_fallback_dict,
    canonicalize_macro_report_direction,
    enrich_macro_direction_post_allocation,
    prepare_allocator_quant_payload,
)
from agents.pipelines.equity_helpers import (
    build_equity_narrative_fallback_dict,
    build_fallback_equity_output,
    build_provisional_equity_output,
    build_synthesized_equity_output,
)
from schemas.macro_schemas import (
    AssetAllocationView,
    AssetStance,
    EconomicState,
    MacroStrategyDirection,
    MarketObservable,
)
from schemas.micro_quant_schemas import (
    EquitySentimentContext,
    MicroQuantOutput,
    QuantSignals,
)


# =====================================================================
# Archivist Pipeline Helpers
# =====================================================================

def test_build_archivist_task_instruction_macro_intel():
    """Macro intel source extracts title from YAML frontmatter and adds deterministic directive."""
    macro_content = "---\ntitle: Macro Strategy Report 2026-10-16\n---\n# Macro Report Content"
    state = {
        "route_meta": {"source": "macro_intel"},
        "messages": [AIMessage(content=macro_content)],
    }
    instruction = build_archivist_task_instruction(
        state,
        extract_reply_fn=lambda msgs: msgs[-1].content,
        has_frontmatter_fn=lambda text: "---" in text,
    )

    assert "write_raw_markdown" in instruction
    assert "filename='Macro Strategy Report 2026-10-16'" in instruction
    assert "[ข้อมูลดิบ]" in instruction


def test_build_archivist_task_instruction_equity_intel_with_news():
    """Equity intel with valid news instructs calling write_raw_markdown twice."""
    report_content = "---\ntitle: NVDA Equity Intel 2026-10-16\n---\n# Analysis"
    news_content = "---\ntitle: NVDA Recent News 2026-10-16\n---\n# News"
    state = {
        "route_meta": {"source": "equity_intel"},
        "equity_news_raw": news_content,
        "messages": [AIMessage(content=report_content)],
    }
    instruction = build_archivist_task_instruction(
        state,
        extract_reply_fn=lambda msgs: msgs[-1].content,
        has_frontmatter_fn=lambda text: "---" in text,
    )

    assert "ทั้งหมด 2 ครั้ง" in instruction
    assert "filename='NVDA Recent News 2026-10-16'" in instruction
    assert "filename='NVDA Equity Intel 2026-10-16'" in instruction


def test_build_archivist_task_instruction_equity_intel_single_call():
    """Equity intel without valid news instructs calling write_raw_markdown once."""
    report_content = "---\ntitle: AAPL Equity Intel 2026-10-16\n---\n# Analysis"
    state = {
        "route_meta": {"source": "equity_intel"},
        "equity_news_raw": None,
        "messages": [AIMessage(content=report_content)],
    }
    instruction = build_archivist_task_instruction(
        state,
        extract_reply_fn=lambda msgs: msgs[-1].content,
        has_frontmatter_fn=lambda text: "---" in text,
    )

    assert "ทั้งหมด 2 ครั้ง" not in instruction
    assert "filename='AAPL Equity Intel 2026-10-16'" in instruction


def test_resolve_archivist_verification_spec():
    """Verifies tool requirement and min call count based on route meta and prompt content."""
    # Macro intel -> 1 call required
    tools, min_calls = resolve_archivist_verification_spec({"source": "macro_intel"}, "บันทึกข้อมูล")
    assert tools == {"write_raw_markdown", "save_memory"}
    assert min_calls == 1

    # Equity intel with 2 calls directive -> 2 calls required
    tools, min_calls = resolve_archivist_verification_spec(
        {"source": "equity_intel"},
        "คุณต้องเรียกใช้เครื่องมือ write_raw_markdown ทั้งหมด 2 ครั้งเพื่อบันทึก",
    )
    assert tools == {"write_raw_markdown", "save_memory"}
    assert min_calls == 2

    # Direct user message -> no tool enforcement
    tools, min_calls = resolve_archivist_verification_spec({"source": "user"}, "สรุปให้หน่อย")
    assert tools is None
    assert min_calls == 1


# =====================================================================
# Macro Pipeline Helpers
# =====================================================================

def test_prepare_allocator_quant_payload():
    """Separates valid and invalid observables and builds evaluated source list."""
    obs_valid = MarketObservable(
        observable_id="obs_gdp_us",
        asset_bucket="fixed_income",
        region="US",
        indicator="GDP Growth",
        value="2.5",
        unit="%",
        observed_at="2026-10-15",
        source_file="us_gdp.md",
        provider="fred",
        is_valid=True,
    )
    obs_invalid = MarketObservable(
        observable_id="obs_stale_cpi",
        asset_bucket="fixed_income",
        region="US",
        indicator="CPI",
        value="3.1",
        unit="%",
        observed_at="2025-01-01",
        source_file="old_cpi.md",
        provider="manual",
        is_valid=False,
        stale_reason="Data is older than 90 days",
    )

    state = {
        "quant_score": {
            "market_observables": [obs_valid.model_dump(), obs_invalid.model_dump()],
            "evaluated_at": "2026-10-16T12:00:00Z",
        },
        "sector_rotation_context": {"leading_sectors": ["Technology"]},
        "sector_rotation_snapshot_id": "snap-123",
        "sector_context_status": {"status": "available"},
    }

    payload, registry, sources, eval_date = prepare_allocator_quant_payload(state, eval_date_override="2026-10-16")

    assert len(payload["valid_observables"]) == 1
    assert payload["valid_observables"][0]["observable_id"] == "obs_gdp_us"
    assert len(payload["invalid_observables"]) == 1
    assert payload["invalid_observables"][0]["observable_id"] == "obs_stale_cpi"
    assert "us_gdp.md" in sources
    assert "Macro_Baseline_2026-10-16.md" in sources
    assert eval_date == "2026-10-16"


def test_enrich_macro_direction_post_allocation_set_equities():
    """Injects SET Equities (Thailand) with UNDERWEIGHT when foreign flow is negative."""
    obs_set = MarketObservable(
        observable_id="obs_set_flow_foreign",
        asset_bucket="equities",
        region="Thailand",
        indicator="Foreign Net Flow",
        value="-1250.5",
        unit="THB Mil",
        observed_at="2026-10-16",
        source_file="set_flow.md",
        provider="settrade",
        is_valid=True,
    )
    obs_breadth = MarketObservable(
        observable_id="obs_set_advance_decline_ratio",
        asset_bucket="equities",
        region="Thailand",
        indicator="Advance Decline Ratio",
        value="0.75",
        unit="x",
        observed_at="2026-10-16",
        source_file="set_flow.md",
        provider="settrade",
        is_valid=True,
    )
    registry = {
        "obs_set_flow_foreign": obs_set,
        "obs_set_advance_decline_ratio": obs_breadth,
    }

    direction = MacroStrategyDirection(
        evaluated_at="2026-10-16T00:00:00Z",
        overall_regime=EconomicState.REFLATION,
        key_assumptions=["Soft landing base case"],
        focus_themes=["AI Capex"],
        conviction_level="medium",
        conviction_rationale="Supported by corporate earnings and moderate inflation",
        quant_narrative_alignment="aligned",
        asset_allocation=[
            AssetAllocationView(
                asset_class="Global Cash",
                asset_bucket="cash",
                region="US",  # Deliberately set to US to test normalization to Global
                stance=AssetStance.NEUTRAL,
                rationale="Liquidity buffer with 5.0% yield",
                confidence="high",
                supporting_data=["Cash yield 5.0%"],
                source_refs=["ref1", "ref2"],
            )
        ],
    )

    allocator_quant = {
        "regions": {
            "Thailand": {
                "market_stance": {
                    "investor_flow": {"foreign_net_mb": -1250.5},
                    "valuation": {"pe_ratio": 16.2},
                    "market_breadth": {"advance_decline_ratio": 0.75},
                }
            }
        }
    }

    enriched = enrich_macro_direction_post_allocation(direction, allocator_quant, registry)

    # Cash region normalized
    cash_view = next(a for a in enriched.asset_allocation if a.asset_bucket == "cash")
    assert cash_view.region == "Global"

    # SET Equities injected and stance is UNDERWEIGHT due to negative flow & breadth < 0.8
    thai_equity = next(a for a in enriched.asset_allocation if a.region == "Thailand" and a.asset_bucket == "equities")
    assert thai_equity.stance == AssetStance.UNDERWEIGHT
    assert "obs_set_flow_foreign" in thai_equity.observable_refs
    assert "set_flow.md" in enriched.source_files


def test_canonicalize_macro_report_direction():
    """Tests canonical report extraction from sidecar result and direction model validation."""
    direction = MacroStrategyDirection(
        evaluated_at="2026-10-16T00:00:00Z",
        overall_regime=EconomicState.REFLATION,
        key_assumptions=["Early recovery"],
        focus_themes=["Rate cuts"],
        conviction_level="medium",
        conviction_rationale="Supported by macroeconomic recovery indicators",
        quant_narrative_alignment="aligned",
        asset_allocation=[],
    )
    sidecar_tuple = (
        "Macro_Report.json",
        {
            **direction.model_dump(mode="json"),
            "strategy_report_id": "rep-456",
            "sector_analysis": {"leading": ["Financials"]},
        },
    )

    canon_dir, canon_report, sector_data = canonicalize_macro_report_direction(
        direction, sidecar_tuple, {"leading": ["Financials"]}
    )

    assert isinstance(canon_dir, MacroStrategyDirection)
    assert canon_report["strategy_report_id"] == "rep-456"
    assert sector_data == {"leading": ["Financials"]}


def test_build_economist_fallback_dict():
    """Economist fallback dictionary has neutral sentiment and evaluated_at timestamp."""
    fallback = build_economist_fallback_dict("Connection timed out")
    assert fallback["market_sentiment"] == "neutral"
    assert "Connection timed out" in fallback["sources_summary"]
    assert "evaluated_at" in fallback


# =====================================================================
# Equity Intel Pipeline Helpers
# =====================================================================

def _create_sample_quant_signals() -> QuantSignals:
    return QuantSignals(
        ticker="TSLA",
        market="US",
        company_name="Tesla Inc",
        evaluated_at="2026-10-16T12:00:00Z",
        value_score=72.5,
        quality_score=85.0,
        momentum_score=68.0,
    )


def test_build_provisional_equity_output():
    """Provisional output creates valid MicroQuantOutput with pending narrative."""
    signals = _create_sample_quant_signals()
    provisional = build_provisional_equity_output(signals, analysis_date="2026-10-16")

    assert provisional.ticker == "TSLA"
    assert provisional.analysis_date == "2026-10-16"
    assert provisional.narrative_status == "pending"
    assert "รอการประมวลผล" in provisional.narrative_analysis


def test_build_equity_narrative_fallback_dict():
    """Equity narrative fallback dict formats neutral sentiment context correctly."""
    fallback = build_equity_narrative_fallback_dict("JSON syntax error")
    assert fallback["market_sentiment"] == "neutral"
    assert "JSON syntax error" in fallback["sources_summary"]


def test_build_synthesized_equity_output():
    """Synthesizer output properly combines quant, narrative context, and narrative text."""
    signals = _create_sample_quant_signals()
    quant_data = signals.model_dump(mode="json")
    narrative_data = {
        "evaluated_at": "2026-10-16T12:00:00Z",
        "market_sentiment": "bullish",
        "key_themes": ["EV adoption", "Robotaxi potential"],
        "tail_risks": ["Tariffs"],
        "sources_summary": "Q3 earnings transcript and SEC filings",
    }
    mock_narrative_result = MagicMock()
    mock_narrative_result.narrative_analysis = "Bullish momentum supported by margins."
    mock_narrative_result.base_case_summary = "Target price revised upwards."

    output = build_synthesized_equity_output(quant_data, narrative_data, mock_narrative_result, analysis_date="2026-10-16")

    assert isinstance(output, MicroQuantOutput)
    assert output.ticker == "TSLA"
    assert output.narrative_analysis == "Bullish momentum supported by margins."
    assert output.sentiment_context.market_sentiment == "bullish"


def test_build_fallback_equity_output_preserves_quant():
    """When synthesis LLM fails, quant signals are preserved with fallback narrative."""
    signals = _create_sample_quant_signals()
    quant_data = signals.model_dump(mode="json")
    narrative_data = {}

    fallback_output = build_fallback_equity_output(quant_data, narrative_data, "RateLimitError", analysis_date="2026-10-16")

    assert fallback_output is not None
    assert fallback_output.ticker == "TSLA"
    assert "LLM_UNAVAILABLE: RateLimitError" in fallback_output.narrative_analysis
    assert fallback_output.quant_signals.ticker == "TSLA"

    # When quant data is empty/invalid, returns None
    assert build_fallback_equity_output({}, {}, "error") is None
