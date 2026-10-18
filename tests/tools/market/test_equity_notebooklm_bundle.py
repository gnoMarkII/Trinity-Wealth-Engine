"""Unit tests for Equity NotebookLM Bundle Builder and Formatters."""
import json
from pathlib import Path
import pytest

from application.equity.notebooklm_export_ports import EquityCorpusSnapshot
from tools.market.notebooklm_bundle_builder import EquityExportBundleBuilder
from tools.market.notebooklm_bundle_formatter import (
    format_earnings_calls,
    format_financials,
    format_historical_analyses,
    format_insider_ownership,
    format_latest_analysis,
    format_market_derivatives,
    format_news_and_references,
    format_price_and_notes,
    format_research_guide,
    format_structured_appendix,
)


@pytest.fixture
def sample_equity_snapshot() -> EquityCorpusSnapshot:
    return EquityCorpusSnapshot(
        ticker="NFLX",
        market="US",
        snapshot_at="2026-10-16T12:00:00Z",
        latest_analysis={
            "ticker": "NFLX",
            "market": "US",
            "company_name": "Netflix Inc.",
            "analysis_date": "2026-10-16",
            "evaluated_at": "2026-10-16T10:00:00Z",
            "quant_signals": {
                "scorecard": {
                    "action_stance": "HOLD_WAIT",
                    "action_stance_reason": "Stage 4 consolidation",
                    "core_conviction_score": 5.5,
                    "business_conviction_score": 6.0,
                    "investment_conviction_score": 5.0,
                    "execution_readiness_score": 3.0,
                },
                "piotroski_f_score": 7,
                "roic_pct": 21.4,
                "dcf_valuation": {
                    "intrinsic_value": 95.71,
                    "current_price": 72.60,
                    "upside_pct": 31.8,
                    "implied_revenue_growth_5y_pct": 4.4,
                },
                "sector_kpis": {
                    "sector": "Communication Services",
                    "subscriber_net_adds": 8.05,
                },
            },
            "portfolio_context": {
                "is_held": True,
                "units": 12.318,
                "avg_cost": 80.03,
                "current_weight_pct": 4.47,
                "unrealized_pnl_pct": -9.3,
            },
            "thesis_delta": {
                "previous_analysis_date": "2026-09-15",
                "price_change_pct": -5.2,
                "conviction_score_delta": 0.5,
                "breached_falsifiers": ["Operating margin dropped below 22%"],
            },
            "base_case_summary": "Streaming dominance with ad-tier scaling.",
            "narrative_analysis": "Valuation has de-risked following recent pullback.",
        },
        historical_analyses=[
            {
                "analysis_date": "2026-09-15",
                "evaluated_at": "2026-09-15T10:00:00Z",
                "quant_signals": {"scorecard": {"action_stance": "HOLD"}},
                "base_case_summary": "Previous neutral stance.",
            }
        ],
        sec_financials={
            "cik": "0001065280",
            "statements": {
                "income_statement": {"Revenues": "9825000000", "OperatingIncome": "2600000000"}
            },
        },
        sec_insider_trades={
            "sentiment": "NEUTRAL",
            "transactions": [
                {
                    "filing_date": "2026-10-01",
                    "insider_name": "Hastings Reed",
                    "title": "Executive Chairman",
                    "transaction_type": "Sale",
                    "shares": 10000,
                    "price": 75.0,
                    "value": 750000,
                }
            ],
        },
        news=[
            {
                "title": "Netflix Expands Live Sports Streaming",
                "published_at": "2026-10-15",
                "source": "Bloomberg",
                "summary": "NFLX secures additional live broadcast events.",
            }
        ],
        metadata={"counts": {"historical_analyses": 1}, "warnings": []},
    )


def test_format_all_parts(sample_equity_snapshot):
    guide = format_research_guide(sample_equity_snapshot)
    assert "NFLX Equity Research Companion Guide" in guide
    assert "Recommended Research Questions" in guide

    latest = format_latest_analysis(sample_equity_snapshot)
    assert "HOLD_WAIT" in latest
    assert "95.71" in latest
    assert "Reverse DCF" in latest
    assert "Portfolio Position Context" in latest
    assert "Thesis Evolution Delta" in latest

    history = format_historical_analyses(sample_equity_snapshot)
    assert "2026-09-15" in history

    fin = format_financials(sample_equity_snapshot)
    assert "SEC EDGAR" in fin
    assert "9825000000" in fin

    insider = format_insider_ownership(sample_equity_snapshot)
    assert "Hastings Reed" in insider

    news = format_news_and_references(sample_equity_snapshot)
    assert "Live Sports Streaming" in news

    app = format_structured_appendix(sample_equity_snapshot)
    assert "NFLX" in app


def test_bundle_builder_creates_ten_sources_and_deterministic_hash(
    sample_equity_snapshot, tmp_path: Path
):
    builder = EquityExportBundleBuilder(export_root=tmp_path)
    bundle_dir, content_hash_1, inv_1 = builder.build_bundle("exp_test_1", sample_equity_snapshot)

    assert bundle_dir.is_dir()
    sources_dir = bundle_dir / "sources"
    assert sources_dir.is_dir()

    files = list(sources_dir.glob("*.md"))
    assert len(files) == 10

    assert (bundle_dir / "corpus.json").is_file()
    assert (bundle_dir / "inventory.json").is_file()
    assert inv_1["total_sources"] == 10

    # Test determinism
    _, content_hash_2, _ = builder.build_bundle("exp_test_2", sample_equity_snapshot)
    assert content_hash_1 == content_hash_2

    # Test inventory coverage flags
    assert inv_1["has_latest_analysis"] is True
    assert "has_financials" in inv_1
    assert "has_sec_filings" in inv_1
    assert "has_insider_trades" in inv_1
    assert "has_options_data" in inv_1
    assert "has_portfolio_holding" in inv_1


def test_format_latest_analysis_none_safety_and_sector_kpis():
    """Verify None-safety for portfolio holding and proper formatting for sector_kpis list."""
    snapshot = EquityCorpusSnapshot(
        ticker="TEST",
        market="US",
        snapshot_at="2026-10-17T12:00:00Z",
        latest_analysis={
            "quant_signals": {
                "sector_kpis": {
                    "sector_name": "Communication Services",
                    "sector_health_score": 92.5,
                    "kpis": [
                        {
                            "name": "Operating Margin",
                            "value": 29.9,
                            "unit": "percent",
                            "benchmark": ">= 15%",
                        }
                    ],
                },
                "portfolio_context": {
                    "is_held": False,
                    "position_action_hint": "WAIT_DO_NOT_ENTER",
                    "units": 0.0,
                    "avg_cost": None,
                    "current_weight_pct": None,
                    "unrealized_pnl_pct": None,
                },
            }
        },
    )

    formatted = format_latest_analysis(snapshot)
    assert "Sector Classification:** `Communication Services`" in formatted
    assert "Sector Health Score:** `92.5 / 100`" in formatted
    assert "Operating Margin:** `29.9%` (Benchmark: `>= 15%`)" in formatted
    assert "Holding Status:** `Not Held`" in formatted
    assert "Units Held:** `0.0000` @ `N/A`" in formatted


def test_format_production_quant_schema_no_na_values():
    """Verify that real production quant signals render without N/A values."""
    prod_snapshot = EquityCorpusSnapshot(
        ticker="NFLX",
        market="US",
        snapshot_at="2026-10-17T01:39:27Z",
        latest_analysis={
            "ticker": "NFLX",
            "market": "US",
            "company_name": "Netflix, Inc.",
            "analysis_date": "2026-10-16",
            "evaluated_at": "2026-10-16T15:16:00Z",
            "quant_signals": {
                "raw_analysis_price": 72.61,
                "revenue_growth_yoy_pct": 15.85,
                "net_income_growth_yoy_pct": 26.05,
                "fcf_yield_pct": 3.69,
                "fcf_margin_pct": 23.06,
                "fcf_cagr_3y": 80.14,
                "interest_coverage": 17.16,
                "net_debt_ebitda": 0.51,
                "roic_pct": 33.69,
                "ocf_to_net_income": 0.88,
                "debt_quality_score": 100.0,
                "solvency_score": 70.4,
                "peer_sector": "Communication Services",
                "deterministic_scorecard": {
                    "action_stance": "HOLD_WAIT",
                    "action_stance_reason": "Business Conviction moderate",
                    "core_conviction_score": 6.2,
                    "business_conviction_score": 5.7,
                    "investment_conviction_score": 6.2,
                    "execution_readiness_score": 4.3,
                    "fundamental_quality_score": 74.3,
                    "valuation_margin_score": 72.5,
                    "guidance_expectation_score": 35.0,
                    "macro_alignment_status": "tailwind",
                    "macro_alignment_note": "Favorable macro",
                },
                "piotroski_breakdown": {
                    "f_score": 7,
                    "roa_positive": True,
                    "cfo_positive": True,
                    "delta_gross_margin_improved": True,
                },
                "reverse_dcf_result": {
                    "intrinsic_value_today": 82.35,
                    "target_price_12m": 89.3,
                    "upside_12m_pct": 23.0,
                    "market_implied_growth_pct": 12.81,
                    "market_implied_margin_pct": 26.24,
                    "consensus_target_price": 91.95,
                    "consensus_target_low": 57.0,
                    "consensus_target_high": 135.0,
                    "analyst_count": 45,
                    "fixed_parameters": {
                        "wacc_pct": 8.21,
                        "terminal_growth_pct": 2.5,
                    },
                },
                "tactical_setup": {
                    "price_stage": "STAGE_4_MARKDOWN",
                    "current_price": 72.61,
                    "sma_50": 75.1,
                    "sma_200": 82.88,
                    "key_support_level": 70.5,
                    "key_resistance_level": 77.2,
                    "invalidation_stop_loss": 69.26,
                    "buy_zone_min": 69.67,
                    "buy_zone_max": 72.15,
                    "tactical_risk_reward_ratio": 1.37,
                    "pullback_entry_status": "between_zone_and_target",
                },
                "insider_conviction": {
                    "open_market_p_count_90d": 0,
                    "open_market_p_value_usd_str": "0.00",
                    "open_market_s_count_90d": 2,
                    "open_market_s_value_usd_str": "150,000.00",
                    "rule_10b5_1_s_count_90d": 1,
                    "rule_10b5_1_s_value_usd_str": "75,000.00",
                    "status": "neutral_no_signal",
                },
                "smart_money_flags": {
                    "institutional_ownership_pct": 84.28,
                    "insider_ownership_pct": 0.57,
                    "short_interest_pct": 1.99,
                    "overall_smart_money_flag": "neutral",
                },
            },
        },
    )

    # 1. Latest Analysis
    latest = format_latest_analysis(prod_snapshot)
    assert "Action Stance:** `HOLD_WAIT`" in latest
    assert "Business Conviction:** `5.7 / 10`" in latest
    assert "Piotroski F-Score: `7 / 9`" in latest
    assert "ROIC: `33.69%`" in latest
    assert "Target Intrinsic Price Today:** `$82.35`" in latest
    assert "12-Month Target Price:** `$89.3`" in latest
    assert "DCF Upside / Downside (12M):** `23.0%`" in latest
    assert "Implied Revenue Growth (Next 5Y):** `12.81%`" in latest
    assert "Implied Operating Margin:** `26.24%`" in latest
    assert "WACC Applied:** `8.21%`" in latest
    assert "STAGE_4_MARKDOWN" in latest
    assert "Wall Street Consensus Target:** `$91.95`" in latest

    # 2. Financials Fallback
    fin = format_financials(prod_snapshot)
    assert "Revenue Growth (YoY)** | `15.85%`" in fin
    assert "Net Income Growth (YoY)** | `26.05%`" in fin
    assert "ROIC (Return on Invested Capital)** | `33.69%`" in fin
    assert "Piotroski F-Score Structural Forensic (`7 / 9`)" in fin

    # 3. Insider Fallback
    insider = format_insider_ownership(prod_snapshot)
    assert "Institutional Ownership:** `84.28%`" in insider
    assert "Open Market Purchases (Code P):** `0`" in insider
    assert "Open Market Sales (Code S):** `2`" in insider
