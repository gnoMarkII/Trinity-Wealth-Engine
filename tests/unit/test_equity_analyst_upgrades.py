"""Unit tests for Institutional Equity Analyst Upgrades across all 5 dimensions.

Covers:
1. Sector-specific KPIs (Banking, Semis, REITs).
2. Sector alternative valuation models (Bank Justified P/BV, REIT DDM) & integration with Reverse DCF.
3. Sector-adjusted valuation scoring & Thai SET-11 peer valuation baskets.
4. Net Shareholder Yield computation & share dilution risk flag.
5. Top-down Macro to Micro scorecard linkage & alternative valuation pillar scoring.
6. Thesis delta computation from vault & portfolio holding context integration.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from schemas.micro_quant_schemas import (
    DeterministicScorecard,
    MarginMetricItem,
    PiotroskiFScoreBreakdown,
    QuantSignals,
    ReverseDCFResult,
    TacticalSetup,
)
from tools.market.dcf_valuation import (
    compute_institutional_reverse_dcf,
    compute_sector_alternative_valuation,
)
from tools.market.equity_rules_engine import (
    compute_deterministic_scorecard,
    load_latest_macro_regime_context,
)
from tools.market.financial_autopsy import FinancialAutopsyPeriod
from tools.market.peer_valuation import fetch_peer_metrics
from tools.market.quant_scoring import (
    compute_net_shareholder_yield,
    compute_value_score,
)
from tools.market.sector_kpi_engine import compute_sector_kpis
from agents.pipelines.equity_helpers import (
    compute_thesis_delta_from_vault,
    load_portfolio_holding_context,
)


# =====================================================================
# Dimension 1: Sector-Specific Operating KPIs
# =====================================================================
def test_sector_kpis_financials_semis_reits():
    """Verify sector-specific KPIs for Banking, Semiconductors, and REITs."""
    # 1. Banking / Financial Services
    p_bank_1 = FinancialAutopsyPeriod(
        fiscal_period_end="2023-12-31",
        net_income=30_000.0,
        total_revenue=100_000.0,
        total_assets=1_000_000.0,
        stockholders_equity=200_000.0,
    )
    p_bank_2 = FinancialAutopsyPeriod(
        fiscal_period_end="2024-12-31",
        net_income=35_000.0,
        total_revenue=110_000.0,
        total_assets=1_100_000.0,
        stockholders_equity=220_000.0,
    )
    res_bank = compute_sector_kpis(sector="Financial Services - Banks", periods=[p_bank_1, p_bank_2])
    assert res_bank.status == "available"
    roe_kpi = next(k for k in res_bank.kpis if "ROE" in k.name)
    assert roe_kpi.value is not None and roe_kpi.value > 0
    buffer_kpi = next(k for k in res_bank.kpis if "Capital Buffer" in k.name)
    assert buffer_kpi.value is not None and buffer_kpi.value > 0
    assert buffer_kpi.benchmark is not None

    # 2. Technology / Semiconductors
    p_semi = FinancialAutopsyPeriod(
        fiscal_period_end="2024-12-31",
        total_revenue=100_000.0,
        gross_profit=62_000.0,
        capital_expenditure=-25_000.0,
        operating_cash_flow=40_000.0,
        free_cash_flow=15_000.0,
    )
    res_semi = compute_sector_kpis(sector="Technology - Semiconductors", periods=[p_semi])
    assert res_semi.status == "available"
    gm_kpi = next(k for k in res_semi.kpis if "Gross Margin" in k.name)
    assert gm_kpi.value == 62.0
    capex_kpi = next(k for k in res_semi.kpis if "CapEx Intensity" in k.name)
    assert capex_kpi.value == 25.0
    fcf_conv_kpi = next(k for k in res_semi.kpis if "FCF Conversion" in k.name)
    assert fcf_conv_kpi.value == 37.5  # 15,000 / 40,000 * 100

    # 3. Real Estate / REIT
    p_reit = FinancialAutopsyPeriod(
        fiscal_period_end="2024-12-31",
        total_revenue=10_000.0,
        operating_cash_flow=6_500.0,
        free_cash_flow=5_500.0,
    )
    res_reit = compute_sector_kpis(sector="Real Estate - REIT", periods=[p_reit])
    assert res_reit.status == "available"
    ocf_kpi = next(k for k in res_reit.kpis if "Operating Cash Flow Margin" in k.name)
    assert ocf_kpi.value == 65.0


# =====================================================================
# Dimension 2: Sector Alternative Valuation & DCF Integration
# =====================================================================
def test_bank_and_reit_alternative_valuation():
    """Verify Justified P/BV for Banks and DDM for REITs, plus DCF integration."""
    # 1. Bank Justified P/BV
    alt_model, alt_intrinsic, alt_target, alt_upside, implied_roe, alt_flags = compute_sector_alternative_valuation(
        sector="Financial Services - Banks",
        current_price=100.0,
        market="TH",
        roe_pct=10.5,
        book_value_per_share=120.0,
    )
    assert alt_model == "justified_pbv"
    assert alt_target is not None and alt_target > 0
    assert alt_upside is not None
    assert implied_roe is not None

    # 2. REIT DDM
    alt_model_reit, alt_intrinsic_reit, alt_target_reit, alt_upside_reit, implied_metric_reit, alt_flags_reit = (
        compute_sector_alternative_valuation(
            sector="Real Estate - REIT",
            current_price=10.0,
            market="TH",
            dividend_yield_pct=8.5,
        )
    )
    assert alt_model_reit == "ddm_yield_spread"
    assert alt_target_reit is not None and alt_target_reit > 0
    assert alt_upside_reit is not None

    # 3. Integration into compute_institutional_reverse_dcf() for Bank
    rd, flags = compute_institutional_reverse_dcf(
        ticker="SCB.BK",
        market="TH",
        sector="Financial Services - Banks",
        current_price=100.0,
        shares_outstanding=3_300_000_000,
        base_revenue=150_000_000_000,
        base_ebit_margin_pct=40.0,
        roe_pct=10.5,
        book_value_per_share=120.0,
    )
    assert rd.status == "not_applicable"
    assert rd.alternative_valuation_model == "justified_pbv"
    assert rd.alternative_target_price_12m is not None
    assert rd.alternative_upside_12m_pct is not None


# =====================================================================
# Dimension 3: Sector-Adjusted Valuation Scoring & Thai Peer Baskets
# =====================================================================
def test_sector_adjusted_value_score_and_th_peers():
    """Verify sector-adjusted P/E score bands and Thai SET peer baskets."""
    # 1. P/E scoring bands
    pe = 30.0
    tech_score = compute_value_score(pe=pe, sector="Technology - Software")
    general_score = compute_value_score(pe=pe, sector="General")
    assert tech_score > general_score, "Tech sector should accommodate higher P/E multiples than general"

    pe_bank = 16.0
    bank_score = compute_value_score(pe=pe_bank, sector="Financial Services - Banks")
    general_bank_score = compute_value_score(pe=pe_bank, sector="General")
    assert bank_score < general_bank_score, "Banking sector should penalize P/E above 15 more heavily than general"

    # 2. Thai Peer Valuation Baskets (mock _get_peer_info_cached to avoid live network)
    with patch("tools.market.peer_valuation._get_peer_info_cached", return_value={"trailingPE": 8.0, "enterpriseToEbitda": 6.5}):
        peer_metrics = fetch_peer_metrics(
            sector="Banking",
            exclude_ticker="SCB.BK",
            market="TH",
        )
    assert isinstance(peer_metrics, list)
    # SCB.BK itself must be excluded from peers
    peer_symbols = [p["symbol"] for p in peer_metrics]
    assert "SCB.BK" not in peer_symbols
    assert "SCB" not in peer_symbols
    assert len(peer_symbols) >= 3


# =====================================================================
# Dimension 4: Net Shareholder Yield & Dilution Flag
# =====================================================================
def test_net_shareholder_yield_and_dilution_flag():
    """Verify net shareholder yield calculation and dilution warning flag."""
    # 1. Net Buyback + Dividend (100M down to 96M shares = 4% buyback)
    net_buyback, net_yield, flag = compute_net_shareholder_yield(
        dividend_yield_pct=3.0,
        latest_shares=96_000_000.0,
        prev_shares=100_000_000.0,
    )
    assert net_buyback == 4.0  # (100M - 96M) / 100M = 4.0%
    assert net_yield == 7.0  # 3.0% + 4.0% = 7.0%
    assert flag is None

    # 2. High Share Dilution (> 3%) (100M up to 105M shares = 5% dilution)
    net_buyback_dil, net_yield_dil, flag_dil = compute_net_shareholder_yield(
        dividend_yield_pct=2.0,
        latest_shares=105_000_000.0,
        prev_shares=100_000_000.0,
    )
    assert net_buyback_dil == -5.0
    assert net_yield_dil == -3.0  # 2.0% - 5.0% dilution = -3.0%
    assert flag_dil == "share_dilution_risk:capital_allocation"


# =====================================================================
# Dimension 5: Top-Down Macro Linkage & Alternative Valuation Scoring
# =====================================================================
def test_macro_to_micro_top_down_linkage():
    """Verify top-down macro regime linkage and alternative valuation in scorecard."""
    signals = QuantSignals(
        ticker="SCB.BK",
        market="TH",
        currency="THB",
        company_name="SCB X Public Company Limited",
        as_of_date="2026-10-16",
        evaluated_at="2026-10-16T10:00:00Z",
        quality_score=80.0,
        roic_pct=10.0,
        ocf_to_net_income=1.1,
        solvency_score=40.0,  # Weak solvency to trigger clamp test
    )
    piotroski = PiotroskiFScoreBreakdown(status="available", f_score=7)

    # 1. Non-applicable DCF with alternative valuation upside
    dcf_alt = ReverseDCFResult(
        status="not_applicable",
        is_eligible=False,
        is_actionable=False,
        actionability_reason="Sector excluded: Financials",
        alternative_valuation_model="justified_pbv",
        alternative_target_price_12m=120.0,
        alternative_upside_12m_pct=20.0,
    )

    tactical = TacticalSetup(
        status="available",
        price_stage="STAGE_2_MARKUP",
        current_price=100.0,
        atr_14=2.5,
        key_support_level=95.0,
        key_resistance_level=110.0,
        buy_zone_min=96.0,
        buy_zone_max=99.0,
        invalidation_stop_loss=92.0,
        tactical_target_price=110.0,
        pullback_entry_status="in_buy_zone",
        is_in_buy_zone=True,
    )

    # Test Headwind clamping
    macro_headwind = {
        "regime_name": "Stagflationary Slowdown",
        "direction": "DEFENSIVE",
        "target_allocations": {"cash": 30.0, "fixed_income": 30.0, "equities": 40.0},
    }
    sc_headwind, falsifiers_headwind, flags_headwind = compute_deterministic_scorecard(
        ticker="SCB.BK",
        market="TH",
        quant_signals=signals,
        piotroski=piotroski,
        dcf=dcf_alt,
        tactical=tactical,
        macro_context=macro_headwind,
    )
    assert sc_headwind.macro_regime == "Stagflationary Slowdown"
    assert sc_headwind.macro_alignment_status == "headwind"
    assert sc_headwind.execution_readiness_score <= 6.0
    assert "macro_regime_headwind:scorecard" in flags_headwind
    # Alternative valuation should provide an actionable valuation score in Pillar 3
    assert sc_headwind.valuation_margin_score is not None
    assert sc_headwind.valuation_margin_score > 0

    # Test Tailwind
    signals_healthy = QuantSignals(
        ticker="SCB.BK",
        market="TH",
        currency="THB",
        company_name="SCB X Public Company Limited",
        as_of_date="2026-10-16",
        evaluated_at="2026-10-16T10:00:00Z",
        quality_score=80.0,
        roic_pct=10.0,
        ocf_to_net_income=1.1,
        solvency_score=85.0,
        peer_sector="Banking",
    )
    macro_tailwind = {
        "regime_name": "Goldilocks Expansion",
        "direction": "AGGRESSIVE_EXPANSION",
        "leading_sectors": ["banking", "financials"],
        "target_allocations": {"equities": 75.0, "cash": 5.0},
    }
    sc_tailwind, falsifiers_tailwind, flags_tailwind = compute_deterministic_scorecard(
        ticker="SCB.BK",
        market="TH",
        quant_signals=signals_healthy,
        piotroski=piotroski,
        dcf=dcf_alt,
        tactical=tactical,
        macro_context=macro_tailwind,
    )
    assert sc_tailwind.macro_alignment_status == "tailwind"


# =====================================================================
# Dimension 6: Thesis Delta & Portfolio Position Context
# =====================================================================
def test_thesis_delta_and_portfolio_holding_context(tmp_path: Path):
    """Verify thesis delta calculation and portfolio holding context loading."""
    # 1. Thesis Delta Calculation using tmp_path vault
    curr_signals = QuantSignals(
        ticker="AAPL",
        market="US",
        currency="USD",
        as_of_date="2026-10-16",
        evaluated_at="2026-10-16T10:00:00Z",
        raw_analysis_price=180.0,
        piotroski_breakdown=PiotroskiFScoreBreakdown(status="available", f_score=8),
        deterministic_scorecard=DeterministicScorecard(
            core_conviction_score=8.2,
            action_stance="ACCUMULATE_NOW",
            coverage_pct=100.0,
        ),
        gaap_operating_margin=MarginMetricItem(value_pct=28.0),
    )

    prev_sidecar_dict = {
        "analysis_date": "2026-09-15",
        "quant_signals": {
            "raw_analysis_price": 170.0,
            "deterministic_scorecard": {
                "core_conviction_score": 7.5,
                "action_stance": "ACCUMULATE_ON_DIP",
            },
            "piotroski_breakdown": {"f_score": 7},
        },
        "thesis_falsifiers": [
            {
                "metric_name": "Operating Margin",
                "threshold_value": 30.0,
                "condition": "< 30%",
            }
        ],
    }

    sidecar_dir = tmp_path / "30_Knowledge_Base" / "Stocks" / "AAPL" / "Analysis"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    latest_file = sidecar_dir / "AAPL Equity Analysis latest.json"
    latest_file.write_text(json.dumps(prev_sidecar_dict), encoding="utf-8")

    delta = compute_thesis_delta_from_vault("AAPL", curr_signals, vault_path=tmp_path)
    assert delta is not None
    assert delta.price_change_pct is not None
    assert delta.price_change_pct > 0  # 180 vs 170
    assert delta.conviction_score_delta == 0.7  # 8.2 - 7.5
    assert delta.f_score_delta == 1  # 8 - 7
    assert delta.previous_action_stance == "ACCUMULATE_ON_DIP"
    assert delta.current_action_stance == "ACCUMULATE_NOW"
    assert len(delta.breached_falsifiers) == 1
    assert "Operating Margin" in delta.breached_falsifiers[0]

    # 2. Portfolio Holding Context
    mock_holding = MagicMock()
    mock_holding.symbol = "AAPL"
    mock_holding.units = 50.0
    mock_holding.market_value_thb = 300_000.0
    mock_holding.avg_cost_usd = 160.0
    mock_holding.unrealized_pnl_percent = 12.5
    mock_holding.bucket_id = "us_equities"

    mock_summary = MagicMock()
    mock_summary.total_value_thb = 3_000_000.0

    mock_portfolio_state = MagicMock()
    mock_portfolio_state.holdings = [mock_holding]
    mock_portfolio_state.summary = mock_summary

    with patch("tools.portfolio.core.get_structured_portfolio_state", return_value=mock_portfolio_state):
        ctx = load_portfolio_holding_context("AAPL", "US", curr_signals)
        assert ctx is not None
        assert ctx.is_held is True
        assert ctx.units == 50.0
        assert ctx.avg_cost == 160.0
        assert ctx.unrealized_pnl_pct == 12.5
        assert ctx.current_weight_pct == 10.0  # 300k / 3000k * 100
        assert ctx.position_action_hint == "ADD_TO_WINNER"
