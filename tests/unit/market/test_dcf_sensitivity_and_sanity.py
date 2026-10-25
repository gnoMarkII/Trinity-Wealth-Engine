"""Unit tests for Wall Street Institutional DCF Upgrades:
- Mid-Year Discounting Convention
- 5x5 Sensitivity Matrix
- Valuation Sanity Audit & Guardrails
- Dual Terminal Value & Implied Exit Multiple
"""
import pytest
from schemas.macro_schemas import MarketObservable
from schemas.micro_quant_schemas import HistoricalFinancialItem
from tools.market.dcf_valuation import (
    compute_institutional_reverse_dcf,
    compute_dcf_valuation,
    audit_valuation_sanity,
    generate_dcf_sensitivity_matrix,
    calculate_synthetic_credit_rating,
    classify_company_archetype_and_normalization,
)


from datetime import datetime, timezone


@pytest.fixture
def mock_macro_registry():
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {
        "obs_dgs10": MarketObservable(
            observable_id="obs_dgs10",
            asset_bucket="fixed_income",
            region="US",
            indicator="10Y Treasury Yield",
            value="4.25",
            unit="%",
            observed_at=today_str,
            source_file="DGS10.csv",
            is_valid=True,
        ),
        "obs_erp_gspc": MarketObservable(
            observable_id="obs_erp_gspc",
            asset_bucket="equities",
            region="US",
            indicator="Equity Risk Premium",
            value="4.50",
            unit="%",
            observed_at=today_str,
            source_file="Damodaran_ERP.csv",
            is_valid=True,
        ),
    }


def test_dcf_mid_year_convention_applied(mock_macro_registry):
    """Verify that compute_dcf_valuation applies mid-year convention and returns sensitivity matrix."""
    result, flags = compute_dcf_valuation(
        ticker="AAPL",
        market="US",
        current_price=220.0,
        beta=1.1,
        fcf_per_share=6.5,
        market_cap=3_300_000_000_000.0,
        total_debt=100_000_000_000.0,
        interest_expense=3_000_000_000.0,
        tax_rate=0.21,
        fcf_cagr_3y=12.0,
        macro_registry=mock_macro_registry,
        cash_and_equivalents=50_000_000_000.0,
    )

    assert result is not None
    assert result.discounting_convention == "mid_year"
    assert result.valuation_sanity_audit is not None
    assert result.valuation_sanity_audit.terminal_value_pct_of_ev is not None
    assert 0.0 < result.valuation_sanity_audit.terminal_value_pct_of_ev < 100.0

    # Sensitivity matrix verification
    matrix = result.sensitivity_matrix
    assert matrix is not None
    assert len(matrix.wacc_values) == 5
    assert len(matrix.terminal_growth_values) == 5
    assert len(matrix.grid) == 5
    for row in matrix.grid:
        assert len(row) == 5
        for cell in row:
            assert cell.wacc_pct > 0
            assert cell.terminal_growth_pct > 0
            assert cell.target_price >= 0

    # Sensitivity summary verification
    summary = result.sensitivity_summary
    assert summary is not None
    assert summary.min_target_price > 0
    assert summary.max_target_price >= summary.min_target_price


def test_reverse_dcf_sanity_audit_and_sensitivity(mock_macro_registry):
    """Verify compute_institutional_reverse_dcf includes sanity audit and sensitivity grid."""
    result, flags = compute_institutional_reverse_dcf(
        ticker="MSFT",
        market="US",
        current_price=410.0,
        shares_outstanding=7_430_000_000,
        base_revenue=245_000_000_000.0,
        base_ebit_margin_pct=42.0,
        tax_rate=0.21,
        reinvestment_rate_pct=15.0,
        beta=1.15,
        total_debt=70_000_000_000.0,
        cash_and_equivalents=80_000_000_000.0,
        interest_expense=2_500_000_000.0,
        macro_registry=mock_macro_registry,
        sector="Technology",
        forecast_revenue_growth_pct=14.0,
    )

    assert result.status == "available"
    assert result.discounting_convention == "mid_year"
    assert result.valuation_sanity_audit is not None
    assert result.valuation_sanity_audit.model_reliability_tier in ("high", "moderate", "terminal_dependent")
    assert result.valuation_sanity_audit.implied_exit_fcf_multiple is not None
    assert result.valuation_sanity_audit.benchmark_sector_multiple == 22.0

    # 5x5 matrix
    assert result.sensitivity_matrix is not None
    assert len(result.sensitivity_matrix.grid) == 5
    assert len(result.sensitivity_matrix.grid[0]) == 5
    assert result.sensitivity_summary is not None
    assert result.sensitivity_summary.max_target_price > result.sensitivity_summary.min_target_price


def test_sanity_audit_flags_high_terminal_dependency():
    """Verify that audit_valuation_sanity flags models where TV > 85% of EV."""
    audit, flags = audit_valuation_sanity(
        sum_pv_fcf=10.0,
        tv_pv=90.0,  # 90 / 100 = 90% of EV
        tv_undisc=150.0,
        last_fcf=2.0,
        wacc_pct=9.0,
        terminal_growth_pct=2.5,
        risk_free_rate_pct=4.25,
        sector="Technology",
    )

    assert audit.terminal_value_pct_of_ev == 90.0
    assert audit.model_reliability_tier == "terminal_dependent"
    assert audit.is_terminal_dependent is True
    assert "high_terminal_value_dependency:dcf" in flags
    assert audit.reliability_score < 75


def test_sensitivity_matrix_singularity_protection():
    """Verify that sensitivity matrix prevents singularity (WACC <= g)."""
    matrix, summary = generate_dcf_sensitivity_matrix(
        base_wacc_pct=5.0,
        base_g_pct=2.5,
        current_price=100.0,
        rf_pct=4.25,
        calc_target_price_fn=lambda w, g: 100.0 / (w - g) if w > g else 0.0,
    )

    assert len(matrix.grid) == 5
    for row in matrix.grid:
        for cell in row:
            # WACC must always exceed terminal growth
            assert cell.wacc_pct > cell.terminal_growth_pct
            assert cell.target_price >= 0.0


def test_dcf_staged_fade_and_reinvestment_reconciliation(mock_macro_registry):
    """Verify Staged Fade Trajectory and exact Reinvestment Breakdown reconciliation:
    Net CapEx + Delta NWC == Total Net Reinvestment.
    """
    result, _ = compute_institutional_reverse_dcf(
        ticker="AAPL",
        market="US",
        current_price=220.0,
        shares_outstanding=15_000_000_000,
        base_revenue=380_000_000_000.0,
        base_ebit_margin_pct=30.0,
        tax_rate=0.21,
        reinvestment_rate_pct=12.0,
        beta=1.10,
        total_debt=100_000_000_000.0,
        cash_and_equivalents=60_000_000_000.0,
        interest_expense=3_500_000_000.0,
        macro_registry=mock_macro_registry,
        sector="Technology",
        forecast_revenue_growth_pct=15.0,
    )

    assert result.status == "available"
    assert len(result.explicit_forecast_5y) == 5
    projections = result.explicit_forecast_5y

    # 1. Staged Fade Trajectory: Y1 growth should be higher than Y5 growth
    assert projections[0].projected_revenue_growth_pct == 15.0
    assert projections[4].projected_revenue_growth_pct < projections[0].projected_revenue_growth_pct

    # 2. Reinvestment Breakdown Reconciliation for all 5 years
    for p in projections:
        # Net CapEx + Delta NWC must equal Total Reinvestment Currency exactly
        recon_sum = round(p.projected_net_capex + p.projected_delta_nwc, 2)
        assert abs(recon_sum - p.projected_reinvestment_currency) < 0.05
        # CapEx - Depreciation must equal Net CapEx exactly
        assert abs(round(p.projected_capex - p.projected_depreciation, 2) - p.projected_net_capex) < 0.05
        # UFCF must equal NOPAT - Total Reinvestment
        expected_fcf = round(max(0.0, p.projected_nopat - p.projected_reinvestment_currency), 2)
        assert abs(p.projected_fcf - expected_fcf) < 0.05


def test_3d_sensitivity_matrices_suite(mock_macro_registry):
    """Verify 3-Dimensional 5x5 Sensitivity Suite (75 Cells Total) with Center Cell Base Case Anchoring."""
    result, _ = compute_institutional_reverse_dcf(
        ticker="GOOGL",
        market="US",
        current_price=175.0,
        shares_outstanding=12_500_000_000,
        base_revenue=300_000_000_000.0,
        base_ebit_margin_pct=28.0,
        tax_rate=0.21,
        reinvestment_rate_pct=15.0,
        beta=1.05,
        total_debt=30_000_000_000.0,
        cash_and_equivalents=110_000_000_000.0,
        interest_expense=1_000_000_000.0,
        macro_registry=mock_macro_registry,
        sector="Technology",
        forecast_revenue_growth_pct=14.0,
    )

    matrices = result.sensitivity_matrices
    assert matrices is not None
    assert matrices.valuation is not None
    assert matrices.operating is not None
    assert matrices.macro is not None

    # Total 75 cells: 3 tables x 25 cells (5x5)
    assert len(matrices.valuation.grid) == 5
    assert len(matrices.valuation.grid[0]) == 5
    assert len(matrices.operating.grid) == 5
    assert len(matrices.operating.grid[0]) == 5
    assert len(matrices.macro.grid) == 5
    assert len(matrices.macro.grid[0]) == 5

    # Center Cell Base Case Anchoring (row 2, col 2)
    val_center = matrices.valuation.grid[2][2]
    op_center = matrices.operating.grid[2][2]
    macro_center = matrices.macro.grid[2][2]

    assert val_center.is_base is True
    assert op_center.is_base is True
    assert macro_center.is_base is True

    # Center cells must have valid prices matching the base case model
    assert val_center.target_price > 0.0
    assert op_center.target_price > 0.0
    assert macro_center.target_price > 0.0
    assert abs(val_center.target_price - op_center.target_price) < 0.05
    assert abs(val_center.target_price - macro_center.target_price) < 0.05


def test_institutional_dcf_validation_criteria(mock_macro_registry):
    """Institutional sanity validation based on validate_dcf.py:
    1. Terminal growth < WACC
    2. Effective WACC in institutional range (5% - 20%)
    3. Terminal Value % of EV is reasonable
    4. Dual Terminal Value calculation
    """
    result, _ = compute_institutional_reverse_dcf(
        ticker="MSFT",
        market="US",
        current_price=400.0,
        shares_outstanding=7_400_000_000,
        base_revenue=245_000_000_000.0,
        base_ebit_margin_pct=42.0,
        tax_rate=0.21,
        reinvestment_rate_pct=15.0,
        beta=1.15,
        total_debt=70_000_000_000.0,
        cash_and_equivalents=80_000_000_000.0,
        interest_expense=2_500_000_000.0,
        macro_registry=mock_macro_registry,
        sector="Technology",
        forecast_revenue_growth_pct=14.0,
    )

    # 1. Effective WACC in 5% - 20%
    assert 5.0 <= result.effective_wacc_pct <= 20.0

    # 2. Terminal Value proportion
    assert result.valuation_sanity_audit is not None
    assert result.valuation_sanity_audit.terminal_value_pct_of_ev is not None
    assert 30.0 <= result.valuation_sanity_audit.terminal_value_pct_of_ev <= 85.0

    # 3. Dual Terminal Value Target Prices
    assert result.target_price_12m is not None
    assert result.target_price_exit_multiple_12m is not None
    assert result.target_price_12m > 0.0
    assert result.target_price_exit_multiple_12m > 0.0
    assert result.exit_multiple_used == 25.0  # Technology benchmark

    # 4. 3Y Historical Baseline Present
    assert len(result.historical_financials_3y) == 3
    assert result.historical_financials_3y[0].fiscal_year.endswith("A")


def test_damodaran_synthetic_credit_rating_engine():
    """Verify Damodaran Synthetic Credit Rating table lookups and edge cases:
    - High ICR (>8.5x) -> AAA (+60 bps)
    - Net Cash / Zero Debt -> AAA (+60 bps)
    - Moderate ICR (2.5x - 3.0x) -> BBB (+160 bps)
    - Low ICR / Distressed (<1.5x or EBIT <= 0) -> CCC / Distressed (+750 bps)
    """
    # 1. AAA Large Cap
    r1, icr1, spread1, kd1 = calculate_synthetic_credit_rating(ebit=100.0, interest_expense=5.0, total_debt=50.0, risk_free_rate_pct=4.25)
    assert r1 == "AAA"
    assert icr1 == 20.0
    assert spread1 == 60.0
    assert kd1 == 4.85

    # 2. Net cash / Zero Debt
    r_nc, icr_nc, spread_nc, kd_nc = calculate_synthetic_credit_rating(ebit=50.0, interest_expense=0.0, total_debt=0.0, risk_free_rate_pct=4.00)
    assert r_nc == "AAA"
    assert spread_nc == 60.0
    assert kd_nc == 4.60

    # 3. BBB Investment Grade
    r_bbb, icr_bbb, spread_bbb, kd_bbb = calculate_synthetic_credit_rating(ebit=27.0, interest_expense=10.0, total_debt=100.0, risk_free_rate_pct=4.25)
    assert r_bbb == "BBB"
    assert icr_bbb == 2.70
    assert spread_bbb == 160.0
    assert kd_bbb == 5.85

    # 4. Negative EBIT / Distressed
    r_dist, icr_dist, spread_dist, kd_dist = calculate_synthetic_credit_rating(ebit=-10.0, interest_expense=5.0, total_debt=100.0, risk_free_rate_pct=4.25)
    assert r_dist == "CCC / Distressed"
    assert icr_dist == 0.0
    assert spread_dist == 750.0
    assert kd_dist == 11.75


def test_functional_opex_breakdown_and_accounting_identity(mock_macro_registry):
    """Verify that when historical R&D and SG&A are provided:
    1. Forecast projections populate projected_rd, projected_sga, projected_gross_profit.
    2. Accounting Identity: Gross Profit - (R&D + SG&A) == EBIT holds identically for all 5 years.
    3. Sensitivity diagnostics (growth_delta, margin_delta, primary_operating_driver, yield shock) are calculated.
    """
    hist = [
        HistoricalFinancialItem(
            fiscal_year="2021A", revenue=100_000.0, ebit=25_000.0, ebit_margin_pct=25.0,
            gross_profit=50_000.0, research_and_development=12_000.0, selling_general_admin=13_000.0,
        ),
        HistoricalFinancialItem(
            fiscal_year="2022A", revenue=110_000.0, ebit=28_000.0, ebit_margin_pct=25.45,
            gross_profit=56_000.0, research_and_development=13_000.0, selling_general_admin=15_000.0,
        ),
        HistoricalFinancialItem(
            fiscal_year="2023A", revenue=120_000.0, ebit=32_000.0, ebit_margin_pct=26.67,
            gross_profit=62_000.0, research_and_development=14_400.0, selling_general_admin=15_600.0,
        ),
    ]

    result, _ = compute_institutional_reverse_dcf(
        ticker="TECH1",
        market="US",
        current_price=150.0,
        shares_outstanding=1_000_000_000,
        base_revenue=120_000.0,
        base_ebit_margin_pct=26.67,
        tax_rate=0.21,
        reinvestment_rate_pct=15.0,
        beta=1.20,
        total_debt=20_000.0,
        cash_and_equivalents=30_000.0,
        interest_expense=800.0,
        macro_registry=mock_macro_registry,
        sector="Technology",
        historical_financials=hist,
    )

    assert result.status == "available"
    assert len(result.explicit_forecast_5y) == 5

    # Verify accounting identity: Gross Profit - (R&D + SG&A) == EBIT
    for f in result.explicit_forecast_5y:
        assert f.projected_gross_profit is not None
        assert f.projected_rd is not None
        assert f.projected_sga is not None
        calculated_ebit = round(f.projected_gross_profit - (f.projected_rd + f.projected_sga), 2)
        assert abs(calculated_ebit - f.projected_ebit) <= 0.05

    # Verify synthetic rating
    assert result.synthetic_credit_rating == "AAA"
    assert result.interest_coverage_ratio is not None and result.interest_coverage_ratio > 20.0
    assert result.credit_spread_bps == 60.0

    # Verify sensitivity diagnostics
    summary = result.sensitivity_summary
    assert summary is not None
    assert summary.growth_delta_per_1pct is not None
    assert summary.margin_delta_per_1pct is not None
    assert summary.primary_operating_driver in ("operating_leverage", "revenue_growth", "balanced")
    assert summary.macro_yield_shock_50bps_delta is not None
    assert summary.macro_yield_shock_100bps_delta is not None


def test_implied_exit_ebitda_and_dual_tv_cross_check(mock_macro_registry):
    """Verify that Terminal Value calculates Year 5 EBITDA, Implied Exit Multiple, and Exit TV cross-check."""
    result, flags = compute_institutional_reverse_dcf(
        ticker="MSFT",
        market="US",
        current_price=410.0,
        shares_outstanding=7_430_000_000,
        base_revenue=245_000_000_000.0,
        base_ebit_margin_pct=42.0,
        tax_rate=0.21,
        reinvestment_rate_pct=15.0,
        beta=1.15,
        total_debt=70_000_000_000.0,
        cash_and_equivalents=80_000_000_000.0,
        interest_expense=2_500_000_000.0,
        macro_registry=mock_macro_registry,
        sector="Technology",
        forecast_revenue_growth_pct=14.0,
    )

    assert result.status == "available"
    # Year 5 Terminal EBITDA must be strictly positive
    assert result.terminal_ebitda is not None and result.terminal_ebitda > 0
    # Implied Exit Multiple must be positive and match the sanity audit
    assert result.implied_exit_ebitda_multiple is not None and result.implied_exit_ebitda_multiple > 0
    assert result.valuation_sanity_audit.implied_exit_ebitda_multiple == result.implied_exit_ebitda_multiple
    # Implied Perpetual Growth from Exit Multiple must be computed
    assert result.implied_perpetual_growth_from_exit_pct is not None
    assert result.exit_multiple_tv_undiscounted is not None and result.exit_multiple_tv_undiscounted > 0

    # Test extreme terminal exit multiple guard in audit_valuation_sanity
    audit_extreme, flags_extreme = audit_valuation_sanity(
        sum_pv_fcf=100.0,
        tv_pv=500.0,
        tv_undisc=800.0,
        last_fcf=30.0,
        wacc_pct=9.0,
        terminal_growth_pct=2.5,
        risk_free_rate_pct=4.25,
        last_ebitda=20.0,  # 800 / 20 = 40.0x exit multiple (> 25x threshold)
    )
    assert "extreme_terminal_exit_ebitda:dcf" in flags_extreme
    assert audit_extreme.implied_exit_ebitda_multiple == 40.0

    # Test depressed terminal exit multiple guard
    audit_depressed, flags_depressed = audit_valuation_sanity(
        sum_pv_fcf=100.0,
        tv_pv=100.0,
        tv_undisc=150.0,
        last_fcf=30.0,
        wacc_pct=9.0,
        terminal_growth_pct=2.5,
        risk_free_rate_pct=4.25,
        last_ebitda=50.0,  # 150 / 50 = 3.0x exit multiple (< 4.0x threshold)
    )
    assert "depressed_terminal_exit_ebitda:dcf" in flags_depressed
    assert audit_depressed.implied_exit_ebitda_multiple == 3.0


def test_dynamic_multi_scenario_forecasts_and_capex_lifecycle(mock_macro_registry):
    """Verify that Reverse DCF generates explicit 5Y forecasts for Bear, Base, and Bull,
    and populates CapEx/D&A coverage ratio with lifecycle phases."""
    result, _ = compute_institutional_reverse_dcf(
        ticker="GOOGL",
        market="US",
        current_price=175.0,
        shares_outstanding=12_000_000_000,
        base_revenue=305_000_000_000.0,
        base_ebit_margin_pct=30.0,
        tax_rate=0.21,
        reinvestment_rate_pct=15.0,
        beta=1.05,
        total_debt=30_000_000_000.0,
        cash_and_equivalents=110_000_000_000.0,
        interest_expense=500_000_000.0,
        macro_registry=mock_macro_registry,
        sector="Technology",
        forecast_revenue_growth_pct=12.0,
    )

    assert result.status == "available"
    scenarios = result.explicit_forecast_5y_scenarios
    assert "bear" in scenarios and "base" in scenarios and "bull" in scenarios
    assert len(scenarios["bear"]) == 5
    assert len(scenarios["base"]) == 5
    assert len(scenarios["bull"]) == 5

    # Growth and revenue ordering: Bear < Base < Bull
    assert scenarios["bear"][0].projected_revenue < scenarios["base"][0].projected_revenue < scenarios["bull"][0].projected_revenue
    assert scenarios["bear"][4].projected_fcf < scenarios["bull"][4].projected_fcf

    # Margin operating leverage ordering: Bear compresses, Bull expands
    assert scenarios["bear"][4].projected_ebit_margin_pct < scenarios["bull"][4].projected_ebit_margin_pct

    # CapEx / D&A coverage ratio on explicit forecasts
    for f in scenarios["base"]:
        assert f.capex_to_depreciation_ratio is not None and f.capex_to_depreciation_ratio > 0
        assert f.reinvestment_lifecycle_phase in ("aggressive_expansion", "sustainable_maintenance", "underinvestment_warning")

    # Verify sanity audit check for capex underinvestment
    audit_under, flags_under = audit_valuation_sanity(
        sum_pv_fcf=100.0,
        tv_pv=100.0,
        tv_undisc=150.0,
        last_fcf=30.0,
        wacc_pct=9.0,
        terminal_growth_pct=2.5,
        risk_free_rate_pct=4.25,
        last_ebitda=40.0,
        capex_to_deprec_ratio=0.72,  # < 0.85 threshold
    )
    assert "capex_underinvestment_risk:dcf" in flags_under


def test_company_archetype_classification_and_mid_cycle_normalization(mock_macro_registry):
    """Verify company archetype profiling (Tech, Mature, Cyclical, Turnaround)
    and mid-cycle margin normalization for cyclical companies at peak earnings."""
    # 1. Cyclical company with peak margin
    hist_cyclical = [
        HistoricalFinancialItem(fiscal_year="2021A", revenue=100.0, ebit=15.0, ebit_margin_pct=15.0, capex=5.0, depreciation=4.0),
        HistoricalFinancialItem(fiscal_year="2022A", revenue=120.0, ebit=18.0, ebit_margin_pct=15.0, capex=6.0, depreciation=5.0),
        HistoricalFinancialItem(fiscal_year="2023A", revenue=150.0, ebit=39.0, ebit_margin_pct=26.0, capex=10.0, depreciation=6.0),
    ]
    # Current LTM margin = 26.0% (Peak), historical median = 15.0%
    arch, mid_m, is_norm, rationale = classify_company_archetype_and_normalization(
        sector="Basic Materials",
        base_ebit_margin_pct=26.0,
        forecast_revenue_growth_pct=8.0,
        historical_financials=hist_cyclical,
    )
    assert arch == "cyclical"
    assert mid_m == 15.0
    assert is_norm is True
    assert "cyclical peak" in rationale.lower()

    # 2. High-Growth Tech Archetype
    arch_tech, _, _, _ = classify_company_archetype_and_normalization(
        sector="Software & Cloud Infrastructure",
        base_ebit_margin_pct=22.0,
        forecast_revenue_growth_pct=25.0,
        historical_financials=None,
    )
    assert arch_tech == "high_growth_tech"

    # 3. Mature Cash Cow Archetype
    arch_cow, _, _, _ = classify_company_archetype_and_normalization(
        sector="Consumer Defensive",
        base_ebit_margin_pct=18.0,
        forecast_revenue_growth_pct=4.0,
        historical_financials=None,
        roe_pct=16.0,
    )
    assert arch_cow == "mature_cash_cow"

    # 4. Turnaround Archetype
    arch_turn, _, _, _ = classify_company_archetype_and_normalization(
        sector="Industrial Machinery",
        base_ebit_margin_pct=1.8,
        forecast_revenue_growth_pct=-5.0,
        historical_financials=None,
    )
    assert arch_turn == "turnaround"

    # 5. Full DCF Integration: Verify cyclical stock normalizes margin downwards from 26% to mid-cycle
    result, flags = compute_institutional_reverse_dcf(
        ticker="XOM",
        market="US",
        current_price=110.0,
        shares_outstanding=4_000_000_000,
        base_revenue=350_000_000_000.0,
        base_ebit_margin_pct=26.0,
        tax_rate=0.21,
        reinvestment_rate_pct=12.0,
        beta=0.95,
        total_debt=40_000_000_000.0,
        cash_and_equivalents=30_000_000_000.0,
        interest_expense=1_500_000_000.0,
        macro_registry=mock_macro_registry,
        sector="Energy",
        forecast_revenue_growth_pct=5.0,
        historical_financials=hist_cyclical,
    )

    assert result.company_archetype == "cyclical"
    assert result.is_mid_cycle_normalized is True
    assert result.normalized_mid_cycle_ebit_margin_pct == 15.0
    assert "cyclical_peak_margin_normalized:dcf" in flags

    # Verify margin fades down toward mid-cycle (Year 1 ~ 26% down to Year 5 ~ 15%)
    forecasts = result.explicit_forecast_5y
    assert forecasts[0].projected_ebit_margin_pct > forecasts[4].projected_ebit_margin_pct
    assert abs(forecasts[4].projected_ebit_margin_pct - 15.0) <= 0.5


def test_institutional_margin_of_safety_entry_price(mock_macro_registry):
    """Verify Margin of Safety (MoS) 15% discount entry price target and buy zone status."""
    # Case 1: Stock is trading at a premium / above MoS entry price (Out of Buy Zone)
    result_premium, flags_premium = compute_institutional_reverse_dcf(
        ticker="AAPL",
        market="US",
        current_price=220.0,
        shares_outstanding=15_000_000_000,
        base_revenue=380_000_000_000.0,
        base_ebit_margin_pct=30.0,
        tax_rate=0.21,
        reinvestment_rate_pct=15.0,
        beta=1.1,
        total_debt=100_000_000_000.0,
        cash_and_equivalents=50_000_000_000.0,
        interest_expense=3_000_000_000.0,
        macro_registry=mock_macro_registry,
        forecast_revenue_growth_pct=8.0,
    )

    assert result_premium.margin_of_safety_pct == 15.0
    assert result_premium.intrinsic_value_today is not None
    assert result_premium.target_price_12m is not None
    assert result_premium.mos_entry_price_today is not None
    assert result_premium.mos_target_price_12m is not None

    expected_mos_today = round(result_premium.intrinsic_value_today * 0.85, 2)
    assert abs(result_premium.mos_entry_price_today - expected_mos_today) <= 0.01

    expected_mos_12m = round(result_premium.target_price_12m * 0.85, 2)
    assert abs(result_premium.mos_target_price_12m - expected_mos_12m) <= 0.01

    # Current price (220.0) is higher than MoS entry price
    if 220.0 > result_premium.mos_entry_price_today:
        assert result_premium.is_in_buy_zone is False
        assert result_premium.required_pullback_pct is not None
        assert result_premium.required_pullback_pct > 0.0
        assert "in_margin_of_safety_buy_zone:dcf" not in flags_premium

    # Case 2: Deep Value Stock trading well below Intrinsic Value (In Buy Zone)
    # Give high FCF / low price so intrinsic value is far above price
    result_bargain, flags_bargain = compute_institutional_reverse_dcf(
        ticker="VALUE",
        market="US",
        current_price=50.0,  # Very cheap
        shares_outstanding=1_000_000_000,
        base_revenue=100_000_000_000.0,
        base_ebit_margin_pct=25.0,
        tax_rate=0.21,
        reinvestment_rate_pct=10.0,
        beta=0.8,
        total_debt=5_000_000_000.0,
        cash_and_equivalents=20_000_000_000.0,
        interest_expense=200_000_000.0,
        macro_registry=mock_macro_registry,
        forecast_revenue_growth_pct=5.0,
    )

    assert result_bargain.is_in_buy_zone is True
    assert result_bargain.required_pullback_pct is None
    assert result_bargain.current_margin_of_safety_pct is not None
    assert result_bargain.current_margin_of_safety_pct >= 15.0
    assert "in_margin_of_safety_buy_zone:dcf" in flags_bargain






