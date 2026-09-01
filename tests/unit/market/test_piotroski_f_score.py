"""Unit tests for 9-point Piotroski F-Score and Sector Exclusions."""
import pytest
from tools.market.financial_autopsy import (
    FinancialAutopsyPeriod,
    calculate_piotroski_f_score,
)


def test_piotroski_f_score_perfect_9():
    """Test a company that meets all 9 Piotroski criteria."""
    p_current = FinancialAutopsyPeriod(
        fiscal_period_end="2025-12-31",
        net_income=150_000_000,
        operating_cash_flow=200_000_000,  # CFO > Net Income (Accrual Quality)
        total_assets=1_000_000_000,       # ROA_t = 0.15 > 0
        current_assets=500_000_000,
        current_liabilities=200_000_000,  # CR_t = 2.5
        long_term_debt=100_000_000,       # Lev_t = 0.10
        total_revenue=800_000_000,
        gross_profit=480_000_000,         # GM_t = 60.0%
        shares_outstanding=100_000_000,
    )
    p_prior = FinancialAutopsyPeriod(
        fiscal_period_end="2024-12-31",
        net_income=100_000_000,
        operating_cash_flow=120_000_000,
        total_assets=900_000_000,         # ROA_t-1 = 0.111 -> Delta ROA > 0
        current_assets=400_000_000,
        current_liabilities=200_000_000,  # CR_t-1 = 2.0 -> Delta Liquidity > 0
        long_term_debt=150_000_000,       # Lev_t-1 = 0.166 -> Delta Lev improved
        total_revenue=600_000_000,
        gross_profit=300_000_000,         # GM_t-1 = 50.0% -> Delta GM > 0
        shares_outstanding=100_000_000,   # No dilution
    )

    res = calculate_piotroski_f_score([p_current, p_prior], sector="Technology")
    assert res.is_eligible is True
    assert res.status == "available"
    assert res.f_score == 9
    assert res.roa_positive is True
    assert res.cfo_positive is True
    assert res.delta_roa_positive is True
    assert res.accrual_quality is True
    assert res.delta_leverage_improved is True
    assert res.delta_liquidity_improved is True
    assert res.no_share_dilution is True
    assert res.delta_gross_margin_improved is True
    assert res.delta_asset_turnover_improved is True


def test_piotroski_f_score_poor_0():
    """Test a distressed company failing all 9 criteria."""
    p_current = FinancialAutopsyPeriod(
        fiscal_period_end="2025-12-31",
        net_income=-50_000_000,
        operating_cash_flow=-80_000_000,  # CFO < Net Income
        total_assets=1_200_000_000,       # ROA_t = -0.0416
        current_assets=200_000_000,
        current_liabilities=300_000_000,  # CR_t = 0.66
        long_term_debt=600_000_000,       # Lev_t = 0.50
        total_revenue=400_000_000,
        gross_profit=80_000_000,          # GM_t = 20.0%
        shares_outstanding=150_000_000,   # Diluted from 100M
    )
    p_prior = FinancialAutopsyPeriod(
        fiscal_period_end="2024-12-31",
        net_income=-10_000_000,
        operating_cash_flow=20_000_000,
        total_assets=1_000_000_000,       # ROA_t-1 = -0.01 -> Delta ROA negative
        current_assets=300_000_000,
        current_liabilities=200_000_000,  # CR_t-1 = 1.5 -> Delta Liquidity worsened
        long_term_debt=200_000_000,       # Lev_t-1 = 0.20 -> Delta Lev worsened
        total_revenue=500_000_000,
        gross_profit=150_000_000,         # GM_t-1 = 30.0% -> Delta GM negative
        shares_outstanding=100_000_000,
    )

    res = calculate_piotroski_f_score([p_current, p_prior], sector="Consumer Cyclical")
    assert res.is_eligible is True
    assert res.status == "available"
    assert res.f_score == 0
    assert res.roa_positive is False
    assert res.cfo_positive is False
    assert res.delta_roa_positive is False
    assert res.accrual_quality is False
    assert res.delta_leverage_improved is False
    assert res.delta_liquidity_improved is False
    assert res.no_share_dilution is False
    assert res.delta_gross_margin_improved is False
    assert res.delta_asset_turnover_improved is False


def test_piotroski_sector_exclusion():
    """Test that financial institutions and REITs are excluded."""
    p1 = FinancialAutopsyPeriod(fiscal_period_end="2025-12-31", net_income=100)
    p2 = FinancialAutopsyPeriod(fiscal_period_end="2024-12-31", net_income=80)

    for sec in ["Financial Services", "Financials", "Real Estate", "Banks", "Insurance", "REIT"]:
        res = calculate_piotroski_f_score([p1, p2], sector=sec)
        assert res.is_eligible is False
        assert res.status == "not_applicable"
        assert res.f_score is None
        assert "excluded" in res.exclusion_reason.lower()


def test_piotroski_insufficient_periods():
    """Test handling when fewer than 2 periods exist."""
    res = calculate_piotroski_f_score([], sector="Technology")
    assert res.status == "unavailable"
    assert res.f_score is None

    p1 = FinancialAutopsyPeriod(fiscal_period_end="2025-12-31", net_income=100)
    res_one = calculate_piotroski_f_score([p1], sector="Technology")
    assert res_one.status == "partial"
    assert res_one.f_score is None
