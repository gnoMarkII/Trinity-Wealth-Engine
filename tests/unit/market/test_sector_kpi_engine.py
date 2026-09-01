"""Unit tests for Declarative Sector KPI Engine (Phase 3 & v3.1)."""
import pytest
from tools.market.financial_autopsy import FinancialAutopsyPeriod
from tools.market.sector_kpi_engine import compute_sector_kpis


def test_saas_rule_of_40_kpi():
    p1 = FinancialAutopsyPeriod(
        fiscal_period_end="2023-12-31",
        total_revenue=100000000.0,
        free_cash_flow=30000000.0,
        gross_profit=75000000.0,
    )
    p2 = FinancialAutopsyPeriod(
        fiscal_period_end="2024-12-31",
        total_revenue=120000000.0,  # 20% YoY growth
        free_cash_flow=36000000.0,       # 30% FCF margin
        gross_profit=96000000.0,  # 80% gross margin
    )

    summary = compute_sector_kpis(sector="Technology - Software (SaaS)", periods=[p1, p2])
    assert summary.sector_name == "Technology - Software (SaaS)"
    assert summary.status == "available"
    
    # Rule of 40 = 20% + 30% = 50%
    r40 = next(k for k in summary.kpis if "Rule of 40" in k.name)
    assert r40.value == 50.0
    assert r40.status == "available"
    assert summary.sector_health_score is not None
    assert summary.sector_health_score >= 90.0


def test_energy_fcf_conversion_kpi():
    p = FinancialAutopsyPeriod(
        fiscal_period_end="2024-09-30",
        total_revenue=500000000.0,
        operating_cash_flow=100000000.0,
        capital_expenditure=-30000000.0,
        free_cash_flow=70000000.0,  # 70% FCF conversion
    )

    summary = compute_sector_kpis(sector="Energy - Oil & Gas", periods=[p])
    assert summary.status == "available"
    
    fcf_conv = next(k for k in summary.kpis if "FCF Conversion" in k.name)
    assert fcf_conv.value == 70.0


def test_insufficient_periods_graceful_fallback():
    summary = compute_sector_kpis(sector="Healthcare", periods=[])
    assert summary.status == "unavailable"
    assert "insufficient_financial_periods" in summary.flags
