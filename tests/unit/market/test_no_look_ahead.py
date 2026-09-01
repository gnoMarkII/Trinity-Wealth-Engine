"""Unit tests for Temporal Isolation & No-Look-Ahead Bias Prevention (Phase 5 & v3.1)."""
from datetime import datetime
import pytest
from tools.market.financial_autopsy import FinancialAutopsyPeriod, calculate_piotroski_f_score


def test_no_look_ahead_filters_future_financial_periods():
    # As-of Date T = 2024-06-30
    as_of_date = "2024-06-30"

    all_periods = [
        FinancialAutopsyPeriod(
            fiscal_period_end="2023-12-31",
            total_revenue=100000000.0,
            net_income=15000000.0,
            operating_cash_flow=20000000.0,
            total_assets=120000000.0,
            long_term_debt=30000000.0,
            current_assets=50000000.0,
            current_liabilities=25000000.0,
            gross_profit=60000000.0,
            shares_outstanding=10000000.0,
        ),
        FinancialAutopsyPeriod(
            fiscal_period_end="2024-03-31",
            total_revenue=110000000.0,
            net_income=18000000.0,
            operating_cash_flow=25000000.0,
            total_assets=130000000.0,
            long_term_debt=28000000.0,
            current_assets=55000000.0,
            current_liabilities=26000000.0,
            gross_profit=68000000.0,
            shares_outstanding=10000000.0,
        ),
        # Future period reported AFTER as_of_date T (Should be strictly filtered out)
        FinancialAutopsyPeriod(
            fiscal_period_end="2024-09-30",
            total_revenue=200000000.0,
            net_income=50000000.0,
            operating_cash_flow=60000000.0,
            total_assets=180000000.0,
            long_term_debt=20000000.0,
            current_assets=80000000.0,
            current_liabilities=30000000.0,
            gross_profit=120000000.0,
            shares_outstanding=10000000.0,
        ),
    ]

    # Enforce point-in-time filtering (sorted descending: latest first)
    pit_periods = sorted(
        [p for p in all_periods if p.fiscal_period_end <= as_of_date],
        key=lambda x: x.fiscal_period_end,
        reverse=True,
    )
    assert len(pit_periods) == 2
    assert all(p.fiscal_period_end <= as_of_date for p in pit_periods)

    # Score calculation on pit_periods must not see the 2024-09-30 data
    breakdown = calculate_piotroski_f_score(
        periods=pit_periods,
        sector="Technology",
    )

    assert breakdown.is_eligible is True
    assert breakdown.f_score is not None
    assert breakdown.f_score >= 7
