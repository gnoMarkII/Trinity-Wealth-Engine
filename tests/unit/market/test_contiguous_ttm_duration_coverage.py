"""Unit tests for contiguous trailing duration TTM validation, 52/53-week, and stub transition periods."""
import pytest
from tools.market.financial_autopsy import (
    deaccumulate_quarterly_cashflows,
    compute_ttm_standardized_fundamentals,
    compute_roic_average_invested_capital,
)


def test_contiguous_ttm_standard_calendar():
    # 4 contiguous quarters covering 365 days
    quarters = [
        {"fiscal_period_start": "2025-07-01", "fiscal_period_end": "2025-09-30", "duration_days": 92, "total_revenue": 1508.0, "operating_income": 490.0, "operating_cash_flow": 600.0, "capital_expenditure": 50.0, "net_income": 400.0, "tax_expense": 80.0, "income_before_tax": 480.0},
        {"fiscal_period_start": "2025-10-01", "fiscal_period_end": "2025-12-31", "duration_days": 92, "total_revenue": 1800.0, "operating_income": 600.0, "operating_cash_flow": 800.0, "capital_expenditure": 70.0, "net_income": 520.0, "tax_expense": 100.0, "income_before_tax": 620.0},
        {"fiscal_period_start": "2026-01-01", "fiscal_period_end": "2026-03-31", "duration_days": 90, "total_revenue": 2000.0, "operating_income": 650.0, "operating_cash_flow": 950.0, "capital_expenditure": 75.0, "net_income": 580.0, "tax_expense": 110.0, "income_before_tax": 690.0},
        {"fiscal_period_start": "2026-04-01", "fiscal_period_end": "2026-06-30", "duration_days": 91, "total_revenue": 2219.4, "operating_income": 702.2, "operating_cash_flow": 1046.1, "capital_expenditure": 84.1, "net_income": 620.7, "tax_expense": 120.0, "income_before_tax": 740.7},
    ]

    res = compute_ttm_standardized_fundamentals(quarters, issuer_reported_fcf=3117.0, market="US", is_us_domestic=True)
    assert res["status"] == "complete"
    assert res["ttm_revenue"] == 7527.4
    assert res["ttm_operating_income"] == 2442.2
    assert res["standardized_ttm_gaap_operating_margin_pct"] == 32.44
    assert res["standardized_ttm_fcf"] == 3117.0
    assert res["ocf_to_net_income"] == 1.60  # 3396.1 / 2120.7 = 1.6014


def test_contiguous_ttm_gap_detection_fails():
    # Quarters with a 3-month gap between Q2 and Q4
    broken_quarters = [
        {"fiscal_period_start": "2025-07-01", "fiscal_period_end": "2025-09-30", "duration_days": 92, "total_revenue": 1500.0, "operating_income": 400.0},
        {"fiscal_period_start": "2025-10-01", "fiscal_period_end": "2025-12-31", "duration_days": 92, "total_revenue": 1600.0, "operating_income": 450.0},
        # Missing Q1 2026!
        {"fiscal_period_start": "2026-04-01", "fiscal_period_end": "2026-06-30", "duration_days": 91, "total_revenue": 1800.0, "operating_income": 500.0},
        {"fiscal_period_start": "2026-07-01", "fiscal_period_end": "2026-09-30", "duration_days": 92, "total_revenue": 1900.0, "operating_income": 550.0},
    ]

    res = compute_ttm_standardized_fundamentals(broken_quarters, market="US")
    assert res["status"] == "partial"
    assert res["exclusion_reason"] == "incomplete_trailing_duration_coverage"
    assert res["has_gap_or_overlap"] is True


def test_deaccumulation_52_53_week_and_stub():
    # Cumulative periods with 13W, 26W, 39W, 52W structure
    cumulative = [
        {"fiscal_year": "2026", "fiscal_period_end": "2026-04-02", "duration_days": 91, "operating_cash_flow": 300.0, "capital_expenditure": 30.0},
        {"fiscal_year": "2026", "fiscal_period_end": "2026-07-02", "duration_days": 182, "operating_cash_flow": 650.0, "capital_expenditure": 65.0},
        {"fiscal_year": "2026", "fiscal_period_end": "2026-10-01", "duration_days": 273, "operating_cash_flow": 1050.0, "capital_expenditure": 100.0},
        {"fiscal_year": "2026", "fiscal_period_end": "2026-12-31", "duration_days": 364, "operating_cash_flow": 1500.0, "capital_expenditure": 140.0},
    ]

    standalone = deaccumulate_quarterly_cashflows(cumulative)
    assert len(standalone) == 4
    assert standalone[0]["operating_cash_flow"] == 300.0
    assert standalone[1]["operating_cash_flow"] == 350.0  # 650 - 300
    assert standalone[2]["operating_cash_flow"] == 400.0  # 1050 - 650
    assert standalone[3]["operating_cash_flow"] == 450.0  # 1500 - 1050


def test_roic_average_invested_capital():
    nopat = 2000.0
    bs_beg = {"total_debt": 1000.0, "stockholders_equity": 4000.0, "cash_and_equivalents": 1000.0}  # IC_beg = 4000
    bs_end = {"total_debt": 1200.0, "stockholders_equity": 5000.0, "cash_and_equivalents": 1200.0}  # IC_end = 5000
    # Average IC = 4500. ROIC = 2000 / 4500 = 44.44%

    roic, meta = compute_roic_average_invested_capital(nopat, bs_beg, bs_end)
    assert roic == 44.44
    assert meta["ic_average"] == 4500.0
