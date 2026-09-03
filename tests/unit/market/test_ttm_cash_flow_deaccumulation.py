"""Unit tests for SEC XBRL Ingestion, YTD Cash Flow De-accumulation, and TTM Standardized Fundamentals (Phase 3 / P0.3)."""
import pytest
from tools.market.financial_autopsy import (
    SecFinancialFilingsAdapter,
    XBRLFactRecord,
    compute_ttm_standardized_fundamentals,
    deaccumulate_quarterly_cashflows,
)


def test_deaccumulate_quarterly_cashflows_exact_reconciliation():
    """Verify conversion of cumulative YTD cash flows to standalone quarters without double counting."""
    cumulative_periods = [
        # Q1 2025 (90d)
        {
            "fiscal_period_end": "2025-03-31",
            "fiscal_quarter": "Q1",
            "fiscal_year": "2025",
            "duration_days": 90,
            "operating_cash_flow": 400.0,
            "capital_expenditure": 50.0,
            "total_revenue": 1800.0,
            "operating_income": 450.0,
            "tax_expense": 80.0,
            "income_before_tax": 440.0,
        },
        # Q2 2025 YTD (180d)
        {
            "fiscal_period_end": "2025-06-30",
            "fiscal_quarter": "Q2",
            "fiscal_year": "2025",
            "duration_days": 180,
            "operating_cash_flow": 950.0,   # Cumulative 6M
            "capital_expenditure": 120.0,  # Cumulative 6M
            "total_revenue": 1900.0,
            "operating_income": 500.0,
            "tax_expense": 90.0,
            "income_before_tax": 490.0,
        },
        # Q3 2025 YTD (270d)
        {
            "fiscal_period_end": "2025-09-30",
            "fiscal_quarter": "Q3",
            "fiscal_year": "2025",
            "duration_days": 270,
            "operating_cash_flow": 1500.0,  # Cumulative 9M
            "capital_expenditure": 180.0,  # Cumulative 9M
            "total_revenue": 2000.0,
            "operating_income": 550.0,
            "tax_expense": 100.0,
            "income_before_tax": 540.0,
        },
        # Q4 2025 / FY Annual (365d)
        {
            "fiscal_period_end": "2025-12-31",
            "fiscal_quarter": "Q4",
            "fiscal_year": "2025",
            "duration_days": 365,
            "operating_cash_flow": 2200.0,  # Full Year FY
            "capital_expenditure": 250.0,  # Full Year FY
            "total_revenue": 2100.0,
            "operating_income": 600.0,
            "tax_expense": 110.0,
            "income_before_tax": 590.0,
        },
    ]

    standalone = deaccumulate_quarterly_cashflows(cumulative_periods)
    assert len(standalone) == 4

    # Q1: 400 - 50 = 350
    assert standalone[0]["operating_cash_flow"] == 400.0
    assert standalone[0]["capital_expenditure"] == 50.0
    assert standalone[0]["free_cash_flow"] == 350.0

    # Q2: 950 - 400 = 550 OCF, 120 - 50 = 70 CapEx -> 480 FCF
    assert standalone[1]["operating_cash_flow"] == 550.0
    assert standalone[1]["capital_expenditure"] == 70.0
    assert standalone[1]["free_cash_flow"] == 480.0

    # Q3: 1500 - 950 = 550 OCF, 180 - 120 = 60 CapEx -> 490 FCF
    assert standalone[2]["operating_cash_flow"] == 550.0
    assert standalone[2]["capital_expenditure"] == 60.0
    assert standalone[2]["free_cash_flow"] == 490.0

    # Q4: 2200 - 1500 = 700 OCF, 250 - 180 = 70 CapEx -> 630 FCF
    assert standalone[3]["operating_cash_flow"] == 700.0
    assert standalone[3]["capital_expenditure"] == 70.0
    assert standalone[3]["free_cash_flow"] == 630.0

    # Sum of standalone quarters must equal annual totals
    total_ocf = sum(q["operating_cash_flow"] for q in standalone)
    total_capex = sum(q["capital_expenditure"] for q in standalone)
    assert total_ocf == 2200.0
    assert total_capex == 250.0
    assert (total_ocf - total_capex) == 1950.0


def test_compute_ttm_standardized_fundamentals_dual_fcf():
    """Verify TTM metrics, GAAP margin, and Dual FCF reconciliation."""
    standalone = [
        {"fiscal_period_end": "2025-09-30", "total_revenue": 1800.0, "operating_income": 450.0, "operating_cash_flow": 550.0, "capital_expenditure": 60.0, "tax_expense": 80.0, "income_before_tax": 440.0},
        {"fiscal_period_end": "2025-12-31", "total_revenue": 2100.0, "operating_income": 600.0, "operating_cash_flow": 700.0, "capital_expenditure": 70.0, "tax_expense": 110.0, "income_before_tax": 590.0},
        {"fiscal_period_end": "2026-03-31", "total_revenue": 1950.0, "operating_income": 520.0, "operating_cash_flow": 600.0, "capital_expenditure": 65.0, "tax_expense": 95.0, "income_before_tax": 510.0},
        {"fiscal_period_end": "2026-06-30", "total_revenue": 2050.0, "operating_income": 580.0, "operating_cash_flow": 650.0, "capital_expenditure": 75.0, "tax_expense": 105.0, "income_before_tax": 560.0},
    ]

    ttm = compute_ttm_standardized_fundamentals(standalone, issuer_reported_fcf=2350.0)
    assert ttm["status"] == "complete"
    assert ttm["latest_fiscal_period_end"] == "2026-06-30"
    assert ttm["ttm_revenue"] == 7900.0
    assert ttm["ttm_operating_income"] == 2150.0
    assert ttm["standardized_ttm_gaap_operating_margin_pct"] == round(2150.0 / 7900.0 * 100.0, 2)
    # Standardized FCF = (550+700+600+650) - (60+70+65+75) = 2500 - 270 = 2230
    assert ttm["standardized_ttm_fcf"] == 2230.0
    assert ttm["issuer_reported_fcf"] == 2350.0
    assert ttm["fcf_reconciliation_delta"] == 120.0
    # Tax rate = (80+110+95+105) / (440+590+510+560) = 390 / 2100 = 18.57%
    assert ttm["ttm_effective_tax_rate_pct"] == 18.57


def test_sec_xbrl_adapter_pit_cutoff_and_restatement():
    """Verify PIT cutoff and amendment superseding in SecFinancialFilingsAdapter."""
    facts = [
        # Original Q1 Revenue filed May 2, 2026
        XBRLFactRecord(
            cik="0001262039",
            form="10-Q",
            accession_number="0001262039-26-000010",
            filed_date="2026-05-02",
            fiscal_period_end="2026-03-31",
            duration_days=90,
            concept="revenue",
            tag="RevenueFromContractWithCustomerExcludingAssessedTax",
            value=1950.0,
        ),
        # Restatement filed Aug 15, 2026
        XBRLFactRecord(
            cik="0001262039",
            form="10-Q/A",
            accession_number="0001262039-26-000045",
            filed_date="2026-08-15",
            fiscal_period_end="2026-03-31",
            duration_days=90,
            concept="revenue",
            tag="RevenueFromContractWithCustomerExcludingAssessedTax",
            value=1960.0,
            lineage_status="restated",
        ),
    ]

    # As of July 31, 2026 (Before restatement) -> must pick original 1950.0
    facts_pit = SecFinancialFilingsAdapter.select_facts(facts, as_of_date="2026-07-31")
    assert len(facts_pit) == 1
    assert facts_pit[0].value == 1950.0

    # As of August 31, 2026 (After restatement) -> must pick restated 1960.0
    facts_latest = SecFinancialFilingsAdapter.select_facts(facts, as_of_date="2026-08-31")
    assert len(facts_latest) == 1
    assert facts_latest[0].value == 1960.0
