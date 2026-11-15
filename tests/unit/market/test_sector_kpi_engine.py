"""Unit tests for Declarative Sector KPI Engine (Phase 3 & v3.1)."""
import pandas as pd
import pytest
from tools.market.financial_autopsy import FinancialAutopsyPeriod, _build_rolling_ttm_period
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


def test_semiconductor_industry_routing_and_ttm_preference():
    """Verify sector='Technology' + industry='Semiconductors' routes to Semi KPIs and uses TTM flows."""
    ann_2025 = FinancialAutopsyPeriod(
        fiscal_period_end="2025-12-31",
        period_type="annual",
        total_revenue=50_000.0,
        gross_profit=17_500.0,         # 35%
        capital_expenditure=15_000.0,  # 30%
        operating_cash_flow=10_000.0,
        free_cash_flow=-5_000.0,       # -50%
    )
    ttm = FinancialAutopsyPeriod(
        fiscal_period_end="2026-09-30 (TTM)",
        period_type="ttm",
        total_revenue=60_000.0,
        gross_profit=27_000.0,         # 45.0%
        capital_expenditure=12_000.0,  # 20.0%
        operating_cash_flow=20_000.0,
        free_cash_flow=8_000.0,        # 40.0%
    )

    summary = compute_sector_kpis(
        sector="Technology",
        periods=[ann_2025],
        industry="Semiconductors",
        ttm_period=ttm,
    )
    assert summary.sector_name == "Technology"
    assert summary.status == "available"
    names = {k.name: k.value for k in summary.kpis}
    assert "Rule of 40 (Growth + FCF Margin)" not in names
    assert names["Gross Margin (%)"] == 45.0
    assert names["CapEx Intensity (CapEx / Revenue %)"] == 20.0
    assert names["FCF Conversion (FCF / OCF %)"] == 40.0


@pytest.mark.parametrize(
    ("industry", "kpi_name"),
    [
        ("Semiconductors", "Gross Margin (%)"),
        ("Software - Infrastructure", "Subscription / Gross Margin"),
    ],
)
def test_gross_margin_is_unavailable_when_ttm_gross_profit_is_missing(industry, kpi_name):
    latest_quarter = FinancialAutopsyPeriod(
        fiscal_period_end="2026-09-30",
        period_type="quarterly",
        total_revenue=100.0,
        gross_profit=50.0,
    )
    ttm_period = FinancialAutopsyPeriod(
        fiscal_period_end="2026-09-30 (TTM)",
        period_type="ttm",
        total_revenue=400.0,
        gross_profit=None,
    )

    summary = compute_sector_kpis(
        sector="Technology",
        periods=[latest_quarter],
        industry=industry,
        ttm_period=ttm_period,
    )
    gross_margin = next(kpi for kpi in summary.kpis if kpi.name == kpi_name)
    assert gross_margin.value is None
    assert gross_margin.status == "unavailable"


def test_ttm_gross_profit_requires_all_four_quarters():
    quarters = pd.to_datetime(["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"])
    partial = pd.DataFrame(
        [[100.0, 100.0, 100.0, 100.0], [50.0, None, None, None]],
        index=["Total Revenue", "Gross Profit"],
        columns=quarters,
    )
    partial_ttm = _build_rolling_ttm_period(partial, None)
    assert partial_ttm is not None
    assert partial_ttm.total_revenue == 400.0
    assert partial_ttm.gross_profit is None

    complete = partial.copy()
    complete.loc["Gross Profit"] = [50.0, 40.0, 30.0, 20.0]
    complete_ttm = _build_rolling_ttm_period(complete, None)
    assert complete_ttm is not None
    assert complete_ttm.gross_profit == 140.0


@pytest.mark.parametrize("missing_quarters", [1, 2, 3])
def test_ttm_revenue_is_unavailable_when_any_of_four_quarters_is_missing(missing_quarters):
    quarters = pd.to_datetime(["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"])
    revenue = [100.0, 100.0, 100.0, 100.0]
    for index in range(missing_quarters):
        revenue[index] = None
    financials = pd.DataFrame(
        [revenue, [50.0, 50.0, 50.0, 50.0]],
        index=["Total Revenue", "Gross Profit"],
        columns=quarters,
    )

    ttm = _build_rolling_ttm_period(financials, None)
    assert ttm is not None
    assert ttm.total_revenue is None
    assert ttm.gross_profit == 200.0

    for industry, kpi_name in (
        ("Semiconductors", "Gross Margin (%)"),
        ("Software - Infrastructure", "Subscription / Gross Margin"),
    ):
        summary = compute_sector_kpis(
            sector="Technology",
            periods=[],
            industry=industry,
            ttm_period=ttm,
        )
        gross_margin = next(kpi for kpi in summary.kpis if kpi.name == kpi_name)
        assert gross_margin.value is None
        assert gross_margin.status == "unavailable"


@pytest.mark.parametrize(
    ("industry", "kpi_name"),
    [
        ("Semiconductors", "Gross Margin (%)"),
        ("Software - Infrastructure", "Subscription / Gross Margin"),
    ],
)
def test_zero_ttm_revenue_does_not_publish_or_score_gross_margin(industry, kpi_name):
    ttm = FinancialAutopsyPeriod(
        fiscal_period_end="2026-09-30 (TTM)",
        period_type="ttm",
        total_revenue=0.0,
        gross_profit=200.0,
    )
    summary = compute_sector_kpis(
        sector="Technology",
        periods=[],
        industry=industry,
        ttm_period=ttm,
    )
    gross_margin = next(kpi for kpi in summary.kpis if kpi.name == kpi_name)
    assert gross_margin.value is None
    assert gross_margin.status == "unavailable"
    assert summary.sector_health_score is None


def test_saas_rule_of_40_uses_ttm_fcf_margin_and_annual_yoy_growth():
    """Rule of 40 mixes TTM FCF margin with comparable annual YoY revenue growth."""
    ann_2024 = FinancialAutopsyPeriod(
        fiscal_period_end="2024-12-31",
        period_type="annual",
        total_revenue=100.0,
        free_cash_flow=10.0,
    )
    ann_2025 = FinancialAutopsyPeriod(
        fiscal_period_end="2025-12-31",
        period_type="annual",
        total_revenue=120.0,  # +20% YoY
        free_cash_flow=12.0,  # 10% annual FCF margin
        gross_profit=90.0,
    )
    ttm = FinancialAutopsyPeriod(
        fiscal_period_end="2026-09-30 (TTM)",
        period_type="ttm",
        total_revenue=140.0,
        free_cash_flow=35.0,  # 25% TTM FCF margin
        gross_profit=112.0,   # 80% TTM gross margin
    )

    summary = compute_sector_kpis(
        sector="Technology",
        periods=[ann_2025, ann_2024],
        industry="Software - Infrastructure",
        ttm_period=ttm,
    )
    names = {k.name: k.value for k in summary.kpis}
    assert names["Rule of 40 (Growth + FCF Margin)"] == 45.0  # 20% + 25%
    assert names["Subscription / Gross Margin"] == 80.0



def test_ttm_alignment_counts_distinct_income_quarters_when_filing_dates_drift():
    from datetime import date

    fin_dates = ["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"]
    cf_dates = [date(2026, 10, 1), date(2026, 7, 1), date(2026, 4, 1), date(2026, 1, 1)]
    financials = pd.DataFrame(
        [[200.0, 100.0, 80.0, 60.0], [80.0, 20.0, 20.0, 20.0]],
        index=["Total Revenue", "Gross Profit"],
        columns=fin_dates,
    )
    cashflow = pd.DataFrame(
        [[50.0, 25.0, 20.0, 15.0]],
        index=["Operating Cash Flow"],
        columns=cf_dates,
    )

    ttm = _build_rolling_ttm_period(financials, cashflow)
    assert ttm is not None
    assert ttm.fiscal_period_end == "2026-09-30 (TTM)"
    assert ttm.total_revenue == 440.0
    assert ttm.gross_profit == 140.0
    assert ttm.operating_cash_flow == 110.0

    for industry, kpi_name in (
        ("Semiconductors", "Gross Margin (%)"),
        ("Software - Infrastructure", "Subscription / Gross Margin"),
    ):
        summary = compute_sector_kpis(
            sector="Technology",
            periods=[],
            industry=industry,
            ttm_period=ttm,
        )
        gross_margin = next(kpi for kpi in summary.kpis if kpi.name == kpi_name)
        assert gross_margin.value == 31.8
        assert gross_margin.status == "available"


@pytest.mark.parametrize(
    ("industry", "kpi_name"),
    [
        ("Semiconductors", "Gross Margin (%)"),
        ("Software - Infrastructure", "Subscription / Gross Margin"),
    ],
)
def test_gap_rejected_ttm_does_not_publish_gross_margin_and_quarter_fallback_is_labeled(industry, kpi_name):
    gapped = pd.DataFrame(
        [[200.0, 100.0, 80.0, 60.0], [80.0, 20.0, 20.0, 20.0]],
        index=["Total Revenue", "Gross Profit"],
        columns=pd.to_datetime(["2026-09-30", "2026-06-30", "2025-12-31", "2025-09-30"]),
    )
    assert _build_rolling_ttm_period(gapped, None) is None
    unavailable = compute_sector_kpis(
        sector="Technology",
        periods=[],
        industry=industry,
        ttm_period=None,
    )
    assert unavailable.status == "unavailable"
    assert not any(kpi.status == "available" for kpi in unavailable.kpis)

    quarter = FinancialAutopsyPeriod(
        fiscal_period_end="2026-09-30",
        period_type="quarterly",
        total_revenue=100.0,
        gross_profit=50.0,
    )
    fallback = compute_sector_kpis(
        sector="Technology",
        periods=[quarter],
        industry=industry,
        ttm_period=None,
    )
    gross_margin = next(kpi for kpi in fallback.kpis if kpi.name == kpi_name)
    assert gross_margin.status == "available"
    assert gross_margin.period_basis == "quarterly"
