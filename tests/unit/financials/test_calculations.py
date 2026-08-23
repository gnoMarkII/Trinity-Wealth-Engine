"""Unit tests for Financial Domain Calculations (Pure Functions)."""
from tools.market.financials.domain.calculations import (
    calc_yoy_growth,
    compute_ratios_and_charts,
    finite_or_none,
)
from tools.market.financials.domain.models import (
    FinancialCellDTO,
    FinancialPeriodDTO,
)


def test_finite_or_none():
    assert finite_or_none(100.5) == 100.5
    assert finite_or_none("250.0") == 250.0
    assert finite_or_none(None) is None
    assert finite_or_none(float("nan")) is None
    assert finite_or_none(float("inf")) is None
    assert finite_or_none("not_a_number") is None


def test_calc_yoy_growth():
    assert calc_yoy_growth(120.0, 100.0) == 20.0
    assert calc_yoy_growth(80.0, 100.0) == -20.0
    assert calc_yoy_growth(100.0, 0.0) is None
    assert calc_yoy_growth(100.0, -50.0) is None
    assert calc_yoy_growth(None, 100.0) is None
    assert calc_yoy_growth(100.0, None) is None


def test_compute_ratios_and_charts():
    p_inc = FinancialPeriodDTO(
        period_key="2024-FY",
        fiscal_year=2024,
        period_end_date="2024-12-31",
        period_kind="duration",
        form_type="10-K",
        items={
            "revenue": FinancialCellDTO(value=1000.0),
            "gross_profit": FinancialCellDTO(value=600.0),
            "operating_income": FinancialCellDTO(value=300.0),
            "net_income": FinancialCellDTO(value=200.0),
        },
    )
    p_bs = FinancialPeriodDTO(
        period_key="2024-FY",
        fiscal_year=2024,
        period_end_date="2024-12-31",
        period_kind="instant",
        form_type="10-K",
        items={
            "current_assets": FinancialCellDTO(value=500.0),
            "current_liabilities": FinancialCellDTO(value=250.0),
            "total_debt": FinancialCellDTO(value=200.0),
            "total_equity": FinancialCellDTO(value=800.0),
        },
    )
    p_cf = FinancialPeriodDTO(
        period_key="2024-FY",
        fiscal_year=2024,
        period_end_date="2024-12-31",
        period_kind="duration",
        form_type="10-K",
        items={
            "calculated_free_cash_flow": FinancialCellDTO(value=250.0),
            "reported_free_cash_flow": FinancialCellDTO(value=250_000_000.0, source_type="reported"),
        },
    )

    charts, ratios = compute_ratios_and_charts([p_inc], [p_bs], [p_cf])

    assert len(charts) == 1
    assert charts[0].revenue == 1000.0
    assert charts[0].gross_profit == 600.0
    assert charts[0].operating_margin_pct == 30.0
    assert charts[0].net_margin_pct == 20.0
    assert charts[0].calculated_free_cash_flow == 250.0

    assert len(ratios) == 1
    assert ratios[0].gross_margin_pct == 60.0
    assert ratios[0].operating_margin_pct == 30.0
    assert ratios[0].current_ratio == 2.0
    assert ratios[0].debt_to_equity == 0.25
