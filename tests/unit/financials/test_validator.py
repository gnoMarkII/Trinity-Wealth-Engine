"""Unit tests for Multi-Statement Validation and Fail-Closed FCF Reconciler."""
from tools.market.financials.domain.models import (
    FinancialCellDTO,
    FinancialPeriodDTO,
    FinancialStatementCategoryDTO,
    LineItemMetaDTO,
)
from tools.market.financials.domain.validator import (
    compute_coverage_metrics,
    validate_periods,
)


def test_balance_sheet_validation_pass():
    p_bs = FinancialPeriodDTO(
        period_key="2024-FY",
        fiscal_year=2024,
        period_end_date="2024-12-31",
        period_kind="instant",
        form_type="10-K",
        items={
            "total_assets": FinancialCellDTO(value=1_000_000.0),
            "total_liabilities": FinancialCellDTO(value=400_000.0),
            "total_equity": FinancialCellDTO(value=600_000.0),
            "stockholders_equity": FinancialCellDTO(value=600_000.0),
            "noncontrolling_interests": FinancialCellDTO(value=0.0, source_type="not_applicable"),
        },
    )
    is_valid, core_w, exp_w = validate_periods([], [p_bs], [], is_quarter=False)
    assert is_valid is True
    assert len(core_w) == 0


def test_balance_sheet_validation_gap():
    p_bs = FinancialPeriodDTO(
        period_key="2024-FY",
        fiscal_year=2024,
        period_end_date="2024-12-31",
        period_kind="instant",
        form_type="10-K",
        items={
            "total_assets": FinancialCellDTO(value=1_000_000.0),
            "total_liabilities": FinancialCellDTO(value=400_000.0),
            "total_equity": FinancialCellDTO(value=200_000.0),  # Gap of $400k
        },
    )
    is_valid, core_w, exp_w = validate_periods([], [p_bs], [], is_quarter=False)
    assert is_valid is False
    assert any("Reconciliation gap" in w for w in core_w)


def test_fcf_fail_closed_margin_percentage_rejected():
    p_cf = FinancialPeriodDTO(
        period_key="2026-Q2",
        fiscal_year=2026,
        fiscal_quarter=2,
        period_end_date="2026-06-30",
        period_kind="duration",
        form_type="10-Q",
        items={
            "operating_cash_flow": FinancialCellDTO(value=986_500_000.0),
            "capital_expenditure": FinancialCellDTO(value=-51_200_000.0),
            "calculated_free_cash_flow": FinancialCellDTO(value=935_300_000.0),
            "reported_free_cash_flow": FinancialCellDTO(value=48.6, source_type="reported"),  # Extracted margin % instead of $M
            "free_cash_flow_adjustments": FinancialCellDTO(value=30_300_000.0),
        },
    )
    is_valid, core_w, exp_w = validate_periods([], [], [p_cf], is_quarter=True)
    rep_cell = p_cf.items["reported_free_cash_flow"]
    assert rep_cell.source_type == "unavailable"
    assert rep_cell.unavailable_reason == "FCF_RECONCILIATION_FAILED"
    assert any("percentage" in w.lower() for w in exp_w)


def test_fcf_reconciliation_pass():
    p_cf = FinancialPeriodDTO(
        period_key="2026-Q2",
        fiscal_year=2026,
        fiscal_quarter=2,
        period_end_date="2026-06-30",
        period_kind="duration",
        form_type="10-Q",
        items={
            "operating_cash_flow": FinancialCellDTO(value=986_500_000.0),
            "capital_expenditure": FinancialCellDTO(value=-51_200_000.0),
            "calculated_free_cash_flow": FinancialCellDTO(value=935_300_000.0),
            "reported_free_cash_flow": FinancialCellDTO(value=965_600_000.0, source_type="reported"),
            "free_cash_flow_adjustments": FinancialCellDTO(value=30_300_000.0),
        },
    )
    is_valid, core_w, exp_w = validate_periods([], [], [p_cf], is_quarter=True)
    rep_cell = p_cf.items["reported_free_cash_flow"]
    assert rep_cell.source_type == "reported"
    assert rep_cell.value == 965_600_000.0
    assert len(exp_w) == 0


def test_compute_coverage_metrics():
    cat = FinancialStatementCategoryDTO(
        statement_type="income",
        period_kind="duration",
        periods=[
            FinancialPeriodDTO(
                period_key="2024-FY",
                fiscal_year=2024,
                period_end_date="2024-12-31",
                period_kind="duration",
                form_type="10-K",
                items={
                    "revenue": FinancialCellDTO(value=100.0),
                    "gross_profit": FinancialCellDTO(value=60.0),
                    "operating_income": FinancialCellDTO(value=20.0),
                    "net_income": FinancialCellDTO(value=15.0),
                    "eps_diluted": FinancialCellDTO(value=1.5),
                },
            )
        ],
        line_items=[],
    )
    core_pct, exp_pct, missing_req, missing_exp = compute_coverage_metrics([cat])
    assert core_pct == 100.0
    assert len(missing_req) == 0
