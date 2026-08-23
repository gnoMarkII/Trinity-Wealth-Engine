"""Financial Domain Validation: Accounting Checks, Fail-Closed FCF Validation, and Coverage Stats."""
import logging
from typing import Optional
from tools.market.financials.domain.constants import (
    REQUIRED_CORE_FIELDS,
    REQUIRED_EXPANDED_FIELDS,
)
from tools.market.financials.domain.models import (
    FinancialCellDTO,
    FinancialPeriodDTO,
    FinancialStatementCategoryDTO,
)

log = logging.getLogger(__name__)


def validate_periods(
    inc_periods: list[FinancialPeriodDTO],
    bs_periods: list[FinancialPeriodDTO],
    cf_periods: list[FinancialPeriodDTO],
    is_quarter: bool = True,
) -> tuple[bool, list[str], list[str]]:
    """Comprehensive Multi-Statement Accounting Validation Engine (Core & Expanded)

    Validates:
    1. Balance Sheet: Total Assets == Total Liabilities + Total Equity (within $0.1M or 0.1%)
    2. Income Statement: Gross Profit == Rev - COGS; OpInc == GP - Opex
    3. Cash Flow: Calculated Free Cash Flow == OCF - |CapEx|
    4. Non-GAAP Reconciliation: Reported FCF == Calc FCF + Disclosed Adjustments

    Returns:
        tuple[bool, list[str], list[str]]: (is_valid, core_warnings, expanded_warnings)
    """
    core_warn: list[str] = []
    exp_warn: list[str] = []
    valid = True

    # 1. Balance Sheet Validations
    for p in bs_periods:
        tot_a = p.items.get("total_assets", FinancialCellDTO()).value
        tot_l = p.items.get("total_liabilities", FinancialCellDTO()).value
        tot_eq = p.items.get("total_equity", FinancialCellDTO()).value
        if tot_eq is None:
            tot_eq = p.items.get("stockholders_equity", FinancialCellDTO()).value
        stk_eq = p.items.get("stockholders_equity", FinancialCellDTO()).value
        nci_cell = p.items.get("noncontrolling_interests", FinancialCellDTO())
        nci = nci_cell.value if (nci_cell.source_type == "reported" and nci_cell.value is not None) else 0.0

        curr_a = p.items.get("current_assets", FinancialCellDTO()).value
        tot_debt = p.items.get("total_debt", FinancialCellDTO()).value
        st_debt = p.items.get("short_term_debt", FinancialCellDTO()).value
        lt_debt = p.items.get("long_term_debt", FinancialCellDTO()).value
        def_rev_curr = p.items.get("deferred_revenue_current", FinancialCellDTO()).value
        def_rev_noncurr = p.items.get("deferred_revenue_noncurrent", FinancialCellDTO()).value

        # Assets ≈ Liabilities + Total Equity (strict tolerance: <= $0.1M or <= 0.1%)
        if tot_a is not None and tot_l is not None and tot_eq is not None:
            exp_a = tot_l + tot_eq
            gap = abs(tot_a - exp_a)
            if gap > 100_000 and (gap / max(abs(tot_a), 1.0)) > 0.001:
                valid = False
                core_warn.append(f"Reconciliation gap in {p.period_key}: Assets (${tot_a:,.0f}) != Liab + Total Equity (${exp_a:,.0f})")

        # Total Equity ≈ Stockholders' Equity + NCI
        if tot_eq is not None and stk_eq is not None and nci_cell.source_type == "reported" and nci > 0:
            exp_tot_eq = stk_eq + nci
            gap_eq = abs(tot_eq - exp_tot_eq)
            if gap_eq > 100_000 and (gap_eq / max(abs(tot_eq), 1.0)) > 0.001:
                valid = False
                core_warn.append(f"Equity reconciliation gap in {p.period_key}: Total Equity (${tot_eq:,.0f}) != Stockholders' Equity + NCI (${exp_tot_eq:,.0f})")

        # Total Debt >= sub debts
        if tot_debt is not None and (st_debt is not None or lt_debt is not None):
            sub_debt = (st_debt or 0.0) + (lt_debt or 0.0)
            if sub_debt > tot_debt * 1.05 and (sub_debt - tot_debt) > 5_000_000:
                valid = False
                core_warn.append(f"Validation gap in {p.period_key}: Sub-debts exceed Total Debt")

        # Current / Non-Current Deferred revenue uniqueness check
        if def_rev_curr is not None and def_rev_noncurr is not None:
            if def_rev_curr == def_rev_noncurr and def_rev_curr > 0:
                exp_warn.append(f"Validation warning in {p.period_key}: Current and Non-Current Deferred Revenue have identical non-zero values")

    # 2. Income Statement Validations
    for p in inc_periods:
        rev = p.items.get("revenue", FinancialCellDTO()).value
        cogs = p.items.get("cost_of_revenue", FinancialCellDTO()).value
        gp = p.items.get("gross_profit", FinancialCellDTO()).value
        opex = p.items.get("operating_expenses", FinancialCellDTO()).value
        op_inc = p.items.get("operating_income", FinancialCellDTO()).value

        # Gross Profit ≈ Revenue - COGS
        if rev is not None and cogs is not None and gp is not None:
            exp_gp = rev - cogs
            gap = abs(gp - exp_gp)
            if gap > 100_000 and (gap / max(abs(gp), 1.0)) > 0.01:
                valid = False
                core_warn.append(f"Income validation gap in {p.period_key}: Gross Profit (${gp:,.0f}) != Rev - COGS (${exp_gp:,.0f})")

        # Operating Income ≈ Gross Profit - Operating Expenses
        if gp is not None and opex is not None and op_inc is not None:
            exp_op = gp - opex
            gap = abs(op_inc - exp_op)
            if gap > 100_000 and (gap / max(abs(op_inc), 1.0)) > 0.01:
                valid = False
                core_warn.append(f"Income validation gap in {p.period_key}: Operating Income (${op_inc:,.0f}) != GP - Opex (${exp_op:,.0f})")

    # 3. Cash Flow Validations
    for p in cf_periods:
        ocf = p.items.get("operating_cash_flow", FinancialCellDTO()).value
        capex = p.items.get("capital_expenditure", FinancialCellDTO()).value
        calc_fcf_cell = p.items.get("calculated_free_cash_flow", FinancialCellDTO())
        calc_fcf = calc_fcf_cell.value if calc_fcf_cell.value is not None else p.items.get("free_cash_flow", FinancialCellDTO()).value
        rep_fcf_cell = p.items.get("reported_free_cash_flow", FinancialCellDTO())
        rep_fcf = rep_fcf_cell.value
        adj_fcf_cell = p.items.get("free_cash_flow_adjustments", FinancialCellDTO())
        adj_fcf = adj_fcf_cell.value

        # Calculated FCF ≈ OCF - |CapEx|
        if ocf is not None and capex is not None and calc_fcf is not None:
            exp_fcf = ocf - abs(capex)
            gap = abs(calc_fcf - exp_fcf)
            if gap > 100_000:
                valid = False
                core_warn.append(f"Cash flow validation gap in {p.period_key}: Calculated FCF (${calc_fcf:,.0f}) != OCF - CapEx (${exp_fcf:,.0f})")

        # Non-GAAP FCF Fail-Closed Validation
        if rep_fcf is not None:
            # Check 1: Percentage check
            if abs(rep_fcf) < 1_000_000 and (ocf is not None and abs(ocf) > 50_000_000):
                p.items["reported_free_cash_flow"] = FinancialCellDTO(
                    value=None,
                    source_type="unavailable",
                    unavailable_reason="FCF_RECONCILIATION_FAILED",
                    source_filing_url=rep_fcf_cell.source_filing_url,
                    is_derived=False,
                )
                exp_warn.append(f"FCF margin percentage rejected in {p.period_key}: Extracted value (${rep_fcf:,.2f}) appears to be margin %")
            elif calc_fcf is not None:
                # Check 2: Mathematical reconciliation check
                exp_rep_fcf = calc_fcf + (adj_fcf or 0.0)
                fcf_gap = abs(rep_fcf - exp_rep_fcf)
                if fcf_gap > 5_000_000 and (fcf_gap / max(abs(rep_fcf), 1.0)) > 0.02:
                    p.items["reported_free_cash_flow"] = FinancialCellDTO(
                        value=None,
                        source_type="unavailable",
                        unavailable_reason="FCF_RECONCILIATION_FAILED",
                        source_filing_url=rep_fcf_cell.source_filing_url,
                        is_derived=False,
                    )
                    exp_warn.append(f"FCF reconciliation failed in {p.period_key}: Reported FCF (${rep_fcf:,.0f}) != Calc FCF + Disclosed Adj (${exp_rep_fcf:,.0f})")

    return valid, core_warn, exp_warn


def compute_coverage_metrics(
    categories: list[FinancialStatementCategoryDTO],
) -> tuple[float, float, list[str], list[str]]:
    """คำนวณ Core & Expanded Coverage Percentages และรายการ Missing Fields"""
    total_core_fields = 0
    present_core_fields = 0
    missing_required: list[str] = []

    total_expanded_fields = 0
    present_expanded_fields = 0
    missing_expanded: list[str] = []

    for cat in categories:
        st_type = cat.statement_type
        req_core = REQUIRED_CORE_FIELDS.get(st_type, [])
        req_exp = REQUIRED_EXPANDED_FIELDS.get(st_type, [])

        for p in cat.periods:
            for field in req_core:
                total_core_fields += 1
                cell = p.items.get(field)
                if cell and cell.value is not None and cell.source_type != "unavailable":
                    present_core_fields += 1
                else:
                    missing_required.append(f"{p.period_key}: {field}")

            for field in req_exp:
                cell = p.items.get(field)
                if cell and cell.source_type == "not_applicable":
                    continue
                total_expanded_fields += 1
                if cell and cell.value is not None and cell.source_type != "unavailable":
                    present_expanded_fields += 1
                else:
                    missing_expanded.append(f"{p.period_key}: {field}")

    core_pct = round((present_core_fields / max(total_core_fields, 1)) * 100.0, 1)
    exp_pct = round((present_expanded_fields / max(total_expanded_fields, 1)) * 100.0, 1)

    return core_pct, exp_pct, missing_required, missing_expanded
