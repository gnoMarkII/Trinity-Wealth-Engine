"""Composite US Financial Provider Adapter.

Composes EdgarSubclient and Sec8KSubclient to fetch 10-K, 10-Q, and 8-K filings,
delegating all parsing, normalization, validation, and calculations to domain services.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from tools.market.financials.adapters.edgar_subclient import EdgarSubclient
from tools.market.financials.adapters.sec_8k_subclient import Sec8KSubclient
from tools.market.financials.domain.calculations import (
    calc_yoy_growth,
    compute_ratios_and_charts,
)
from tools.market.financials.domain.constants import CANONICAL_LINE_ITEMS
from tools.market.financials.domain.models import (
    FinancialCellDTO,
    FinancialPeriodDTO,
    FinancialStatementCategoryDTO,
    FinancialStatementsDTO,
    LineItemMetaDTO,
)
from tools.market.financials.domain.normalizer import (
    collect_distinct_authoritative_filings,
    extract_canonical_cell_from_df,
    resolve_sub_line_items,
    select_xbrl_column,
    validate_sec_url,
)
from tools.market.financials.domain.validator import (
    compute_coverage_metrics,
    validate_periods,
)
from tools.market.financials.ports.provider_port import FinancialStatementProviderPort

log = logging.getLogger(__name__)


class CompositeUsFinancialProvider(FinancialStatementProviderPort):
    """Implementation of FinancialStatementProviderPort for US Market (SEC EDGAR + 8-K Press Release)."""

    def __init__(
        self,
        edgar_subclient: Optional[EdgarSubclient] = None,
        sec_8k_subclient: Optional[Sec8KSubclient] = None,
    ):
        self._edgar_client = edgar_subclient or EdgarSubclient()
        self._sec_8k_client = sec_8k_subclient or Sec8KSubclient()

    def fetch_statements(self, ticker: str, provider_symbol: str) -> Optional[FinancialStatementsDTO]:
        """ดึงข้อมูลงบการเงินของหุ้น US ผ่าน SEC EDGAR และ 8-K Press Release"""
        filings_tuple = self._edgar_client.get_company_filings(ticker)
        if not filings_tuple:
            return None

        all_10k, all_10q, all_8k = filings_tuple
        authoritative_10k = collect_distinct_authoritative_filings(all_10k, target_distinct_count=5)
        authoritative_10q = collect_distinct_authoritative_filings(all_10q, target_distinct_count=12)

        if not authoritative_10k and not authoritative_10q:
            log.warning("No 10-K or 10-Q filings found for %s", ticker)
            return None

        filing_meta_map: dict[str, dict[str, Any]] = {}
        for f in authoritative_10k + authoritative_10q:
            p = str(getattr(f, "period_of_report", ""))[:10]
            if p:
                filing_meta_map[p] = {
                    "accession_no": getattr(f, "accession_no", None) or getattr(f, "accession_number", ""),
                    "filing_url": validate_sec_url(getattr(f, "url", None) or getattr(f, "filing_url", None)),
                    "form_type": str(getattr(f, "form", "") or getattr(f, "form_type", "")),
                }

        warnings: list[str] = []
        income_annual_dict: dict[str, FinancialPeriodDTO] = {}
        bs_annual_dict: dict[str, FinancialPeriodDTO] = {}
        cf_annual_dict: dict[str, FinancialPeriodDTO] = {}

        # 1. 10-K Processing (Annual Statements)
        for filing in authoritative_10k:
            end_date = str(getattr(filing, "period_of_report", ""))[:10]
            if not end_date:
                continue
            try:
                f_year = int(end_date[:4])
            except Exception:
                f_year = 2024
            p_key = f"{f_year}-FY"
            f_meta = filing_meta_map.get(end_date, {})
            filing_url = f_meta.get("filing_url")
            form_type = f_meta.get("form_type", "10-K")

            p_inc = FinancialPeriodDTO(
                period_key=p_key,
                fiscal_year=f_year,
                fiscal_quarter=None,
                period_end_date=end_date,
                period_kind="duration",
                duration_days=365,
                form_type=form_type,
                filing_url=filing_url,
                is_derived=False,
            )
            p_bs = FinancialPeriodDTO(
                period_key=p_key,
                fiscal_year=f_year,
                fiscal_quarter=None,
                period_end_date=end_date,
                period_kind="instant",
                duration_days=None,
                form_type=form_type,
                filing_url=filing_url,
                is_derived=False,
            )
            p_cf = FinancialPeriodDTO(
                period_key=p_key,
                fiscal_year=f_year,
                fiscal_quarter=None,
                period_end_date=end_date,
                period_kind="duration",
                duration_days=365,
                form_type=form_type,
                filing_url=filing_url,
                is_derived=False,
            )

            try:
                xb = filing.xbrl()
                if xb and xb.statements:
                    inc_stmt = xb.statements.income_statement()
                    if inc_stmt:
                        df = inc_stmt.to_dataframe()
                        target_col, dur_days, col_lbl = select_xbrl_column(df.columns.tolist(), end_date, "income", is_quarterly=False)
                        if target_col:
                            for meta in CANONICAL_LINE_ITEMS["income"]:
                                val, concept = extract_canonical_cell_from_df(df, meta["key"], target_col)
                                p_inc.items[meta["key"]] = FinancialCellDTO(
                                    value=val,
                                    source_type="reported" if val is not None else "unavailable",
                                    source_concept=concept,
                                    source_filing_url=filing_url,
                                    is_derived=False,
                                )
                            resolve_sub_line_items("income", p_inc.items, df=df, target_col=target_col, filing_url=filing_url)

                    bs_stmt = xb.statements.balance_sheet()
                    if bs_stmt:
                        df = bs_stmt.to_dataframe()
                        target_col, dur_days, col_lbl = select_xbrl_column(df.columns.tolist(), end_date, "balance_sheet", is_quarterly=False)
                        if target_col:
                            for meta in CANONICAL_LINE_ITEMS["balance_sheet"]:
                                val, concept = extract_canonical_cell_from_df(df, meta["key"], target_col)
                                p_bs.items[meta["key"]] = FinancialCellDTO(
                                    value=val,
                                    source_type="reported" if val is not None else "unavailable",
                                    source_concept=concept,
                                    source_filing_url=filing_url,
                                    is_derived=False,
                                )
                            resolve_sub_line_items("balance_sheet", p_bs.items, df=df, target_col=target_col, filing_url=filing_url)

                    cf_stmt = xb.statements.cash_flow_statement()
                    if cf_stmt:
                        df = cf_stmt.to_dataframe()
                        target_col, dur_days, col_lbl = select_xbrl_column(df.columns.tolist(), end_date, "cash_flow", is_quarterly=False)
                        if target_col:
                            for meta in CANONICAL_LINE_ITEMS["cash_flow"]:
                                val, concept = extract_canonical_cell_from_df(df, meta["key"], target_col)
                                p_cf.items[meta["key"]] = FinancialCellDTO(
                                    value=val,
                                    source_type="reported" if val is not None else "unavailable",
                                    source_concept=concept,
                                    source_filing_url=filing_url,
                                    is_derived=False,
                                )

                            ocf = p_cf.items.get("operating_cash_flow", FinancialCellDTO()).value
                            capex = p_cf.items.get("capital_expenditure", FinancialCellDTO()).value
                            calc_fcf = None
                            if ocf is not None and capex is not None:
                                calc_fcf = round(ocf - abs(capex), 2)
                                p_cf.items["calculated_free_cash_flow"] = FinancialCellDTO(
                                    value=calc_fcf,
                                    source_type="derived",
                                    source_concept=None,
                                    source_filing_url=filing_url,
                                    derivation="Operating Cash Flow - |CapEx|",
                                    formula="OCF - |CapEx|",
                                    input_items=["operating_cash_flow", "capital_expenditure"],
                                    is_derived=True,
                                )
                                p_cf.items["free_cash_flow"] = FinancialCellDTO(
                                    value=calc_fcf,
                                    source_type="derived",
                                    source_concept=None,
                                    source_filing_url=filing_url,
                                    derivation="Operating Cash Flow - |CapEx|",
                                    is_derived=True,
                                )

                            # 8-K Exhibit 99.1 Non-GAAP FCF Reconciliation for FY
                            _, adj_val, rep_fcf, rep_url, adj_expl, unavail_reason = self._sec_8k_client.extract_fcf_reconciliation(
                                all_8k, f_year, end_date, is_quarterly=False
                            )
                            if rep_fcf is not None:
                                p_cf.items["reported_free_cash_flow"] = FinancialCellDTO(
                                    value=rep_fcf,
                                    source_type="reported",
                                    source_concept="8-K Exhibit 99.1 Press Release Non-GAAP Reconciliation",
                                    source_filing_url=rep_url or filing_url,
                                    is_derived=False,
                                )
                                p_cf.items["free_cash_flow_adjustments"] = FinancialCellDTO(
                                    value=adj_val if adj_val is not None else 0.0,
                                    source_type="reported",
                                    source_concept=adj_expl or "8-K Disclosed Non-GAAP adjustments",
                                    source_filing_url=rep_url or filing_url,
                                    is_derived=False,
                                )
                            else:
                                if calc_fcf is not None:
                                    p_cf.items["reported_free_cash_flow"] = FinancialCellDTO(
                                        value=calc_fcf,
                                        source_type="derived",
                                        source_concept=None,
                                        source_filing_url=filing_url,
                                        derivation="Equal to Calculated FCF (No Non-GAAP adjustments disclosed)",
                                        is_derived=True,
                                    )
                                    p_cf.items["free_cash_flow_adjustments"] = FinancialCellDTO(
                                        value=0.0,
                                        source_type="not_applicable",
                                        source_concept=None,
                                        source_filing_url=filing_url,
                                        derivation="No Non-GAAP adjustments disclosed",
                                        is_derived=False,
                                    )
                                else:
                                    p_cf.items["reported_free_cash_flow"] = FinancialCellDTO(
                                        value=None,
                                        source_type="unavailable",
                                        unavailable_reason=unavail_reason or "FCF_RECONCILIATION_NOT_FOUND",
                                        source_filing_url=filing_url,
                                        is_derived=False,
                                    )
                                    p_cf.items["free_cash_flow_adjustments"] = FinancialCellDTO(
                                        value=None,
                                        source_type="unavailable",
                                        unavailable_reason=unavail_reason or "FCF_RECONCILIATION_NOT_FOUND",
                                        source_filing_url=filing_url,
                                        is_derived=False,
                                    )
            except Exception as e:
                log.warning("Error extracting 10-K %s: %s", end_date, e)

            income_annual_dict[p_key] = p_inc
            bs_annual_dict[p_key] = p_bs
            cf_annual_dict[p_key] = p_cf

        # 2. 10-Q Processing (Quarterly Statements)
        income_quarterly_dict: dict[str, FinancialPeriodDTO] = {}
        bs_quarterly_dict: dict[str, FinancialPeriodDTO] = {}
        cf_quarterly_dict: dict[str, FinancialPeriodDTO] = {}
        raw_ytd_cf: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}

        for filing in authoritative_10q:
            end_date = str(getattr(filing, "period_of_report", ""))[:10]
            if not end_date:
                continue
            try:
                f_year = int(end_date[:4])
                month = int(end_date[5:7])
                q_num = (month - 1) // 3 + 1
            except Exception:
                f_year = 2024
                q_num = 1
            if q_num == 4:
                continue

            p_key = f"{f_year}-Q{q_num}"
            f_meta = filing_meta_map.get(end_date, {})
            filing_url = f_meta.get("filing_url")
            form_type = f_meta.get("form_type", "10-Q")

            p_inc = FinancialPeriodDTO(
                period_key=p_key,
                fiscal_year=f_year,
                fiscal_quarter=q_num,
                period_end_date=end_date,
                period_kind="duration",
                duration_days=91,
                form_type=form_type,
                filing_url=filing_url,
                is_derived=False,
            )
            p_bs = FinancialPeriodDTO(
                period_key=p_key,
                fiscal_year=f_year,
                fiscal_quarter=q_num,
                period_end_date=end_date,
                period_kind="instant",
                duration_days=None,
                form_type=form_type,
                filing_url=filing_url,
                is_derived=False,
            )
            p_cf = FinancialPeriodDTO(
                period_key=p_key,
                fiscal_year=f_year,
                fiscal_quarter=q_num,
                period_end_date=end_date,
                period_kind="duration",
                duration_days=91,
                form_type=form_type,
                filing_url=filing_url,
                is_derived=False,
            )

            try:
                xb = filing.xbrl()
                if xb and xb.statements:
                    inc_stmt = xb.statements.income_statement()
                    if inc_stmt:
                        df = inc_stmt.to_dataframe()
                        target_col, dur_days, col_lbl = select_xbrl_column(df.columns.tolist(), end_date, "income", is_quarterly=True, fiscal_quarter=q_num)
                        if target_col:
                            for meta in CANONICAL_LINE_ITEMS["income"]:
                                val, concept = extract_canonical_cell_from_df(df, meta["key"], target_col)
                                p_inc.items[meta["key"]] = FinancialCellDTO(
                                    value=val,
                                    source_type="reported" if val is not None else "unavailable",
                                    source_concept=concept,
                                    source_filing_url=filing_url,
                                    is_derived=False,
                                )
                            resolve_sub_line_items("income", p_inc.items, df=df, target_col=target_col, filing_url=filing_url)

                    bs_stmt = xb.statements.balance_sheet()
                    if bs_stmt:
                        df = bs_stmt.to_dataframe()
                        target_col, dur_days, col_lbl = select_xbrl_column(df.columns.tolist(), end_date, "balance_sheet", is_quarterly=True, fiscal_quarter=q_num)
                        if target_col:
                            for meta in CANONICAL_LINE_ITEMS["balance_sheet"]:
                                val, concept = extract_canonical_cell_from_df(df, meta["key"], target_col)
                                p_bs.items[meta["key"]] = FinancialCellDTO(
                                    value=val,
                                    source_type="reported" if val is not None else "unavailable",
                                    source_concept=concept,
                                    source_filing_url=filing_url,
                                    is_derived=False,
                                )
                            resolve_sub_line_items("balance_sheet", p_bs.items, df=df, target_col=target_col, filing_url=filing_url)

                    cf_stmt = xb.statements.cash_flow_statement()
                    if cf_stmt:
                        df = cf_stmt.to_dataframe()
                        target_col, dur_days, col_lbl = select_xbrl_column(df.columns.tolist(), end_date, "cash_flow", is_quarterly=True, fiscal_quarter=q_num)
                        if target_col:
                            ytd_vals: dict[str, dict[str, Any]] = {}
                            for meta in CANONICAL_LINE_ITEMS["cash_flow"]:
                                val, concept = extract_canonical_cell_from_df(df, meta["key"], target_col)
                                ytd_vals[meta["key"]] = {"value": val, "concept": concept}
                            raw_ytd_cf[(f_year, q_num)] = ytd_vals
            except Exception as e:
                log.warning("Error extracting 10-Q %s: %s", end_date, e)

            income_quarterly_dict[p_key] = p_inc
            bs_quarterly_dict[p_key] = p_bs
            cf_quarterly_dict[p_key] = p_cf

        # Standalone 3M Cash Flow from YTD Deltas
        for (f_year, q_num), ytd_vals in raw_ytd_cf.items():
            p_key = f"{f_year}-Q{q_num}"
            p_cf = cf_quarterly_dict.get(p_key)
            if not p_cf:
                continue

            prev_q_num = q_num - 1
            prev_ytd_vals = raw_ytd_cf.get((f_year, prev_q_num))

            for meta in CANONICAL_LINE_ITEMS["cash_flow"]:
                k = meta["key"]
                if k in ["free_cash_flow", "calculated_free_cash_flow", "reported_free_cash_flow", "free_cash_flow_adjustments"]:
                    continue

                curr_cell = ytd_vals.get(k, {})
                curr_val = curr_cell.get("value")
                curr_concept = curr_cell.get("concept")

                if q_num == 1:
                    p_cf.items[k] = FinancialCellDTO(
                        value=curr_val,
                        source_type="reported" if curr_val is not None else "unavailable",
                        source_concept=curr_concept,
                        source_filing_url=p_cf.filing_url,
                        is_derived=False,
                    )
                else:
                    prev_val = prev_ytd_vals.get(k, {}).get("value") if prev_ytd_vals else None
                    if curr_val is not None and prev_val is not None:
                        delta_val = round(curr_val - prev_val, 2)
                        p_cf.items[k] = FinancialCellDTO(
                            value=delta_val,
                            source_type="derived",
                            source_concept=curr_concept,
                            source_filing_url=p_cf.filing_url,
                            derivation=f"YTD Q{q_num} minus YTD Q{prev_q_num}",
                            formula=f"YTD_Q{q_num} - YTD_Q{prev_q_num}",
                            is_derived=True,
                        )
                    else:
                        p_cf.items[k] = FinancialCellDTO(
                            value=None,
                            source_type="unavailable",
                            source_concept=None,
                            source_filing_url=p_cf.filing_url,
                            is_derived=False,
                        )

            # CapEx & Calculated FCF Calculation
            ocf = p_cf.items.get("operating_cash_flow", FinancialCellDTO()).value
            capex = p_cf.items.get("capital_expenditure", FinancialCellDTO()).value
            calc_fcf = None
            if ocf is not None and capex is not None:
                calc_fcf = round(ocf - abs(capex), 2)
                p_cf.items["calculated_free_cash_flow"] = FinancialCellDTO(
                    value=calc_fcf,
                    source_type="derived",
                    source_concept=None,
                    source_filing_url=p_cf.filing_url,
                    derivation="Operating Cash Flow - |CapEx|",
                    formula="OCF - |CapEx|",
                    input_items=["operating_cash_flow", "capital_expenditure"],
                    is_derived=True,
                )
                p_cf.items["free_cash_flow"] = FinancialCellDTO(
                    value=calc_fcf,
                    source_type="derived",
                    source_concept=None,
                    source_filing_url=p_cf.filing_url,
                    derivation="Operating Cash Flow - |CapEx|",
                    is_derived=True,
                )

            # 8-K Exhibit 99.1 Non-GAAP FCF for Q1-Q3
            _, adj_val, rep_fcf, rep_url, adj_expl, unavail_reason = self._sec_8k_client.extract_fcf_reconciliation(
                all_8k, f_year, p_cf.period_end_date, is_quarterly=True, fiscal_quarter=q_num
            )
            if rep_fcf is not None:
                p_cf.items["reported_free_cash_flow"] = FinancialCellDTO(
                    value=rep_fcf,
                    source_type="reported",
                    source_concept="8-K Exhibit 99.1 Press Release Non-GAAP Reconciliation",
                    source_filing_url=rep_url or p_cf.filing_url,
                    is_derived=False,
                )
                p_cf.items["free_cash_flow_adjustments"] = FinancialCellDTO(
                    value=adj_val if adj_val is not None else 0.0,
                    source_type="reported",
                    source_concept=adj_expl or "8-K Disclosed Non-GAAP adjustments",
                    source_filing_url=rep_url or p_cf.filing_url,
                    is_derived=False,
                )
            else:
                if calc_fcf is not None:
                    p_cf.items["reported_free_cash_flow"] = FinancialCellDTO(
                        value=calc_fcf,
                        source_type="derived",
                        source_concept=None,
                        source_filing_url=p_cf.filing_url,
                        derivation="Equal to Calculated FCF (No Non-GAAP adjustments disclosed)",
                        is_derived=True,
                    )
                    p_cf.items["free_cash_flow_adjustments"] = FinancialCellDTO(
                        value=0.0,
                        source_type="not_applicable",
                        source_concept=None,
                        source_filing_url=p_cf.filing_url,
                        derivation="No Non-GAAP adjustments disclosed",
                        is_derived=False,
                    )
                else:
                    p_cf.items["reported_free_cash_flow"] = FinancialCellDTO(
                        value=None,
                        source_type="unavailable",
                        unavailable_reason=unavail_reason or "FCF_RECONCILIATION_NOT_FOUND",
                        source_filing_url=p_cf.filing_url,
                        is_derived=False,
                    )
                    p_cf.items["free_cash_flow_adjustments"] = FinancialCellDTO(
                        value=None,
                        source_type="unavailable",
                        unavailable_reason=unavail_reason or "FCF_RECONCILIATION_NOT_FOUND",
                        source_filing_url=p_cf.filing_url,
                        is_derived=False,
                    )

        # 3. Derive Q4 Statements (FY minus Q1+Q2+Q3)
        for p_key_fy, p_inc_fy in list(income_annual_dict.items()):
            f_year = p_inc_fy.fiscal_year
            q1_key = f"{f_year}-Q1"
            q2_key = f"{f_year}-Q2"
            q3_key = f"{f_year}-Q3"
            q4_key = f"{f_year}-Q4"

            q1_inc = income_quarterly_dict.get(q1_key)
            q2_inc = income_quarterly_dict.get(q2_key)
            q3_inc = income_quarterly_dict.get(q3_key)

            if q1_inc and q2_inc and q3_inc:
                # Q4 Income Statement Derivation
                p_q4_inc = FinancialPeriodDTO(
                    period_key=q4_key,
                    fiscal_year=f_year,
                    fiscal_quarter=4,
                    period_end_date=p_inc_fy.period_end_date,
                    period_kind="duration",
                    duration_days=91,
                    form_type=p_inc_fy.form_type,
                    filing_url=p_inc_fy.filing_url,
                    is_derived=True,
                )
                for meta in CANONICAL_LINE_ITEMS["income"]:
                    k = meta["key"]
                    if k in ["eps_basic", "eps_diluted"]:
                        continue
                    fy_val = p_inc_fy.items.get(k, FinancialCellDTO()).value
                    q1_val = q1_inc.items.get(k, FinancialCellDTO()).value
                    q2_val = q2_inc.items.get(k, FinancialCellDTO()).value
                    q3_val = q3_inc.items.get(k, FinancialCellDTO()).value

                    if fy_val is not None and q1_val is not None and q2_val is not None and q3_val is not None:
                        q4_val = round(fy_val - (q1_val + q2_val + q3_val), 2)
                        p_q4_inc.items[k] = FinancialCellDTO(
                            value=q4_val,
                            source_type="derived",
                            source_concept=None,
                            source_filing_url=p_inc_fy.filing_url,
                            derivation="FY minus (Q1 + Q2 + Q3)",
                            formula="FY - (Q1 + Q2 + Q3)",
                            input_items=[k],
                            is_derived=True,
                        )
                    else:
                        p_q4_inc.items[k] = FinancialCellDTO(
                            value=None,
                            source_type="unavailable",
                            source_concept=None,
                            source_filing_url=p_inc_fy.filing_url,
                            is_derived=False,
                        )

                # Q4 EPS resolution
                basic_eps, diluted_eps, eps_url = self._sec_8k_client.extract_q4_eps(all_8k, f_year, p_inc_fy.period_end_date)
                if diluted_eps is not None:
                    p_q4_inc.items["eps_diluted"] = FinancialCellDTO(
                        value=diluted_eps,
                        source_type="reported",
                        source_concept="8-K Exhibit 99.1 Press Release",
                        source_filing_url=eps_url or p_inc_fy.filing_url,
                        is_derived=False,
                    )
                else:
                    p_q4_inc.items["eps_diluted"] = FinancialCellDTO(
                        value=None,
                        source_type="unavailable",
                        source_concept=None,
                        source_filing_url=p_inc_fy.filing_url,
                        is_derived=False,
                    )

                if basic_eps is not None:
                    p_q4_inc.items["eps_basic"] = FinancialCellDTO(
                        value=basic_eps,
                        source_type="reported",
                        source_concept="8-K Exhibit 99.1 Press Release",
                        source_filing_url=eps_url or p_inc_fy.filing_url,
                        is_derived=False,
                    )
                else:
                    p_q4_inc.items["eps_basic"] = FinancialCellDTO(
                        value=None,
                        source_type="unavailable",
                        source_concept=None,
                        source_filing_url=p_inc_fy.filing_url,
                        is_derived=False,
                    )

                income_quarterly_dict[q4_key] = p_q4_inc

                # Q4 Balance Sheet (Snapshot from 10-K)
                p_bs_fy = bs_annual_dict.get(p_key_fy)
                if p_bs_fy:
                    p_q4_bs = FinancialPeriodDTO(
                        period_key=q4_key,
                        fiscal_year=f_year,
                        fiscal_quarter=4,
                        period_end_date=p_bs_fy.period_end_date,
                        period_kind="instant",
                        duration_days=None,
                        form_type=p_bs_fy.form_type,
                        filing_url=p_bs_fy.filing_url,
                        is_derived=False,
                        items=dict(p_bs_fy.items),
                    )
                    bs_quarterly_dict[q4_key] = p_q4_bs

                # Q4 Cash Flow Derivation
                p_cf_fy = cf_annual_dict.get(p_key_fy)
                q1_cf = cf_quarterly_dict.get(q1_key)
                q2_cf = cf_quarterly_dict.get(q2_key)
                q3_cf = cf_quarterly_dict.get(q3_key)

                if p_cf_fy:
                    p_q4_cf = FinancialPeriodDTO(
                        period_key=q4_key,
                        fiscal_year=f_year,
                        fiscal_quarter=4,
                        period_end_date=p_cf_fy.period_end_date,
                        period_kind="duration",
                        duration_days=91,
                        form_type=p_cf_fy.form_type,
                        filing_url=p_cf_fy.filing_url,
                        is_derived=True,
                    )

                    for meta in CANONICAL_LINE_ITEMS["cash_flow"]:
                        k = meta["key"]
                        if k in ["free_cash_flow", "calculated_free_cash_flow", "reported_free_cash_flow", "free_cash_flow_adjustments"]:
                            continue
                        fy_val = p_cf_fy.items.get(k, FinancialCellDTO()).value
                        q1_val = q1_cf.items.get(k, FinancialCellDTO()).value if q1_cf else None
                        q2_val = q2_cf.items.get(k, FinancialCellDTO()).value if q2_cf else None
                        q3_val = q3_cf.items.get(k, FinancialCellDTO()).value if q3_cf else None

                        if fy_val is not None and q1_val is not None and q2_val is not None and q3_val is not None:
                            q4_val = round(fy_val - (q1_val + q2_val + q3_val), 2)
                            p_q4_cf.items[k] = FinancialCellDTO(
                                value=q4_val,
                                source_type="derived",
                                source_concept=None,
                                source_filing_url=p_cf_fy.filing_url,
                                derivation="FY minus (Q1 + Q2 + Q3)",
                                formula="FY - (Q1 + Q2 + Q3)",
                                is_derived=True,
                            )
                        else:
                            p_q4_cf.items[k] = FinancialCellDTO(
                                value=None,
                                source_type="unavailable",
                                source_concept=None,
                                source_filing_url=p_cf_fy.filing_url,
                                is_derived=False,
                            )

                    ocf = p_q4_cf.items.get("operating_cash_flow", FinancialCellDTO()).value
                    capex = p_q4_cf.items.get("capital_expenditure", FinancialCellDTO()).value
                    calc_fcf = None
                    if ocf is not None and capex is not None:
                        calc_fcf = round(ocf - abs(capex), 2)
                        p_q4_cf.items["calculated_free_cash_flow"] = FinancialCellDTO(
                            value=calc_fcf,
                            source_type="derived",
                            source_concept=None,
                            source_filing_url=p_cf_fy.filing_url,
                            derivation="Operating Cash Flow - |CapEx|",
                            formula="OCF - |CapEx|",
                            input_items=["operating_cash_flow", "capital_expenditure"],
                            is_derived=True,
                        )
                        p_q4_cf.items["free_cash_flow"] = FinancialCellDTO(
                            value=calc_fcf,
                            source_type="derived",
                            source_concept=None,
                            source_filing_url=p_cf_fy.filing_url,
                            derivation="Operating Cash Flow - |CapEx|",
                            is_derived=True,
                        )

                    # 8-K Exhibit 99.1 Non-GAAP FCF for Q4
                    _, adj_val, rep_fcf, rep_url, adj_expl, unavail_reason = self._sec_8k_client.extract_fcf_reconciliation(
                        all_8k, f_year, p_cf_fy.period_end_date, is_quarterly=True, fiscal_quarter=4
                    )
                    if rep_fcf is not None:
                        p_q4_cf.items["reported_free_cash_flow"] = FinancialCellDTO(
                            value=rep_fcf,
                            source_type="reported",
                            source_concept="8-K Exhibit 99.1 Press Release Non-GAAP Reconciliation",
                            source_filing_url=rep_url or p_cf_fy.filing_url,
                            is_derived=False,
                        )
                        p_q4_cf.items["free_cash_flow_adjustments"] = FinancialCellDTO(
                            value=adj_val if adj_val is not None else 0.0,
                            source_type="reported",
                            source_concept=adj_expl or "8-K Disclosed Non-GAAP adjustments",
                            source_filing_url=rep_url or p_cf_fy.filing_url,
                            is_derived=False,
                        )
                    else:
                        if calc_fcf is not None:
                            p_q4_cf.items["reported_free_cash_flow"] = FinancialCellDTO(
                                value=calc_fcf,
                                source_type="derived",
                                source_concept=None,
                                source_filing_url=p_cf_fy.filing_url,
                                derivation="Equal to Calculated FCF (No Non-GAAP adjustments disclosed)",
                                is_derived=True,
                            )
                            p_q4_cf.items["free_cash_flow_adjustments"] = FinancialCellDTO(
                                value=0.0,
                                source_type="not_applicable",
                                source_concept=None,
                                source_filing_url=p_cf_fy.filing_url,
                                derivation="No Non-GAAP adjustments disclosed",
                                is_derived=False,
                            )
                        else:
                            p_q4_cf.items["reported_free_cash_flow"] = FinancialCellDTO(
                                value=None,
                                source_type="unavailable",
                                unavailable_reason=unavail_reason or "FCF_RECONCILIATION_NOT_FOUND",
                                source_filing_url=p_cf_fy.filing_url,
                                is_derived=False,
                            )
                            p_q4_cf.items["free_cash_flow_adjustments"] = FinancialCellDTO(
                                value=None,
                                source_type="unavailable",
                                unavailable_reason=unavail_reason or "FCF_RECONCILIATION_NOT_FOUND",
                                source_filing_url=p_cf_fy.filing_url,
                                is_derived=False,
                            )

                    cf_quarterly_dict[q4_key] = p_q4_cf

        # Sort and limit periods
        sorted_annual_keys = sorted(income_annual_dict.keys(), reverse=True)[:5]
        sorted_quarterly_keys = sorted(income_quarterly_dict.keys(), reverse=True)[:12]

        annual_inc = [income_annual_dict[k] for k in sorted_annual_keys]
        annual_bs = [bs_annual_dict[k] for k in sorted_annual_keys if k in bs_annual_dict]
        annual_cf = [cf_annual_dict[k] for k in sorted_annual_keys if k in cf_annual_dict]

        quarterly_inc = [income_quarterly_dict[k] for k in sorted_quarterly_keys]
        quarterly_bs = [bs_quarterly_dict[k] for k in sorted_quarterly_keys if k in bs_quarterly_dict]
        quarterly_cf = [cf_quarterly_dict[k] for k in sorted_quarterly_keys if k in cf_quarterly_dict]

        # YoY Growth calculations
        self._calculate_yoy_growth_for_categories(annual_inc, annual_bs, annual_cf, is_quarter=False)
        self._calculate_yoy_growth_for_categories(quarterly_inc, quarterly_bs, quarterly_cf, is_quarter=True)

        # Multi-Statement Accounting & FCF Fail-Closed Validation
        val_ann, core_w_ann, exp_w_ann = validate_periods(annual_inc, annual_bs, annual_cf, is_quarter=False)
        val_q, core_w_q, exp_w_q = validate_periods(quarterly_inc, quarterly_bs, quarterly_cf, is_quarter=True)

        validation_warnings = list(set(core_w_ann + core_w_q))
        expanded_validation_warnings = list(set(exp_w_ann + exp_w_q))

        # Build Categories
        ann_cat_inc = FinancialStatementCategoryDTO(statement_type="income", period_kind="duration", periods=annual_inc, line_items=[LineItemMetaDTO(canonical_key=m["key"], display_label=m["label"], unit_type=m["unit_type"], is_primary_highlight=m["is_primary"]) for m in CANONICAL_LINE_ITEMS["income"]])
        ann_cat_bs = FinancialStatementCategoryDTO(statement_type="balance_sheet", period_kind="instant", periods=annual_bs, line_items=[LineItemMetaDTO(canonical_key=m["key"], display_label=m["label"], unit_type=m["unit_type"], is_primary_highlight=m["is_primary"]) for m in CANONICAL_LINE_ITEMS["balance_sheet"]])
        ann_cat_cf = FinancialStatementCategoryDTO(statement_type="cash_flow", period_kind="duration", periods=annual_cf, line_items=[LineItemMetaDTO(canonical_key=m["key"], display_label=m["label"], unit_type=m["unit_type"], is_primary_highlight=m["is_primary"]) for m in CANONICAL_LINE_ITEMS["cash_flow"]])

        q_cat_inc = FinancialStatementCategoryDTO(statement_type="income", period_kind="duration", periods=quarterly_inc, line_items=[LineItemMetaDTO(canonical_key=m["key"], display_label=m["label"], unit_type=m["unit_type"], is_primary_highlight=m["is_primary"]) for m in CANONICAL_LINE_ITEMS["income"]])
        q_cat_bs = FinancialStatementCategoryDTO(statement_type="balance_sheet", period_kind="instant", periods=quarterly_bs, line_items=[LineItemMetaDTO(canonical_key=m["key"], display_label=m["label"], unit_type=m["unit_type"], is_primary_highlight=m["is_primary"]) for m in CANONICAL_LINE_ITEMS["balance_sheet"]])
        q_cat_cf = FinancialStatementCategoryDTO(statement_type="cash_flow", period_kind="duration", periods=quarterly_cf, line_items=[LineItemMetaDTO(canonical_key=m["key"], display_label=m["label"], unit_type=m["unit_type"], is_primary_highlight=m["is_primary"]) for m in CANONICAL_LINE_ITEMS["cash_flow"]])

        annual_categories = [ann_cat_inc, ann_cat_bs, ann_cat_cf]
        quarterly_categories = [q_cat_inc, q_cat_bs, q_cat_cf]

        # Coverage Metrics
        core_pct, exp_pct, missing_req, missing_exp = compute_coverage_metrics(annual_categories + quarterly_categories)

        core_status = "complete" if core_pct >= 90.0 and not missing_req else "partial"
        exp_status = "complete" if exp_pct >= 85.0 and not missing_exp else "partial"
        data_status = "ok" if (core_status == "complete" and not validation_warnings) else "partial"

        # Summary Chart & Ratios
        chart_ann, ratios_ann = compute_ratios_and_charts(annual_inc, annual_bs, annual_cf)
        chart_q, ratios_q = compute_ratios_and_charts(quarterly_inc, quarterly_bs, quarterly_cf)

        return FinancialStatementsDTO(
            schema_version=6,
            ticker=ticker.upper(),
            market="US",
            currency="USD",
            provider="edgartools",
            provider_symbol=provider_symbol.upper(),
            data_status=data_status,
            coverage_status=core_status,
            core_coverage_status=core_status,
            expanded_coverage_status=exp_status,
            expanded_data_status=exp_status,
            core_coverage_pct=core_pct,
            expanded_coverage_pct=exp_pct,
            missing_required_items=missing_req,
            missing_expanded_items=missing_exp,
            validation_warnings=validation_warnings,
            expanded_validation_warnings=expanded_validation_warnings,
            expanded_error_count=len(expanded_validation_warnings),
            warnings=warnings,
            annual=annual_categories,
            quarterly=quarterly_categories,
            summary_chart_annual=chart_ann,
            summary_chart_quarterly=chart_q,
            ratios_annual=ratios_ann,
            ratios_quarterly=ratios_q,
            synced_at=datetime.now(timezone.utc).isoformat(),
        )

    def _calculate_yoy_growth_for_categories(
        self,
        inc_periods: list[FinancialPeriodDTO],
        bs_periods: list[FinancialPeriodDTO],
        cf_periods: list[FinancialPeriodDTO],
        is_quarter: bool = True,
    ) -> None:
        """คำนวณ YoY Growth % เปรียบเทียบกับปีก่อนหน้า (Q vs Q-4 หรือ FY vs FY-1)"""
        for periods, cat_type in [(inc_periods, "income"), (bs_periods, "balance_sheet"), (cf_periods, "cash_flow")]:
            period_map = {p.period_key: p for p in periods}
            for p in periods:
                f_yr = p.fiscal_year
                q_num = p.fiscal_quarter
                base_key = f"{f_yr - 1}-Q{q_num}" if (is_quarter and q_num is not None) else f"{f_yr - 1}-FY"
                base_p = period_map.get(base_key)

                for meta in CANONICAL_LINE_ITEMS[cat_type]:
                    k = meta["key"]
                    cell = p.items.get(k)
                    if not cell or cell.value is None:
                        continue
                    base_cell = base_p.items.get(k) if base_p else None
                    base_val = base_cell.value if (base_cell and base_cell.value is not None) else None
                    cell.yoy_growth_pct = calc_yoy_growth(cell.value, base_val)
