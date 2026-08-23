"""Thai & Fallback Market Financial Statements Provider using yfinance."""
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional

import pandas as pd
import yfinance as yf

from tools.market.financials.domain.calculations import (
    calc_yoy_growth,
    compute_ratios_and_charts,
    finite_or_none,
)
from tools.market.financials.domain.constants import (
    CANONICAL_LINE_ITEMS,
    YFINANCE_CONCEPT_MAP,
)
from tools.market.financials.domain.models import (
    FinancialCellDTO,
    FinancialPeriodDTO,
    FinancialStatementCategoryDTO,
    FinancialStatementsDTO,
    LineItemMetaDTO,
)
from tools.market.financials.domain.validator import compute_coverage_metrics
from tools.market.financials.ports.provider_port import FinancialStatementProviderPort

log = logging.getLogger(__name__)


def get_period_key_from_date(date_str: str, is_quarter: bool) -> tuple[str, int, Optional[int]]:
    """แปลงวันที่ YYYY-MM-DD เป็น Period Key, Year, Quarter"""
    try:
        dt = pd.to_datetime(date_str)
        yr = dt.year
        mo = dt.month
        if is_quarter:
            q = (mo - 1) // 3 + 1
            return f"{yr}-Q{q}", yr, q
        return f"{yr}-FY", yr, None
    except Exception:
        return "2024-FY", 2024, None


def extract_cell_from_yf_df(df: pd.DataFrame, canonical_key: str, col_name: Any) -> tuple[Optional[float], Optional[str]]:
    """ดึงค่าจาก DataFrame ของ yfinance โดยแมป Concept Name"""
    synonyms = YFINANCE_CONCEPT_MAP.get(canonical_key, [])
    for syn in synonyms:
        if syn in df.index:
            row = df.loc[syn]
            val = row[col_name] if col_name in df.columns else None
            f_val = finite_or_none(val)
            if f_val is not None:
                return f_val, syn
    return None, None


class ThaiSetFinancialProvider(FinancialStatementProviderPort):
    """Provider สำหรับดึงงบการเงินตลาดไทย (TH) หรือ Fallback ผ่าน yfinance"""

    def __init__(self, market: Literal["US", "TH"] = "TH", ticker_factory: Optional[Callable[[str], Any]] = None):
        self._market = market
        self._ticker_factory = ticker_factory or (lambda s: yf.Ticker(s))

    def fetch_statements(self, ticker: str, provider_symbol: str) -> Optional[FinancialStatementsDTO]:
        try:
            t = self._ticker_factory(provider_symbol)

            q_inc = getattr(t, "quarterly_income_stmt", None)
            q_bs = getattr(t, "quarterly_balance_sheet", None)
            q_cf = getattr(t, "quarterly_cashflow", None)

            a_inc = getattr(t, "income_stmt", None)
            a_bs = getattr(t, "balance_sheet", None)
            a_cf = getattr(t, "cashflow", None)

            if (q_inc is None or q_inc.empty) and (a_inc is None or a_inc.empty):
                log.warning("yfinance returned empty financials for %s", provider_symbol)
                return None

            currency = "THB" if self._market == "TH" else "USD"

            def _extract_yf_category_periods(
                inc_df: Optional[pd.DataFrame],
                bs_df: Optional[pd.DataFrame],
                cf_df: Optional[pd.DataFrame],
                is_quarter: bool = True,
            ) -> tuple[list[FinancialPeriodDTO], list[FinancialPeriodDTO], list[FinancialPeriodDTO]]:
                inc_periods: list[FinancialPeriodDTO] = []
                bs_periods: list[FinancialPeriodDTO] = []
                cf_periods: list[FinancialPeriodDTO] = []

                if inc_df is not None and not inc_df.empty:
                    for col in inc_df.columns:
                        col_str = str(col)[:10]
                        p_key, f_year, q_num = get_period_key_from_date(col_str, is_quarter)
                        p_dto = FinancialPeriodDTO(
                            period_key=p_key,
                            fiscal_year=f_year,
                            fiscal_quarter=q_num,
                            period_end_date=col_str,
                            period_kind="duration",
                            duration_days=91 if is_quarter else 365,
                            form_type="yfinance",
                            is_derived=False,
                        )
                        for meta in CANONICAL_LINE_ITEMS["income"]:
                            k = meta["key"]
                            val, raw_label = extract_cell_from_yf_df(inc_df, k, col)
                            p_dto.items[k] = FinancialCellDTO(
                                value=val,
                                source_type="reported" if val is not None else "unavailable",
                                source_concept=raw_label,
                                is_derived=False,
                            )
                        inc_periods.append(p_dto)

                if bs_df is not None and not bs_df.empty:
                    for col in bs_df.columns:
                        col_str = str(col)[:10]
                        p_key, f_year, q_num = get_period_key_from_date(col_str, is_quarter)
                        p_dto = FinancialPeriodDTO(
                            period_key=p_key,
                            fiscal_year=f_year,
                            fiscal_quarter=q_num,
                            period_end_date=col_str,
                            period_kind="instant",
                            duration_days=None,
                            form_type="yfinance",
                            is_derived=False,
                        )
                        for meta in CANONICAL_LINE_ITEMS["balance_sheet"]:
                            k = meta["key"]
                            val, raw_label = extract_cell_from_yf_df(bs_df, k, col)
                            p_dto.items[k] = FinancialCellDTO(
                                value=val,
                                source_type="reported" if val is not None else "unavailable",
                                source_concept=raw_label,
                                is_derived=False,
                            )
                        tot_eq = p_dto.items.get("total_equity", FinancialCellDTO()).value
                        stk_eq = p_dto.items.get("stockholders_equity", FinancialCellDTO()).value
                        nci_cell = p_dto.items.get("noncontrolling_interests", FinancialCellDTO())
                        if nci_cell.value is None:
                            p_dto.items["noncontrolling_interests"] = FinancialCellDTO(
                                value=0.0,
                                source_type="not_applicable",
                                derivation="No Non-controlling Interests disclosed",
                                is_derived=False,
                            )
                            if tot_eq is None and stk_eq is not None:
                                p_dto.items["total_equity"] = FinancialCellDTO(
                                    value=stk_eq,
                                    source_type="derived",
                                    derivation="Total Equity = Stockholders' Equity",
                                    is_derived=True,
                                )
                        bs_periods.append(p_dto)

                if cf_df is not None and not cf_df.empty:
                    for col in cf_df.columns:
                        col_str = str(col)[:10]
                        p_key, f_year, q_num = get_period_key_from_date(col_str, is_quarter)
                        p_dto = FinancialPeriodDTO(
                            period_key=p_key,
                            fiscal_year=f_year,
                            fiscal_quarter=q_num,
                            period_end_date=col_str,
                            period_kind="duration",
                            duration_days=91 if is_quarter else 365,
                            form_type="yfinance",
                            is_derived=False,
                        )
                        for meta in CANONICAL_LINE_ITEMS["cash_flow"]:
                            k = meta["key"]
                            if k in ["free_cash_flow", "calculated_free_cash_flow", "reported_free_cash_flow", "free_cash_flow_adjustments"]:
                                continue
                            val, raw_label = extract_cell_from_yf_df(cf_df, k, col)
                            p_dto.items[k] = FinancialCellDTO(
                                value=val,
                                source_type="reported" if val is not None else "unavailable",
                                source_concept=raw_label,
                                is_derived=False,
                            )
                        ocf = p_dto.items.get("operating_cash_flow", FinancialCellDTO()).value
                        capex = p_dto.items.get("capital_expenditure", FinancialCellDTO()).value
                        if ocf is not None and capex is not None:
                            calc_fcf = round(ocf - abs(capex), 2)
                            p_dto.items["calculated_free_cash_flow"] = FinancialCellDTO(
                                value=calc_fcf,
                                source_type="derived",
                                derivation="Operating Cash Flow - |CapEx|",
                                is_derived=True,
                            )
                            p_dto.items["free_cash_flow"] = FinancialCellDTO(
                                value=calc_fcf,
                                source_type="derived",
                                derivation="Operating Cash Flow - |CapEx|",
                                is_derived=True,
                            )
                        p_dto.items["reported_free_cash_flow"] = FinancialCellDTO(
                            value=None,
                            source_type="unavailable",
                            unavailable_reason="Not available via yfinance",
                            is_derived=False,
                        )
                        p_dto.items["free_cash_flow_adjustments"] = FinancialCellDTO(
                            value=None,
                            source_type="unavailable",
                            unavailable_reason="Not available via yfinance",
                            is_derived=False,
                        )
                        cf_periods.append(p_dto)

                inc_periods.sort(key=lambda p: p.period_end_date, reverse=True)
                bs_periods.sort(key=lambda p: p.period_end_date, reverse=True)
                cf_periods.sort(key=lambda p: p.period_end_date, reverse=True)
                return inc_periods, bs_periods, cf_periods

            q_inc_p, q_bs_p, q_cf_p = _extract_yf_category_periods(q_inc, q_bs, q_cf, is_quarter=True)
            a_inc_p, a_bs_p, a_cf_p = _extract_yf_category_periods(a_inc, a_bs, a_cf, is_quarter=False)

            def _calc_list_yoy(p_list: list[FinancialPeriodDTO], yoy_offset: int) -> None:
                for i, curr_p in enumerate(p_list):
                    if i + yoy_offset < len(p_list):
                        base_p = p_list[i + yoy_offset]
                        for k, cell in curr_p.items.items():
                            cell.yoy_growth_pct = calc_yoy_growth(cell.value, base_p.items.get(k, FinancialCellDTO()).value)

            _calc_list_yoy(q_inc_p, 4)
            _calc_list_yoy(q_bs_p, 4)
            _calc_list_yoy(q_cf_p, 4)

            _calc_list_yoy(a_inc_p, 1)
            _calc_list_yoy(a_bs_p, 1)
            _calc_list_yoy(a_cf_p, 1)

            q_categories = [
                FinancialStatementCategoryDTO(statement_type="income", period_kind="duration", periods=q_inc_p, line_items=[LineItemMetaDTO(canonical_key=m["key"], display_label=m["label"], unit_type=m["unit_type"], is_primary_highlight=m["is_primary"]) for m in CANONICAL_LINE_ITEMS["income"]]),
                FinancialStatementCategoryDTO(statement_type="balance_sheet", period_kind="instant", periods=q_bs_p, line_items=[LineItemMetaDTO(canonical_key=m["key"], display_label=m["label"], unit_type=m["unit_type"], is_primary_highlight=m["is_primary"]) for m in CANONICAL_LINE_ITEMS["balance_sheet"]]),
                FinancialStatementCategoryDTO(statement_type="cash_flow", period_kind="duration", periods=q_cf_p, line_items=[LineItemMetaDTO(canonical_key=m["key"], display_label=m["label"], unit_type=m["unit_type"], is_primary_highlight=m["is_primary"]) for m in CANONICAL_LINE_ITEMS["cash_flow"]]),
            ]
            a_categories = [
                FinancialStatementCategoryDTO(statement_type="income", period_kind="duration", periods=a_inc_p, line_items=[LineItemMetaDTO(canonical_key=m["key"], display_label=m["label"], unit_type=m["unit_type"], is_primary_highlight=m["is_primary"]) for m in CANONICAL_LINE_ITEMS["income"]]),
                FinancialStatementCategoryDTO(statement_type="balance_sheet", period_kind="instant", periods=a_bs_p, line_items=[LineItemMetaDTO(canonical_key=m["key"], display_label=m["label"], unit_type=m["unit_type"], is_primary_highlight=m["is_primary"]) for m in CANONICAL_LINE_ITEMS["balance_sheet"]]),
                FinancialStatementCategoryDTO(statement_type="cash_flow", period_kind="duration", periods=a_cf_p, line_items=[LineItemMetaDTO(canonical_key=m["key"], display_label=m["label"], unit_type=m["unit_type"], is_primary_highlight=m["is_primary"]) for m in CANONICAL_LINE_ITEMS["cash_flow"]]),
            ]

            core_pct, exp_pct, missing_req, missing_exp = compute_coverage_metrics(a_categories + q_categories)
            core_status = "complete" if core_pct >= 85.0 else "partial"
            exp_status = "complete" if exp_pct >= 70.0 else "partial"

            chart_ann, ratios_ann = compute_ratios_and_charts(a_inc_p, a_bs_p, a_cf_p)
            chart_q, ratios_q = compute_ratios_and_charts(q_inc_p, q_bs_p, q_cf_p)

            return FinancialStatementsDTO(
                schema_version=6,
                ticker=ticker.upper(),
                market=self._market,
                currency=currency,
                provider="yfinance",
                provider_symbol=provider_symbol.upper(),
                data_status="ok" if core_status == "complete" else "partial",
                coverage_status=core_status,
                core_coverage_status=core_status,
                expanded_coverage_status=exp_status,
                expanded_data_status=exp_status,
                core_coverage_pct=core_pct,
                expanded_coverage_pct=exp_pct,
                missing_required_items=missing_req,
                missing_expanded_items=missing_exp,
                validation_warnings=[],
                expanded_validation_warnings=[],
                expanded_error_count=0,
                warnings=[],
                annual=a_categories,
                quarterly=q_categories,
                summary_chart_annual=chart_ann,
                summary_chart_quarterly=chart_q,
                ratios_annual=ratios_ann,
                ratios_quarterly=ratios_q,
                synced_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            log.warning("yfinance fetch error for %s: %s", provider_symbol, e)
            return None
