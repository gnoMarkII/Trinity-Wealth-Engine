"""Financial Domain Calculations: Margins, Ratios, YoY Growth, and Summary Chart Points."""
import math
from typing import Any, Optional
from tools.market.financials.domain.models import (
    FinancialCellDTO,
    FinancialPeriodDTO,
    FinancialRatioPointDTO,
    FinancialSummaryChartPointDTO,
)


def finite_or_none(val: Any) -> Optional[float]:
    """แปลงค่าตัวเลขเป็น float หรือ None หากเป็น NaN / Inf"""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def calc_yoy_growth(current_val: Optional[float], base_val: Optional[float]) -> Optional[float]:
    """คำนวณ YoY Growth % (คืน None เมื่อ base <= 0 หรือค่าติดลบที่ทำให้สับสน)"""
    if current_val is None or base_val is None:
        return None
    if base_val <= 0 or current_val < 0:
        return None
    try:
        pct = ((current_val - base_val) / base_val) * 100.0
        if math.isnan(pct) or math.isinf(pct):
            return None
        return round(pct, 2)
    except Exception:
        return None


def compute_ratios_and_charts(
    inc_periods: list[FinancialPeriodDTO],
    bs_periods: list[FinancialPeriodDTO],
    cf_periods: list[FinancialPeriodDTO],
) -> tuple[list[FinancialSummaryChartPointDTO], list[FinancialRatioPointDTO]]:
    """คำนวณ Summary Chart Points และ Financial Ratios สำหรับงวดต่างๆ"""
    chart_points: list[FinancialSummaryChartPointDTO] = []
    ratios_list: list[FinancialRatioPointDTO] = []

    inc_map = {p.period_key: p for p in inc_periods}
    bs_map = {p.period_key: p for p in bs_periods}
    cf_map = {p.period_key: p for p in cf_periods}

    all_keys = list(set(list(inc_map.keys()) + list(bs_map.keys()) + list(cf_map.keys())))

    def _get_date(k: str) -> str:
        return (
            inc_map.get(k, FinancialPeriodDTO(period_key="", fiscal_year=0, period_end_date="", period_kind="duration", form_type="")).period_end_date
            or bs_map.get(k, FinancialPeriodDTO(period_key="", fiscal_year=0, period_end_date="", period_kind="instant", form_type="")).period_end_date
            or cf_map.get(k, FinancialPeriodDTO(period_key="", fiscal_year=0, period_end_date="", period_kind="duration", form_type="")).period_end_date
            or ""
        )

    all_keys.sort(key=_get_date, reverse=True)

    for k in all_keys:
        p_inc = inc_map.get(k, FinancialPeriodDTO(period_key=k, fiscal_year=0, period_end_date=_get_date(k), period_kind="duration", form_type=""))
        p_bs = bs_map.get(k, FinancialPeriodDTO(period_key=k, fiscal_year=0, period_end_date=_get_date(k), period_kind="instant", form_type=""))
        p_cf = cf_map.get(k, FinancialPeriodDTO(period_key=k, fiscal_year=0, period_end_date=_get_date(k), period_kind="duration", form_type=""))

        p_date = _get_date(k)
        rev = p_inc.items.get("revenue", FinancialCellDTO()).value
        gp = p_inc.items.get("gross_profit", FinancialCellDTO()).value
        op_inc = p_inc.items.get("operating_income", FinancialCellDTO()).value
        net_inc = p_inc.items.get("net_income", FinancialCellDTO()).value
        calc_fcf = p_cf.items.get("calculated_free_cash_flow", FinancialCellDTO()).value or p_cf.items.get("free_cash_flow", FinancialCellDTO()).value
        rep_cell = p_cf.items.get("reported_free_cash_flow", FinancialCellDTO())
        rep_fcf = rep_cell.value if (rep_cell.source_type == "reported" and rep_cell.value is not None and abs(rep_cell.value) >= 1_000_000) else None
        curr_a = p_bs.items.get("current_assets", FinancialCellDTO()).value
        curr_l = p_bs.items.get("current_liabilities", FinancialCellDTO()).value
        tot_debt = p_bs.items.get("total_debt", FinancialCellDTO()).value
        tot_eq = p_bs.items.get("total_equity", FinancialCellDTO()).value
        equity = tot_eq if (tot_eq is not None and tot_eq > 0) else p_bs.items.get("stockholders_equity", FinancialCellDTO()).value

        # Margins
        gm_pct = round((gp / rev) * 100.0, 2) if rev and gp is not None and rev > 0 else None
        om_pct = round((op_inc / rev) * 100.0, 2) if rev and op_inc is not None and rev > 0 else None
        nm_pct = round((net_inc / rev) * 100.0, 2) if rev and net_inc is not None and rev > 0 else None
        fcf_m_pct = round((calc_fcf / rev) * 100.0, 2) if rev and calc_fcf is not None and rev > 0 else None

        # Balance Sheet Ratios
        curr_ratio = round(curr_a / curr_l, 2) if curr_a is not None and curr_l and curr_l > 0 else None
        de_ratio = round(tot_debt / equity, 2) if tot_debt is not None and equity and equity > 0 else None

        ratios_list.append(
            FinancialRatioPointDTO(
                period_key=k,
                period_end_date=p_date,
                gross_margin_pct=gm_pct,
                operating_margin_pct=om_pct,
                net_margin_pct=nm_pct,
                fcf_margin_pct=fcf_m_pct,
                debt_to_equity=de_ratio,
                current_ratio=curr_ratio,
            )
        )

        chart_points.append(
            FinancialSummaryChartPointDTO(
                period_key=k,
                date=p_date,
                revenue=rev,
                gross_profit=gp,
                operating_income=op_inc,
                net_income=net_inc,
                free_cash_flow=calc_fcf,
                calculated_free_cash_flow=calc_fcf,
                reported_free_cash_flow=rep_fcf,
                operating_margin_pct=om_pct,
                net_margin_pct=nm_pct,
            )
        )

    return chart_points, ratios_list
