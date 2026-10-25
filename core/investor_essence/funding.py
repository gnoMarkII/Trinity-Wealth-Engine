"""Deterministic funding plan and cash flow calculation engine (WP09).

Calculates bucket gap, funding progress, inflow cash routing (fixed vs priority fill),
emergency reserve requirements (accounting for outside reserves), and baseline contributions.
"""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from core.investor_essence.bucket_policy import BucketPolicy, InflowRoutingMode
from core.investor_essence.models import (
    GoalAdjustmentLevers,
    GoalPriorityTier,
    GoalTensionLevel,
    GoalTensionReport,
)

ZERO_MONEY = Decimal("0.00")
HUNDRED = Decimal("100.00")



def to_money(val: Any) -> Decimal:
    """Coerce value to Decimal quantized to 2 decimal places."""
    if val is None:
        return ZERO_MONEY
    if isinstance(val, Decimal):
        return val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    try:
        return Decimal(str(val).strip().replace(",", "")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    except Exception:
        return ZERO_MONEY


def calculate_bucket_gap(
    target_amount: Optional[Decimal],
    current_value: Decimal,
    minimum_amount: Optional[Decimal] = None,
) -> Decimal:
    """
    Calculates the unmet funding gap for a bucket.
    gap = max(0, effective_target - current_value).
    """
    effective_target: Optional[Decimal] = None
    if target_amount is not None and target_amount > ZERO_MONEY:
        effective_target = target_amount
    elif minimum_amount is not None and minimum_amount > ZERO_MONEY:
        effective_target = minimum_amount

    if effective_target is None:
        return ZERO_MONEY

    gap = effective_target - current_value
    return max(ZERO_MONEY, gap.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def calculate_funding_progress(
    current_value: Decimal,
    target_amount: Optional[Decimal],
) -> Optional[Decimal]:
    """Calculates funding progress percentage = current_value / target_amount * 100."""
    if target_amount is None or target_amount <= ZERO_MONEY:
        return None
    progress = (current_value / target_amount) * HUNDRED
    return progress.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def calculate_emergency_reserve_requirement(
    essential_monthly_expense: Decimal,
    emergency_months: Decimal,
    reserves_outside_portfolio: Decimal = Decimal("0.00"),
    funding_scope: str = "portfolio",
) -> Dict[str, Any]:
    """
    Calculates emergency reserve requirement considering reserves held outside the portfolio.
    Gross = expense * months.
    Net portfolio required = max(0, Gross - outside_reserves).
    """
    gross = (essential_monthly_expense * emergency_months).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    outside = to_money(reserves_outside_portfolio)
    net_required = max(ZERO_MONEY, gross - outside)

    return {
        "essential_monthly_expense": essential_monthly_expense,
        "emergency_months": emergency_months,
        "gross_required": gross,
        "reserves_outside_portfolio": outside,
        "net_portfolio_required": net_required,
        "funding_scope": funding_scope,
        "basis": f"{essential_monthly_expense} x {emergency_months} months",
        "is_fully_covered_outside": outside >= gross,
    }


def calculate_baseline_contribution(
    gap: Decimal,
    months_remaining: Optional[int],
) -> Dict[str, Any]:
    """
    Calculates baseline monthly contribution = gap / months_remaining without return assumptions.
    """
    gap_clean = to_money(gap)
    if gap_clean <= ZERO_MONEY:
        return {
            "gap": ZERO_MONEY,
            "months_remaining": months_remaining,
            "monthly_contribution": ZERO_MONEY,
            "status": "funded",
            "message": "เป้าหมายนี้ได้รับเงินทุนครบถ้วนแล้ว",
        }

    if months_remaining is None:
        return {
            "gap": gap_clean,
            "months_remaining": None,
            "monthly_contribution": None,
            "status": "insufficient_data",
            "message": "ไม่ได้ระบุกำหนดเวลา จึงไม่สามารถคำนวณเงินสะสมต่อเดือนได้",
        }

    if months_remaining <= 0:
        return {
            "gap": gap_clean,
            "months_remaining": months_remaining,
            "monthly_contribution": None,
            "status": "overdue",
            "message": "กำหนดเวลาผ่านพ้นไปแล้วหรือไม่เพียงพอ",
        }

    monthly = (gap_clean / Decimal(str(months_remaining))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return {
        "gap": gap_clean,
        "months_remaining": months_remaining,
        "monthly_contribution": monthly,
        "status": "on_track",
        "message": f"ต้องการสะสมเดือนละ {monthly:,.2f} บาท เป็นเวลา {months_remaining} เดือน",
    }


def calculate_inflow_routing(
    monthly_budget: Decimal,
    policies: List[BucketPolicy],
    current_values: Optional[Dict[str, Decimal]] = None,
) -> Dict[str, Any]:
    """
    Allocates incoming monthly investment cash according to bucket policies.
    Supports FIXED_RATIO and PRIORITY_FILL modes.
    """
    budget = to_money(monthly_budget)
    curr_map = {k: to_money(v) for k, v in (current_values or {}).items()}
    routing_amounts: Dict[str, Decimal] = {p.bucket_id: ZERO_MONEY for p in policies}
    issues: List[str] = []

    if budget <= ZERO_MONEY:
        return {
            "monthly_budget": budget,
            "routing_amounts": routing_amounts,
            "total_allocated": ZERO_MONEY,
            "remaining_budget": ZERO_MONEY,
            "is_valid": True,
            "issues": [],
        }

    # Separate policies by routing mode
    fixed_policies = [p for p in policies if p.inflow_route_mode == InflowRoutingMode.FIXED_RATIO]
    priority_policies = [p for p in policies if p.inflow_route_mode == InflowRoutingMode.PRIORITY_FILL]

    allocated_sum = ZERO_MONEY

    # 1. Fixed Ratio Mode
    total_fixed_pct = sum((p.inflow_route_percent for p in fixed_policies), ZERO_MONEY)
    if total_fixed_pct > HUNDRED:
        issues.append(
            f"สัดส่วน Fixed Inflow Routing ({total_fixed_pct}%) เกิน 100% ของงบรายเดือน"
        )

    for p in fixed_policies:
        amt = (budget * (p.inflow_route_percent / HUNDRED)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        routing_amounts[p.bucket_id] = amt
        allocated_sum += amt

    # 2. Priority Fill Mode
    remaining_budget = max(ZERO_MONEY, budget - allocated_sum)
    if priority_policies and remaining_budget > ZERO_MONEY:
        sorted_priority = sorted(priority_policies, key=lambda p: p.inflow_priority)
        for p in sorted_priority:
            if remaining_budget <= ZERO_MONEY:
                break

            curr_val = curr_map.get(p.bucket_id, ZERO_MONEY)
            gap = calculate_bucket_gap(p.target_amount, curr_val, p.minimum_amount)

            if gap > ZERO_MONEY:
                fill_amt = min(remaining_budget, gap)
                routing_amounts[p.bucket_id] += fill_amt
                remaining_budget -= fill_amt
                allocated_sum += fill_amt
            else:
                # Target full -> check redirect
                if p.when_full_redirect_bucket_id and p.when_full_redirect_bucket_id in routing_amounts:
                    redirect_target = p.when_full_redirect_bucket_id
                    routing_amounts[redirect_target] += remaining_budget
                    allocated_sum += remaining_budget
                    remaining_budget = ZERO_MONEY

    final_remaining = max(ZERO_MONEY, budget - allocated_sum)
    is_valid = len(issues) == 0 and allocated_sum <= budget

    return {
        "monthly_budget": budget,
        "routing_amounts": routing_amounts,
        "total_allocated": allocated_sum,
        "remaining_budget": final_remaining,
        "is_valid": is_valid,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Goal-Based Wealth Engine: Three Faces of Risk, R_req, Levers & Tension
# ---------------------------------------------------------------------------


def calculate_required_return(
    pv: Any,
    fv: Any,
    pmt: Any = Decimal("0.00"),
    months: Optional[int] = None,
) -> Optional[Decimal]:
    """Calculates annualized Required Rate of Return (CAGR %) using Bisection numerical solver.

    Formula:
        FV = PV * (1 + r_m)^N + PMT * [((1 + r_m)^N - 1) / r_m]
    Returns None if insufficient data (months <= 0 or target <= 0 or (pv <= 0 and pmt <= 0)).
    Returns 0.00% if already funded or achievable purely through cash savings.
    """
    if months is None or months <= 0:
        return None

    pv_dec = to_money(pv)
    fv_dec = to_money(fv)
    pmt_dec = to_money(pmt)

    if fv_dec <= ZERO_MONEY:
        return ZERO_MONEY
    if pv_dec >= fv_dec:
        return ZERO_MONEY

    # If savings alone are enough
    total_savings_capacity = pv_dec + (pmt_dec * Decimal(months))
    if total_savings_capacity >= fv_dec:
        return ZERO_MONEY

    if pv_dec <= ZERO_MONEY and pmt_dec <= ZERO_MONEY:
        return None

    # Case A: Lump-sum only (pmt == 0)
    if pmt_dec <= ZERO_MONEY:
        if pv_dec <= ZERO_MONEY:
            return None
        ratio = float(fv_dec / pv_dec)
        cagr_float = (ratio ** (12.0 / months) - 1.0) * 100.0
        return Decimal(str(round(cagr_float, 2))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # Case B: Future value of annuity with initial lump sum
    pv_f = float(pv_dec)
    fv_f = float(fv_dec)
    pmt_f = float(pmt_dec)

    def calc_fv(r_m: float) -> float:
        if abs(r_m) < 1e-12:
            return pv_f + pmt_f * months
        try:
            if r_m > 0 and months * math.log1p(r_m) > 700:
                return float("inf")
            comp = (1.0 + r_m) ** months
            return pv_f * comp + pmt_f * ((comp - 1.0) / r_m)
        except OverflowError:
            return float("inf")

    low = 0.0
    high = 0.5
    while calc_fv(high) < fv_f and high < 5.0:
        high *= 2.0

    for _ in range(36):
        mid = (low + high) / 2.0
        if calc_fv(mid) < fv_f:
            low = mid
        else:
            high = mid

    r_monthly = (low + high) / 2.0
    try:
        annual_cagr = ((1.0 + r_monthly) ** 12.0 - 1.0) * 100.0
    except OverflowError:
        annual_cagr = 9999.0
    return Decimal(str(round(annual_cagr, 2))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def calculate_goal_adjustment_levers(
    pv: Any,
    fv: Any,
    pmt: Any = Decimal("0.00"),
    months: Optional[int] = None,
    expected_annual_return: Any = Decimal("7.00"),
    required_cagr: Optional[Decimal] = None,
) -> GoalAdjustmentLevers:
    """Computes the 4 actionable adjustment levers to resolve funding tension."""
    pv_dec = to_money(pv)
    fv_dec = to_money(fv)
    pmt_dec = to_money(pmt)
    exp_ret_dec = to_money(expected_annual_return)

    if months is None or months <= 0 or fv_dec <= ZERO_MONEY:
        return GoalAdjustmentLevers(
            save_more_monthly_thb=ZERO_MONEY,
            total_monthly_savings_needed_thb=pmt_dec,
            extend_timeline_months=0,
            total_months_needed=months or 0,
            downsized_target_thb=pv_dec,
            required_risk_cagr_pct=required_cagr,
        )

    r_annual_float = float(exp_ret_dec) / 100.0
    if r_annual_float <= 0.0:
        r_m = 0.0
    else:
        r_m = (1.0 + r_annual_float) ** (1.0 / 12.0) - 1.0

    pv_f = float(pv_dec)
    fv_f = float(fv_dec)
    pmt_f = float(pmt_dec)

    # 1. Lever 1: Save More Monthly (PMT needed)
    if r_m > 1e-9:
        comp_n = (1.0 + r_m) ** months
        denom = (comp_n - 1.0) / r_m
        if denom > 0:
            pmt_needed_f = (fv_f - pv_f * comp_n) / denom
        else:
            pmt_needed_f = (fv_f - pv_f) / months
    else:
        pmt_needed_f = (fv_f - pv_f) / months

    pmt_needed = max(ZERO_MONEY, to_money(pmt_needed_f))
    save_more = max(ZERO_MONEY, pmt_needed - pmt_dec)

    # 2. Lever 2: Extend Timeline (Months needed at current PMT and expected return)
    if r_m > 1e-9 and (pv_f * r_m + pmt_f) > 0 and (fv_f * r_m + pmt_f) > 0:
        ratio = (fv_f * r_m + pmt_f) / (pv_f * r_m + pmt_f)
        if ratio > 1.0:
            months_needed_f = math.log(ratio) / math.log(1.0 + r_m)
            total_months_needed = int(math.ceil(months_needed_f))
        else:
            total_months_needed = months
    else:
        if pmt_f > 0:
            gap_f = max(0.0, fv_f - pv_f)
            total_months_needed = int(math.ceil(gap_f / pmt_f))
        else:
            total_months_needed = months

    extend_months = max(0, total_months_needed - months)

    # 3. Lever 3: Downsized Target (FV achievable at current PMT and expected return)
    if r_m > 1e-9:
        comp_n = (1.0 + r_m) ** months
        achievable_f = pv_f * comp_n + pmt_f * ((comp_n - 1.0) / r_m)
    else:
        achievable_f = pv_f + pmt_f * months
    achievable_fv = max(pv_dec, to_money(achievable_f))

    return GoalAdjustmentLevers(
        save_more_monthly_thb=save_more,
        total_monthly_savings_needed_thb=pmt_needed,
        extend_timeline_months=extend_months,
        total_months_needed=total_months_needed,
        downsized_target_thb=achievable_fv,
        required_risk_cagr_pct=required_cagr,
    )


def evaluate_goal_tension(
    current_amount: Any,
    target_amount: Any,
    monthly_savings: Any = Decimal("0.00"),
    months_remaining: Optional[int] = None,
    expected_annual_return: Any = Decimal("7.00"),
    priority_tier: str = "wants",
) -> GoalTensionReport:
    """Evaluates mathematical tension between goal requirements and portfolio return baseline."""
    pv = to_money(current_amount)
    fv = to_money(target_amount)
    pmt = to_money(monthly_savings)
    exp_return = to_money(expected_annual_return)

    if months_remaining is None or months_remaining <= 0:
        levers = calculate_goal_adjustment_levers(pv, fv, pmt, None, exp_return, None)
        return GoalTensionReport(
            required_return_cagr_pct=None,
            expected_return_cagr_pct=exp_return,
            tension_level=GoalTensionLevel.INSUFFICIENT_DATA,
            headline_message="ยังไม่ได้ระบุกำหนดเวลาเป้าหมาย",
            detailed_message="กรุณาระบุกำหนดเวลา (Deadline) หรือจำนวนปี เพื่อให้ระบบคำนวณอัตราผลตอบแทนและคานงัดการปรับแผน",
            levers=levers,
        )

    r_req = calculate_required_return(pv, fv, pmt, months_remaining)
    levers = calculate_goal_adjustment_levers(pv, fv, pmt, months_remaining, exp_return, r_req)

    norm_tier = str(priority_tier).lower()
    if norm_tier in ("needs", "essential"):
        threshold_crit = Decimal("8.00")
        threshold_mod = Decimal("5.00")
    elif norm_tier in ("wishes", "aspirational"):
        threshold_crit = Decimal("15.00")
        threshold_mod = Decimal("10.00")
    else:  # wants / important
        threshold_crit = Decimal("12.00")
        threshold_mod = Decimal("8.00")

    if r_req is None:
        return GoalTensionReport(
            required_return_cagr_pct=None,
            expected_return_cagr_pct=exp_return,
            tension_level=GoalTensionLevel.INSUFFICIENT_DATA,
            headline_message="ข้อมูลเป้าหมายไม่เพียงพอ",
            detailed_message="ไม่สามารถคำนวณอัตราผลตอบแทนที่ต้องการได้เนื่องจากไม่มีเงินต้นและไม่มีเงินออมสะสม",
            levers=levers,
        )

    if r_req <= ZERO_MONEY:
        return GoalTensionReport(
            required_return_cagr_pct=ZERO_MONEY,
            expected_return_cagr_pct=exp_return,
            tension_level=GoalTensionLevel.BALANCED,
            headline_message="เป้าหมายอยู่ในเกณฑ์ที่เป็นไปได้สูง",
            detailed_message="เงินออมสะสมและมูลค่าปัจจุบันเพียงพอต่อการบรรลุเป้าหมายโดยไม่ต้องพึ่งพาความเสี่ยงจากผลตอบแทนสูง",
            levers=levers,
        )

    if r_req > threshold_crit:
        return GoalTensionReport(
            required_return_cagr_pct=r_req,
            expected_return_cagr_pct=exp_return,
            tension_level=GoalTensionLevel.CRITICAL_TENSION,
            headline_message=f"เป้าหมายมีความตึงเครียดสูง (ต้องการ {r_req}% ต่อปี)",
            detailed_message=(
                f"อัตราผลตอบแทนที่เป้าหมายเรียกร้อง ({r_req}% ต่อปี) สูงเกินกว่าระดับความเสี่ยงที่เหมาะสมของเป้าหมายระดับ {norm_tier} "
                f"ควรพิจารณาใช้คานงัด: ออมเพิ่มเดือนละ ฿{levers.save_more_monthly_thb:,.2f} หรือขยายเวลาอีก {levers.extend_timeline_months} เดือน"
            ),
            levers=levers,
        )

    if r_req > threshold_mod:
        if exp_return is not None and r_req > exp_return:
            detailed_msg = (
                f"เป้าหมายต้องการผลตอบแทน {r_req}% ต่อปี ซึ่งสูงกว่าผลตอบแทนคาดหวังพื้นฐาน ({exp_return}%) เล็กน้อย "
                f"สามารถปรับพอร์ตให้เน้นการเติบโตขึ้น หรือทยอยออมเพิ่มเดือนละ ฿{levers.save_more_monthly_thb:,.2f}"
            )
        else:
            detailed_msg = (
                f"เป้าหมายต้องการผลตอบแทน {r_req}% ต่อปี ซึ่งต้องติดตามความเสี่ยงสำหรับเป้าหมายระดับ {norm_tier} "
                f"สามารถปรับพอร์ตให้เน้นการเติบโตขึ้น หรือทยอยออมเพิ่มเดือนละ ฿{levers.save_more_monthly_thb:,.2f}"
            )
        return GoalTensionReport(
            required_return_cagr_pct=r_req,
            expected_return_cagr_pct=exp_return,
            tension_level=GoalTensionLevel.MODERATE_TENSION,
            headline_message=f"เป้าหมายต้องพึ่งพาผลตอบแทนเชิงรุก ({r_req}% ต่อปี)",
            detailed_message=detailed_msg,
            levers=levers,
        )

    return GoalTensionReport(
        required_return_cagr_pct=r_req,
        expected_return_cagr_pct=exp_return,
        tension_level=GoalTensionLevel.BALANCED,
        headline_message=f"เป้าหมายสมดุลดี (ต้องการ {r_req}% ต่อปี)",
        detailed_message=f"แผนการลงทุนอยู่ในระดับที่สอดคล้องกับพอร์ตการลงทุนที่คาดหวัง ({exp_return}% ต่อปี) และบรรลุได้จริง",
        levers=levers,
    )

