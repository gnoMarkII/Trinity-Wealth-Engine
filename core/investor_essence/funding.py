"""Deterministic funding plan and cash flow calculation engine (WP09).

Calculates bucket gap, funding progress, inflow cash routing (fixed vs priority fill),
emergency reserve requirements (accounting for outside reserves), and baseline contributions.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from core.investor_essence.bucket_policy import BucketPolicy, InflowRoutingMode

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
