"""Pure domain rules for Investment Axis validation and confirmation.

Rules:
- Validates the 8 mandatory sections from prompt command 2:
  1. Basic Policy (นโยบายพื้นฐาน)
  2. Risk Limits (ระดับความเสี่ยง: MDD สูงสุดต่อปี, ขีดจำกัดผลขาดทุนต่อ 1 การซื้อขาย)
  3. Targets (เป้าหมาย: ลงทุน / ไม่ลงทุน)
  4. Methods (วิธีการ: หลัก / เสริม)
  5. Horizon (กรอบเวลาในการลงทุน)
  6. Asset Allocation (การจัดสรรสินทรัพย์: สัดส่วนรวม 100% +- 0.01, ความถี่ Rebalance)
  7. Role Models (นักลงทุนต้นแบบ หรือระบุว่าไม่ยึดบุคคลใด)
  8. Non-actions (สิ่งที่จะไม่ทำ: อย่างน้อย 3 ข้อที่ไม่ซ้ำกัน)
- Ensures all numeric policy fields have confirmed values and units
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any, Dict, List

from core.investor_essence.models import (
    HUNDRED_PERCENT,
    PERCENT_TOLERANCE,
    AllocationBasis,
    AllocationPlanRow,
    ArtifactRef,
    ConfirmedInvestmentAxis,
    InvestmentAxisDraft,
)


def validate_axis_completeness(draft: InvestmentAxisDraft) -> List[str]:
    """Audits draft for all 8 mandatory sections and numerical constraints."""
    errors: List[str] = []

    # 1. Basic Policy
    if not draft.basic_policy.strip():
        errors.append("Section 1 (นโยบายพื้นฐาน): ต้องระบุข้อความนโยบายพื้นฐาน")

    # 2. Risk Limits (support canonical and alias keys)
    mdd = (
        draft.risk_limits.get("mdd_max_annual")
        or draft.risk_limits.get("mdd_percent")
        or draft.risk_limits.get("max_drawdown_percent")
    )
    if mdd is None or mdd.value is None or not mdd.is_confirmed:
        errors.append("Section 2 (ระดับความเสี่ยง): ต้องระบุและยืนยันค่า MDD สูงสุดต่อปี")
    elif mdd.unit == "%":
        try:
            mdd_dec = Decimal(str(mdd.value))
            if not (Decimal("0") <= mdd_dec <= Decimal("100")):
                errors.append(f"Section 2 (ระดับความเสี่ยง): ค่า MDD สูงสุดต่อปีต้องอยู่ระหว่าง 0% ถึง 100% (ปัจจุบัน {mdd.value}%)")
        except Exception:
            errors.append(f"Section 2 (ระดับความเสี่ยง): ค่า MDD ต้องเป็นตัวเลขที่ถูกต้อง (ปัจจุบัน '{mdd.value}')")

    trade_limit = (
        draft.risk_limits.get("max_loss_per_trade")
        or draft.risk_limits.get("max_loss_per_trade_percent")
        or draft.risk_limits.get("max_single_loss_percent")
    )
    if trade_limit is None:
        errors.append("Section 2 (ระดับความเสี่ยง): ต้องระบุและยืนยันขีดจำกัดผลขาดทุนต่อ 1 การซื้อขาย")
    else:
        is_na = (
            trade_limit.unit == "not_applicable"
            or trade_limit.calculation_basis == "not_applicable"
            or "not_applicable" in (trade_limit.assumptions or "").lower()
        )
        if not is_na:
            if trade_limit.value is None or not trade_limit.is_confirmed:
                errors.append("Section 2 (ระดับความเสี่ยง): ต้องระบุและยืนยันขีดจำกัดผลขาดทุนต่อ 1 การซื้อขาย")
            elif trade_limit.unit == "%":
                try:
                    tl_dec = Decimal(str(trade_limit.value))
                    if not (Decimal("0") <= tl_dec <= Decimal("100")):
                        errors.append(f"Section 2 (ระดับความเสี่ยง): ขีดจำกัดขาดทุนต่อการซื้อขายต้องอยู่ระหว่าง 0% ถึง 100% (ปัจจุบัน {trade_limit.value}%)")
                except Exception:
                    errors.append(f"Section 2 (ระดับความเสี่ยง): ค่าขีดจำกัดขาดทุนต่อการซื้อขายต้องเป็นตัวเลขที่ถูกต้อง (ปัจจุบัน '{trade_limit.value}')")

    # 3. Targets (invest / exclude)
    invest_t = [t.strip() for t in draft.invest_targets if t.strip()]
    if not invest_t:
        errors.append("Section 3 (เป้าหมาย): ต้องระบุรายการสินทรัพย์หรือหมวดที่ลงทุน")
    exclude_t = [t.strip() for t in draft.exclude_targets if t.strip()]
    if not exclude_t:
        errors.append("Section 3 (เป้าหมาย): ต้องระบุรายการสินทรัพย์หรือหมวดที่ไม่ลงทุน")

    # 4. Methods (primary / secondary)
    prim_m = [m.strip() for m in draft.primary_methods if m.strip()]
    if not prim_m:
        errors.append("Section 4 (วิธีการ): ต้องระบุวิธีการลงทุนหลัก")
    sec_m = [m.strip() for m in draft.secondary_methods if m.strip()]
    if not sec_m:
        errors.append("Section 4 (วิธีการ): ต้องระบุวิธีการลงทุนเสริม")

    # 5. Horizon
    if not draft.investment_horizon.strip():
        errors.append("Section 5 (กรอบเวลา): ต้องระบุกรอบเวลาในการลงทุน")

    # 6. Asset Allocation
    if not draft.rebalance_frequency.strip():
        errors.append("Section 6 (การจัดสรรสินทรัพย์): ต้องระบุความถี่ในการ Rebalance")
    if not draft.allocation_rows:
        errors.append("Section 6 (การจัดสรรสินทรัพย์): ต้องมีสัดส่วนการจัดสรรสินทรัพย์อย่างน้อย 1 หมวด")
    else:
        alloc_ids = set()
        for r in draft.allocation_rows:
            if r.allocation_id in alloc_ids:
                errors.append(f"Section 6 (การจัดสรรสินทรัพย์): allocation_id ซ้ำกัน ({r.allocation_id})")
            alloc_ids.add(r.allocation_id)
            if r.target_percent < Decimal("0.00"):
                errors.append(f"Section 6 (การจัดสรรสินทรัพย์): สัดส่วนหมวด '{r.category_name}' ต้องไม่ติดลบ ({r.target_percent}%)")

        total_percent = sum((r.target_percent for r in draft.allocation_rows), Decimal("0.00"))
        variance = abs(total_percent - HUNDRED_PERCENT)
        if variance > PERCENT_TOLERANCE:
            errors.append(
                f"Section 6 (การจัดสรรสินทรัพย์): สัดส่วนรวมต้องเท่ากับ 100% (ปัจจุบันรวมได้ {total_percent}%)"
            )

    # 7. Role Models
    rm = [r.strip() for r in draft.role_models if r.strip()]
    if not rm:
        errors.append("Section 7 (นักลงทุนต้นแบบ): ต้องระบุนักลงทุนต้นแบบหรือระบุว่าไม่ยึดบุคคลใดเป็นต้นแบบ")

    # 8. Non-actions (at least 3 unique)
    unique_non_actions = {na.strip() for na in draft.non_actions if na.strip()}
    if len(unique_non_actions) < 3:
        errors.append(
            f"Section 8 (สิ่งที่จะไม่ทำ): ต้องระบุสิ่งที่จะไม่ทำอย่างน้อย 3 ข้อที่ไม่ซ้ำกัน (ปัจจุบันมี {len(unique_non_actions)} ข้อ)"
        )

    return errors


def build_confirmed_axis(
    confirmation_id: str,
    draft: InvestmentAxisDraft,
    confirmed_at_iso: str,
) -> ConfirmedInvestmentAxis:
    """Compiles a fully-validated draft into an immutable ConfirmedInvestmentAxis."""
    errors = validate_axis_completeness(draft)
    if errors:
        raise ValueError("Cannot confirm investment axis with incomplete sections:\n" + "\n".join(errors))

    complete_sections: Dict[str, Any] = {
        "basic_policy": draft.basic_policy,
        "risk_limits": {
            k: {
                "field_id": v.field_id,
                "value": str(v.value) if v.value is not None else None,
                "unit": v.unit,
                "calculation_basis": v.calculation_basis,
                "origin": v.origin.value,
                "is_confirmed": v.is_confirmed,
            }
            for k, v in draft.risk_limits.items()
        },
        "invest_targets": draft.invest_targets,
        "exclude_targets": draft.exclude_targets,
        "primary_methods": draft.primary_methods,
        "secondary_methods": draft.secondary_methods,
        "investment_horizon": draft.investment_horizon,
        "rebalance_frequency": draft.rebalance_frequency,
        "role_models": draft.role_models,
        "non_actions": draft.non_actions,
    }

    from core.investor_essence.models import compute_semantic_hash

    semantic_payload: Dict[str, Any] = {
        "sections": complete_sections,
        "allocation_basis": draft.allocation_basis.value,
        "allocation_rows": [
            {
                "allocation_id": r.allocation_id,
                "category_name": r.category_name,
                "target_percent": str(r.target_percent),
                "role_description": r.role_description,
            }
            for r in draft.allocation_rows
        ],
        "essence_ref": draft.essence_ref.content_hash,
        "context_ref": draft.context_ref,
    }
    content_hash = compute_semantic_hash(semantic_payload)

    return ConfirmedInvestmentAxis(
        confirmation_id=confirmation_id,
        portfolio_id=draft.portfolio_id,
        accepted_essence_ref=draft.essence_ref,
        context_ref=draft.context_ref,
        complete_sections=complete_sections,
        allocation_basis=draft.allocation_basis,
        allocation_rows=list(draft.allocation_rows),
        non_actions=list(draft.non_actions),
        confirmed_at_iso=confirmed_at_iso,
        content_hash=content_hash,
    )
