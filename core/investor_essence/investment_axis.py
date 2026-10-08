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

    # 2. Risk Limits
    mdd = draft.risk_limits.get("mdd_max_annual")
    if mdd is None or mdd.value is None or not mdd.is_confirmed:
        errors.append("Section 2 (ระดับความเสี่ยง): ต้องระบุและยืนยันค่า MDD สูงสุดต่อปี")

    trade_limit = draft.risk_limits.get("max_loss_per_trade")
    if trade_limit is None or trade_limit.value is None or not trade_limit.is_confirmed:
        errors.append("Section 2 (ระดับความเสี่ยง): ต้องระบุและยืนยันขีดจำกัดผลขาดทุนต่อ 1 การซื้อขาย")

    # 3. Targets (invest / exclude)
    if not draft.invest_targets:
        errors.append("Section 3 (เป้าหมาย): ต้องระบุรายการสินทรัพย์หรือหมวดที่ลงทุน")
    if not draft.exclude_targets:
        errors.append("Section 3 (เป้าหมาย): ต้องระบุรายการสินทรัพย์หรือหมวดที่ไม่ลงทุน")

    # 4. Methods (primary / secondary)
    if not draft.primary_methods:
        errors.append("Section 4 (วิธีการ): ต้องระบุวิธีการลงทุนหลัก")
    if not draft.secondary_methods:
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
        total_percent = sum((r.target_percent for r in draft.allocation_rows), Decimal("0.00"))
        variance = abs(total_percent - HUNDRED_PERCENT)
        if variance > PERCENT_TOLERANCE:
            errors.append(
                f"Section 6 (การจัดสรรสินทรัพย์): สัดส่วนรวมต้องเท่ากับ 100% (ปัจจุบันรวมได้ {total_percent}%)"
            )

    # 7. Role Models
    if not draft.role_models:
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

    serialized = json.dumps(complete_sections, sort_keys=True, ensure_ascii=False)
    content_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

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
