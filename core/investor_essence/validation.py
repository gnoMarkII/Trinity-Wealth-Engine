"""Pure validation helpers for Investor Essence and Allocation Plans."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple

from core.investor_essence.models import HUNDRED_PERCENT, PERCENT_TOLERANCE, AllocationMappingCell


def to_decimal_2dp(value: Any) -> Decimal:
    """Converts int, float, or str to Decimal quantized to 2 decimal places."""
    if value is None:
        raise ValueError("Cannot convert None to Decimal")
    try:
        d = Decimal(str(value))
        return d.quantize(Decimal("0.01"))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"Invalid decimal value: {value}") from exc


def validate_percent_sum(
    percentages: List[Decimal],
    target: Decimal = HUNDRED_PERCENT,
    tolerance: Decimal = PERCENT_TOLERANCE,
) -> Tuple[bool, Decimal]:
    """Validates that a list of percentages sums to target within tolerance."""
    total = sum(percentages, Decimal("0.00"))
    diff = abs(total - target)
    return (diff <= tolerance, total)


def validate_allocation_matrix(
    category_targets: Dict[str, Decimal],
    bucket_targets: Dict[str, Decimal],
    mapping_cells: List[AllocationMappingCell],
    tolerance: Decimal = PERCENT_TOLERANCE,
) -> List[str]:
    """Validates an N:N allocation matrix against category and bucket targets."""
    errors: List[str] = []

    # 1. Total cells sum
    all_weights = [c.portfolio_weight_percent for c in mapping_cells]
    total_valid, total_sum = validate_percent_sum(all_weights, HUNDRED_PERCENT, tolerance)
    if not total_valid:
        errors.append(f"ผลรวมน้ำหนักในเมทริกซ์ทั้งหมดต้องเท่ากับ 100% (ปัจจุบันได้ {total_sum}%)")

    # 2. Row sums (axis categories)
    row_sums: Dict[str, Decimal] = {cat_id: Decimal("0.00") for cat_id in category_targets}
    for cell in mapping_cells:
        if cell.axis_allocation_id not in row_sums:
            errors.append(f"Axis category ID {cell.axis_allocation_id} ในเมทริกซ์ไม่อยู่ในแผนจัดสรรหลัก")
            continue
        row_sums[cell.axis_allocation_id] += cell.portfolio_weight_percent

    for cat_id, expected_weight in category_targets.items():
        actual_weight = row_sums.get(cat_id, Decimal("0.00"))
        if abs(actual_weight - expected_weight) > tolerance:
            errors.append(
                f"หมวด {cat_id}: ผลรวมในเมทริกซ์ ({actual_weight}%) ไม่ตรงกับแผนแกนหลัก ({expected_weight}%)"
            )

    # 3. Column sums (purpose buckets)
    col_sums: Dict[str, Decimal] = {b_id: Decimal("0.00") for b_id in bucket_targets}
    for cell in mapping_cells:
        if cell.bucket_id not in col_sums:
            errors.append(f"Bucket ID {cell.bucket_id} ในเมทริกซ์ไม่อยู่ในรายชื่อ Target Buckets")
            continue
        col_sums[cell.bucket_id] += cell.portfolio_weight_percent

    for b_id, expected_target in bucket_targets.items():
        actual_target = col_sums.get(b_id, Decimal("0.00"))
        if abs(actual_target - expected_target) > tolerance:
            errors.append(
                f"Bucket {b_id}: ผลรวมที่ได้รับในเมทริกซ์ ({actual_target}%) ไม่ตรงกับเป้าหมาย Bucket ({expected_target}%)"
            )

    return errors
