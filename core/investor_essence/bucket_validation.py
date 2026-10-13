"""Domain validator for Bucket Plans, Matrix weights, Remappings, and Inflow Routings (WP10).

Ensures:
1. Unique bucket IDs, non-empty names/roles.
2. Target percentages sum to 100%.
3. Inflow routing percentages (DCA) do not exceed 100%.
4. Remapping destinations point to existing bucket IDs or None (unassigned).
5. N:N Matrix weights match both Axis row targets and Bucket column targets (AC12).
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from core.investor_essence.models import (
    AllocationMappingCell,
    AllocationPlanRow,
    BucketPlanDraft,
    BucketRemappingItem,
    PurposeBucketDraft,
    HUNDRED_PERCENT,
    PERCENT_TOLERANCE,
)
from core.investor_essence.validation import validate_allocation_matrix, validate_percent_sum


def validate_bucket_plan_completeness(
    draft: BucketPlanDraft,
    axis_allocation_targets: Optional[Dict[str, Decimal]] = None,
) -> List[str]:
    """
    Validates all invariants of a BucketPlanDraft.
    Returns a list of human-readable blocking validation errors (empty if valid).
    """
    errors: List[str] = []

    if not draft.purpose_buckets:
        return ["ต้องมี Purpose Buckets อย่างน้อย 1 ใบ"]

    # 1. Unique bucket IDs & basic fields
    seen_ids: Set[str] = set()
    bucket_targets: Dict[str, Decimal] = {}
    fixed_inflow_pct_sum = Decimal("0.00")

    for idx, b in enumerate(draft.purpose_buckets, 1):
        if not b.bucket_id or not b.bucket_id.strip():
            errors.append(f"Bucket #{idx}: ต้องมี bucket_id ที่ไม่ว่างเปล่า")
        elif b.bucket_id in seen_ids:
            errors.append(f"Bucket #{idx}: รหัส '{b.bucket_id}' ซ้ำกับ Bucket อื่น")
        else:
            seen_ids.add(b.bucket_id)

        if not b.name or not b.name.strip():
            errors.append(f"Bucket #{idx}: ต้องระบุชื่อ (name)")
        if not b.role or not b.role.strip():
            errors.append(f"Bucket '{b.name or idx}': ต้องระบุบทบาทและเหตุผล (role)")

        # Target percent bounds
        if b.target_percent < Decimal("0.00") or b.target_percent > HUNDRED_PERCENT:
            errors.append(
                f"Bucket '{b.name}': สัดส่วนเป้าหมาย ({b.target_percent}%) ต้องอยู่ระหว่าง 0% ถึง 100%"
            )
        bucket_targets[b.bucket_id] = b.target_percent

        # Cash Inflow routing bounds (DCA)
        if b.cash_inflow_routing_pct is not None:
            try:
                r_pct = Decimal(str(b.cash_inflow_routing_pct).strip())
                if r_pct < Decimal("0.00"):
                    errors.append(f"Bucket '{b.name}': สัดส่วนเงินเข้าใหม่ (DCA) ต้องไม่ติดลบ")
                else:
                    fixed_inflow_pct_sum += r_pct
            except Exception:
                errors.append(f"Bucket '{b.name}': สัดส่วนเงินเข้าใหม่ '{b.cash_inflow_routing_pct}' ไม่ใช่ตัวเลขที่ถูกต้อง")

    # 2. Total target percent sum == 100%
    is_valid_sum, actual_sum = validate_percent_sum(
        list(bucket_targets.values()), HUNDRED_PERCENT, PERCENT_TOLERANCE
    )
    if not is_valid_sum:
        errors.append(f"ผลรวมสัดส่วน Buckets ({actual_sum}%) ต้องเท่ากับ 100%")

    # 3. Cash Inflow routing sum <= 100% (AC13: e.g. DCA 180% rejected)
    if fixed_inflow_pct_sum > (HUNDRED_PERCENT + PERCENT_TOLERANCE):
        errors.append(
            f"ผลรวมสัดส่วนเงินเข้าใหม่ (DCA) ({fixed_inflow_pct_sum}%) เกิน 100% ของงบลงทุน"
        )

    # 4. Remapping destinations must exist in target buckets or be None (Unassigned)
    for r in draft.remapping:
        if r.target_bucket_id is not None and r.target_bucket_id not in seen_ids:
            errors.append(
                f"การย้ายหมวดของ '{r.old_bucket_id}': ปลายทาง '{r.target_bucket_id}' ไม่มีอยู่ในเป้าหมาย Buckets"
            )

    # 5. Matrix weights validation (AC12)
    if draft.mapping_weights:
        # Check matrix total sum
        matrix_cells_sum = sum(
            (c.portfolio_weight_percent for c in draft.mapping_weights), Decimal("0.00")
        )
        if abs(matrix_cells_sum - HUNDRED_PERCENT) > PERCENT_TOLERANCE:
            errors.append(
                f"ผลรวมน้ำหนักในเมทริกซ์การจัดสรร ({matrix_cells_sum}%) ต้องเท่ากับ 100%"
            )

        # Check column sums match purpose bucket targets
        col_sums: Dict[str, Decimal] = {b_id: Decimal("0.00") for b_id in seen_ids}
        for c in draft.mapping_weights:
            if c.bucket_id not in col_sums:
                errors.append(f"เมทริกซ์อ้างถึง Bucket ID '{c.bucket_id}' ที่ไม่มีอยู่จริง")
            else:
                col_sums[c.bucket_id] += c.portfolio_weight_percent

        for b_id, expected_pct in bucket_targets.items():
            actual_col_pct = col_sums.get(b_id, Decimal("0.00"))
            if abs(actual_col_pct - expected_pct) > PERCENT_TOLERANCE:
                errors.append(
                    f"เมทริกซ์คอลัมน์ Bucket '{b_id}': ผลรวมน้ำหนัก ({actual_col_pct}%) ไม่ตรงกับเป้าหมาย Bucket ({expected_pct}%)"
                )

        # Check row sums match axis targets if provided
        if axis_allocation_targets:
            row_sums: Dict[str, Decimal] = {cat: Decimal("0.00") for cat in axis_allocation_targets}
            for c in draft.mapping_weights:
                if c.axis_allocation_id in row_sums:
                    row_sums[c.axis_allocation_id] += c.portfolio_weight_percent

            for cat, expected_row_pct in axis_allocation_targets.items():
                actual_row_pct = row_sums.get(cat, Decimal("0.00"))
                if abs(actual_row_pct - expected_row_pct) > PERCENT_TOLERANCE:
                    errors.append(
                        f"เมทริกซ์แถวหมวด '{cat}': ผลรวมน้ำหนัก ({actual_row_pct}%) ไม่ตรงกับเป้าหมายแกนหลัก ({expected_row_pct}%)"
                    )

    return errors


def score_affinity(cat_text: str, bucket_text: str) -> int:
    cat_lower = cat_text.lower()
    b_lower = bucket_text.lower()
    score = 0

    # Cash / Emergency
    cash_kws = ["เงินสด", "ตลาดเงิน", "cash", "mmf", "money"]
    emg_kws = ["สำรอง", "ฉุกเฉิน", "สภาพคล่อง", "emergency", "reserve", "liquidity"]
    if any(k in cat_lower for k in cash_kws) and any(k in b_lower for k in emg_kws):
        score += 100

    # Equity / Growth
    eq_kws = ["หุ้นโลก", "หุ้นเติบโต", "ดัชนี", "etf", "equity", "stock", "global", "growth", "s&p"]
    growth_kws = ["กองหน้า", "เติบโต", "10 ล้าน", "growth", "wealth", "ขับเคลื่อน"]
    if any(k in cat_lower for k in eq_kws) and any(k in b_lower for k in growth_kws):
        score += 80

    # Fixed income / Income / Stability
    bond_kws = ["ตราสารหนี้", "พันธบัตร", "หุ้นกู้", "reit", "อสังหา", "bond", "fixed", "income"]
    inc_kws = ["ฐานราก", "กระแสเงินสด", "กันชน", "ปันผล", "income", "stability", "buffer"]
    if any(k in cat_lower for k in bond_kws) and any(k in b_lower for k in inc_kws):
        score += 80

    # Tactical / Opportunity / Single stock
    tact_kws = ["รายตัว", "เทคโนโลยี", "tech", "alpha", "tactical", "theme"]
    opp_kws = ["สำรวจ", "โอกาส", "เทรนด์", "alpha", "tactical", "satellite"]
    if any(k in cat_lower for k in tact_kws) and any(k in b_lower for k in opp_kws):
        score += 90

    return score


def reconcile_allocation_matrix(
    axis_rows: List[Dict[str, Any] | AllocationPlanRow],
    purpose_buckets: List[Dict[str, Any] | PurposeBucketDraft],
    existing_mapping: Optional[List[AllocationMappingCell]] = None,
) -> List[AllocationMappingCell]:
    """Reconciles or solves an N:N allocation matrix matching axis row targets and bucket column targets."""
    axis_targets: Dict[str, Decimal] = {}
    axis_info: Dict[str, str] = {}
    for r in axis_rows:
        if isinstance(r, dict):
            alloc_id = r.get("allocation_id") or r.get("id", "")
            pct = Decimal(str(r.get("target_percent", 0)))
            text = f"{r.get('category', '')} {r.get('category_name', '')} {r.get('name', '')} {r.get('role_description', '')}"
        else:
            alloc_id = r.allocation_id
            pct = r.target_percent
            text = f"{r.category} {r.category_name} {r.role_description}"
        if alloc_id:
            axis_targets[alloc_id] = pct
            axis_info[alloc_id] = text

    bucket_targets: Dict[str, Decimal] = {}
    bucket_info: Dict[str, str] = {}
    for b in purpose_buckets:
        if isinstance(b, dict):
            b_id = b.get("bucket_id") or b.get("id", "")
            pct = Decimal(str(b.get("target_percent", 0)))
            text = f"{b.get('bucket_id', '')} {b.get('name', '')} {b.get('role', '')} {b.get('linked_goal_name', '')}"
        else:
            b_id = b.bucket_id
            pct = b.target_percent
            text = f"{b.bucket_id} {b.name} {b.role} {b.linked_goal_name or ''}"
        if b_id:
            bucket_targets[b_id] = pct
            bucket_info[b_id] = text

    # Check if existing mapping is already 100% valid
    if existing_mapping:
        all_cells_valid = True
        for c in existing_mapping:
            if c.bucket_id not in bucket_targets or c.axis_allocation_id not in axis_targets:
                all_cells_valid = False
                break
        if all_cells_valid:
            errs = validate_allocation_matrix(axis_targets, bucket_targets, existing_mapping)
            if not errs:
                return list(existing_mapping)

    if not axis_targets or not bucket_targets:
        return []

    # Affinity pairs
    pairs = []
    for cat_id, cat_pct in axis_targets.items():
        for b_id, b_pct in bucket_targets.items():
            sc = score_affinity(axis_info.get(cat_id, ""), bucket_info.get(b_id, ""))
            if existing_mapping:
                for em in existing_mapping:
                    if em.axis_allocation_id == cat_id and (em.bucket_id in b_id or b_id in em.bucket_id):
                        sc += 50
            pairs.append((sc, cat_id, b_id))

    pairs.sort(key=lambda x: x[0], reverse=True)

    remaining_axis = dict(axis_targets)
    remaining_bucket = dict(bucket_targets)
    cells_dict: Dict[tuple, Decimal] = {}

    for sc, cat_id, b_id in pairs:
        if remaining_axis[cat_id] > Decimal("0.00") and remaining_bucket[b_id] > Decimal("0.00"):
            alloc = min(remaining_axis[cat_id], remaining_bucket[b_id])
            if alloc > Decimal("0.00"):
                key = (cat_id, b_id)
                cells_dict[key] = cells_dict.get(key, Decimal("0.00")) + alloc
                remaining_axis[cat_id] -= alloc
                remaining_bucket[b_id] -= alloc

    for cat_id in list(remaining_axis.keys()):
        while remaining_axis[cat_id] > Decimal("0.00"):
            target_b = next((bid for bid, rem in remaining_bucket.items() if rem > Decimal("0.00")), None)
            if not target_b:
                break
            alloc = min(remaining_axis[cat_id], remaining_bucket[target_b])
            if alloc > Decimal("0.00"):
                key = (cat_id, target_b)
                cells_dict[key] = cells_dict.get(key, Decimal("0.00")) + alloc
                remaining_axis[cat_id] -= alloc
                remaining_bucket[target_b] -= alloc
            else:
                break

    return [
        AllocationMappingCell(
            axis_allocation_id=cat_id,
            bucket_id=b_id,
            portfolio_weight_percent=amt,
        )
        for (cat_id, b_id), amt in cells_dict.items()
        if amt > Decimal("0.00")
    ]


def reconcile_bucket_remapping(
    remapping: List[BucketRemappingItem],
    purpose_buckets: List[Dict[str, Any] | PurposeBucketDraft],
) -> List[BucketRemappingItem]:
    """Reconciles remapping targets so they consistently point to active purpose bucket IDs."""
    bucket_map: Dict[str, str] = {}
    for b in purpose_buckets:
        if isinstance(b, dict):
            b_id = b.get("bucket_id") or ""
            text = f"{b_id} {b.get('name', '')} {b.get('role', '')}".lower()
        else:
            b_id = b.bucket_id
            text = f"{b_id} {b.name} {b.role}".lower()
        if b_id:
            bucket_map[b_id] = text

    valid_bids = set(bucket_map.keys())
    result = []
    for r in remapping:
        target = r.target_bucket_id
        if target and target in valid_bids:
            result.append(r)
            continue

        # Try to match old_bucket_id with active purpose buckets
        old_lower = r.old_bucket_id.lower()
        matched_id = None
        for b_id, b_text in bucket_map.items():
            if old_lower in b_id.lower() or b_id.lower() in old_lower:
                matched_id = b_id
                break
            if any(k in old_lower for k in ["emergency", "สำรอง", "ฉุกเฉิน"]) and any(k in b_text for k in ["emergency", "สำรอง", "ฉุกเฉิน"]):
                matched_id = b_id
                break
            if any(k in old_lower for k in ["growth", "เติบโต"]) and any(k in b_text for k in ["growth", "เติบโต"]):
                matched_id = b_id
                break
            if any(k in old_lower for k in ["income", "stability", "กันชน", "กระแสเงินสด"]) and any(k in b_text for k in ["income", "stability", "กันชน", "กระแสเงินสด"]):
                matched_id = b_id
                break

        result.append(
            BucketRemappingItem(
                old_bucket_id=r.old_bucket_id,
                target_bucket_id=matched_id or (next(iter(valid_bids)) if valid_bids else None),
                affected_holding_count=r.affected_holding_count,
            )
        )
    return result
