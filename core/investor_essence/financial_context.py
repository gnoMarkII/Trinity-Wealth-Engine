"""FinancialContext readiness and validation domain logic.

Rules:
- Separates known values, range estimates, explicit none/zeros, and unknown fields
- Audits blocking issues (MISSING_HORIZON, UNKNOWN_RESERVES, UNKNOWN_OBLIGATIONS, MISSING_WITHDRAWAL_PLAN)
- Computes is_ready_for_numeric_policy
"""
from __future__ import annotations

from decimal import Decimal
from typing import List, Optional

from core.investor_essence.models import (
    FinancialContextSnapshot,
    FinancialReadinessIssue,
)


def evaluate_financial_readiness(
    snapshot: FinancialContextSnapshot,
) -> List[FinancialReadinessIssue]:
    """Audits the financial context for readiness to propose numeric policy values."""
    issues: List[FinancialReadinessIssue] = []

    # 1. Horizon check
    if snapshot.horizon_years is None and "horizon_years" not in snapshot.unknown_fields:
        issues.append(
            FinancialReadinessIssue(
                code="MISSING_HORIZON",
                severity="blocking",
                message="กรอบเวลาการลงทุน (Investment Horizon) ยังไม่ได้ระบุ",
                field_path="horizon_years",
            )
        )

    # 2. Emergency reserves check
    has_reserves = (
        snapshot.emergency_reserves_amount is not None
        or snapshot.emergency_reserves_months is not None
    )
    if not has_reserves and "emergency_reserves" in snapshot.unknown_fields:
        issues.append(
            FinancialReadinessIssue(
                code="UNKNOWN_RESERVES",
                severity="blocking",
                message="เงินสำรองฉุกเฉินยังไม่ทราบสถานะที่ชัดเจน",
                field_path="emergency_reserves",
            )
        )

    # 3. Obligations check
    if snapshot.obligations_monthly is None and "obligations" in snapshot.unknown_fields:
        issues.append(
            FinancialReadinessIssue(
                code="UNKNOWN_OBLIGATIONS",
                severity="blocking",
                message="ภาระผูกพันหรือค่าใช้จ่ายจำเป็นยังไม่ทราบจำนวน",
                field_path="obligations",
            )
        )

    # 4. Target use / Withdrawal check
    has_target_use = bool(snapshot.target_use_amount or snapshot.target_use_range or snapshot.target_use_timeline)
    has_withdrawal = bool(snapshot.withdrawal_frequency or snapshot.withdrawal_amount)
    if not has_target_use and not has_withdrawal and "withdrawal_plan" in snapshot.unknown_fields:
        issues.append(
            FinancialReadinessIssue(
                code="MISSING_WITHDRAWAL_PLAN",
                severity="warning",
                message="ยังไม่มีแผนการถอนเงินหรือการใช้เงินตามกำหนด",
                field_path="withdrawal_plan",
            )
        )

    return issues


def update_readiness(
    snapshot: FinancialContextSnapshot,
) -> FinancialContextSnapshot:
    """Returns a new snapshot with evaluated readiness issues and status."""
    issues = evaluate_financial_readiness(snapshot)
    has_blocking = any(i.severity == "blocking" for i in issues)
    return FinancialContextSnapshot(
        snapshot_id=snapshot.snapshot_id,
        portfolio_id=snapshot.portfolio_id,
        horizon_years=snapshot.horizon_years,
        target_use_amount=snapshot.target_use_amount,
        target_use_range=snapshot.target_use_range,
        target_use_timeline=snapshot.target_use_timeline,
        emergency_reserves_amount=snapshot.emergency_reserves_amount,
        emergency_reserves_months=snapshot.emergency_reserves_months,
        obligations_monthly=snapshot.obligations_monthly,
        obligations_description=snapshot.obligations_description,
        withdrawal_frequency=snapshot.withdrawal_frequency,
        withdrawal_amount=snapshot.withdrawal_amount,
        experience_description=snapshot.experience_description,
        unknown_fields=list(snapshot.unknown_fields),
        as_of=snapshot.as_of,
        source=snapshot.source,
        portfolio_checkpoint_refs=dict(snapshot.portfolio_checkpoint_refs),
        readiness_issues=issues,
        is_ready_for_numeric_policy=not has_blocking,
    )
