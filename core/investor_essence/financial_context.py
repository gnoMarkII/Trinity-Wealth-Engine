"""FinancialContext readiness, facts synthesis, and validation domain logic.

Rules:
- Robust parsing of numeric facts with comma and whitespace stripping
- Explicit distinction between known (including 0), none, and unknown
- Audits blocking issues vs qualitative warnings for numeric policy readiness
- Builds structured FinancialFact records with units and provenance
"""
from __future__ import annotations

import decimal
from decimal import Decimal
from typing import Any, Dict, List, Optional

from core.investor_essence.models import (
    FinancialContextSnapshot,
    FinancialFact,
    FinancialReadinessIssue,
)


def parse_numeric_fact(field_name: str, val: Any) -> Optional[Decimal]:
    """Parses numeric input cleanly, handling commas, spaces, and optional unit suffixes.
    
    Raises ValueError with field name if the string is non-empty and non-numeric.
    Preserves 0 as Decimal('0').
    """
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str:
        return None

    import re
    cleaned = val_str.replace(",", "").strip()
    if field_name == "horizon_years":
        cleaned = re.sub(r"\s*(?:ปี|years?|yrs?\.?)$", "", cleaned, flags=re.IGNORECASE).strip()
    elif any(k in field_name for k in ("amount", "expense", "monthly", "reserves")):
        cleaned = re.sub(r"\s*(?:บาท|thb)$", "", cleaned, flags=re.IGNORECASE).strip()

    try:
        dec = Decimal(cleaned)
        if not dec.is_finite():
            raise ValueError(f"Value for field '{field_name}' must be finite")
        return dec
    except (decimal.InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for field '{field_name}': '{val}'") from exc


def extract_years_from_text(text: Optional[str]) -> Optional[Decimal]:
    """Extracts numeric years from horizon text such as 'ระยะยาว 10 ปี', '10 years', '5-7 ปี'."""
    if not text:
        return None
    import re
    cleaned = str(text).strip()
    # Match specific pattern: digits followed by year/ปี
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:ปี|years?|yrs?\b)", cleaned, re.IGNORECASE)
    if match:
        try:
            val = Decimal(match.group(1))
            if val > 0:
                return val
        except Exception:
            pass
    # Match standalone number
    match_num = re.search(r"\b(\d+(?:\.\d+)?)\b", cleaned)
    if match_num:
        try:
            val = Decimal(match_num.group(1))
            if val > 0:
                return val
        except Exception:
            pass
    return None


def evaluate_financial_readiness(
    snapshot: FinancialContextSnapshot,
) -> List[FinancialReadinessIssue]:
    """Audits the financial context for readiness to propose and confirm numeric policy values."""
    issues: List[FinancialReadinessIssue] = []

    # 1. Horizon check
    if snapshot.horizon_years is None:
        if "horizon_years" in snapshot.unknown_fields:
            issues.append(
                FinancialReadinessIssue(
                    code="UNKNOWN_HORIZON",
                    severity="blocking",
                    message="กรอบเวลาการลงทุนยังไม่ทราบแน่ชัด จำเป็นต้องระบุก่อนคำนวณนโยบายตัวเลข",
                    field_path="horizon_years",
                )
            )
        else:
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
        or snapshot.reserves_outside_portfolio is not None
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
    elif not has_reserves and "emergency_reserves" not in snapshot.unknown_fields:
        issues.append(
            FinancialReadinessIssue(
                code="MISSING_RESERVES",
                severity="warning",
                message="ยังไม่ได้ระบุเงินสำรองฉุกเฉินหรือเงินสำรองนอกพอร์ต",
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
    has_target_use = bool(
        snapshot.target_use_amount is not None
        or snapshot.target_use_range
        or snapshot.target_use_timeline
    )
    has_withdrawal = bool(
        snapshot.withdrawal_frequency
        or snapshot.withdrawal_amount is not None
    )
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


def build_financial_facts(snapshot: FinancialContextSnapshot) -> Dict[str, FinancialFact]:
    """Extracts structured, typed FinancialFact objects for each financial dimension."""
    facts: Dict[str, FinancialFact] = {}
    now_as_of = snapshot.as_of

    def make_fact(key: str, numeric_val: Optional[Decimal], text_val: Optional[str], unit: str) -> FinancialFact:
        if key in snapshot.unknown_fields:
            status = "unknown"
        elif numeric_val is not None or text_val is not None:
            status = "known"
        else:
            status = "none"

        return FinancialFact(
            field_key=key,
            status=status,
            value_numeric=numeric_val,
            value_text=text_val or (str(numeric_val) if numeric_val is not None else None),
            unit=unit,
            source=snapshot.source,
            as_of=now_as_of,
        )

    facts["horizon_years"] = make_fact("horizon_years", snapshot.horizon_years, None, "years")
    facts["target_use_amount"] = make_fact("target_use_amount", snapshot.target_use_amount, snapshot.target_use_range, "THB")
    facts["emergency_reserves_amount"] = make_fact("emergency_reserves_amount", snapshot.emergency_reserves_amount, None, "THB")
    facts["emergency_reserves_months"] = make_fact("emergency_reserves_months", snapshot.emergency_reserves_months, None, "months")
    facts["obligations_monthly"] = make_fact("obligations_monthly", snapshot.obligations_monthly, snapshot.obligations_description, "THB/month")
    facts["withdrawal_amount"] = make_fact("withdrawal_amount", snapshot.withdrawal_amount, snapshot.withdrawal_frequency, "THB")
    facts["monthly_investable_contribution"] = make_fact("monthly_investable_contribution", snapshot.monthly_investable_contribution, None, "THB/month")
    facts["essential_monthly_expense"] = make_fact("essential_monthly_expense", snapshot.essential_monthly_expense, None, "THB/month")
    facts["reserves_outside_portfolio"] = make_fact("reserves_outside_portfolio", snapshot.reserves_outside_portfolio, None, "THB")

    return facts


def update_readiness(
    snapshot: FinancialContextSnapshot,
) -> FinancialContextSnapshot:
    """Returns a new snapshot with evaluated readiness issues, synthesized facts, and status."""
    issues = evaluate_financial_readiness(snapshot)
    facts = build_financial_facts(snapshot)
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
        monthly_investable_contribution=snapshot.monthly_investable_contribution,
        essential_monthly_expense=snapshot.essential_monthly_expense,
        reserves_outside_portfolio=snapshot.reserves_outside_portfolio,
        facts=facts,
        unknown_fields=list(snapshot.unknown_fields),
        as_of=snapshot.as_of,
        source=snapshot.source,
        portfolio_checkpoint_refs=dict(snapshot.portfolio_checkpoint_refs),
        readiness_issues=issues,
        is_ready_for_numeric_policy=not has_blocking,
    )
