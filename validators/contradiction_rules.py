"""Contradiction guardrail rules for Macro Strategy Direction."""
from dataclasses import dataclass
from typing import Any
from schemas.warning_registry import (
    WarningMessage,
    GOLD_CONTRADICTION,
    GOLD_RATIONALE_WARNING,
    US_EQUITY_CONTRADICTION,
    REGIME_CONTRADICTION,
    CONVICTION_CONTRADICTION,
)


@dataclass
class ContradictionFinding:
    """Structured finding from a contradiction check rule (no in-place mutation)."""
    warning: WarningMessage
    target_object: Any
    downgrade_confidence: bool = False
    downgrade_conviction: bool = False


def _add_warning(warnings_list: list[str], warning: Any) -> None:
    w_str = str(warning)
    if w_str not in warnings_list:
        warnings_list.append(w_str)


def check_gold_contradiction(direction: Any) -> list[ContradictionFinding]:
    """Pure check rule for Gold rationale contradiction."""
    findings: list[ContradictionFinding] = []
    for a in getattr(direction, "asset_allocation", []):
        stance_str = str(getattr(a, "stance", "")).split(".")[-1].upper()
        if stance_str == "OVERWEIGHT":
            if any(k in getattr(a, "asset_class", "").lower() for k in ["gold", "precious metal", "ทองคำ", "โลหะมีค่า"]):
                text_lower = (getattr(a, "rationale", "") + " " + " ".join(getattr(a, "supporting_data", []))).lower()
                has_macro_anchor = any(k in text_lower for k in ["yield", "fed", "rate", "inflation", "cpi", "monetary", "real interest", "ดอกเบี้ย", "เงินเฟ้อ"])
                has_geopolitics = any(k in text_lower for k in ["geopolit", "war", "conflict", "tension", "สงคราม", "ภูมิรัฐศาสตร์"])
                if has_geopolitics and not has_macro_anchor:
                    findings.append(ContradictionFinding(
                        warning=WarningMessage(GOLD_CONTRADICTION),
                        target_object=a,
                        downgrade_confidence=True
                    ))
                    findings.append(ContradictionFinding(
                        warning=WarningMessage(GOLD_RATIONALE_WARNING),
                        target_object=direction,
                        downgrade_conviction=True
                    ))
    return findings


def check_us_equity_contradiction(direction: Any) -> list[ContradictionFinding]:
    """Pure check rule for US Equity conflicting signals contradiction."""
    findings: list[ContradictionFinding] = []
    for a in getattr(direction, "asset_allocation", []):
        stance_str = str(getattr(a, "stance", "")).split(".")[-1].upper()
        if stance_str == "OVERWEIGHT":
            if any(k in getattr(a, "asset_class", "").lower() for k in ["equit", "stock", "growth", "หุ้น"]):
                text_lower = (getattr(a, "rationale", "") + " " + " ".join(getattr(a, "supporting_data", []))).lower()
                conflicting = any(k in text_lower for k in ["yields rising", "housing starts weak", "consumer sentiment low", "housing starts ลดลง", "sentiment ต่ำ", "cpi > 3.0%"])
                if conflicting:
                    findings.append(ContradictionFinding(
                        warning=WarningMessage(US_EQUITY_CONTRADICTION),
                        target_object=a,
                        downgrade_confidence=True,
                        downgrade_conviction=True
                    ))
    return findings


def check_regime_contradiction(direction: Any) -> list[ContradictionFinding]:
    """Pure check rule for overall regime contradiction."""
    findings: list[ContradictionFinding] = []
    regime_str = str(getattr(direction, "overall_regime", "")).split(".")[-1].upper()
    if regime_str in ["GOLDILOCKS", "REFLATION"]:
        all_text = (getattr(direction, "conviction_rationale", "") + " " + " ".join([getattr(a, "rationale", "") + " " + " ".join(getattr(a, "supporting_data", [])) for a in getattr(direction, "asset_allocation", [])])).lower()
        if any(k in all_text for k in ["cpi > 3", "sticky inflation", "high inflation", "เงินเฟ้อสูง"]):
            findings.append(ContradictionFinding(
                warning=WarningMessage(REGIME_CONTRADICTION),
                target_object=direction,
                downgrade_conviction=True
            ))
    return findings


def check_barbell_contradiction(direction: Any) -> list[ContradictionFinding]:
    """Pure check rule for Barbell portfolio contradiction."""
    findings: list[ContradictionFinding] = []
    eq_overweight = []
    bond_overweight = []
    for a in getattr(direction, "asset_allocation", []):
        stance_str = str(getattr(a, "stance", "")).split(".")[-1].upper()
        if stance_str == "OVERWEIGHT":
            ac = getattr(a, "asset_class", "").lower()
            if any(k in ac for k in ["equit", "stock", "growth", "หุ้น"]):
                eq_overweight.append(a)
            if any(k in ac for k in ["bond", "duration", "treasur", "fixed income", "พันธบัตร"]):
                bond_overweight.append(a)
    if eq_overweight and bond_overweight:
        reconcile_keywords = ["barbell", "reconcil", "protection", "duration hedge", "hedged", "hedging", "with hedge", "ป้องกันความเสี่ยง"]
        combined_text = f"{getattr(direction, 'conviction_rationale', '')} {getattr(direction, 'divergence_note', '')} " + " ".join(getattr(a, "rationale", "") for a in eq_overweight + bond_overweight)
        text_lower = combined_text.lower()
        has_reconcile = any(k in text_lower for k in reconcile_keywords)
        if ("without hedge" in text_lower or "no hedge" in text_lower) and not any(k in text_lower for k in ["barbell", "reconcil", "protection"]):
            has_reconcile = False
        if not has_reconcile:
            warn_msg = WarningMessage(CONVICTION_CONTRADICTION)
            findings.append(ContradictionFinding(
                warning=warn_msg,
                target_object=direction,
                downgrade_conviction=True
            ))
            for a in eq_overweight + bond_overweight:
                findings.append(ContradictionFinding(
                    warning=warn_msg,
                    target_object=a,
                    downgrade_confidence=True
                ))
    return findings


def apply_contradiction_findings(findings: list[ContradictionFinding], direction: Any = None) -> None:
    """Validation layer helper to apply decisions (downgrade confidence/conviction and attach warnings)."""
    for f in findings:
        _add_warning(getattr(f.target_object, "validation_warnings", []), str(f.warning))
        if f.downgrade_confidence and getattr(f.target_object, "confidence", "") == "high":
            f.target_object.confidence = "medium"
        if f.downgrade_conviction:
            target_dir = direction if direction is not None else (f.target_object if hasattr(f.target_object, "conviction_level") else None)
            if target_dir is not None and getattr(target_dir, "conviction_level", "") == "high":
                target_dir.conviction_level = "medium"


def validate_gold_contradiction(direction: Any) -> list[ContradictionFinding]:
    findings = check_gold_contradiction(direction)
    apply_contradiction_findings(findings, direction)
    return findings


def validate_us_equity_contradiction(direction: Any) -> list[ContradictionFinding]:
    findings = check_us_equity_contradiction(direction)
    apply_contradiction_findings(findings, direction)
    return findings


def validate_regime_contradiction(direction: Any) -> list[ContradictionFinding]:
    findings = check_regime_contradiction(direction)
    apply_contradiction_findings(findings, direction)
    return findings


def validate_barbell_contradiction(direction: Any) -> list[ContradictionFinding]:
    findings = check_barbell_contradiction(direction)
    apply_contradiction_findings(findings, direction)
    return findings


def validate_all_contradictions(direction: Any) -> list[ContradictionFinding]:
    """Run all contradiction validation checks and apply decisions."""
    findings: list[ContradictionFinding] = []
    findings.extend(check_gold_contradiction(direction))
    findings.extend(check_us_equity_contradiction(direction))
    findings.extend(check_regime_contradiction(direction))
    findings.extend(check_barbell_contradiction(direction))
    try:
        from validators.valuation_guardrails import check_valuation_contradiction, check_credit_spread_warning
        findings.extend(check_valuation_contradiction(direction))
        findings.extend(check_credit_spread_warning(direction))
    except Exception:
        pass
    apply_contradiction_findings(findings, direction)
    return findings
