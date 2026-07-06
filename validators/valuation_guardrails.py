"""Valuation and Credit Spread guardrail rules for Macro Strategy Direction."""
import re
from typing import Any, Optional
from schemas.warning_registry import (
    WarningMessage,
    VALUATION_RICH_WARNING,
    CREDIT_SPREAD_WARNING,
)
from tools.macro.ticker_config import (
    VALUATION_RICH_ERP_THRESHOLD,
    CREDIT_SPREAD_DANGER_THRESHOLD,
    CREDIT_SPREAD_WIDENING_3M_BPS,
)
from validators.contradiction_rules import ContradictionFinding, apply_contradiction_findings


def _parse_val_to_float(val_str: Any) -> Optional[float]:
    if isinstance(val_str, (int, float)):
        return float(val_str)
    if not isinstance(val_str, str) or not val_str.strip() or val_str.strip() == "-" or val_str.strip().upper() == "N/A":
        return None
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(val_str))
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def check_valuation_contradiction(direction: Any, observable_registry: Optional[dict[str, Any]] = None) -> list[ContradictionFinding]:
    """
    Check if Equity Risk Premium (ERP) is below VALUATION_RICH_ERP_THRESHOLD (1.5%)
    when US Equities is Overweight with HIGH confidence and without hedging/defensive rationale.
    """
    findings: list[ContradictionFinding] = []
    reg = observable_registry if observable_registry is not None else getattr(direction, "observable_registry", {})
    if not reg and hasattr(direction, "market_observables"):
        reg = {o.observable_id: o for o in getattr(direction, "market_observables", [])}

    # Locate ERP observable
    erp_obs = None
    if isinstance(reg, dict):
        erp_obs = reg.get("obs_erp_gspc")
        if not erp_obs:
            for obs in reg.values():
                if "erp" in getattr(obs, "observable_id", "").lower() or "equity risk premium" in getattr(obs, "indicator", "").lower():
                    erp_obs = obs
                    break
    elif isinstance(reg, list):
        for obs in reg:
            if "erp" in getattr(obs, "observable_id", "").lower() or "equity risk premium" in getattr(obs, "indicator", "").lower():
                erp_obs = obs
                break

    if not erp_obs or not getattr(erp_obs, "is_valid", True):
        return findings

    # Get ERP decimal
    erp_dec: Optional[float] = None
    meta = getattr(erp_obs, "metadata", {})
    if isinstance(meta, dict) and "erp_decimal" in meta and meta["erp_decimal"] is not None:
        erp_dec = float(meta["erp_decimal"])
    else:
        val_num = _parse_val_to_float(getattr(erp_obs, "value", None))
        if val_num is not None:
            # If value is e.g. 0.75 and unit is %, convert to decimal
            unit = str(getattr(erp_obs, "unit", "")).strip()
            erp_dec = val_num / 100.0 if unit == "%" or val_num > 0.20 else val_num

    if erp_dec is None or erp_dec >= VALUATION_RICH_ERP_THRESHOLD:
        return findings

    # Check US Equities allocation
    for a in getattr(direction, "asset_allocation", []):
        bucket = str(getattr(a, "asset_bucket", "")).lower()
        ac = str(getattr(a, "asset_class", "")).lower()
        stance = str(getattr(a, "stance", "")).split(".")[-1].upper()
        conf = str(getattr(a, "confidence", "")).lower()

        is_us_eq = (bucket == "equities" or "equity" in ac or "equities" in ac or "หุ้น" in ac) and ("us" in ac or "สหรัฐ" in ac or "s&p" in ac or "nasdaq" in ac or bucket == "equities")
        if is_us_eq and stance == "OVERWEIGHT" and conf == "high":
            # Check keywords for hedging or earnings revision
            text_lower = (
                str(getattr(a, "rationale", "")) + " " +
                " ".join(str(x) for x in getattr(a, "supporting_data", [])) + " " +
                str(getattr(a, "why_not_high", ""))
            ).lower()
            hedge_keywords = ["hedge", "defensive", "option", "put", "collar", "barbell", "earnings revision", "ป้องกันความเสี่ยง", "ปรับประมาณการ", "ประกันความเสี่ยง"]
            if not any(k in text_lower for k in hedge_keywords):
                findings.append(ContradictionFinding(
                    warning=WarningMessage(VALUATION_RICH_WARNING),
                    target_object=a,
                    downgrade_confidence=True,
                ))

    return findings


def check_credit_spread_warning(direction: Any, observable_registry: Optional[dict[str, Any]] = None) -> list[ContradictionFinding]:
    """
    Check High Yield Credit Spread observable against CREDIT_SPREAD_DANGER_THRESHOLD (5.0%)
    or widening > 100 bps over 3 months. Warning-only (no confidence/conviction downgrade).
    """
    findings: list[ContradictionFinding] = []
    reg = observable_registry if observable_registry is not None else getattr(direction, "observable_registry", {})
    if not reg and hasattr(direction, "market_observables"):
        reg = {o.observable_id: o for o in getattr(direction, "market_observables", [])}

    hy_obs = None
    if isinstance(reg, dict):
        hy_obs = reg.get("obs_hy_spread")
        if not hy_obs:
            for obs in reg.values():
                if "bamlh0a0hym2" in getattr(obs, "observable_id", "").lower() or "high yield" in getattr(obs, "indicator", "").lower():
                    hy_obs = obs
                    break
    elif isinstance(reg, list):
        for obs in reg:
            if "bamlh0a0hym2" in getattr(obs, "observable_id", "").lower() or "high yield" in getattr(obs, "indicator", "").lower():
                hy_obs = obs
                break

    if not hy_obs or not getattr(hy_obs, "is_valid", True):
        return findings

    val_num: Optional[float] = None
    meta = getattr(hy_obs, "metadata", {})
    widening_3m = 0.0
    if isinstance(meta, dict):
        if "hy_spread_pct" in meta and meta["hy_spread_pct"] is not None:
            val_num = float(meta["hy_spread_pct"])
        if "widening_3m_bps" in meta and meta["widening_3m_bps"] is not None:
            widening_3m = float(meta["widening_3m_bps"])

    if val_num is None:
        val_num = _parse_val_to_float(getattr(hy_obs, "value", None))

    if val_num is not None:
        if val_num > CREDIT_SPREAD_DANGER_THRESHOLD or widening_3m > CREDIT_SPREAD_WIDENING_3M_BPS:
            findings.append(ContradictionFinding(
                warning=WarningMessage(CREDIT_SPREAD_WARNING),
                target_object=direction,
                downgrade_confidence=False,
                downgrade_conviction=False,
            ))

    return findings


def validate_valuation_guardrails(direction: Any) -> list[ContradictionFinding]:
    findings = check_valuation_contradiction(direction)
    findings.extend(check_credit_spread_warning(direction))
    apply_contradiction_findings(findings, direction)
    return findings
