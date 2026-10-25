"""Opportunity Finder Policy & Investor Axis Evaluation.

Architecture Layer: core/opportunity_finder/ (Pure Domain)
Evaluates candidates against confirmed Investor Axis constraints and Applied Bucket Policies.
Enforces Hard Rule: Prohibitions cannot be compensated by positive scores in other dimensions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.opportunity_finder.models import OpportunityCandidate, PolicyFitStatus


def evaluate_policy_admissibility(
    candidate: OpportunityCandidate,
    confirmed_axis: Optional[Dict[str, Any]] = None,
    applied_buckets: Optional[List[Dict[str, Any]]] = None,
    sidecar_data: Optional[Dict[str, Any]] = None,
) -> Tuple[PolicyFitStatus, List[str], Optional[str], Optional[str]]:
    """Evaluate candidate admissibility against confirmed axis and bucket policy.

    Returns:
        (policy_fit, reasons, suggested_bucket_id, applied_bucket_name)
    """
    reasons: List[str] = []

    # 1. If no confirmed axis exists, flag for review (D08: unconfirmed policies require review)
    if not confirmed_axis:
        reasons.append("No confirmed Investor Axis for this portfolio; candidate admissibility requires review.")
        suggested_bucket_id, bucket_name = _match_bucket(candidate, applied_buckets)
        return "needs_review", reasons, suggested_bucket_id, bucket_name

    # 2. Check Exclusions (Hard Exclusions cannot be bypassed)
    exclusions = confirmed_axis.get("exclusions") or []
    prohibited_sectors = confirmed_axis.get("prohibited_sectors") or []
    prohibited_symbols = confirmed_axis.get("prohibited_symbols") or []
    exclude_targets = confirmed_axis.get("exclude_targets") or []
    non_actions = confirmed_axis.get("non_actions") or []

    sym_clean = candidate.symbol.upper()
    sector_clean = candidate.sector_name.lower()
    comp_clean = candidate.company_name.lower()

    for p_sym in prohibited_symbols:
        if str(p_sym).strip().upper() == sym_clean:
            reasons.append(f"Excluded by Investor Axis: Symbol '{sym_clean}' is in prohibited symbols list.")
            return "fail", reasons, None, None

    for p_sec in prohibited_sectors:
        p_sec_clean = str(p_sec).strip().lower()
        if p_sec_clean in sector_clean or sector_clean in p_sec_clean:
            reasons.append(f"Excluded by Investor Axis: Sector '{candidate.sector_name}' is in prohibited sectors list.")
            return "fail", reasons, None, None

    # Check exclude_targets (symbols, company names, or sectors)
    for target in exclude_targets:
        t_clean = str(target).strip()
        if not t_clean:
            continue
        if t_clean.upper() == sym_clean:
            reasons.append(f"Excluded by Investor Axis: Symbol '{sym_clean}' is in exclude targets list.")
            return "fail", reasons, None, None
        t_lower = t_clean.lower()
        if t_lower in comp_clean or t_lower in sector_clean or sector_clean in t_lower:
            reasons.append(f"Excluded by Investor Axis target exclusion: '{t_clean}'.")
            return "fail", reasons, None, None

    # Check non_actions rules / constraints
    for na in non_actions:
        na_clean = str(na).strip()
        if not na_clean:
            continue
        if na_clean.upper() == sym_clean:
            reasons.append(f"Excluded by Investor Axis non-action rule for symbol '{sym_clean}': '{na_clean}'.")
            return "fail", reasons, None, None
        if len(na_clean) >= 3 and na_clean.lower() in comp_clean:
            reasons.append(f"Excluded by Investor Axis non-action rule: '{na_clean}'.")
            return "fail", reasons, None, None

    # Check keyword exclusions
    for exc in exclusions:
        exc_str = str(exc).lower()
        if exc_str in comp_clean or exc_str in sector_clean:
            reasons.append(f"Excluded by Investor Axis constraint: '{exc}'.")
            return "fail", reasons, None, None

    # 3. Check quality / threshold rules if defined in confirmed axis
    min_score = confirmed_axis.get("min_composite_score")
    if min_score is not None and sidecar_data:
        comp_score = sidecar_data.get("composite_score")
        if comp_score is not None and float(comp_score) < float(min_score):
            reasons.append(f"Composite score ({comp_score:.1f}) is below Axis minimum threshold ({min_score}).")
            return "fail", reasons, None, None

    # 4. Suggest bucket from applied bucket policies (D10: Must match real applied policy)
    suggested_bucket_id, bucket_name = _match_bucket(candidate, applied_buckets)

    reasons.append("Admissible under confirmed Investor Axis rules.")
    return "pass", reasons, suggested_bucket_id, bucket_name


def _match_bucket(
    candidate: OpportunityCandidate,
    applied_buckets: Optional[Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Match candidate to an applied bucket based on sector and asset characteristics."""
    if not applied_buckets:
        return None, None

    buckets_list: List[Dict[str, Any]] = []
    if isinstance(applied_buckets, dict):
        buckets_list = applied_buckets.get("purpose_buckets") or applied_buckets.get("buckets") or []
    elif isinstance(applied_buckets, list):
        buckets_list = applied_buckets

    if not buckets_list:
        return None, None

    sector = candidate.sector_name.lower()
    asset_type = candidate.asset_type.upper()

    def _get_id_and_name(b: Dict[str, Any]) -> Tuple[str, str]:
        b_id = str(b.get("id") or b.get("bucket_id") or b.get("purpose_bucket_id") or "")
        b_name = str(b.get("name") or b.get("bucket_name") or b.get("purpose_name") or b_id)
        return b_id, b_name

    # Rule 1: REIT / Income / Dividend
    if asset_type == "REIT" or "utilities" in sector or "real estate" in sector:
        for b in buckets_list:
            b_id, b_name = _get_id_and_name(b)
            if any(k in b_id.lower() or k in b_name.lower() for k in ("income", "dividend", "yield", "cashflow", "reit")):
                return b_id, b_name

    # Rule 2: Growth / Tech / Innovation
    if "technology" in sector or "communication" in sector or "discretionary" in sector:
        for b in buckets_list:
            b_id, b_name = _get_id_and_name(b)
            if any(k in b_id.lower() or k in b_name.lower() for k in ("growth", "innovat", "satellite", "tech")):
                return b_id, b_name

    # Rule 3: Core / Compounder default
    for b in buckets_list:
        b_id, b_name = _get_id_and_name(b)
        if any(k in b_id.lower() or k in b_name.lower() for k in ("core", "compound", "anchor", "main")):
            return b_id, b_name

    # Fallback to first bucket
    first = buckets_list[0]
    f_id, f_name = _get_id_and_name(first)
    return f_id, f_name
