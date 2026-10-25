"""Opportunity Finder Portfolio Capacity & Exposure Model.

Architecture Layer: core/opportunity_finder/ (Pure Domain)
Evaluates remaining capital capacity and single-stock/sector room using Decimal math.
Enforces Hard Rules:
- Deterministic formulas: issuer_room = cV - E, sector_room = sV - S
- Unknown capacity flagged when coverage is incomplete
- Negative room results in 'constrained' status and remaining capacity 0
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Tuple

from core.opportunity_finder.models import CapacityStatus, OpportunityCandidate


def calculate_candidate_capacity(
    candidate: OpportunityCandidate,
    total_nav_thb: Decimal,
    available_cash_thb: Decimal,
    current_sector_exposure_thb: Decimal,
    single_stock_cap_pct: Decimal = Decimal("15.0"),
    sector_cap_pct: Decimal = Decimal("30.0"),
    coverage_pct: Decimal = Decimal("100.0"),
) -> Tuple[CapacityStatus, Optional[Decimal], Optional[Decimal], Optional[Decimal], List[str]]:
    """Compute capacity and exposure room for candidate under existing cash scenario.

    Returns:
        (capacity_status, usable_capacity_thb, issuer_room_thb, sector_room_thb, reasons)
    """
    reasons: List[str] = []

    if total_nav_thb <= Decimal("0"):
        reasons.append("Portfolio NAV is zero; capacity not applicable.")
        return "not_applicable", Decimal("0"), Decimal("0"), Decimal("0"), reasons

    # 1. Check data coverage completeness (D06 / 8.2)
    # If fund constituent coverage is below 60%, exact room cannot be guaranteed
    if coverage_pct < Decimal("60.0"):
        reasons.append(
            f"Underlying fund look-through coverage is partial ({coverage_pct:.1f}%); "
            "capacity room is conservatively marked as unknown."
        )
        return "unknown", None, None, None, reasons

    # 2. Issuer Room: cV - E
    issuer_cap_val = (total_nav_thb * single_stock_cap_pct) / Decimal("100.0")
    current_issuer_exp = candidate.current_total_exposure_thb
    raw_issuer_room = issuer_cap_val - current_issuer_exp

    # 3. Sector Room: sV - S
    sector_cap_val = (total_nav_thb * sector_cap_pct) / Decimal("100.0")
    raw_sector_room = sector_cap_val - current_sector_exposure_thb

    # Quantize to 2 decimals
    issuer_room = raw_issuer_room.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    sector_room = raw_sector_room.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    deployable_cash = max(Decimal("0"), available_cash_thb).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    # 4. Check Constraints
    is_issuer_constrained = raw_issuer_room <= Decimal("0")
    is_sector_constrained = raw_sector_room <= Decimal("0")
    is_cash_constrained = deployable_cash <= Decimal("0")

    if is_issuer_constrained:
        reasons.append(
            f"Issuer room exhausted: existing economic exposure for {candidate.symbol} is {current_issuer_exp:,.0f} THB "
            f"({(current_issuer_exp / total_nav_thb * 100):.1f}%), exceeding/at {single_stock_cap_pct}% Cap."
        )
    if is_sector_constrained:
        reasons.append(
            f"Sector room exhausted: sector exposure for {candidate.sector_name} is {current_sector_exposure_thb:,.0f} THB "
            f"({(current_sector_exposure_thb / total_nav_thb * 100):.1f}%), exceeding/at {sector_cap_pct}% Cap."
        )
    if is_cash_constrained:
        reasons.append(
            f"Deployable cash balance is {deployable_cash:,.0f} THB; cash addition currently constrained under existing-cash scenario."
        )

    if is_issuer_constrained or is_sector_constrained:
        status: CapacityStatus = "constrained"
        usable_capacity = Decimal("0")
    elif is_cash_constrained:
        # Cash account is 0, but portfolio has structural room under policy caps
        status = "available"
        usable_capacity = Decimal("0")
        structural_room = max(Decimal("0"), min(raw_issuer_room, raw_sector_room)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        reasons.append(
            f"Structural capacity available: up to {structural_room:,.0f} THB under {single_stock_cap_pct}% single-stock "
            f"and {sector_cap_pct}% sector limits (pending cash addition/rebalancing)."
        )
    else:
        status = "available"
        usable_capacity = max(Decimal("0"), min(raw_issuer_room, raw_sector_room, deployable_cash)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        reasons.append(
            f"Capacity available: up to {usable_capacity:,.0f} THB under {single_stock_cap_pct}% single-stock "
            f"and {sector_cap_pct}% sector limits."
        )

    return status, usable_capacity, max(Decimal("0"), issuer_room), max(Decimal("0"), sector_room), reasons
