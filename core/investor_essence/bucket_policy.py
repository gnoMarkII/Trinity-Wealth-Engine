"""Bucket policy domain models and structure definitions (WP09).

Defines structured operational purpose bucket policies, goal links, inflow/outflow
rules, and asset eligibility constraints matching schema v2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class InflowRoutingMode(str, Enum):
    """Mode determining how incoming monthly investment cash is allocated."""
    FIXED_RATIO = "fixed_ratio"
    PRIORITY_FILL = "priority_fill"


@dataclass(frozen=True)
class GoalFundingLink:
    """Ties a purpose bucket to a specific structured investor goal."""
    goal_id: str
    goal_name: str
    target_amount: Optional[Decimal] = None
    need_by_date: Optional[str] = None
    funding_scope: str = "portfolio"  # 'portfolio' or 'overall'
    priority: int = 1
    revision: int = 1


@dataclass(frozen=True)
class BucketPolicy:
    """Comprehensive operational policy definition for an investment purpose bucket."""
    # Identity
    bucket_id: str
    name: str
    purpose: str
    role: str
    color: str = "#3B82F6"
    schema_version: str = "2.0"

    # Goals
    goal_links: List[GoalFundingLink] = field(default_factory=list)
    explicit_purpose: Optional[str] = None

    # Targets
    target_percent: Decimal = Decimal("0.00")
    target_amount: Optional[Decimal] = None
    minimum_amount: Optional[Decimal] = None
    calculation_basis: str = "NAV"
    currency: str = "THB"

    # Time & Liquidity
    need_by_date: Optional[str] = None
    horizon_years: Optional[Decimal] = None
    liquidity_requirement: str = "normal"  # 'immediate', 'high', 'normal', 'low'
    withdrawal_schedule: Optional[str] = None

    # Inflow
    inflow_route_mode: InflowRoutingMode = InflowRoutingMode.FIXED_RATIO
    inflow_route_percent: Decimal = Decimal("0.00")
    inflow_priority: int = 1
    when_full_redirect_bucket_id: Optional[str] = None

    # Outflow
    withdrawal_order: int = 1
    withdrawal_condition: Optional[str] = None
    refill_rule: Optional[str] = None

    # Assets & Category mappings
    eligible_assets: List[str] = field(default_factory=list)
    excluded_assets: List[str] = field(default_factory=list)
    source_axis_allocation_ids: List[str] = field(default_factory=list)
    source_value_ids: List[str] = field(default_factory=list)

    # Structured Rules
    rebalance_threshold_pct: Optional[Decimal] = None
    action_rules: List[str] = field(default_factory=list)

    # Provenance
    source_refs: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)


def purpose_bucket_to_policy(draft: Any) -> BucketPolicy:
    """Converts a PurposeBucketDraft domain object to a full BucketPolicy."""
    goal_links: List[GoalFundingLink] = []
    if getattr(draft, "linked_goal_name", None):
        goal_id = getattr(draft, "goal_id", None) or f"goal_{draft.bucket_id}"
        goal_links.append(
            GoalFundingLink(
                goal_id=goal_id,
                goal_name=draft.linked_goal_name,
                target_amount=getattr(draft, "target_amount", None),
                need_by_date=getattr(draft, "need_by_date", None),
                funding_scope="portfolio",
                priority=getattr(draft, "inflow_priority", 1) or 1,
            )
        )

    # Inflow route percent
    route_pct_str = getattr(draft, "cash_inflow_routing_pct", None)
    if route_pct_str is not None:
        try:
            route_pct = Decimal(str(route_pct_str))
        except Exception:
            route_pct = Decimal("0.00")
    else:
        route_pct = draft.target_percent

    route_mode_str = getattr(draft, "inflow_route_mode", "fixed_ratio")
    try:
        route_mode = InflowRoutingMode(route_mode_str)
    except Exception:
        route_mode = InflowRoutingMode.FIXED_RATIO

    return BucketPolicy(
        bucket_id=draft.bucket_id,
        name=draft.name,
        purpose=draft.role,
        role=draft.role,
        color=draft.color,
        schema_version="2.0",
        goal_links=goal_links,
        explicit_purpose=draft.role if not goal_links else None,
        target_percent=draft.target_percent,
        target_amount=getattr(draft, "target_amount", None),
        minimum_amount=getattr(draft, "minimum_amount", None),
        calculation_basis="NAV",
        currency="THB",
        need_by_date=getattr(draft, "need_by_date", None),
        liquidity_requirement="normal",
        inflow_route_mode=route_mode,
        inflow_route_percent=route_pct,
        inflow_priority=getattr(draft, "inflow_priority", 1) or 1,
        when_full_redirect_bucket_id=getattr(draft, "when_full_redirect_bucket_id", None),
        eligible_assets=list(draft.eligible_assets),
        excluded_assets=[],
        source_axis_allocation_ids=list(draft.source_axis_allocation_ids),
        source_value_ids=list(getattr(draft, "source_value_ids", [])),
        action_rules=list(draft.action_rules),
    )
