"""Tests for Deliverable D5: WP08 (Holdings planning snapshot) and WP09 (BucketPolicy & funding calculations)."""
from decimal import Decimal
from typing import Any, Dict, List
import pytest

from core.investor_essence.bucket_policy import (
    BucketPolicy,
    GoalFundingLink,
    InflowRoutingMode,
    purpose_bucket_to_policy,
)
from core.investor_essence.funding import (
    calculate_baseline_contribution,
    calculate_bucket_gap,
    calculate_emergency_reserve_requirement,
    calculate_funding_progress,
    calculate_inflow_routing,
)
from core.investor_essence.models import (
    AllocationBasis,
    ArtifactRef,
    HoldingPlanningSnapshot,
    PurposeBucketDraft,
)
from application.investor_essence.bucket_planning_service import BucketPlanningService
from application.investor_essence.dto import (
    BucketPlanProposal,
    PurposeBucketProposal,
)
from application.investor_essence.testing.fakes import (
    FakeBucketGenerator,
    FakeClock,
    FakeIdGenerator,
    FakeInvestorRuntimeUowFactory,
    FakePortfolioPlanningAdapter,
)


def test_holding_planning_snapshot_calculation_and_no_truncation():
    """WP08 / AC11: 900,000 from NAV 1,000,000 is 90%, and >10 holdings coverage is fully preserved."""
    nav = Decimal("1000000.00")
    # Build 12 holdings
    holdings: List[HoldingPlanningSnapshot] = [
        HoldingPlanningSnapshot(
            holding_id="port-1:SPY",
            symbol="SPY",
            asset_type="ETF",
            market_value_thb=Decimal("900000.00"),
            weight_percent=Decimal("90.00"),
            existing_bucket_id="b_growth",
        )
    ]
    for i in range(2, 13):
        holdings.append(
            HoldingPlanningSnapshot(
                holding_id=f"port-1:STK{i}",
                symbol=f"STK{i}",
                asset_type="Stock",
                market_value_thb=Decimal("10000.00"),
                weight_percent=Decimal("1.00"),
                existing_bucket_id=None,
            )
        )

    adapter = FakePortfolioPlanningAdapter(holdings=holdings, nav_thb=nav)
    snapshot = adapter.snapshot("port-1")

    # 1. Exact 90% weight on the 900,000 holding
    spy_holding = next(h for h in snapshot.holdings if h.symbol == "SPY")
    assert spy_holding.market_value_thb == Decimal("900000.00")
    assert spy_holding.weight_percent == Decimal("90.00")

    # 2. Coverage count: all 12 items preserved (NO silent truncation to 10)
    assert len(snapshot.holdings) == 12
    assert len(snapshot.holdings_summary) == 12

    # 3. NAV=0 empty portfolio handled gracefully
    adapter_empty = FakePortfolioPlanningAdapter(holdings=[], nav_thb=Decimal("0.00"))
    empty_snap = adapter_empty.snapshot("port-empty")
    assert empty_snap.nav_thb == Decimal("0.00")
    assert len(empty_snap.holdings) == 0


def test_bucket_service_sends_complete_holdings_to_generator():
    """WP08: Bucket planning service transmits full holdings coverage and actual values to generator."""
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock()
    id_gen = FakeIdGenerator()

    # Pre-seed confirmed investment axis and essence
    with uow_factory.open() as uow:
        scope = "portfolio:port-12"
        uow.planning.set_confirmed_pointer(
            scope,
            "investment_axis",
            "axis-conf-1",
            None,
            snapshot={
                "basic_policy": "Disciplined global growth",
                "allocation_basis": "asset_class",
                "allocation_rows": [
                    {"allocation_id": "a-1", "category_name": "Global Equities", "target_percent": "70.00"}
                ],
            },
        )
        uow.planning.set_confirmed_pointer(
            "workspace",
            "investor_essence",
            "ess-conf-1",
            None,
            snapshot={"accepted_claims": [{"claim_id": "c-1", "text": "เน้นเติบโตระยะยาว"}]},
        )
        uow.commit()

    # Create 14 holdings in portfolio
    holdings = [
        HoldingPlanningSnapshot(
            holding_id=f"port-12:SYM{i}",
            symbol=f"SYM{i}",
            asset_type="ETF",
            market_value_thb=Decimal(str(i * 10000)),
            weight_percent=Decimal(str(i)),
            existing_bucket_id=None,
        )
        for i in range(1, 15)
    ]
    portfolio_adapter = FakePortfolioPlanningAdapter(holdings=holdings, nav_thb=Decimal("1050000.00"))

    generator = FakeBucketGenerator()
    service = BucketPlanningService(
        uow_factory=uow_factory,
        bucket_generator=generator,
        portfolio_port=portfolio_adapter,
        clock=clock,
        id_gen=id_gen,
    )

    draft_view = service.create_bucket_plan_draft("port-12")
    assert draft_view.portfolio_id == "port-12"

    # Verify generator call payload received all 14 holdings
    assert len(generator.call_history) == 1
    call = generator.call_history[0]
    p_snap_payload = call["portfolio_snapshot"]
    assert p_snap_payload["holdings_count"] == 14
    assert len(p_snap_payload["holdings"]) == 14
    assert Decimal(p_snap_payload["holdings"][0]["market_value_thb"]) == Decimal("10000")
    assert Decimal(p_snap_payload["holdings"][0]["weight_percent"]) == Decimal("1")


def test_purpose_bucket_to_policy_mapping():
    """WP09: Conversion between PurposeBucketDraft and BucketPolicy preserves structured fields."""
    draft = PurposeBucketDraft(
        bucket_id="b-growth",
        name="กองหน้าขับเคลื่อน 10 ล้าน",
        role="เร่งผลตอบแทนทบต้นเพื่อพิชิตพอร์ต 10 ล้าน",
        color="#3B82F6",
        target_percent=Decimal("60.00"),
        source_value_ids=["v-1"],
        source_axis_allocation_ids=["a-1"],
        linked_goal_name="พอร์ต 10 ล้านบาท",
        eligible_assets=["หุ้นดัชนีโลก S&P500"],
        cash_inflow_routing_pct="60.00",
        action_rules=["รับเงิน DCA รายเดือนเป็นหลัก"],
        target_amount=Decimal("10000000.00"),
        inflow_route_mode="fixed_ratio",
        inflow_priority=1,
    )

    policy = purpose_bucket_to_policy(draft)
    assert policy.bucket_id == "b-growth"
    assert policy.name == "กองหน้าขับเคลื่อน 10 ล้าน"
    assert policy.target_percent == Decimal("60.00")
    assert policy.target_amount == Decimal("10000000.00")
    assert policy.inflow_route_percent == Decimal("60.00")
    assert policy.inflow_route_mode == InflowRoutingMode.FIXED_RATIO
    assert len(policy.goal_links) == 1
    assert policy.goal_links[0].goal_name == "พอร์ต 10 ล้านบาท"


def test_funding_emergency_reserve_requirement_deducts_outside_reserves():
    """WP09 / AC09: Emergency reserve calculation accounts for reserves held outside portfolio without double-counting."""
    # Essential monthly expense 30,000 THB, 6 months reserve = 180,000 THB gross requirement
    # Outside reserves = 80,000 THB -> Portfolio needs 100,000 THB
    result = calculate_emergency_reserve_requirement(
        essential_monthly_expense=Decimal("30000.00"),
        emergency_months=Decimal("6.0"),
        reserves_outside_portfolio=Decimal("80000.00"),
    )
    assert result["gross_required"] == Decimal("180000.00")
    assert result["reserves_outside_portfolio"] == Decimal("80000.00")
    assert result["net_portfolio_required"] == Decimal("100000.00")
    assert result["is_fully_covered_outside"] is False

    # Fully covered outside: 200,000 THB outside reserves -> Portfolio needs 0 THB
    covered = calculate_emergency_reserve_requirement(
        essential_monthly_expense=Decimal("30000.00"),
        emergency_months=Decimal("6.0"),
        reserves_outside_portfolio=Decimal("200000.00"),
    )
    assert covered["net_portfolio_required"] == Decimal("0.00")
    assert covered["is_fully_covered_outside"] is True


def test_funding_gap_progress_and_baseline_contribution():
    """WP09: Gap, progress %, and baseline monthly contribution calculate deterministically."""
    # 1. Gap
    gap = calculate_bucket_gap(
        target_amount=Decimal("1000000.00"),
        current_value=Decimal("400000.00"),
    )
    assert gap == Decimal("600000.00")

    # 2. Funding progress %
    progress = calculate_funding_progress(
        current_value=Decimal("400000.00"),
        target_amount=Decimal("1000000.00"),
    )
    assert progress == Decimal("40.00")

    # 3. Baseline monthly contribution: 600,000 over 60 months (5 years) = 10,000 THB/month
    contrib = calculate_baseline_contribution(gap=gap, months_remaining=60)
    assert contrib["status"] == "on_track"
    assert contrib["monthly_contribution"] == Decimal("10000.00")

    # 4. Unknown deadline or overdue status
    no_time = calculate_baseline_contribution(gap=gap, months_remaining=None)
    assert no_time["status"] == "insufficient_data"
    assert no_time["monthly_contribution"] is None

    overdue = calculate_baseline_contribution(gap=gap, months_remaining=0)
    assert overdue["status"] == "overdue"


def test_funding_inflow_routing_fixed_and_priority_fill():
    """WP09 / AC13: Inflow routing allocations respect budget and handle fixed ratio vs priority fill."""
    policies = [
        BucketPolicy(
            bucket_id="b-growth",
            name="เติบโต",
            purpose="สะสม",
            role="กองหน้า",
            inflow_route_mode=InflowRoutingMode.FIXED_RATIO,
            inflow_route_percent=Decimal("70.00"),
        ),
        BucketPolicy(
            bucket_id="b-safety",
            name="กันชน",
            purpose="สำรอง",
            role="กันชน",
            inflow_route_mode=InflowRoutingMode.FIXED_RATIO,
            inflow_route_percent=Decimal("30.00"),
        ),
    ]

    # 1. 20,000 monthly budget at 70/30 -> 14,000 to growth, 6,000 to safety
    routing = calculate_inflow_routing(
        monthly_budget=Decimal("20000.00"),
        policies=policies,
    )
    assert routing["is_valid"] is True
    assert routing["routing_amounts"]["b-growth"] == Decimal("14000.00")
    assert routing["routing_amounts"]["b-safety"] == Decimal("6000.00")
    assert routing["total_allocated"] == Decimal("20000.00")

    # 2. AC13: Exceeding 100% routing flags error and validation issue
    invalid_policies = [
        BucketPolicy(
            bucket_id="b-1",
            name="A",
            purpose="A",
            role="A",
            inflow_route_mode=InflowRoutingMode.FIXED_RATIO,
            inflow_route_percent=Decimal("90.00"),
        ),
        BucketPolicy(
            bucket_id="b-2",
            name="B",
            purpose="B",
            role="B",
            inflow_route_mode=InflowRoutingMode.FIXED_RATIO,
            inflow_route_percent=Decimal("90.00"),
        ),
    ]
    invalid_routing = calculate_inflow_routing(
        monthly_budget=Decimal("20000.00"),
        policies=invalid_policies,
    )
    assert invalid_routing["is_valid"] is False
    assert any("เกิน 100%" in err for err in invalid_routing["issues"])

    # 3. Priority Fill with redirect when full
    p_fill_policies = [
        BucketPolicy(
            bucket_id="b-emergency",
            name="ฉุกเฉิน",
            purpose="ฉุกเฉิน",
            role="เสบียง",
            inflow_route_mode=InflowRoutingMode.PRIORITY_FILL,
            inflow_priority=1,
            target_amount=Decimal("50000.00"),
            when_full_redirect_bucket_id="b-growth",
        ),
        BucketPolicy(
            bucket_id="b-growth",
            name="เติบโต",
            purpose="เติบโต",
            role="กองหน้า",
            inflow_route_mode=InflowRoutingMode.PRIORITY_FILL,
            inflow_priority=2,
            target_amount=Decimal("1000000.00"),
        ),
    ]
    # Current emergency is 40,000 (gap is 10,000). Budget is 25,000.
    # Emergency gets 10,000 (fills gap). Remaining 15,000 redirected to growth!
    fill_routing = calculate_inflow_routing(
        monthly_budget=Decimal("25000.00"),
        policies=p_fill_policies,
        current_values={"b-emergency": Decimal("40000.00"), "b-growth": Decimal("100000.00")},
    )
    assert fill_routing["routing_amounts"]["b-emergency"] == Decimal("10000.00")
    assert fill_routing["routing_amounts"]["b-growth"] == Decimal("15000.00")
    assert fill_routing["total_allocated"] == Decimal("25000.00")
