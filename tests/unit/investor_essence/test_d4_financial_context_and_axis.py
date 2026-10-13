"""Unit tests for Deliverable D4 (WP06 + WP07):
- Financial context: parse_numeric_fact, commas/spaces, known 0 vs unknown vs none
- FinancialContextService: round-trip, facts synthesis, validation failure on invalid inputs
- InvestmentAxis: completeness checks, risk limit ranges, not_applicable limit, non-negative allocations
- InvestmentAxisService.confirm_axis: draft revision check, readiness check, origin retention
"""
from decimal import Decimal
import pytest

from core.investor_essence.models import (
    AllocationBasis,
    AllocationPlanRow,
    ArtifactRef,
    FinancialContextSnapshot,
    InvestmentAxisDraft,
    NumericPolicyField,
    NumericPolicyOrigin,
)
from core.investor_essence.financial_context import (
    build_financial_facts,
    evaluate_financial_readiness,
    parse_numeric_fact,
    update_readiness,
)
from core.investor_essence.investment_axis import validate_axis_completeness
from application.investor_essence.financial_context_service import FinancialContextService
from application.investor_essence.investment_axis_service import InvestmentAxisService
from application.investor_essence.errors import (
    AxisIncompleteError,
    RevisionConflictError,
    ValidationFailedError,
)
from application.investor_essence.testing.fakes import (
    FakeAxisGenerator,
    FakeClock,
    FakeIdGenerator,
    FakeInvestorRuntimeUowFactory,
)


def test_parse_numeric_fact_handles_commas_and_known_zero():
    # 1. Comma and whitespace stripping
    parsed = parse_numeric_fact("target_use_amount", " 1,250,000.75 ")
    assert parsed == Decimal("1250000.75")

    # 2. Known zero is preserved as Decimal(0), NOT None
    zero_parsed = parse_numeric_fact("obligations_monthly", "0")
    assert zero_parsed == Decimal("0")
    zero_num = parse_numeric_fact("obligations_monthly", 0)
    assert zero_num == Decimal("0")

    # 3. None / empty returns None
    assert parse_numeric_fact("horizon_years", None) is None
    assert parse_numeric_fact("horizon_years", "   ") is None

    # 4. Invalid strings raise ValueError with field name
    with pytest.raises(ValueError, match="field 'horizon_years'"):
        parse_numeric_fact("horizon_years", "ten_years")


def test_financial_context_facts_and_readiness_audit():
    # 1. Context with known zero and known numbers
    snap = FinancialContextSnapshot(
        snapshot_id="snap1",
        portfolio_id="port1",
        horizon_years=Decimal("10"),
        emergency_reserves_amount=Decimal("300000"),
        obligations_monthly=Decimal("0"),  # known 0
        reserves_outside_portfolio=Decimal("150000"),
        unknown_fields=[],
        as_of="2026-10-10T20:00:00Z",
    )
    evaluated = update_readiness(snap)
    assert evaluated.is_ready_for_numeric_policy is True
    assert len([i for i in evaluated.readiness_issues if i.severity == "blocking"]) == 0

    facts = evaluated.facts
    assert facts["obligations_monthly"].status == "known"
    assert facts["obligations_monthly"].value_numeric == Decimal("0")
    assert facts["reserves_outside_portfolio"].status == "known"
    assert facts["horizon_years"].status == "known"
    assert facts["monthly_investable_contribution"].status == "none"

    # 2. Context with unknown horizon -> blocking issue
    snap_unknown = FinancialContextSnapshot(
        snapshot_id="snap2",
        portfolio_id="port1",
        horizon_years=None,
        unknown_fields=["horizon_years", "emergency_reserves"],
    )
    evaluated_unknown = update_readiness(snap_unknown)
    assert evaluated_unknown.is_ready_for_numeric_policy is False
    blocking_codes = [i.code for i in evaluated_unknown.readiness_issues if i.severity == "blocking"]
    assert "UNKNOWN_HORIZON" in blocking_codes
    assert "UNKNOWN_RESERVES" in blocking_codes
    assert evaluated_unknown.facts["horizon_years"].status == "unknown"


def test_financial_context_service_updates_and_rejects_invalid_input():
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock()
    id_gen = FakeIdGenerator()
    service = FinancialContextService(uow_factory=uow_factory, clock=clock, id_gen=id_gen)

    # 1. Valid update with comma separated string and extra WP06 fields
    res = service.update_financial_context(
        portfolio_id="port-test",
        updates={
            "horizon_years": " 10 ",
            "emergency_reserves_amount": " 250,000 ",
            "obligations_monthly": "0",
            "monthly_investable_contribution": "15,000",
            "reserves_outside_portfolio": "100,000",
            "unknown_fields": [],
        },
    )
    assert res.horizon_years == "10"
    assert res.emergency_reserves_amount == "250000"
    assert res.obligations_monthly == "0"
    assert res.monthly_investable_contribution == "15000"
    assert res.reserves_outside_portfolio == "100000"
    assert res.is_ready_for_numeric_policy is True
    assert res.facts["obligations_monthly"]["status"] == "known"

    # 2. Invalid number raises ValidationFailedError
    with pytest.raises(ValidationFailedError, match="field 'emergency_reserves_amount'"):
        service.update_financial_context(
            portfolio_id="port-test",
            updates={"emergency_reserves_amount": "invalid_number"},
        )


def _make_sample_axis_draft(
    mdd_val=Decimal("15"),
    tl_val=Decimal("2"),
    tl_unit="%",
    alloc_rows=None,
    non_actions=None,
    revision=1,
) -> InvestmentAxisDraft:
    if alloc_rows is None:
        alloc_rows = [
            AllocationPlanRow("a1", "Equities", Decimal("70.00"), "Growth"),
            AllocationPlanRow("a2", "Bonds", Decimal("30.00"), "Safety"),
        ]
    if non_actions is None:
        non_actions = ["No panic selling", "No crypto futures", "No leverage"]

    risk_limits = {
        "mdd_max_annual": NumericPolicyField(
            field_id="mdd_max_annual",
            value=mdd_val,
            unit="%",
            calculation_basis="NAV",
            origin=NumericPolicyOrigin.AI_PROPOSAL,
            is_confirmed=True,
        ),
        "max_loss_per_trade": NumericPolicyField(
            field_id="max_loss_per_trade",
            value=tl_val,
            unit=tl_unit,
            calculation_basis="per_trade" if tl_unit != "not_applicable" else "not_applicable",
            origin=NumericPolicyOrigin.AI_PROPOSAL,
            assumptions="Buy and hold ETF index" if tl_unit == "not_applicable" else "",
            is_confirmed=True,
        ),
    }

    return InvestmentAxisDraft(
        draft_id="draft-axis-1",
        portfolio_id="port-test",
        essence_ref=ArtifactRef("doc-1", "note-1", "rev-1", "hash-1", "art-1"),
        context_ref="ctx-1",
        basic_policy="Disciplined long term growth",
        risk_limits=risk_limits,
        invest_targets=["Global Equities", "Government Bonds"],
        exclude_targets=["Penny stocks", "Cryptocurrency"],
        primary_methods=["Dollar Cost Averaging"],
        secondary_methods=["Quarterly Rebalancing"],
        investment_horizon="10 years",
        allocation_basis=AllocationBasis.ASSET_CLASS,
        allocation_rows=alloc_rows,
        rebalance_frequency="Quarterly",
        role_models=["John Bogle"],
        non_actions=non_actions,
        numeric_fields=risk_limits,
        revision=revision,
    )


def test_validate_axis_completeness_and_rules():
    # 1. Valid draft passes with zero issues
    draft = _make_sample_axis_draft()
    assert validate_axis_completeness(draft) == []

    # 2. MDD > 100% fails
    draft_bad_mdd = _make_sample_axis_draft(mdd_val=Decimal("150"))
    errs = validate_axis_completeness(draft_bad_mdd)
    assert any("MDD สูงสุดต่อปีต้องอยู่ระหว่าง 0% ถึง 100%" in e for e in errs)

    # 3. Not applicable trade limit passes without number
    draft_na_trade = _make_sample_axis_draft(tl_val=None, tl_unit="not_applicable")
    assert validate_axis_completeness(draft_na_trade) == []

    # 4. Negative allocation fails
    draft_neg_alloc = _make_sample_axis_draft(
        alloc_rows=[
            AllocationPlanRow("a1", "Equities", Decimal("110.00"), "Growth"),
            AllocationPlanRow("a2", "Bonds", Decimal("-10.00"), "Safety"),
        ]
    )
    errs = validate_axis_completeness(draft_neg_alloc)
    assert any("ต้องไม่ติดลบ" in e for e in errs)

    # 5. Duplicate allocation ID fails
    draft_dup_alloc = _make_sample_axis_draft(
        alloc_rows=[
            AllocationPlanRow("a1", "Equities", Decimal("50.00"), "Growth"),
            AllocationPlanRow("a1", "Bonds", Decimal("50.00"), "Safety"),
        ]
    )
    errs = validate_axis_completeness(draft_dup_alloc)
    assert any("allocation_id ซ้ำกัน" in e for e in errs)

    # 6. Non-actions fewer than 3 unique non-empty items fails
    draft_few_na = _make_sample_axis_draft(non_actions=["No panic", "   ", "No panic"])
    errs = validate_axis_completeness(draft_few_na)
    assert any("อย่างน้อย 3 ข้อ" in e for e in errs)


def test_investment_axis_confirm_checks_revision_readiness_and_provenance():
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock()
    id_gen = FakeIdGenerator()
    axis_gen = FakeAxisGenerator()

    service = InvestmentAxisService(
        uow_factory=uow_factory,
        axis_generator=axis_gen,
        clock=clock,
        id_gen=id_gen,
    )

    draft = _make_sample_axis_draft(revision=2)
    # Save draft and blocking context
    with uow_factory.open() as uow:
        uow.planning.save_axis_draft(draft)
        blocking_ctx = FinancialContextSnapshot(
            snapshot_id="ctx-blocking",
            portfolio_id="port-test",
            horizon_years=None,
            unknown_fields=["horizon_years"],
        )
        uow.planning.save_context_snapshot(update_readiness(blocking_ctx))
        uow.commit()

    # 1. Revision mismatch raises RevisionConflictError
    with pytest.raises(RevisionConflictError):
        service.confirm_axis(
            draft_id="draft-axis-1",
            expected_draft_revision=1,  # actual is 2
        )

    # 2. Financial context not ready raises AxisIncompleteError
    with pytest.raises(AxisIncompleteError, match="Financial context blocking"):
        service.confirm_axis(
            draft_id="draft-axis-1",
            expected_draft_revision=2,
        )

    # 3. Resolve financial context readiness and confirm successfully
    with uow_factory.open() as uow:
        ready_ctx = FinancialContextSnapshot(
            snapshot_id="ctx-ready",
            portfolio_id="port-test",
            horizon_years=Decimal("10"),
            emergency_reserves_amount=Decimal("300000"),
            unknown_fields=[],
        )
        uow.planning.save_context_snapshot(update_readiness(ready_ctx))
        uow.commit()

    receipt = service.confirm_axis(
        draft_id="draft-axis-1",
        expected_draft_revision=2,
    )
    assert receipt.status == "committed"

    # 4. Check confirmed axis retains AI_PROPOSAL provenance
    confirmed_axis = service.get_current_confirmed_axis("port-test")
    assert confirmed_axis is not None
    assert confirmed_axis["risk_limits"]["mdd_max_annual"]["origin"] == "ai_proposal"
