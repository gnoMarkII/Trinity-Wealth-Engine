"""Unit tests for Deliverable D6:
- WP10: Matrix weights, purpose buckets, remappings, and inflow routing completeness validation (AC12, AC13, AC14).
- WP11: Preview metadata and Apply revision/checkpoint conflict consistency (AC16, AC17, AC19, AC20).
"""
from decimal import Decimal
from typing import Any, Dict, List, Optional
import pytest

from core.investor_essence.bucket_validation import validate_bucket_plan_completeness
from core.investor_essence.models import (
    AllocationBasis,
    AllocationMappingCell,
    ArtifactRef,
    BucketPlanDraft,
    BucketPlanStatus,
    BucketRemappingItem,
    PurposeBucketDraft,
)
from application.investor_essence.bucket_planning_service import BucketPlanningService
from application.investor_essence.dto import (
    BucketPlanProposal,
    PurposeBucketProposal,
)
from application.investor_essence.errors import (
    IdempotencyConflictError,
    PortfolioConflictError,
    RevisionConflictError,
    ValidationFailedError,
)
from application.investor_essence.testing.fakes import (
    FakeBucketGenerator,
    FakeClock,
    FakeIdGenerator,
    FakeInvestorRuntimeUowFactory,
    FakePortfolioPlanningAdapter,
)


def _make_bucket(
    bucket_id: str,
    name: str,
    role: str,
    target_percent: Decimal,
    color: str = "#3B82F6",
    cash_inflow_routing_pct: Optional[str] = None,
    **kwargs: Any,
) -> PurposeBucketDraft:
    return PurposeBucketDraft(
        bucket_id=bucket_id,
        name=name,
        role=role,
        color=color,
        target_percent=target_percent,
        cash_inflow_routing_pct=cash_inflow_routing_pct,
        **kwargs,
    )


def _build_test_draft(
    purpose_buckets: List[PurposeBucketDraft],
    mapping_weights: List[AllocationMappingCell] = None,
    remapping: List[BucketRemappingItem] = None,
    revision: int = 1,
    status: BucketPlanStatus = BucketPlanStatus.DRAFT,
) -> BucketPlanDraft:
    return BucketPlanDraft(
        draft_id="draft-test-1",
        portfolio_id="port-alpha",
        essence_ref=ArtifactRef("doc_ess", "note_ess", "rev_ess", "hash_ess", "set_hash_ess"),
        axis_ref=ArtifactRef("doc_axis", "note_axis", "rev_axis", "hash_axis", "set_hash_axis"),
        context_ref="ctx_snapshot_1",
        portfolio_checkpoint={"sequence": 1, "state_hash": "chk-hash-1"},
        purpose_buckets=purpose_buckets,
        allocation_basis=AllocationBasis.PURPOSE,
        mapping_weights=mapping_weights or [],
        constraints=[],
        remapping=remapping or [],
        status=status,
        revision=revision,
        created_at_iso="2026-10-10T10:00:00Z",
    )


# ==============================================================================
# WP10: Matrix & Inflow Routing Validation Tests (AC12, AC13, AC14)
# ==============================================================================

def test_validate_bucket_plan_ac12_matrix_column_mismatch_rejected():
    """AC12: Purpose buckets are 90/10, but matrix column weights sum to 70/30 -> rejected."""
    buckets = [
        _make_bucket("b_growth", "เติบโตระยะยาว", "เป้าหมายเกษียณ", Decimal("90.00"), color="#3B82F6"),
        _make_bucket("b_safety", "สำรองฉุกเฉิน", "ความมั่นคง", Decimal("10.00"), color="#10B981"),
    ]
    # Matrix cells sum to 100%, but growth is 70% and safety is 30% (mismatch with 90/10)
    matrix = [
        AllocationMappingCell(
            axis_allocation_id="axis_stocks",
            bucket_id="b_growth",
            portfolio_weight_percent=Decimal("70.00"),
        ),
        AllocationMappingCell(
            axis_allocation_id="axis_bonds",
            bucket_id="b_safety",
            portfolio_weight_percent=Decimal("30.00"),
        ),
    ]
    draft = _build_test_draft(purpose_buckets=buckets, mapping_weights=matrix)

    issues = validate_bucket_plan_completeness(draft)

    assert len(issues) > 0
    growth_err = next((e for e in issues if "b_growth" in e and "70.00%" in e and "90.00%" in e), None)
    assert growth_err is not None, f"Expected column mismatch error for b_growth, got: {issues}"
    safety_err = next((e for e in issues if "b_safety" in e and "30.00%" in e and "10.00%" in e), None)
    assert safety_err is not None, f"Expected column mismatch error for b_safety, got: {issues}"


def test_validate_bucket_plan_matrix_total_not_100_rejected():
    """Matrix cells that do not sum to 100% are rejected."""
    buckets = [
        _make_bucket("b_growth", "เติบโต", "เกษียณ", Decimal("70.00")),
        _make_bucket("b_safety", "สำรอง", "ฉุกเฉิน", Decimal("30.00")),
    ]
    # Matrix cells sum to 95.00%
    matrix = [
        AllocationMappingCell(
            axis_allocation_id="axis_stocks",
            bucket_id="b_growth",
            portfolio_weight_percent=Decimal("65.00"),
        ),
        AllocationMappingCell(
            axis_allocation_id="axis_bonds",
            bucket_id="b_safety",
            portfolio_weight_percent=Decimal("30.00"),
        ),
    ]
    draft = _build_test_draft(purpose_buckets=buckets, mapping_weights=matrix)

    issues = validate_bucket_plan_completeness(draft)
    assert any("ผลรวมน้ำหนักในเมทริกซ์การจัดสรร (95.00%) ต้องเท่ากับ 100%" in e for e in issues)


def test_validate_bucket_plan_matrix_row_mismatch_rejected():
    """Matrix rows that do not sum to confirmed Axis targets are rejected."""
    buckets = [
        _make_bucket("b_growth", "เติบโต", "เกษียณ", Decimal("60.00")),
        _make_bucket("b_safety", "สำรอง", "ฉุกเฉิน", Decimal("40.00")),
    ]
    # Matrix matches bucket targets: growth=60%, safety=40%
    # But row axis_stocks has 50% and axis_bonds has 50%
    matrix = [
        AllocationMappingCell(
            axis_allocation_id="axis_stocks",
            bucket_id="b_growth",
            portfolio_weight_percent=Decimal("50.00"),
        ),
        AllocationMappingCell(
            axis_allocation_id="axis_bonds",
            bucket_id="b_growth",
            portfolio_weight_percent=Decimal("10.00"),
        ),
        AllocationMappingCell(
            axis_allocation_id="axis_bonds",
            bucket_id="b_safety",
            portfolio_weight_percent=Decimal("40.00"),
        ),
    ]
    draft = _build_test_draft(purpose_buckets=buckets, mapping_weights=matrix)

    # Axis target specifies axis_stocks=70%, axis_bonds=30%
    axis_targets = {
        "axis_stocks": Decimal("70.00"),
        "axis_bonds": Decimal("30.00"),
    }
    issues = validate_bucket_plan_completeness(draft, axis_allocation_targets=axis_targets)

    assert any("เมทริกซ์แถวหมวด 'axis_stocks'" in e for e in issues)
    assert any("เมทริกซ์แถวหมวด 'axis_bonds'" in e for e in issues)


def test_validate_bucket_plan_ac13_inflow_routing_exceeding_100_rejected():
    """AC13: Inflow DCA routing total of 180% is rejected."""
    buckets = [
        _make_bucket("b_growth", "เติบโต", "เกษียณ", Decimal("80.00"), cash_inflow_routing_pct="100.00"),
        _make_bucket("b_safety", "สำรอง", "ฉุกเฉิน", Decimal("20.00"), cash_inflow_routing_pct="80.00"),
    ]
    draft = _build_test_draft(purpose_buckets=buckets)

    issues = validate_bucket_plan_completeness(draft)
    assert any("ผลรวมสัดส่วนเงินเข้าใหม่ (DCA) (180.00%) เกิน 100% ของงบลงทุน" in e for e in issues)


def test_validate_bucket_plan_ac14_duplicate_bucket_ids_rejected():
    """AC14: Duplicate bucket IDs are rejected."""
    buckets = [
        _make_bucket("b_dup", "Bucket 1", "Role 1", Decimal("50.00")),
        _make_bucket("b_dup", "Bucket 2", "Role 2", Decimal("50.00")),
    ]
    draft = _build_test_draft(purpose_buckets=buckets)

    issues = validate_bucket_plan_completeness(draft)
    assert any("รหัส 'b_dup' ซ้ำกับ Bucket อื่น" in e for e in issues)


def test_validate_bucket_plan_ac14_dangling_remapping_rejected():
    """AC14: Remapping to a nonexistent bucket ID is rejected."""
    buckets = [
        _make_bucket("b_main", "Bucket หลัก", "ลงทุน", Decimal("100.00")),
    ]
    remapping = [
        BucketRemappingItem(
            old_bucket_id="old_bucket_legacy",
            target_bucket_id="non_existent_bucket_id",
            affected_holding_count=2,
        )
    ]
    draft = _build_test_draft(purpose_buckets=buckets, remapping=remapping)

    issues = validate_bucket_plan_completeness(draft)
    assert any("ปลายทาง 'non_existent_bucket_id' ไม่มีอยู่ในเป้าหมาย Buckets" in e for e in issues)


def test_validate_bucket_plan_ac14_remapping_to_none_unassigned_allowed():
    """AC14: Remapping to None (Unassigned) is allowed and valid."""
    buckets = [
        _make_bucket("b_main", "Bucket หลัก", "ลงทุน", Decimal("100.00")),
    ]
    remapping = [
        BucketRemappingItem(
            old_bucket_id="old_bucket_legacy",
            target_bucket_id=None,  # Explicit Unassigned
            affected_holding_count=1,
        )
    ]
    draft = _build_test_draft(purpose_buckets=buckets, remapping=remapping)

    issues = validate_bucket_plan_completeness(draft)
    assert issues == []


# ==============================================================================
# WP11: Preview and Apply Consistency Tests (AC16, AC17, AC19, AC20)
# ==============================================================================

def test_preview_bucket_plan_generates_id_and_tracks_revision():
    """WP11 / AC16: preview_bucket_plan returns preview_id, draft_id, draft_revision, and aggregates issues."""
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock("2026-10-10T12:00:00Z")
    id_gen = FakeIdGenerator()
    portfolio_adapter = FakePortfolioPlanningAdapter()
    bucket_gen = FakeBucketGenerator()

    service = BucketPlanningService(
        uow_factory=uow_factory,
        bucket_generator=bucket_gen,
        portfolio_port=portfolio_adapter,
        clock=clock,
        id_gen=id_gen,
    )

    # Create draft with mismatched matrix to verify issue aggregation
    buckets = [
        _make_bucket("b_growth", "เติบโต", "เกษียณ", Decimal("90.00")),
        _make_bucket("b_safety", "สำรอง", "ฉุกเฉิน", Decimal("10.00")),
    ]
    matrix = [
        AllocationMappingCell(
            axis_allocation_id="axis_stocks",
            bucket_id="b_growth",
            portfolio_weight_percent=Decimal("70.00"),
        ),
        AllocationMappingCell(
            axis_allocation_id="axis_bonds",
            bucket_id="b_safety",
            portfolio_weight_percent=Decimal("30.00"),
        ),
    ]
    draft = _build_test_draft(purpose_buckets=buckets, mapping_weights=matrix, revision=3)

    with uow_factory.open() as uow:
        uow.planning.save_bucket_draft(draft)
        uow.commit()

    preview = service.preview_bucket_plan(draft.draft_id)

    assert preview.preview_id.startswith("prev_")
    assert preview.draft_id == draft.draft_id
    assert preview.draft_revision == 3
    # Check that domain issues were captured in preview
    assert len(preview.issues) > 0
    assert any("70.00%" in iss and "90.00%" in iss for iss in preview.issues)


def test_apply_bucket_plan_ac16_checkpoint_sequence_conflict_raises_409():
    """AC16: When portfolio checkpoint advances after preview/planning, Apply raises PortfolioConflictError (HTTP 409)."""
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock("2026-10-10T12:00:00Z")
    id_gen = FakeIdGenerator()

    # Portfolio sequence is currently 2
    portfolio_adapter = FakePortfolioPlanningAdapter(checkpoint_sequence=2)
    bucket_gen = FakeBucketGenerator()

    service = BucketPlanningService(
        uow_factory=uow_factory,
        bucket_generator=bucket_gen,
        portfolio_port=portfolio_adapter,
        clock=clock,
        id_gen=id_gen,
    )

    buckets = [
        _make_bucket("b_main", "พอร์ตหลัก", "ลงทุน", Decimal("100.00"))
    ]
    draft = _build_test_draft(purpose_buckets=buckets, revision=1)

    with uow_factory.open() as uow:
        uow.planning.save_bucket_draft(draft)
        uow.commit()

    # Client expects checkpoint sequence 1 (stale), but portfolio is now at sequence 2
    with pytest.raises(PortfolioConflictError) as exc_info:
        service.apply_bucket_plan(
            draft_id=draft.draft_id,
            expected_checkpoint_sequence=1,
            expected_draft_revision=1,
        )

    assert exc_info.value.expected_seq == 1
    assert exc_info.value.actual_seq == 2


def test_apply_bucket_plan_ac16_ac17_draft_revision_conflict_raises_409():
    """AC16 & AC17: When draft revision differs from expected_draft_revision, Apply raises RevisionConflictError (HTTP 409)."""
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock("2026-10-10T12:00:00Z")
    id_gen = FakeIdGenerator()
    portfolio_adapter = FakePortfolioPlanningAdapter(checkpoint_sequence=1)
    bucket_gen = FakeBucketGenerator()

    service = BucketPlanningService(
        uow_factory=uow_factory,
        bucket_generator=bucket_gen,
        portfolio_port=portfolio_adapter,
        clock=clock,
        id_gen=id_gen,
    )

    buckets = [
        _make_bucket("b_main", "พอร์ตหลัก", "ลงทุน", Decimal("100.00"))
    ]
    draft = _build_test_draft(purpose_buckets=buckets, revision=2)

    with uow_factory.open() as uow:
        uow.planning.save_bucket_draft(draft)
        uow.commit()

    # User previewed revision 1, but draft was updated to revision 2
    with pytest.raises(RevisionConflictError) as exc_info:
        service.apply_bucket_plan(
            draft_id=draft.draft_id,
            expected_draft_revision=1,
            expected_checkpoint_sequence=1,
        )

    assert exc_info.value.expected_revision == 1
    assert exc_info.value.actual_revision == 2


def test_apply_bucket_plan_validation_failed_raises_422():
    """AC17: Apply rejects a draft with invalid configuration (e.g. DCA > 100%)."""
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock("2026-10-10T12:00:00Z")
    id_gen = FakeIdGenerator()
    portfolio_adapter = FakePortfolioPlanningAdapter(checkpoint_sequence=1)
    bucket_gen = FakeBucketGenerator()

    service = BucketPlanningService(
        uow_factory=uow_factory,
        bucket_generator=bucket_gen,
        portfolio_port=portfolio_adapter,
        clock=clock,
        id_gen=id_gen,
    )

    # Invalid: DCA 150%
    buckets = [
        _make_bucket("b_main", "พอร์ตหลัก", "ลงทุน", Decimal("100.00"), cash_inflow_routing_pct="150.00")
    ]
    draft = _build_test_draft(purpose_buckets=buckets, revision=1)

    with uow_factory.open() as uow:
        uow.planning.save_bucket_draft(draft)
        uow.commit()

    with pytest.raises(ValidationFailedError) as exc_info:
        service.apply_bucket_plan(
            draft_id=draft.draft_id,
            expected_draft_revision=1,
            expected_checkpoint_sequence=1,
        )

    assert "เกิน 100% ของงบลงทุน" in str(exc_info.value)


def test_apply_bucket_plan_success_and_idempotency_ac19_ac20():
    """AC19 & AC20: Successful apply marks draft APPLIED, caches receipt, and prevents duplicate commits."""
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock("2026-10-10T12:00:00Z")
    id_gen = FakeIdGenerator()
    portfolio_adapter = FakePortfolioPlanningAdapter(checkpoint_sequence=1)
    bucket_gen = FakeBucketGenerator()

    service = BucketPlanningService(
        uow_factory=uow_factory,
        bucket_generator=bucket_gen,
        portfolio_port=portfolio_adapter,
        clock=clock,
        id_gen=id_gen,
    )

    buckets = [
        _make_bucket("b_growth", "เติบโต", "เกษียณ", Decimal("70.00"), cash_inflow_routing_pct="70.00"),
        _make_bucket("b_safety", "สำรอง", "ฉุกเฉิน", Decimal("30.00"), cash_inflow_routing_pct="30.00"),
    ]
    matrix = [
        AllocationMappingCell(
            axis_allocation_id="axis_stocks",
            bucket_id="b_growth",
            portfolio_weight_percent=Decimal("70.00"),
        ),
        AllocationMappingCell(
            axis_allocation_id="axis_bonds",
            bucket_id="b_safety",
            portfolio_weight_percent=Decimal("30.00"),
        ),
    ]
    draft = _build_test_draft(purpose_buckets=buckets, mapping_weights=matrix, revision=1)

    with uow_factory.open() as uow:
        uow.planning.save_bucket_draft(draft)
        uow.commit()

    # 1. Apply successfully
    receipt = service.apply_bucket_plan(
        draft_id=draft.draft_id,
        idempotency_key="idemp-key-123",
        expected_draft_revision=1,
        expected_checkpoint_sequence=1,
    )

    assert receipt.canonical_status == "committed"
    assert receipt.applied_sequence == 2

    # Check draft state in UoW: should be APPLIED and revision incremented to 2
    with uow_factory.open() as uow:
        saved_draft = uow.planning.get_bucket_draft(draft.draft_id)
        assert saved_draft.status == BucketPlanStatus.APPLIED
        assert saved_draft.revision == 2

    # Portfolio adapter received exactly 1 apply command
    assert len(portfolio_adapter.applied_commands) == 1

    # 2. AC19: Replay with same idempotency key and same expected revision & checkpoint sequence returns cached receipt
    cached_receipt = service.apply_bucket_plan(
        draft_id=draft.draft_id,
        idempotency_key="idemp-key-123",
        expected_draft_revision=1,
        expected_checkpoint_sequence=1,
    )
    assert cached_receipt.command_id == receipt.command_id
    assert cached_receipt.applied_sequence == receipt.applied_sequence
    # Still only 1 call reached portfolio adapter (no duplicate commit)
    assert len(portfolio_adapter.applied_commands) == 1

    # 3. AC20: Reusing same idempotency key with different request parameters raises IdempotencyConflictError
    with pytest.raises(IdempotencyConflictError) as exc_info:
        service.apply_bucket_plan(
            draft_id=draft.draft_id,
            idempotency_key="idemp-key-123",
            expected_draft_revision=2,  # Different payload hash
            expected_checkpoint_sequence=1,
        )
    assert exc_info.value.key == "idemp-key-123"
