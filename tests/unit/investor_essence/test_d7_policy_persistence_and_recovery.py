"""Tests for Deliverable D7 (WP12):
- Policy and origin persistence in PortfolioState (WP12 tasks 1, 2, 11).
- Durable command receipts and idempotency in PortfolioTransactionStore (WP12 tasks 3, 4, 5).
- Runtime intent persistence and fault recovery / reconciliation (WP12 tasks 6, 7).
- Verifiable projection repair and evidence auditing (WP12 tasks 8, 9).
"""
from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from core.investor_essence.models import (
    AllocationBasis,
    AllocationMappingCell,
    ArtifactRef,
    BucketPlanDraft,
    BucketPlanStatus,
    PurposeBucketDraft,
)
from application.investor_essence.bucket_planning_service import BucketPlanningService
from application.investor_essence.dto import (
    ApplyAllocationCommand,
)
from application.investor_essence.errors import (
    IdempotencyConflictError,
    PortfolioConflictError,
    ResourceNotFoundError,
)
from application.investor_essence.testing.fakes import (
    FakeBucketGenerator,
    FakeClock,
    FakeIdGenerator,
    FakeInvestorRuntimeUowFactory,
    FakePortfolioPlanningAdapter,
)
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.models import AllocationTarget, Holding, PortfolioState
from tools.portfolio.services.portfolio_state_service import PortfolioStateService
from tools.portfolio.transaction_store import PortfolioTransactionStore
from tools.portfolio.transactional_repository import (
    TransactionalPortfolioRepository,
)
from tools.investor_essence.adapters.portfolio_planning_adapter import (
    PortfolioPlanningAdapter,
)


class FakeUnderlyingRepo:
    def __init__(self, state: PortfolioState):
        self._state = state

    def unit_of_work(self, portfolio_id: str):
        repo = self
        class FakeUow:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def load_state(self):
                return repo._state
            def read_trade_log_locked(self):
                return []
            def commit(self, state, ledger_change=None):
                repo._state = state
            def rollback(self):
                pass
        return FakeUow()

    def load_state(self, portfolio_id: str):
        return self._state

    def read_trade_log(self, portfolio_id: str, symbol=None):
        return []

    def backup_and_reset_clean_slate(self, portfolio_id: str):
        return self._state


def _create_test_state() -> PortfolioState:
    return PortfolioState(
        last_updated="2026-10-10T12:00:00Z",
        name="test_portfolio",
        holdings=[
            Holding(
                symbol="AAPL",
                asset_type="Stock",
                units=10.0,
                avg_cost_thb=5000.0,
                market_value_thb=60000.0,
                bucket_id="b_old_tech",
            ),
            Holding(
                symbol="BND",
                asset_type="Bond",
                units=50.0,
                avg_cost_thb=3000.0,
                market_value_thb=40000.0,
                bucket_id="b_old_safe",
            ),
        ],
        allocation_targets=[
            AllocationTarget(bucket_id="b_old_tech", name="Tech", target_percent=60.0),
            AllocationTarget(bucket_id="b_old_safe", name="Safe", target_percent=40.0),
        ],
    )


def test_wp12_policy_persistence_and_unmodified_holdings(tmp_path: Path):
    """WP12 Tasks 1 & 11: Apply writes allocation_policy and allocation_plan_origin, without mutating units, costs, or ledger."""
    store = PortfolioTransactionStore(runtime_root=tmp_path / "runtime")
    underlying = FakeUnderlyingRepo(_create_test_state())
    repo = TransactionalPortfolioRepository(underlying, store=store)
    adapter = PortfolioPlanningAdapter(repository=repo)

    cmd = ApplyAllocationCommand(
        command_id="cmd-persist-1",
        request_hash="hash-req-1",
        portfolio_id="default",
        expected_checkpoint_sequence=0,
        expected_checkpoint_state_hash="",
        target_rows=[
            {"bucket_id": "b_growth", "name": "เติบโตระยะยาว", "target_percent": 70.0, "color": "#3B82F6"},
            {"bucket_id": "b_safety", "name": "สำรองฉุกเฉิน", "target_percent": 30.0, "color": "#10B981"},
        ],
        mapping_weights=[
            {"axis_allocation_id": "stocks", "bucket_id": "b_growth", "portfolio_weight_percent": 70.0},
            {"axis_allocation_id": "bonds", "bucket_id": "b_safety", "portfolio_weight_percent": 30.0},
        ],
        remapping=[
            {"old_bucket_id": "b_old_tech", "target_bucket_id": "b_growth"},
            {"old_bucket_id": "b_old_safe", "target_bucket_id": "b_safety"},
        ],
        accepted_policy_snapshot={
            "draft_id": "draft-policy-123",
            "allocation_basis": "purpose",
            "purpose_buckets": [
                {"bucket_id": "b_growth", "name": "เติบโตระยะยาว", "target_percent": "70.00"},
                {"bucket_id": "b_safety", "name": "สำรองฉุกเฉิน", "target_percent": "30.00"},
            ],
        },
        source_refs={
            "draft_id": "draft-policy-123",
            "essence_ref": "ess-hash-abc",
            "axis_ref": "axis-hash-xyz",
            "context_ref": "ctx-ref-001",
        },
        idempotency_key="idemp-key-wp12",
    )

    receipt = adapter.apply(cmd)
    assert receipt.canonical_status == "committed"

    # Reload state from store
    saved_state = repo.load_state("default")

    # 1. Allocation targets updated
    assert len(saved_state.allocation_targets) == 2
    assert saved_state.allocation_targets[0].bucket_id == "b_growth"
    assert saved_state.allocation_targets[0].target_percent == 70.0

    # 2. Holdings remapped
    aapl = next(h for h in saved_state.holdings if h.symbol == "AAPL")
    bnd = next(h for h in saved_state.holdings if h.symbol == "BND")
    assert aapl.bucket_id == "b_growth"
    assert bnd.bucket_id == "b_safety"

    # 3. Holding units, costs, and market values are untouched (WP12 task 11)
    assert aapl.units == 10.0
    assert aapl.avg_cost_thb == 5000.0
    assert aapl.market_value_thb == 60000.0
    assert bnd.units == 50.0

    # 4. allocation_policy persisted
    assert saved_state.allocation_policy is not None
    assert saved_state.allocation_policy["draft_id"] == "draft-policy-123"
    assert saved_state.allocation_policy["allocation_basis"] == "purpose"
    assert len(saved_state.allocation_policy["purpose_buckets"]) == 2

    # 5. allocation_plan_origin persisted
    assert saved_state.allocation_plan_origin is not None
    origin = saved_state.allocation_plan_origin
    assert origin["applied_draft_id"] == "draft-policy-123"
    assert origin["command_id"] == "cmd-persist-1"
    assert origin["is_modified_since_apply"] is False


def test_wp12_manual_edit_targets_marks_modified_since_apply(tmp_path: Path):
    """WP12 Task 2: Manual edit of targets updates is_modified_since_apply to True."""
    store = PortfolioTransactionStore(runtime_root=tmp_path / "runtime")
    state = _create_test_state()
    state.allocation_plan_origin = {
        "applied_draft_id": "draft-policy-1",
        "command_id": "cmd-1",
        "is_modified_since_apply": False,
    }
    underlying = FakeUnderlyingRepo(state)
    repo = TransactionalPortfolioRepository(underlying, store=store)
    state_service = PortfolioStateService(repo=repo)

    # Manually edit targets
    new_targets = [
        AllocationTarget(bucket_id="b_growth", name="Growth", target_percent=80.0),
        AllocationTarget(bucket_id="b_safety", name="Safety", target_percent=20.0),
    ]
    updated = state_service.structured_upsert_allocation_targets(new_targets, portfolio_id="default")

    assert updated.allocation_plan_origin is not None
    assert updated.allocation_plan_origin["is_modified_since_apply"] is True


def test_wp12_durable_receipt_idempotency_and_conflict(tmp_path: Path):
    """WP12 Tasks 3, 4, 5: Durable receipt check before checkpoint sequence conflict."""
    store = PortfolioTransactionStore(runtime_root=tmp_path / "runtime")
    underlying = FakeUnderlyingRepo(_create_test_state())
    repo = TransactionalPortfolioRepository(underlying, store=store)
    adapter = PortfolioPlanningAdapter(repository=repo)

    cmd = ApplyAllocationCommand(
        command_id="cmd-idem-1",
        request_hash="hash-req-idem",
        portfolio_id="default",
        expected_checkpoint_sequence=0,
        expected_checkpoint_state_hash="",
        target_rows=[
            {"bucket_id": "b_growth", "name": "Growth", "target_percent": 100.0, "color": "#3B82F6"},
        ],
        mapping_weights=[],
        remapping=[],
        accepted_policy_snapshot={"draft_id": "draft-1", "allocation_basis": "purpose"},
        source_refs={"draft_id": "draft-1"},
        idempotency_key="idemp-key-shared",
    )

    # 1. Apply for the first time
    receipt1 = adapter.apply(cmd)
    assert receipt1.canonical_status == "committed"

    # Advance the sequence in store by appending an unrelated state event
    store.append_event("default", "unrelated_event", {"info": "ping"})

    # 2. WP12 Task 5: Replaying with same idempotency key and same request_hash succeeds with duplicate_reused
    # even though checkpoint sequence has advanced!
    receipt_replay = adapter.apply(cmd)
    assert receipt_replay.canonical_status == "duplicate_reused"
    assert receipt_replay.command_id == receipt1.command_id

    # 3. Calling with same idempotency key but different request_hash raises IdempotencyConflictError
    altered_cmd = ApplyAllocationCommand(
        command_id="cmd-idem-2",
        request_hash="different-hash",
        portfolio_id="default",
        expected_checkpoint_sequence=0,
        expected_checkpoint_state_hash="",
        target_rows=[
            {"bucket_id": "b_growth", "name": "Growth", "target_percent": 100.0, "color": "#3B82F6"},
        ],
        mapping_weights=[],
        remapping=[],
        accepted_policy_snapshot={"draft_id": "draft-1", "allocation_basis": "purpose"},
        source_refs={"draft_id": "draft-1"},
        idempotency_key="idemp-key-shared",
    )
    with pytest.raises(IdempotencyConflictError) as exc_info:
        adapter.apply(altered_cmd)
    assert exc_info.value.key == "idemp-key-shared"


def test_wp12_reconcile_receipt_when_runtime_save_failed():
    """WP12 Tasks 6 & 7: If portfolio committed but runtime save failed, apply_bucket_plan reconciles without duplicate commit."""
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
        PurposeBucketDraft(
            bucket_id="b_main",
            name="พอร์ตหลัก",
            role="ลงทุน",
            color="#3B82F6",
            target_percent=Decimal("100.00"),
        )
    ]
    draft = BucketPlanDraft(
        draft_id="draft-reconcile-1",
        portfolio_id="port-alpha",
        essence_ref=ArtifactRef("doc_ess", "note_ess", "rev_ess", "hash_ess", "set_hash_ess"),
        axis_ref=ArtifactRef("doc_axis", "note_axis", "rev_axis", "hash_axis", "set_hash_axis"),
        context_ref="ctx_ref_1",
        portfolio_checkpoint={"sequence": 1, "state_hash": "chk-1"},
        purpose_buckets=buckets,
        allocation_basis=AllocationBasis.PURPOSE,
        mapping_weights=[],
        constraints=[],
        remapping=[],
        status=BucketPlanStatus.DRAFT,
        revision=1,
    )

    with uow_factory.open() as uow:
        uow.planning.save_bucket_draft(draft)
        uow.commit()

    # Apply 1: Normal commit
    receipt1 = service.apply_bucket_plan(
        draft_id=draft.draft_id,
        idempotency_key="recon-key-1",
        expected_draft_revision=1,
        expected_checkpoint_sequence=1,
    )
    assert receipt1.canonical_status == "committed"
    assert len(portfolio_adapter.applied_commands) == 1

    # Simulate scenario where runtime command receipt was missing (e.g. process crash before step 7),
    # but intent was saved in step 5 and portfolio store has the durable receipt.
    scope = f"portfolio:{draft.portfolio_id}"
    with uow_factory.open() as uow:
        uow.intents.command_receipts.pop(f"{scope}:apply_bucket_plan:recon-key-1", None)
        uow.commit()

    # Call apply_bucket_plan again: Reconciler should detect prior intent, fetch receipt from adapter,
    # and finalize draft without issuing a second command to portfolio adapter!
    reconciled_receipt = service.apply_bucket_plan(
        draft_id=draft.draft_id,
        idempotency_key="recon-key-1",
        expected_draft_revision=1,
        expected_checkpoint_sequence=1,
    )

    assert reconciled_receipt.command_id == receipt1.command_id
    assert len(portfolio_adapter.applied_commands) == 1  # NO duplicate commit to portfolio adapter!


def test_wp12_repair_projection_with_evidence(tmp_path: Path):
    """WP12 Tasks 8 & 9: repair_projection verifies command ID evidence in transaction store."""
    store = PortfolioTransactionStore(runtime_root=tmp_path / "runtime")
    underlying = FakeUnderlyingRepo(_create_test_state())
    repo = TransactionalPortfolioRepository(underlying, store=store)
    adapter = PortfolioPlanningAdapter(repository=repo)

    cmd = ApplyAllocationCommand(
        command_id="cmd-repair-valid",
        request_hash="hash-req-repair",
        portfolio_id="default",
        expected_checkpoint_sequence=0,
        expected_checkpoint_state_hash="",
        target_rows=[
            {"bucket_id": "b_main", "name": "Main", "target_percent": 100.0, "color": "#3B82F6"},
        ],
        mapping_weights=[],
        remapping=[],
        accepted_policy_snapshot={"draft_id": "draft-repair-1", "allocation_basis": "purpose"},
        source_refs={"draft_id": "draft-repair-1"},
        idempotency_key="repair-idem-key",
    )

    receipt = adapter.apply(cmd)

    # 1. Valid command_id: projection repaired cleanly
    repaired = adapter.repair_projection("default", "cmd-repair-valid")
    assert repaired.command_id == "cmd-repair-valid"
    assert repaired.canonical_status == "committed"
    assert repaired.projection_status == "ready"

    # 2. Unknown command_id: rejected with ResourceNotFoundError (not falsely reporting repaired)
    with pytest.raises(ResourceNotFoundError):
        adapter.repair_projection("default", "non-existent-cmd-id")
