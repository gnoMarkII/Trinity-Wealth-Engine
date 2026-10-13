"""Unit tests for Deliverable D2 (WP02 + WP03):
- Typed contracts & Decimal 2dp normalization in compute_semantic_hash
- Immutable history in investor_confirmed_snapshots
- Elimination of latest-draft recovery forgery
- Downstream lineage reading confirmed snapshots only
"""
from decimal import Decimal
import json
import sqlite3
import pytest

from core.investor_essence.models import (
    AllocationBasis,
    AllocationPlanRow,
    ArtifactRef,
    ConfirmedSnapshotRecord,
    FinancialFact,
    GoalFundingLink,
    HoldingPlanningSnapshot,
    InvestmentAxisDraft,
    NumericPolicyField,
    NumericPolicyOrigin,
    SourceRef,
    SourceRefKind,
    ValidationIssue,
    compute_semantic_hash,
)
from core.investor_essence.investment_axis import build_confirmed_axis
from api.db.connection import init_schema
from api.db.repositories.investor_essence_repository import SqlitePlanningRepository


def test_compute_semantic_hash_determinism_and_normalization():
    # 1. Ephemeral metadata like request_id, idempotency_key, timestamp ignored
    payload_a = {
        "basic_policy": "Growth",
        "allocation_rows": [
            {"category": "Equities", "target_percent": Decimal("70.00")},
        ],
        "request_id": "req-123",
        "idempotency_key": "idem-abc",
        "timestamp": "2026-10-10T12:00:00Z",
    }
    payload_b = {
        "basic_policy": "Growth",
        "allocation_rows": [
            # Decimal("70") normalizes to "70.00"
            {"category": "Equities", "target_percent": Decimal("70")},
        ],
        "request_id": "different-req",
        "idempotency_key": "different-key",
        "timestamp": "2026-10-10T14:30:00Z",
    }
    hash_a = compute_semantic_hash(payload_a)
    hash_b = compute_semantic_hash(payload_b)
    assert hash_a == hash_b

    # 2. Meaningful changes alter hash
    payload_c = {
        "basic_policy": "Growth",
        "allocation_rows": [
            {"category": "Equities", "target_percent": Decimal("75.00")},
        ],
    }
    hash_c = compute_semantic_hash(payload_c)
    assert hash_a != hash_c


def test_axis_content_hash_mutates_on_allocation_change():
    rf = {
        "max_drawdown_percent": NumericPolicyField(
            field_id="max_drawdown_percent",
            value=Decimal("15.00"),
            unit="percent",
            calculation_basis="annual_nav_drawdown",
            origin=NumericPolicyOrigin.USER_INPUT,
            is_confirmed=True,
        ),
        "max_single_loss_percent": NumericPolicyField(
            field_id="max_single_loss_percent",
            value=Decimal("2.00"),
            unit="percent",
            calculation_basis="portfolio_nav_at_entry",
            origin=NumericPolicyOrigin.USER_INPUT,
            is_confirmed=True,
        ),
    }
    non_actions = [
        "จะไม่ใช้ Leverage หรือกู้ยืมเงินมาลงทุนเด็ดขาด",
        "จะไม่ซื้อหุ้นตามกระแสโซเชียลมีเดียโดยไม่ศึกษาข้อมูล",
        "จะไม่ขายหุ้นเพราะความตื่นตระหนกของตลาดระยะสั้น",
    ]
    draft1 = InvestmentAxisDraft(
        draft_id="d1",
        portfolio_id="p1",
        essence_ref=ArtifactRef("doc", "n1", "1", "hash1", "hash1"),
        context_ref="ctx1",
        basic_policy="Policy A",
        risk_limits=rf,
        invest_targets=["Equities"],
        exclude_targets=["Derivatives"],
        primary_methods=["DCA"],
        secondary_methods=["Rebalancing"],
        investment_horizon="10y",
        allocation_basis=AllocationBasis.PURPOSE,
        allocation_rows=[
            AllocationPlanRow("r1", "Core", Decimal("70.00"), "Growth"),
            AllocationPlanRow("r2", "Buffer", Decimal("30.00"), "Safety"),
        ],
        rebalance_frequency="Annual",
        role_models=["Warren Buffett"],
        non_actions=non_actions,
        numeric_fields=rf,
        revision=1,
        created_at_iso="2026-10-10T10:00:00Z",
    )
    conf1 = build_confirmed_axis("conf1", draft1, "2026-10-10T12:00:00Z")

    # Change allocation target_percent
    draft2 = InvestmentAxisDraft(
        draft_id="d2",
        portfolio_id="p1",
        essence_ref=ArtifactRef("doc", "n1", "1", "hash1", "hash1"),
        context_ref="ctx1",
        basic_policy="Policy A",
        risk_limits=rf,
        invest_targets=["Equities"],
        exclude_targets=["Derivatives"],
        primary_methods=["DCA"],
        secondary_methods=["Rebalancing"],
        investment_horizon="10y",
        allocation_basis=AllocationBasis.PURPOSE,
        allocation_rows=[
            AllocationPlanRow("r1", "Core", Decimal("60.00"), "Growth"),
            AllocationPlanRow("r2", "Buffer", Decimal("40.00"), "Safety"),
        ],
        rebalance_frequency="Annual",
        role_models=["Warren Buffett"],
        non_actions=non_actions,
        numeric_fields=rf,
        revision=1,
        created_at_iso="2026-10-10T10:00:00Z",
    )
    conf2 = build_confirmed_axis("conf2", draft2, "2026-10-10T12:00:00Z")

    assert conf1.content_hash != conf2.content_hash


def test_sqlite_repository_confirmed_snapshots_history_and_no_draft_fallback():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_schema(conn)
    repo = SqlitePlanningRepository(conn)

    # 1. Initially no confirmed snapshot exists
    assert repo.get_confirmed_snapshot("portfolio:p1", "investment_axis") is None

    # 2. Save an unconfirmed draft
    draft = InvestmentAxisDraft(
        draft_id="draft-1",
        portfolio_id="p1",
        essence_ref=ArtifactRef("doc", "n1", "1", "hash1", "hash1"),
        context_ref="ctx1",
        basic_policy="Draft policy",
        risk_limits={},
        invest_targets=[],
        exclude_targets=[],
        primary_methods=[],
        secondary_methods=[],
        investment_horizon="5y",
        allocation_basis=AllocationBasis.PURPOSE,
        allocation_rows=[],
        rebalance_frequency="Annual",
        role_models=[],
        non_actions=[],
        numeric_fields={},
        revision=1,
        created_at_iso="2026-10-10T10:00:00Z",
    )
    repo.save_axis_draft(draft)

    # CRITICAL TEST: Previously, get_confirmed_snapshot synthesized a fake snapshot from draft!
    # Must now strictly return None because no confirmation has happened.
    assert repo.get_confirmed_snapshot("portfolio:p1", "investment_axis") is None

    # 3. Save first confirmed snapshot record
    rec1 = ConfirmedSnapshotRecord(
        snapshot_id="snap-v1",
        scope="portfolio:p1",
        scope_id="p1",
        kind="investment_axis",
        source_id="draft-1",
        source_revision=1,
        content_hash="hash-v1",
        snapshot_json=json.dumps({"version": 1, "policy": "First"}),
        confirmed_at_iso="2026-10-10T11:00:00Z",
    )
    repo.save_confirmed_snapshot(rec1)

    # 4. Save second confirmed snapshot record
    rec2 = ConfirmedSnapshotRecord(
        snapshot_id="snap-v2",
        scope="portfolio:p1",
        scope_id="p1",
        kind="investment_axis",
        source_id="draft-1",
        source_revision=2,
        content_hash="hash-v2",
        snapshot_json=json.dumps({"version": 2, "policy": "Second"}),
        confirmed_at_iso="2026-10-10T12:00:00Z",
    )
    repo.save_confirmed_snapshot(rec2)

    # 5. Verify immutable history retention
    history = repo.list_confirmed_snapshots("portfolio:p1", "p1", "investment_axis")
    assert len(history) == 2
    assert history[0].snapshot_id == "snap-v2"
    assert history[1].snapshot_id == "snap-v1"

    # 6. Verify get_latest and get_by_id
    latest = repo.get_latest_confirmed_snapshot("portfolio:p1", "p1", "investment_axis")
    assert latest is not None
    assert latest.snapshot_id == "snap-v2"

    by_id = repo.get_confirmed_snapshot_by_id("snap-v1")
    assert by_id is not None
    assert by_id.content_hash == "hash-v1"

    # 7. get_confirmed_snapshot returns latest record
    snap = repo.get_confirmed_snapshot("portfolio:p1", "investment_axis")
    assert snap is not None
    assert snap["version"] == 2


def test_downstream_axis_draft_reads_confirmed_essence_claims():
    from application.investor_essence.investment_axis_service import InvestmentAxisService
    from application.investor_essence.testing.fakes import (
        FakeAxisGenerator,
        FakeClock,
        FakeIdGenerator,
        FakeInvestorRuntimeUowFactory,
    )

    uow_factory = FakeInvestorRuntimeUowFactory()
    axis_gen = FakeAxisGenerator()
    clock = FakeClock()
    id_gen = FakeIdGenerator()
    service = InvestmentAxisService(
        uow_factory=uow_factory,
        axis_generator=axis_gen,
        clock=clock,
        id_gen=id_gen,
    )

    # 1. Without confirmed essence, drafting axis fails
    with pytest.raises(Exception) as exc:
        service.create_axis_draft(portfolio_id="p1")
    assert "no confirmed investor essence" in str(exc.value)

    # 2. Setup confirmed essence snapshot in UoW
    with uow_factory.open() as uow:
        snap_dict = {
            "confirmation_id": "conf_ess_1",
            "content_hash": "hash_ess_123",
            "accepted_claims": [
                {"claim_id": "c1", "effective_text": "เน้นลงทุนหุ้นเติบโต"},
                {"claim_id": "c2", "effective_text": "ไม่ลงทุนอนุพันธ์เสี่ยงสูง"},
            ],
        }
        uow.planning.set_confirmed_pointer(
            scope="workspace",
            kind="investor_essence",
            artifact_ref="vault:investor_essence:conf_ess_1:hash_ess_123",
            expected_ref=None,
            snapshot=snap_dict,
        )
        uow.commit()

    # 3. Create axis draft should now pick up confirmed claims
    draft_view = service.create_axis_draft(portfolio_id="p1")
    assert draft_view.essence_ref["note_id"] == "conf_ess_1"
    assert draft_view.essence_ref["content_hash"] == "hash_ess_123"

    # Generator received confirmed claims
    assert len(axis_gen.call_history) == 1
    passed_claims = axis_gen.call_history[0]["accepted_claims"]
    assert len(passed_claims) == 2
    assert passed_claims[0]["text"] == "เน้นลงทุนหุ้นเติบโต"

    # 4. Confirm axis saves ConfirmedSnapshotRecord
    conf_view = service.confirm_axis(draft_id=draft_view.draft_id)
    assert conf_view.status == "committed"

    with uow_factory.open() as uow:
        axis_rec = uow.planning.get_latest_confirmed_snapshot(
            scope="portfolio:p1", scope_id="p1", kind="investment_axis"
        )
        assert axis_rec is not None
        assert axis_rec.snapshot_id == conf_view.confirmation_id
        assert len(axis_rec.content_hash) > 0

