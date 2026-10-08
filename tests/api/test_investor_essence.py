"""HTTP API integration tests for Investor Essence, Investment Axis, and Purpose Buckets."""
from __future__ import annotations

from decimal import Decimal
import pytest

from api.dependencies import (
    get_bucket_planning_service,
    get_claim_review_service,
    get_confirmation_service,
    get_financial_context_service,
    get_interview_service,
    get_investment_axis_service,
)
from api.main import app
from application.investor_essence import (
    BucketPlanningService,
    ClaimReviewService,
    ConfirmationService,
    FinancialContextService,
    InterviewService,
    InvestmentAxisService,
)
from application.investor_essence.dto import (
    AllocationApplyReceipt,
    AllocationPreview,
    ApplyAllocationCommand,
    ContentOptionProposal,
    PortfolioPlanningSnapshot,
    PreviewAllocationCommand,
    QuestionProposal,
)
from application.investor_essence.ports import PortfolioPlanningPort
from application.investor_essence.testing.fakes import (
    FakeAxisGenerator,
    FakeBucketGenerator,
    FakeClock,
    FakeEssenceGenerator,
    FakeIdGenerator,
    FakeInterviewGenerator,
    FakeInvestorRuntimeUowFactory,
)


class FakePortfolioPlanningPort(PortfolioPlanningPort):
    def snapshot(self, portfolio_id: str) -> PortfolioPlanningSnapshot:
        return PortfolioPlanningSnapshot(
            portfolio_id=portfolio_id,
            name=portfolio_id,
            checkpoint_sequence=1,
            checkpoint_state_hash="hash-1",
            base_currency="THB",
            nav_thb=Decimal("1000000.00"),
            cash_thb=Decimal("100000.00"),
            as_of="2026-10-08T12:00:00Z",
            targets=[
                {"bucket_id": "old_growth", "name": "Growth", "target_percent": 60.0, "color": "#3B82F6"},
                {"bucket_id": "old_cash", "name": "Cash", "target_percent": 40.0, "color": "#10B981"},
            ],
            holdings_summary=[
                {"symbol": "AAPL", "asset_type": "stock", "market_value_thb": 600000.0, "bucket_id": "old_growth"},
                {"symbol": "THB", "asset_type": "cash", "market_value_thb": 400000.0, "bucket_id": "old_cash"},
            ],
        )

    def preview(self, command: PreviewAllocationCommand) -> AllocationPreview:
        return AllocationPreview(
            checkpoint_sequence=command.expected_checkpoint_sequence,
            checkpoint_state_hash=command.expected_checkpoint_state_hash,
            validated_targets=command.target_rows,
            affected_holdings=[],
            before_allocation={"old_growth": 60.0, "old_cash": 40.0},
            after_allocation={t["bucket_id"]: float(t["target_percent"]) for t in command.target_rows},
            issues=[],
            payload_hash="preview-hash",
        )

    def apply(self, command: ApplyAllocationCommand) -> AllocationApplyReceipt:
        return AllocationApplyReceipt(
            command_id=command.command_id,
            portfolio_id=command.portfolio_id,
            request_hash=command.request_hash,
            applied_sequence=command.expected_checkpoint_sequence + 1,
            applied_state_hash="applied-hash-2",
            applied_at_iso="2026-10-08T12:05:00Z",
            canonical_status="committed",
            projection_status="ready",
            warnings=[],
        )

    def get_apply_receipt(self, portfolio_id: str, command_id: str):
        return None

    def repair_projection(self, portfolio_id: str, command_id: str):
        return AllocationApplyReceipt(
            command_id=command_id,
            portfolio_id=portfolio_id,
            request_hash="repaired",
            applied_sequence=1,
            applied_state_hash="applied-hash-repaired",
            applied_at_iso="2026-10-08T12:05:00Z",
            canonical_status="committed",
            projection_status="ready",
            warnings=[],
        )


def _build_10_proposals() -> list[QuestionProposal]:
    return [
        QuestionProposal(
            text=f"คำถามทดสอบ {i}",
            options=[
                ContentOptionProposal(key="A", text=f"ตัวเลือก A{i}"),
                ContentOptionProposal(key="B", text=f"ตัวเลือก B{i}"),
                ContentOptionProposal(key="C", text=f"ตัวเลือก C{i}"),
                ContentOptionProposal(key="D", text=f"ตัวเลือก D{i}"),
            ],
            coverage_topics=[f"topic_{i}"],
            evidence_type="self_report",
        )
        for i in range(1, 11)
    ]


@pytest.fixture
def essence_services():
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock()
    id_gen = FakeIdGenerator()
    proposals = _build_10_proposals()
    interview_gen = FakeInterviewGenerator(proposals)
    essence_gen = FakeEssenceGenerator()
    axis_gen = FakeAxisGenerator()
    bucket_gen = FakeBucketGenerator()
    portfolio_port = FakePortfolioPlanningPort()

    interview_service = InterviewService(
        uow_factory=uow_factory,
        generator=interview_gen,
        clock=clock,
        id_gen=id_gen,
    )
    claim_service = ClaimReviewService(
        uow_factory=uow_factory,
        essence_generator=essence_gen,
        interview_generator=interview_gen,
        clock=clock,
        id_gen=id_gen,
    )
    confirmation_service = ConfirmationService(
        uow_factory=uow_factory,
        clock=clock,
        id_gen=id_gen,
    )
    financial_context_service = FinancialContextService(
        uow_factory=uow_factory,
        clock=clock,
        id_gen=id_gen,
        portfolio_port=portfolio_port,
    )
    investment_axis_service = InvestmentAxisService(
        uow_factory=uow_factory,
        axis_generator=axis_gen,
        clock=clock,
        id_gen=id_gen,
    )
    bucket_planning_service = BucketPlanningService(
        uow_factory=uow_factory,
        bucket_generator=bucket_gen,
        portfolio_port=portfolio_port,
        clock=clock,
        id_gen=id_gen,
    )

    app.dependency_overrides[get_interview_service] = lambda: interview_service
    app.dependency_overrides[get_claim_review_service] = lambda: claim_service
    app.dependency_overrides[get_confirmation_service] = lambda: confirmation_service
    app.dependency_overrides[get_financial_context_service] = lambda: financial_context_service
    app.dependency_overrides[get_investment_axis_service] = lambda: investment_axis_service
    app.dependency_overrides[get_bucket_planning_service] = lambda: bucket_planning_service

    yield {
        "interview": interview_service,
        "claim": claim_service,
        "confirmation": confirmation_service,
        "financial_context": financial_context_service,
        "investment_axis": investment_axis_service,
        "bucket_planning": bucket_planning_service,
    }

    app.dependency_overrides.pop(get_interview_service, None)
    app.dependency_overrides.pop(get_claim_review_service, None)
    app.dependency_overrides.pop(get_confirmation_service, None)
    app.dependency_overrides.pop(get_financial_context_service, None)
    app.dependency_overrides.pop(get_investment_axis_service, None)
    app.dependency_overrides.pop(get_bucket_planning_service, None)


def test_interview_config(authed_client, essence_services):
    res = authed_client.get("/api/investor/essence/interview-config")
    assert res.status_code == 200
    data = res.json()
    assert data["questions_count"] == 10
    assert data["options_per_question"] == 4
    assert len(data["topics"]) >= 5


def test_full_essence_api_lifecycle(authed_client, essence_services):
    # 1. Start Session
    res = authed_client.post("/api/investor/essence/sessions", json={"scope": "workspace"})
    assert res.status_code == 201
    session = res.json()
    session_id = session["session_id"]
    assert session["status"] == "interviewing"
    assert session["questions_count"] == 1
    assert session["current_question"]["sequence_no"] == 1
    assert len(session["current_question"]["options"]) == 4

    # 2. Get Current Session
    res = authed_client.get("/api/investor/essence/sessions/current")
    assert res.status_code == 200
    assert res.json()["session_id"] == session_id

    # 3. Answer 9 questions and advance
    for seq in range(1, 10):
        curr_q = session["current_question"]
        qid = curr_q["question_id"]
        opt_id = curr_q["options"][0]["option_id"]

        # Record answer
        res = authed_client.put(
            f"/api/investor/essence/sessions/{session_id}/answers/{qid}",
            json={"answer_kind": "choice", "option_id": opt_id},
        )
        assert res.status_code == 200

        # Advance
        res = authed_client.post(f"/api/investor/essence/sessions/{session_id}/next-question")
        assert res.status_code == 200
        session = res.json()
        assert session["questions_count"] == seq + 1

    # 4. Answer 10th question
    curr_q = session["current_question"]
    res = authed_client.put(
        f"/api/investor/essence/sessions/{session_id}/answers/{curr_q['question_id']}",
        json={"answer_kind": "choice", "option_id": curr_q["options"][0]["option_id"]},
    )
    assert res.status_code == 200
    assert res.json()["is_complete"] is True

    # 5. Summarize
    res = authed_client.post(f"/api/investor/essence/sessions/{session_id}/summarize")
    assert res.status_code == 200
    summary = res.json()
    assert summary["session_id"] == session_id
    assert len(summary["claims"]) >= 2

    claim1_id = summary["claims"][0]["claim_id"]
    claim2_id = summary["claims"][1]["claim_id"]

    # 6. Review Claims
    # Rate claim 1
    res = authed_client.put(
        f"/api/investor/essence/sessions/{session_id}/summary/claims/{claim1_id}",
        json={"fit_rating": "exact"},
    )
    assert res.status_code == 200
    assert res.json()["claims"][0]["is_accepted"] is True

    # Edit claim 2
    res = authed_client.put(
        f"/api/investor/essence/sessions/{session_id}/summary/claims/{claim2_id}",
        json={"edited_text": "แก้ไขข้อความ: อิสรภาพและความมั่นคงอย่างยั่งยืน"},
    )
    assert res.status_code == 200
    assert res.json()["claims"][1]["is_accepted"] is True
    assert res.json()["claims"][1]["source_kind"] == "user_edited"

    # 7. Confirm Essence
    res = authed_client.post(
        f"/api/investor/essence/sessions/{session_id}/confirm",
        json={
            "accepted_claim_ids": [claim1_id, claim2_id],
            "idempotency_key": "api-idem-test-1",
        },
    )
    assert res.status_code == 200
    confirmation = res.json()
    assert confirmation["status"] == "committed"
    assert confirmation["accepted_claims_count"] == 2

    # 8. Get Current Confirmed Essence
    res = authed_client.get("/api/investor/essence/current")
    assert res.status_code == 200
    data = res.json()
    assert data["scope"] == "workspace"
    assert "pointer" in data

    # 9. Idempotent replay of confirmation
    res_replay = authed_client.post(
        f"/api/investor/essence/sessions/{session_id}/confirm",
        json={
            "accepted_claim_ids": [claim1_id, claim2_id],
            "idempotency_key": "api-idem-test-1",
        },
    )
    assert res_replay.status_code == 200
    assert res_replay.json()["confirmation_id"] == confirmation["confirmation_id"]


def test_milestone2_financial_context_and_axis_flow(authed_client, essence_services):
    # Set up confirmed essence first
    res = authed_client.post("/api/investor/essence/sessions", json={"scope": "workspace"})
    session_id = res.json()["session_id"]
    for _ in range(10):
        curr_session = authed_client.get(f"/api/investor/essence/sessions/{session_id}").json()
        curr_q = curr_session["current_question"]
        authed_client.put(
            f"/api/investor/essence/sessions/{session_id}/answers/{curr_q['question_id']}",
            json={"answer_kind": "choice", "option_id": curr_q["options"][0]["option_id"]},
        )
        if not authed_client.get(f"/api/investor/essence/sessions/{session_id}").json()["is_complete"]:
            authed_client.post(f"/api/investor/essence/sessions/{session_id}/next-question")

    summary = authed_client.post(f"/api/investor/essence/sessions/{session_id}/summarize").json()
    c1 = summary["claims"][0]["claim_id"]
    authed_client.put(
        f"/api/investor/essence/sessions/{session_id}/summary/claims/{c1}",
        json={"fit_rating": "exact"},
    )
    authed_client.post(
        f"/api/investor/essence/sessions/{session_id}/confirm",
        json={"accepted_claim_ids": [c1]},
    )

    portfolio_id = "port-alpha"

    # 1. Get initial Financial Context (defaults with unknown fields)
    res = authed_client.get(f"/api/investor/financial-context?portfolio_id={portfolio_id}")
    assert res.status_code == 200
    fc = res.json()
    assert fc["portfolio_id"] == portfolio_id
    assert fc["is_ready_for_numeric_policy"] is False

    # 2. Update Financial Context to provide horizon & reserves
    res = authed_client.put(
        f"/api/investor/financial-context?portfolio_id={portfolio_id}",
        json={
            "horizon_years": "7.0",
            "emergency_reserves_amount": "300000.00",
            "obligations_monthly": "25000.00",
            "unknown_fields": [],
        },
    )
    assert res.status_code == 200
    fc_updated = res.json()
    assert fc_updated["horizon_years"] == "7.0"
    assert fc_updated["is_ready_for_numeric_policy"] is True

    # 3. Create Axis Draft (Command 2)
    res = authed_client.post(
        "/api/investor/investment-axis/drafts",
        json={"portfolio_id": portfolio_id},
    )
    assert res.status_code == 201
    draft = res.json()
    draft_id = draft["draft_id"]
    assert draft["portfolio_id"] == portfolio_id
    assert len(draft["non_actions"]) >= 3

    # 4. Get Axis Draft
    res = authed_client.get(f"/api/investor/investment-axis/drafts/{draft_id}")
    assert res.status_code == 200
    assert res.json()["draft_id"] == draft_id

    # 5. Update Axis Draft (confirm risk limits for completeness)
    res = authed_client.put(
        f"/api/investor/investment-axis/drafts/{draft_id}",
        json={
            "risk_limits": {
                "mdd_max_annual": {
                    "field_id": "mdd_max_annual",
                    "value": "15.00",
                    "unit": "%",
                    "calculation_basis": "NAV",
                    "is_confirmed": True,
                },
                "max_loss_per_trade": {
                    "field_id": "max_loss_per_trade",
                    "value": "2.00",
                    "unit": "%",
                    "calculation_basis": "trade_capital",
                    "is_confirmed": True,
                },
            },
            "expected_revision": 1,
        },
    )
    assert res.status_code == 200
    updated_draft = res.json()
    assert updated_draft["revision"] == 2
    assert updated_draft["is_complete"] is True

    # 6. Confirm Investment Axis
    res = authed_client.post(
        f"/api/investor/investment-axis/drafts/{draft_id}/confirm",
        json={"idempotency_key": "axis-confirm-idem-1"},
    )
    assert res.status_code == 200
    axis_conf = res.json()
    assert axis_conf["status"] == "committed"

    # 7. Get Current Confirmed Axis
    res = authed_client.get(f"/api/investor/investment-axis/current?portfolio_id={portfolio_id}")
    assert res.status_code == 200
    assert "pointer" in res.json()


def test_milestone3_bucket_planning_lifecycle(authed_client, essence_services):
    portfolio_id = "port-beta"

    # Set up confirmed essence
    res = authed_client.post("/api/investor/essence/sessions", json={"scope": "workspace"})
    session_id = res.json()["session_id"]
    for _ in range(10):
        curr_session = authed_client.get(f"/api/investor/essence/sessions/{session_id}").json()
        curr_q = curr_session["current_question"]
        authed_client.put(
            f"/api/investor/essence/sessions/{session_id}/answers/{curr_q['question_id']}",
            json={"answer_kind": "choice", "option_id": curr_q["options"][0]["option_id"]},
        )
        if not authed_client.get(f"/api/investor/essence/sessions/{session_id}").json()["is_complete"]:
            authed_client.post(f"/api/investor/essence/sessions/{session_id}/next-question")

    summary = authed_client.post(f"/api/investor/essence/sessions/{session_id}/summarize").json()
    c1 = summary["claims"][0]["claim_id"]
    authed_client.put(
        f"/api/investor/essence/sessions/{session_id}/summary/claims/{c1}",
        json={"fit_rating": "exact"},
    )
    authed_client.post(
        f"/api/investor/essence/sessions/{session_id}/confirm",
        json={"accepted_claim_ids": [c1]},
    )

    # Set up confirmed axis
    authed_client.put(
        f"/api/investor/financial-context?portfolio_id={portfolio_id}",
        json={"horizon_years": "5.0", "unknown_fields": []},
    )
    draft = authed_client.post(
        "/api/investor/investment-axis/drafts",
        json={"portfolio_id": portfolio_id},
    ).json()
    draft_id = draft["draft_id"]
    authed_client.put(
        f"/api/investor/investment-axis/drafts/{draft_id}",
        json={
            "risk_limits": {
                "mdd_max_annual": {"field_id": "mdd_max_annual", "value": "15.00", "unit": "%", "is_confirmed": True},
                "max_loss_per_trade": {"field_id": "max_loss_per_trade", "value": "2.00", "unit": "%", "is_confirmed": True},
            },
        },
    )
    authed_client.post(f"/api/investor/investment-axis/drafts/{draft_id}/confirm")

    # 1. Create Bucket Plan Draft from confirmed axis
    res = authed_client.post("/api/investor/bucket-plans", json={"portfolio_id": portfolio_id})
    assert res.status_code == 201
    bplan = res.json()
    bp_id = bplan["draft_id"]
    assert bplan["portfolio_id"] == portfolio_id
    assert len(bplan["purpose_buckets"]) >= 2
    assert len(bplan["mapping_weights"]) >= 2

    # 2. Get Bucket Plan Draft
    res = authed_client.get(f"/api/investor/bucket-plans/{bp_id}")
    assert res.status_code == 200
    assert res.json()["draft_id"] == bp_id

    # 3. Update Bucket Plan Draft
    res = authed_client.put(
        f"/api/investor/bucket-plans/{bp_id}",
        json={
            "purpose_buckets": [
                {
                    "bucket_id": "b-growth",
                    "name": "เติบโตระยะยาว (อัปเดต)",
                    "role": "ลงทุนระยะยาว",
                    "color": "#3B82F6",
                    "target_percent": "70.00",
                },
                {
                    "bucket_id": "b-safety",
                    "name": "เงินสำรองและสภาพคล่อง (อัปเดต)",
                    "role": "รองรับความผันผวน",
                    "color": "#10B981",
                    "target_percent": "30.00",
                },
            ],
            "expected_revision": 1,
        },
    )
    assert res.status_code == 200
    updated_bp = res.json()
    assert updated_bp["revision"] == 2
    assert updated_bp["is_valid"] is True

    # 4. Preview Bucket Plan
    res = authed_client.get(f"/api/investor/bucket-plans/{bp_id}/preview")
    assert res.status_code == 200
    prev = res.json()
    assert prev["checkpoint_sequence"] >= 1
    assert len(prev["validated_targets"]) == 2

    # 5. Apply Bucket Plan (Atomic commit)
    res = authed_client.post(
        f"/api/investor/bucket-plans/{bp_id}/apply",
        json={"idempotency_key": "apply-idem-key-1"},
    )
    assert res.status_code == 200
    receipt = res.json()
    assert receipt["portfolio_id"] == portfolio_id
    assert receipt["canonical_status"] == "committed"

    # 6. Replay apply with same idempotency key
    res_replay = authed_client.post(
        f"/api/investor/bucket-plans/{bp_id}/apply",
        json={"idempotency_key": "apply-idem-key-1"},
    )
    assert res_replay.status_code == 200
    assert res_replay.json()["command_id"] == receipt["command_id"]
