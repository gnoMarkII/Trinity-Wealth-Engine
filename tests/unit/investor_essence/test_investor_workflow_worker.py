"""Unit tests for InvestorWorkflowWorker."""
from typing import Any, Dict
import pytest

from api.workers.investor_workflow_worker import InvestorWorkflowWorker
from application.investor_essence.testing.fakes import (
    FakeAxisGenerator,
    FakeBucketGenerator,
    FakeClock,
    FakeEssenceGenerator,
    FakeIdGenerator,
    FakeInterviewGenerator,
    FakeInvestorRuntimeUowFactory,
)
from application.investor_essence.interview_service import InterviewService
from application.investor_essence.claim_review_service import ClaimReviewService
from application.investor_essence.investment_axis_service import InvestmentAxisService
from application.investor_essence.bucket_planning_service import BucketPlanningService


@pytest.fixture
def worker_bundle():
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock()
    id_gen = FakeIdGenerator()
    interview_gen = FakeInterviewGenerator()
    essence_gen = FakeEssenceGenerator()
    axis_gen = FakeAxisGenerator()
    bucket_gen = FakeBucketGenerator()

    class FakeKnowledgeReader:
        def read_exact(self, ref):
            return None
        def read_confirmed_versions(self, scope, entity_type):
            return []

    class FakeKnowledgeWriter:
        def submit(self, cmd):
            return {"command_id": "cmd-1", "status": "committed"}
        def get_receipt(self, cmd_id):
            return None

    class FakePortfolioPlanner:
        def get_planning_snapshot(self, portfolio_id):
            return {"sequence": 1, "state_hash": "hash-1", "targets": [], "holdings": [], "nav_thb": "1000000"}
        def validate_compiled_plan(self, plan):
            return {"is_valid": True, "issues": []}
        def apply_compiled_plan(self, plan):
            from application.investor_essence.ports import AllocationApplyReceipt
            return AllocationApplyReceipt(
                command_id="cmd-1",
                portfolio_id=plan.portfolio_id,
                request_hash="req-1",
                applied_sequence=2,
                applied_state_hash="hash-2",
                applied_at_iso="2026-10-08T12:00:00Z",
                canonical_status="committed",
                projection_status="committed",
            )

    knowledge_reader = FakeKnowledgeReader()
    knowledge_writer = FakeKnowledgeWriter()
    portfolio_planner = FakePortfolioPlanner()

    interview_service = InterviewService(
        uow_factory=uow_factory,
        generator=interview_gen,
        clock=clock,
        id_gen=id_gen,
    )
    claim_review_service = ClaimReviewService(
        uow_factory=uow_factory,
        essence_generator=essence_gen,
        interview_generator=interview_gen,
        clock=clock,
        id_gen=id_gen,
    )
    axis_service = InvestmentAxisService(
        uow_factory=uow_factory,
        axis_generator=axis_gen,
        clock=clock,
        id_gen=id_gen,
        reader_port=knowledge_reader,
        knowledge_write=knowledge_writer,
    )
    bucket_service = BucketPlanningService(
        uow_factory=uow_factory,
        bucket_generator=bucket_gen,
        portfolio_port=portfolio_planner,
        clock=clock,
        id_gen=id_gen,
    )

    worker = InvestorWorkflowWorker(
        uow_factory=uow_factory,
        interview_service=interview_service,
        claim_review_service=claim_review_service,
        axis_service=axis_service,
        bucket_service=bucket_service,
        worker_id="test-worker-1",
    )

    return worker, uow_factory, interview_service


def test_worker_processes_advance_question(worker_bundle):
    worker, uow_factory, interview_service = worker_bundle

    # 1. Start a session
    sess_view = interview_service.start_session()
    sess_id = sess_view.session_id

    # 2. Enqueue an operation in UoW
    with uow_factory.open() as uow:
        uow.operations.enqueue(
            operation_id="op-1",
            stage="interview_next",
            resource_id=sess_id,
            resource_revision=sess_view.revision,
            input_hash="hash-1",
            prompt_version="1.0",
            frozen_input={},
        )

    # 3. Process one batch
    processed = worker.process_one_batch(limit=1)
    assert processed == 1

    # 4. Verify operation is succeeded
    with uow_factory.open() as uow:
        op = uow.operations.get("op-1")
        assert op is not None
        assert op.status == "succeeded"
        assert op.result is not None
        assert "questions_count" in op.result


def test_worker_handles_unknown_task_type(worker_bundle):
    worker, uow_factory, _ = worker_bundle

    with uow_factory.open() as uow:
        uow.operations.enqueue(
            operation_id="op-unknown",
            stage="invalid_stage_xyz",
            resource_id="res-1",
            resource_revision=1,
            input_hash="hash-x",
            prompt_version="1.0",
            frozen_input={},
        )

    processed = worker.process_one_batch(limit=1)
    assert processed == 0

    with uow_factory.open() as uow:
        op = uow.operations.get("op-unknown")
        assert op is not None
        assert op.status in ("failed", "retryable")
