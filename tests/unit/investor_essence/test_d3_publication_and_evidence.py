"""Unit tests for Deliverable D3 (WP04 + WP05):
- KnowledgePublicationAdapter producing typed KnowledgeWriteCommands
- EvidenceBuilder with option key/text preservation & informative coverage topics
- ClaimReviewService validating evidence lineage and moving unsupported claims to unresolved_topics
- Clarification request and answering flow
"""
from decimal import Decimal
import pytest
from typing import Any, Dict, List, Optional

from core.investor_essence.models import (
    Answer,
    AnswerKind,
    ContentOption,
    CoverageItem,
    EssenceClaim,
    EvidenceRef,
    EvidenceType,
    FitRating,
    GeneratedQuestion,
    SessionStatus,
    SourceKind,
)
from core.investor_essence.interview import EssenceSession
from core.investor_essence.evidence_builder import (
    format_qa_item,
    compute_informative_coverage_topics,
    build_evidence_snapshot_from_session,
)
from application.knowledge.write_models import KnowledgeWriteCommand, KnowledgeWriteReceipt
from tools.investor_essence.adapters.knowledge_publication_adapter import (
    KnowledgePublicationAdapter,
    build_essence_markdown_body,
    build_axis_markdown_body,
)
from application.investor_essence.claim_review_service import ClaimReviewService
from application.investor_essence.dto import ClaimProposal, EssenceSummaryProposal, QuestionProposal, ContentOptionProposal
from application.investor_essence.testing.fakes import (
    FakeClock,
    FakeIdGenerator,
    FakeInterviewGenerator,
    FakeInvestorRuntimeUowFactory,
)


class FakeBroker:
    def __init__(self) -> None:
        self.submitted_commands: List[KnowledgeWriteCommand] = []

    def submit(self, command: KnowledgeWriteCommand) -> KnowledgeWriteReceipt:
        self.submitted_commands.append(command)
        return KnowledgeWriteReceipt(
            command_id=command.idempotency_key,
            idempotency_key=command.idempotency_key,
            status="committed",
            operation=command.operation,
            producer=command.producer,
            document_key=command.document_key,
            note_id=command.payload["metadata"]["note_id"],
            revision_id="rev-1",
            content_hash="hash-1",
            artifact_set_hash="artifact-1",
        )


class FakeEssenceGenerator:
    def __init__(self, proposal: Optional[EssenceSummaryProposal] = None) -> None:
        self.proposal = proposal or EssenceSummaryProposal(
            statement="Test statement",
            claims=[],
            unresolved_topics=[],
            coverage_report=[],
        )

    def generate_summary(self, evidence: Any, prompt_version: str = "1.0") -> EssenceSummaryProposal:
        return self.proposal


def test_knowledge_publication_adapter_builds_valid_commands():
    broker = FakeBroker()
    adapter = KnowledgePublicationAdapter(write_port=broker)

    # 1. Test publish_essence
    essence_snapshot = {
        "statement": "Invest for long term freedom",
        "accepted_claims": [
            {
                "claim_id": "c1",
                "text": "Focus on high quality equities",
                "effective_text": "Focus on high quality equities",
                "source_kind": "direct",
                "evidence_refs": [{"quote": "High quality only"}],
            }
        ],
        "unresolved_topics": ["crypto_leverage"],
        "content_hash": "hash123",
        "scope": "workspace",
        "confirmed_at_iso": "2026-10-10T20:00:00Z",
    }
    receipt = adapter.publish_essence(
        confirmation_id="conf-ess-1",
        snapshot_dict=essence_snapshot,
    )
    assert receipt is not None
    assert receipt["document_key"] == "investor-essence-conf-ess-1"
    assert len(broker.submitted_commands) == 1
    cmd = broker.submitted_commands[0]
    assert isinstance(cmd, KnowledgeWriteCommand)
    assert cmd.entity_type == "investor_essence"
    assert cmd.payload["metadata"]["schema_version"] == 2
    assert cmd.payload["metadata"]["entity_type"] == "investor_essence"
    assert len(cmd.payload["metadata"]["accepted_claims"]) == 1
    assert "Focus on high quality equities" in cmd.payload["body"]
    assert "High quality only" in cmd.payload["body"]

    # 2. Test publish_axis
    axis_snapshot = {
        "portfolio_id": "port-main",
        "basic_policy": "Balanced Growth with Drawdown Cap",
        "risk_limits": {
            "mdd_percent": {"value": "15", "unit": "%"},
            "max_loss_per_trade": {"value": "2", "unit": "%"},
        },
        "invest_targets": ["Global ETFs", "Thai Large Caps"],
        "exclude_targets": ["Crypto Futures", "Penny Stocks"],
        "primary_methods": ["DCA", "Rebalancing"],
        "secondary_methods": ["Tactical Tilt"],
        "investment_horizon": "10+ years",
        "rebalance_frequency": "Quarterly",
        "allocation_rows": [
            {"category_name": "Equities", "target_percent": "70.00", "role_description": "Growth engine"},
            {"category_name": "Bonds", "target_percent": "30.00", "role_description": "Safety anchor"},
        ],
        "role_models": ["Warren Buffett"],
        "non_actions": ["No panic selling", "No leverage"],
        "content_hash": "axis123",
        "scope": "portfolio",
        "scope_id": "port-main",
        "confirmed_at_iso": "2026-10-10T20:05:00Z",
    }
    axis_receipt = adapter.publish_axis(
        confirmation_id="conf-axis-1",
        portfolio_id="port-main",
        snapshot_dict=axis_snapshot,
    )
    assert axis_receipt is not None
    assert axis_receipt["document_key"] == "investment-axis-port-main"
    assert len(broker.submitted_commands) == 2
    axis_cmd = broker.submitted_commands[1]
    assert axis_cmd.entity_type == "investment_axis"
    assert axis_cmd.payload["metadata"]["schema_version"] == 2
    assert axis_cmd.payload["metadata"]["entity_type"] == "investment_axis"
    assert "Balanced Growth" in axis_cmd.payload["body"]
    assert "Global ETFs" in axis_cmd.payload["body"]
    assert "Equities" in axis_cmd.payload["body"]
    assert "No panic selling" in axis_cmd.payload["body"]


def test_evidence_builder_formatting_and_informative_coverage():
    opts = [
        ContentOption(option_id="opt1", option_key="A", text="Option A Text"),
        ContentOption(option_id="opt2", option_key="B", text="Option B Text"),
        ContentOption(option_id="opt3", option_key="C", text="Option C Text"),
        ContentOption(option_id="opt4", option_key="D", text="Option D Text"),
    ]
    q1 = GeneratedQuestion(
        question_id="q1",
        sequence_no=1,
        text="Question 1?",
        options=opts,
        evidence_type=EvidenceType.SELF_REPORT,
        coverage_topics=["goals", "lifestyle"],
    )
    ans1 = Answer(
        answer_id="ans1",
        question_id="q1",
        answer_kind=AnswerKind.CHOICE,
        option_id="opt2",
        free_text="My extra detail",
        evidence_type=EvidenceType.SELF_REPORT,
    )

    # 1. Format QA item
    qa_dict = format_qa_item(q1, ans1)
    assert qa_dict["question_id"] == "q1"
    assert qa_dict["answer"]["selected_option_key"] == "B"
    assert qa_dict["answer"]["selected_option_text"] == "Option B Text"
    assert qa_dict["answer"]["free_text"] == "My extra detail"
    assert "Option B Text" in qa_dict["answer"]["quote"]
    assert "My extra detail" in qa_dict["answer"]["quote"]
    assert qa_dict["answer"]["is_informative"] is True

    # 2. Coverage calculation: unsure and skipped should not count
    q2 = GeneratedQuestion(
        question_id="q2",
        sequence_no=2,
        text="Question 2?",
        options=opts,
        evidence_type=EvidenceType.SELF_REPORT,
        coverage_topics=["risk_tolerance"],
    )
    ans2 = Answer(
        answer_id="ans2",
        question_id="q2",
        answer_kind=AnswerKind.UNSURE,
        option_id=None,
    )

    q3 = GeneratedQuestion(
        question_id="q3",
        sequence_no=3,
        text="Question 3?",
        options=opts,
        evidence_type=EvidenceType.SELF_REPORT,
        coverage_topics=["liquidity"],
    )
    ans3 = Answer(
        answer_id="ans3",
        question_id="q3",
        answer_kind=AnswerKind.SKIPPED,
        option_id=None,
    )

    pairs = [(q1, ans1), (q2, ans2), (q3, ans3)]
    covered = compute_informative_coverage_topics(pairs)
    assert "goals" in covered
    assert "lifestyle" in covered
    assert "risk_tolerance" not in covered
    assert "liquidity" not in covered


def _create_completed_10_question_session(uow_factory) -> str:
    opts = [
        ContentOption(option_id="opt1", option_key="A", text="Option A"),
        ContentOption(option_id="opt2", option_key="B", text="Option B"),
        ContentOption(option_id="opt3", option_key="C", text="Option C"),
        ContentOption(option_id="opt4", option_key="D", text="Option D"),
    ]
    session = EssenceSession(session_id="sess-d3-test", revision=1)
    for i in range(1, 11):
        q = GeneratedQuestion(
            question_id=f"q{i}",
            sequence_no=i,
            text=f"Question {i}?",
            options=opts,
            evidence_type=EvidenceType.SELF_REPORT,
            coverage_topics=[f"topic_{i}"],
        )
        session.add_question(q)
        ans = Answer(
            answer_id=f"ans{i}",
            question_id=f"q{i}",
            answer_kind=AnswerKind.CHOICE,
            option_id="opt1",
            evidence_type=EvidenceType.SELF_REPORT,
        )
        session.record_answer(ans)

    with uow_factory.open() as uow:
        uow.sessions.save(session)
        uow.commit()
    return session.session_id


def test_claim_review_enforces_evidence_lineage_and_unresolved_topics():
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock()
    id_gen = FakeIdGenerator()
    interview_gen = FakeInterviewGenerator()

    # Proposal with 1 claim having valid evidence (q1) and 1 claim having fake/invalid evidence (q99)
    proposal = EssenceSummaryProposal(
        statement="Summary statement",
        claims=[
            ClaimProposal(
                text="Valid claim supported by Q1",
                source_kind="direct",
                supporting_question_ids=["q1"],
                quote="Option A",
                evidence_type="self_report",
            ),
            ClaimProposal(
                text="Unsupported claim referencing non-existent question",
                source_kind="ai_inferred",
                supporting_question_ids=["q99_invalid"],
                quote="",
                evidence_type="self_report",
            ),
        ],
        unresolved_topics=["existing_unresolved"],
        coverage_report=[],
    )
    essence_gen = FakeEssenceGenerator(proposal)

    service = ClaimReviewService(
        uow_factory=uow_factory,
        essence_generator=essence_gen,
        interview_generator=interview_gen,
        clock=clock,
        id_gen=id_gen,
    )

    session_id = _create_completed_10_question_session(uow_factory)
    summary_view = service.generate_summary(session_id)

    # Valid claim is kept
    assert len(summary_view.claims) == 1
    assert summary_view.claims[0]["text"] == "Valid claim supported by Q1"

    # Unsupported claim is demoted to unresolved_topics
    assert "Unsupported claim referencing non-existent question" in summary_view.unresolved_topics
    assert "existing_unresolved" in summary_view.unresolved_topics


def test_clarification_request_and_answering_flow():
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock()
    id_gen = FakeIdGenerator()
    interview_gen = FakeInterviewGenerator()
    essence_gen = FakeEssenceGenerator()

    service = ClaimReviewService(
        uow_factory=uow_factory,
        essence_generator=essence_gen,
        interview_generator=interview_gen,
        clock=clock,
        id_gen=id_gen,
    )

    session_id = _create_completed_10_question_session(uow_factory)

    # 1. Request clarification
    clarification_q = service.request_clarification(
        session_id=session_id,
        unresolved_topic="emergency_fund_depth",
    )
    assert clarification_q["is_clarification"] is True
    assert clarification_q["coverage_topics"] == ["emergency_fund_depth"]
    assert len(clarification_q["options"]) == 4

    # 2. Check session has clarification recorded and revision incremented
    with uow_factory.open() as uow:
        sess = uow.sessions.get(session_id)
        assert sess is not None
        assert len(sess.clarifications) == 1
        assert sess.clarifications[0].question_id == clarification_q["question_id"]
        assert sess.revision == 2

        # 3. Answer clarification
        cq_id = clarification_q["question_id"]
        c_opt_id = clarification_q["options"][0]["option_id"]
        c_ans = Answer(
            answer_id="ans_clarify_1",
            question_id=cq_id,
            answer_kind=AnswerKind.CHOICE,
            option_id=c_opt_id,
            free_text="6 months saved",
            evidence_type=EvidenceType.SELF_REPORT,
        )
        sess.record_answer(c_ans)
        uow.sessions.save(sess)
        uow.commit()

    # 4. Check evidence snapshot includes clarification question and answer
    with uow_factory.open() as uow:
        sess_after = uow.sessions.get(session_id)
        assert sess_after is not None
        ev = build_evidence_snapshot_from_session(sess_after)
        # 10 base questions + 1 clarification = 11 items
        assert len(ev.qa_pairs) == 11
        assert "emergency_fund_depth" in ev.coverage_topics
        clarify_item = [p for p in ev.qa_pairs if p["question_id"] == cq_id][0]
        assert clarify_item["answer"] is not None
        assert clarify_item["answer"]["free_text"] == "6 months saved"
