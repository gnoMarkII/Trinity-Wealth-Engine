"""End-to-end Application Flow tests for Milestone 1 (Adaptive Interview -> Review -> Confirmation)."""
from __future__ import annotations

import pytest

from core.investor_essence.models import (
    AnswerKind,
    FitRating,
    SessionStatus,
    SourceKind,
)
from application.investor_essence.dto import (
    ContentOptionProposal,
    QuestionProposal,
)
from application.investor_essence.errors import (
    IdempotencyConflictError,
    ResourceNotFoundError,
    RevisionConflictError,
    ValidationFailedError,
)
from application.investor_essence.interview_service import InterviewService
from application.investor_essence.claim_review_service import ClaimReviewService
from application.investor_essence.confirmation_service import ConfirmationService
from application.investor_essence.testing.fakes import (
    FakeClock,
    FakeEssenceGenerator,
    FakeIdGenerator,
    FakeInterviewGenerator,
    FakeInvestorRuntimeUowFactory,
)


def _build_10_proposals() -> list[QuestionProposal]:
    """Helper creating 10 distinct adaptive question proposals."""
    return [
        QuestionProposal(
            text=f"คำถามทดสอบข้อที่ {i}",
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


def test_full_milestone1_interview_to_confirmation() -> None:
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock()
    id_gen = FakeIdGenerator()
    proposals = _build_10_proposals()
    interview_gen = FakeInterviewGenerator(proposals)
    essence_gen = FakeEssenceGenerator()

    interview_service = InterviewService(
        uow_factory=uow_factory,
        interview_generator=interview_gen,
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

    # 1. Start session -> Q1 generated
    session_view = interview_service.start_session(scope="workspace")
    assert session_view.session_id.startswith("test-id-")
    assert session_view.status == SessionStatus.INTERVIEWING.value
    assert session_view.questions_count == 1
    assert session_view.answers_count == 0
    assert not session_view.is_complete
    assert session_view.current_question is not None
    assert session_view.current_question["sequence_no"] == 1

    session_id = session_view.session_id

    # 2. Answer Q1 through Q9 and advance
    for seq in range(1, 10):
        current_q = session_view.current_question
        assert current_q is not None
        qid = current_q["question_id"]
        opt_id = current_q["options"][0]["option_id"]

        # Record answer
        session_view = interview_service.record_answer(
            session_id=session_id,
            question_id=qid,
            answer_kind=AnswerKind.CHOICE,
            option_id=opt_id,
        )
        assert session_view.answers_count == seq

        # Advance to next question
        session_view = interview_service.advance_next_question(session_id=session_id)
        assert session_view.questions_count == seq + 1
        assert session_view.current_question["sequence_no"] == seq + 1

    # 3. Answer Q10 (the final question)
    q10 = session_view.current_question
    assert q10 is not None
    assert q10["sequence_no"] == 10
    session_view = interview_service.record_answer(
        session_id=session_id,
        question_id=q10["question_id"],
        answer_kind=AnswerKind.CHOICE,
        option_id=q10["options"][0]["option_id"],
    )
    assert session_view.answers_count == 10
    assert session_view.is_complete

    # Attempting to advance beyond Q10 must raise ValidationFailedError (no Q11)
    with pytest.raises(ValidationFailedError, match="Base interview already complete"):
        interview_service.advance_next_question(session_id=session_id)

    # 4. Generate Summary Draft
    summary_view = claim_service.generate_summary(session_id=session_id)
    assert summary_view.session_id == session_id
    assert len(summary_view.claims) >= 2
    assert "statement" in summary_view.__dataclass_fields__
    assert summary_view.revision == 1

    claim1_id = summary_view.claims[0]["claim_id"]
    claim2_id = summary_view.claims[1]["claim_id"]

    # 5. Review Claims
    # Rate claim 1 as EXACT
    summary_view = claim_service.rate_claim(
        session_id=session_id,
        claim_id=claim1_id,
        fit_rating=FitRating.EXACT,
    )
    assert summary_view.claims[0]["is_accepted"] is True
    assert summary_view.claims[0]["fit_rating"] == FitRating.EXACT.value

    # Edit claim 2 text
    summary_view = claim_service.edit_claim(
        session_id=session_id,
        claim_id=claim2_id,
        new_text="ปรับปรุงข้อความ: ลงทุนเพื่อความมั่นคงและอิสรภาพอย่างสมดุล",
    )
    assert summary_view.claims[1]["is_accepted"] is True
    assert summary_view.claims[1]["edited_text"] == "ปรับปรุงข้อความ: ลงทุนเพื่อความมั่นคงและอิสรภาพอย่างสมดุล"
    assert summary_view.claims[1]["effective_text"] == "ปรับปรุงข้อความ: ลงทุนเพื่อความมั่นคงและอิสรภาพอย่างสมดุล"
    assert summary_view.claims[1]["source_kind"] == SourceKind.USER_EDITED.value

    # 6. Confirm Essence
    conf_view = confirmation_service.confirm_essence(
        session_id=session_id,
        accepted_claim_ids=[claim1_id, claim2_id],
        idempotency_key="test-idem-key-1",
    )
    assert conf_view.status == "committed"
    assert conf_view.accepted_claims_count == 2
    assert conf_view.artifact_ref is not None
    assert "content_hash" in conf_view.artifact_ref

    # 7. Check current confirmed essence pointer
    curr_essence = confirmation_service.get_current_confirmed_essence()
    assert curr_essence is not None
    assert curr_essence["scope"] == "workspace"
    assert "vault:investor_essence:" in curr_essence["pointer"]

    # 8. Test Idempotency: exact same payload returns identical result
    replay_view = confirmation_service.confirm_essence(
        session_id=session_id,
        accepted_claim_ids=[claim1_id, claim2_id],
        idempotency_key="test-idem-key-1",
    )
    assert replay_view.confirmation_id == conf_view.confirmation_id
    assert replay_view.accepted_claims_count == conf_view.accepted_claims_count

    # 9. Test Idempotency Conflict: same key with different payload raises IdempotencyConflictError
    with pytest.raises(IdempotencyConflictError):
        confirmation_service.confirm_essence(
            session_id=session_id,
            accepted_claim_ids=[claim1_id],  # Different payload
            idempotency_key="test-idem-key-1",
        )


def test_cannot_confirm_rejected_or_unaccepted_claims() -> None:
    uow_factory = FakeInvestorRuntimeUowFactory()
    clock = FakeClock()
    id_gen = FakeIdGenerator()
    proposals = _build_10_proposals()
    interview_gen = FakeInterviewGenerator(proposals)
    essence_gen = FakeEssenceGenerator()

    interview_service = InterviewService(uow_factory, interview_gen, clock, id_gen)
    claim_service = ClaimReviewService(uow_factory, essence_gen, interview_gen, clock, id_gen)
    confirmation_service = ConfirmationService(uow_factory, clock, id_gen)

    session_view = interview_service.start_session()
    session_id = session_view.session_id

    for _ in range(10):
        q = session_view.current_question
        interview_service.record_answer(
            session_id=session_id,
            question_id=q["question_id"],
            answer_kind=AnswerKind.CHOICE,
            option_id=q["options"][0]["option_id"],
        )
        if not interview_service.get_session(session_id).is_complete:
            session_view = interview_service.advance_next_question(session_id=session_id)

    summary_view = claim_service.generate_summary(session_id=session_id)
    c1 = summary_view.claims[0]["claim_id"]
    c2 = summary_view.claims[1]["claim_id"]

    # Exclude c1
    claim_service.exclude_claim(session_id=session_id, claim_id=c1)

    # Trying to confirm including rejected c1 should fail
    with pytest.raises(ValidationFailedError, match="Rejected claim"):
        confirmation_service.confirm_essence(
            session_id=session_id,
            accepted_claim_ids=[c1],
        )

    # Trying to confirm with empty list should fail
    with pytest.raises(ValidationFailedError, match="at least 1 accepted claim is required"):
        confirmation_service.confirm_essence(
            session_id=session_id,
            accepted_claim_ids=[],
        )
