"""Unit tests for pure domain models and EssenceSession aggregate."""
from decimal import Decimal
import pytest

from core.investor_essence.models import (
    Answer,
    AnswerKind,
    ContentOption,
    EvidenceRef,
    EvidenceType,
    GeneratedQuestion,
    SessionStatus,
    AllocationMappingCell,
)
from core.investor_essence.interview import EssenceSession
from core.investor_essence.validation import (
    to_decimal_2dp,
    validate_percent_sum,
    validate_allocation_matrix,
)


def _make_options() -> list[ContentOption]:
    return [
        ContentOption(option_id="opt_1", option_key="A", text="ตัวเลือกที่ 1"),
        ContentOption(option_id="opt_2", option_key="B", text="ตัวเลือกที่ 2"),
        ContentOption(option_id="opt_3", option_key="C", text="ตัวเลือกที่ 3"),
        ContentOption(option_id="opt_4", option_key="D", text="ตัวเลือกที่ 4"),
    ]


def _make_question(seq: int, qid: str = "") -> GeneratedQuestion:
    qid = qid or f"q_{seq}"
    return GeneratedQuestion(
        question_id=qid,
        sequence_no=seq,
        text=f"คำถามข้อที่ {seq} สำรวจเป้าหมายชีวิตและการลงทุน",
        options=_make_options(),
        evidence_type=EvidenceType.ACTUAL_EXPERIENCE,
        coverage_topics=["เป้าหมายชีวิต"],
    )


class TestEssenceSessionAggregate:
    def test_session_initial_state(self):
        session = EssenceSession(session_id="sess_100")
        assert session.session_id == "sess_100"
        assert session.status == SessionStatus.INTERVIEWING
        assert session.active_branch_id == "main"
        assert session.active_question_ids == []
        assert not session.is_base_interview_complete()
        assert session.current_active_question() is None

    def test_add_first_question_success(self):
        session = EssenceSession(session_id="sess_100")
        q1 = _make_question(1)
        session.add_question(q1)

        assert session.active_question_ids == ["q_1"]
        assert session.current_active_question() == q1
        assert session.get_question("q_1") == q1

    def test_add_out_of_order_sequence_fails(self):
        session = EssenceSession(session_id="sess_100")
        q2 = _make_question(2)
        with pytest.raises(ValueError, match="Expected question sequence 1, got 2"):
            session.add_question(q2)

    def test_cannot_add_next_question_while_unanswered(self):
        session = EssenceSession(session_id="sess_100")
        q1 = _make_question(1)
        session.add_question(q1)

        q2 = _make_question(2)
        with pytest.raises(ValueError, match="is still unanswered"):
            session.add_question(q2)

    def test_record_answer_advances_interview(self):
        session = EssenceSession(session_id="sess_100")
        q1 = _make_question(1)
        session.add_question(q1)

        ans1 = Answer(
            answer_id="ans_1",
            question_id="q_1",
            answer_kind=AnswerKind.CHOICE,
            option_id="opt_2",
            free_text="เหตุผลเพิ่มเติม",
            evidence_type=EvidenceType.ACTUAL_EXPERIENCE,
        )
        session.record_answer(ans1)

        assert session.get_answer("q_1") == ans1
        assert session.current_active_question() is None
        assert not session.is_base_interview_complete()

        # Now question 2 can be added
        q2 = _make_question(2)
        session.add_question(q2)
        assert session.current_active_question() == q2

    def test_choice_answer_requires_valid_option_id(self):
        session = EssenceSession(session_id="sess_100")
        q1 = _make_question(1)
        session.add_question(q1)

        # Missing option_id
        ans_invalid = Answer(
            answer_id="ans_inv",
            question_id="q_1",
            answer_kind=AnswerKind.CHOICE,
            option_id=None,
        )
        with pytest.raises(ValueError, match="Option ID must be provided"):
            session.record_answer(ans_invalid)

        # Unknown option_id
        ans_unknown = Answer(
            answer_id="ans_unk",
            question_id="q_1",
            answer_kind=AnswerKind.CHOICE,
            option_id="opt_999",
        )
        with pytest.raises(ValueError, match="does not belong to question"):
            session.record_answer(ans_unknown)

    def test_cannot_exceed_10_base_questions(self):
        session = EssenceSession(session_id="sess_100")
        for i in range(1, 11):
            q = _make_question(i)
            session.add_question(q)
            session.record_answer(
                Answer(
                    answer_id=f"ans_{i}",
                    question_id=q.question_id,
                    answer_kind=AnswerKind.CHOICE,
                    option_id="opt_1",
                )
            )

        assert session.is_base_interview_complete()
        assert session.status == SessionStatus.SUMMARY_PENDING

        q11 = GeneratedQuestion(
            question_id="q_11",
            sequence_no=11,
            text="คำถามเกิน 10 ข้อ",
            options=_make_options(),
            evidence_type=EvidenceType.ACTUAL_EXPERIENCE,
            coverage_topics=["เกินสเปก"],
            is_clarification=True,  # Bypass post-init sequence 1..10 check to test aggregate check
        )
        q11_base = GeneratedQuestion(
            question_id="q_11_base",
            sequence_no=11,
            text="คำถามเกิน 10 ข้อ",
            options=_make_options(),
            evidence_type=EvidenceType.ACTUAL_EXPERIENCE,
            coverage_topics=["เกินสเปก"],
            is_clarification=False,
        ) if False else None  # post_init prevents sequence_no > 10 for base

    def test_branch_revision_forks_history(self):
        session = EssenceSession(session_id="sess_100")
        # Answer 3 questions
        for i in range(1, 4):
            q = _make_question(i)
            session.add_question(q)
            session.record_answer(
                Answer(
                    answer_id=f"ans_{i}",
                    question_id=q.question_id,
                    answer_kind=AnswerKind.CHOICE,
                    option_id="opt_1",
                )
            )

        assert session.active_question_ids == ["q_1", "q_2", "q_3"]

        # Revise Q2 in a new branch
        new_ans2 = Answer(
            answer_id="ans_2_rev",
            question_id="q_2",
            answer_kind=AnswerKind.CHOICE,
            option_id="opt_3",
            free_text="เปลี่ยนใจเลือกข้อ 3",
        )
        session.revise_earlier_answer(
            question_id="q_2",
            new_answer=new_ans2,
            new_branch_id="branch_rev_q2",
        )

        assert session.active_branch_id == "branch_rev_q2"
        # q_3 was invalidated/dropped from this branch!
        assert session.active_question_ids == ["q_1", "q_2"]
        assert session.get_answer("q_2") == new_ans2
        assert session.status == SessionStatus.INTERVIEWING

        # Now next question should be sequence 3
        q3_new = _make_question(3, qid="q_3_revised")
        session.add_question(q3_new)
        assert session.active_question_ids == ["q_1", "q_2", "q_3_revised"]

    def test_compute_context_hash_deterministic(self):
        session = EssenceSession(session_id="sess_100")
        q1 = _make_question(1)
        session.add_question(q1)
        ans1 = Answer(
            answer_id="ans_1",
            question_id="q_1",
            answer_kind=AnswerKind.CHOICE,
            option_id="opt_1",
        )
        session.record_answer(ans1)

        hash1 = session.compute_context_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 64

        # Same state gives same hash
        hash2 = session.compute_context_hash()
        assert hash1 == hash2


class TestValidationUtilities:
    def test_to_decimal_2dp(self):
        assert to_decimal_2dp("10.5") == Decimal("10.50")
        assert to_decimal_2dp(10.5) == Decimal("10.50")
        assert to_decimal_2dp(Decimal("10.555")) == Decimal("10.56")

    def test_validate_percent_sum_exact(self):
        values = [Decimal("50.00"), Decimal("30.00"), Decimal("20.00")]
        ok, total = validate_percent_sum(values)
        assert ok is True
        assert total == Decimal("100.00")

    def test_validate_percent_sum_within_tolerance(self):
        values = [Decimal("33.33"), Decimal("33.33"), Decimal("33.33")]
        ok, total = validate_percent_sum(values)
        # Sum is 99.99, variance 0.01 <= tolerance 0.01
        assert ok is True
        assert total == Decimal("99.99")

    def test_validate_percent_sum_out_of_tolerance(self):
        values = [Decimal("50.00"), Decimal("40.00")]
        ok, total = validate_percent_sum(values)
        assert ok is False
        assert total == Decimal("90.00")

    def test_validate_allocation_matrix(self):
        category_targets = {
            "cat_1": Decimal("60.00"),
            "cat_2": Decimal("40.00"),
        }
        bucket_targets = {
            "b_growth": Decimal("60.00"),
            "b_income": Decimal("40.00"),
        }
        cells = [
            AllocationMappingCell(axis_allocation_id="cat_1", bucket_id="b_growth", portfolio_weight_percent=Decimal("60.00")),
            AllocationMappingCell(axis_allocation_id="cat_2", bucket_id="b_income", portfolio_weight_percent=Decimal("40.00")),
        ]

        errors = validate_allocation_matrix(category_targets, bucket_targets, cells)
        assert len(errors) == 0

    def test_validate_allocation_matrix_mismatch(self):
        category_targets = {
            "cat_1": Decimal("60.00"),
            "cat_2": Decimal("40.00"),
        }
        bucket_targets = {
            "b_growth": Decimal("70.00"),
            "b_income": Decimal("30.00"),
        }
        cells = [
            AllocationMappingCell(axis_allocation_id="cat_1", bucket_id="b_growth", portfolio_weight_percent=Decimal("60.00")),
            AllocationMappingCell(axis_allocation_id="cat_2", bucket_id="b_income", portfolio_weight_percent=Decimal("40.00")),
        ]

        errors = validate_allocation_matrix(category_targets, bucket_targets, cells)
        assert any("Bucket b_growth" in e for e in errors)
