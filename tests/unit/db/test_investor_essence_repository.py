"""Unit tests for Investor Essence SQLite repositories and Unit of Work."""
from decimal import Decimal
import tempfile
from pathlib import Path
import pytest

from core.investor_essence.models import (
    AllocationBasis,
    AllocationPlanRow,
    ArtifactRef,
    Answer,
    AnswerKind,
    BucketPlanDraft,
    BucketPlanStatus,
    ContentOption,
    CoverageItem,
    EssenceClaim,
    EssenceSummaryDraft,
    EvidenceRef,
    EvidenceType,
    FinancialContextSnapshot,
    FitRating,
    GeneratedQuestion,
    InvestmentAxisDraft,
    NumericPolicyField,
    NumericPolicyOrigin,
    PurposeBucketDraft,
    SessionStatus,
    SourceKind,
)
from core.investor_essence.interview import EssenceSession
from api.db.investor_essence_adapters import (
    SqliteInvestorRuntimeUow,
    SqliteInvestorRuntimeUowFactory,
)


@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test_investor_essence.sqlite")


class TestSqliteInvestorEssenceRepositories:
    def test_session_repository_save_and_get(self, temp_db_path):
        factory = SqliteInvestorRuntimeUowFactory(db_path=temp_db_path)

        with factory.open() as uow:
            session = EssenceSession(session_id="sess_1")
            q1 = GeneratedQuestion(
                question_id="q_1",
                sequence_no=1,
                text="เป้าหมายชีวิตของคุณคืออะไร?",
                options=[
                    ContentOption(option_id="opt_1", option_key="A", text="อิสระ"),
                    ContentOption(option_id="opt_2", option_key="B", text="ความมั่นคง"),
                    ContentOption(option_id="opt_3", option_key="C", text="รายได้"),
                    ContentOption(option_id="opt_4", option_key="D", text="ส่งต่อ"),
                ],
                evidence_type=EvidenceType.SELF_REPORT,
                coverage_topics=["เป้าหมาย"],
            )
            session.add_question(q1)
            session.record_answer(
                Answer(
                    answer_id="ans_1",
                    question_id="q_1",
                    answer_kind=AnswerKind.CHOICE,
                    option_id="opt_2",
                    evidence_type=EvidenceType.SELF_REPORT,
                )
            )
            uow.sessions.save(session)
            uow.commit()

        # Read back in new transaction
        with factory.open() as uow:
            retrieved = uow.sessions.get("sess_1")
            assert retrieved is not None
            assert retrieved.session_id == "sess_1"
            assert len(retrieved.questions) == 1
            assert retrieved.questions[0].question_id == "q_1"
            assert len(retrieved.answers) == 1
            assert retrieved.answers["q_1"].option_id == "opt_2"

            current = uow.sessions.get_current("workspace")
            assert current is not None
            assert current.session_id == "sess_1"

    def test_summary_draft_save_and_get(self, temp_db_path):
        factory = SqliteInvestorRuntimeUowFactory(db_path=temp_db_path)

        with factory.open() as uow:
            claim = EssenceClaim(
                claim_id="c_1",
                text="ต้องการความมั่นคงของเงินต้น",
                source_kind=SourceKind.USER_STATED,
                evidence_refs=[
                    EvidenceRef(
                        answer_id="ans_1",
                        question_id="q_1",
                        revision=1,
                        quote="ความมั่นคง",
                        evidence_type=EvidenceType.SELF_REPORT,
                    )
                ],
                fit_rating=FitRating.EXACT,
                is_accepted=True,
            )
            draft = EssenceSummaryDraft(
                summary_id="summary_sess_1",
                session_id="sess_1",
                statement="แก่นแท้คือความมั่นคง",
                claims=[claim],
                unresolved_topics=["กรอบเวลา"],
                coverage_report=[
                    CoverageItem(topic="เป้าหมาย", status="covered", supporting_answer_ids=["ans_1"])
                ],
                context_hash="hash_123",
            )
            uow.sessions.save_summary(draft)
            uow.commit()

        with factory.open() as uow:
            retrieved = uow.sessions.get_summary("sess_1")
            assert retrieved is not None
            assert retrieved.statement == "แก่นแท้คือความมั่นคง"
            assert len(retrieved.claims) == 1
            assert retrieved.claims[0].claim_id == "c_1"
            assert retrieved.claims[0].fit_rating == FitRating.EXACT
            assert retrieved.claims[0].is_accepted is True
            assert retrieved.unresolved_topics == ["กรอบเวลา"]

    def test_planning_repository_cas_pointers(self, temp_db_path):
        factory = SqliteInvestorRuntimeUowFactory(db_path=temp_db_path)

        with factory.open() as uow:
            # 1. Initial insert with expected_ref=None -> succeeds
            ok = uow.planning.set_confirmed_pointer("workspace", "investor-essence", "art_v1", None)
            assert ok is True
            uow.commit()

        with factory.open() as uow:
            current = uow.planning.get_confirmed_pointer("workspace", "investor-essence")
            assert current == "art_v1"

            # 2. Update with mismatched expected_ref -> fails (CAS protection)
            ok_conflict = uow.planning.set_confirmed_pointer(
                "workspace", "investor-essence", "art_v2", "wrong_ref"
            )
            assert ok_conflict is False

            # 3. Update with matching expected_ref -> succeeds
            ok_update = uow.planning.set_confirmed_pointer(
                "workspace", "investor-essence", "art_v2", "art_v1"
            )
            assert ok_update is True
            uow.commit()

        with factory.open() as uow:
            current2 = uow.planning.get_confirmed_pointer("workspace", "investor-essence")
            assert current2 == "art_v2"

    def test_operation_repository_lease_and_fence(self, temp_db_path):
        factory = SqliteInvestorRuntimeUowFactory(db_path=temp_db_path)

        with factory.open() as uow:
            uow.operations.enqueue(
                operation_id="op_1",
                stage="interview_next",
                resource_id="sess_1",
                resource_revision=1,
                input_hash="hash_in",
                prompt_version="1.0",
                frozen_input={"key": "val"},
            )
            uow.commit()

        with factory.open() as uow:
            op_view = uow.operations.get("op_1")
            assert op_view is not None
            assert op_view.status == "queued"

            # Worker 1 claims lease
            claimed = uow.operations.claim_lease("worker_A", lease_seconds=30)
            assert claimed is not None
            assert claimed["operation_id"] == "op_1"
            assert claimed["fencing_token"] == 1
            uow.commit()

        with factory.open() as uow:
            # Try to complete with old fencing token -> fails
            stale_ok = uow.operations.complete_with_fence("op_1", fence=99, result={"ok": True})
            assert stale_ok is False

            # Complete with correct fencing token -> succeeds
            ok = uow.operations.complete_with_fence("op_1", fence=1, result={"ok": True})
            assert ok is True
            uow.commit()

        with factory.open() as uow:
            finished = uow.operations.get("op_1")
            assert finished is not None
            assert finished.status == "succeeded"
            assert finished.result == {"ok": True}

    def test_intent_and_command_receipt_idempotency(self, temp_db_path):
        factory = SqliteInvestorRuntimeUowFactory(db_path=temp_db_path)

        with factory.open() as uow:
            uow.intents.save_command_receipt(
                scope="workspace",
                use_case="confirm_essence",
                idempotency_key="idem_1",
                request_hash="req_hash_1",
                result={"status": "confirmed", "art_id": "art_100"},
            )
            uow.commit()

        with factory.open() as uow:
            receipt = uow.intents.get_command_receipt("workspace", "confirm_essence", "idem_1")
            assert receipt is not None
            assert receipt["request_hash"] == "req_hash_1"
            assert receipt["result"]["status"] == "confirmed"

            # Unknown key returns None
            none_receipt = uow.intents.get_command_receipt("workspace", "confirm_essence", "unknown_key")
            assert none_receipt is None

    def test_uow_rollback_discards_changes(self, temp_db_path):
        factory = SqliteInvestorRuntimeUowFactory(db_path=temp_db_path)

        with pytest.raises(RuntimeError):
            with factory.open() as uow:
                session = EssenceSession(session_id="sess_rollback")
                uow.sessions.save(session)
                raise RuntimeError("Force rollback")

        with factory.open() as uow:
            retrieved = uow.sessions.get("sess_rollback")
            assert retrieved is None
