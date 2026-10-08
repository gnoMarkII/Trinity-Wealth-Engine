"""Unit tests for application test fakes and in-memory repositories."""
import pytest

from core.investor_essence.models import (
    Answer,
    AnswerKind,
    ContentOption,
    GeneratedQuestion,
    SessionStatus,
)
from core.investor_essence.interview import EssenceSession
from application.investor_essence.dto import EvidenceSnapshot
from application.investor_essence.testing.fakes import (
    FakeClock,
    FakeIdGenerator,
    FakeInterviewGenerator,
    InMemorySessionRepository,
    InMemoryPlanningRepository,
    InMemoryOperationRepository,
    InMemoryIntentRepository,
    FakeInvestorRuntimeUow,
    FakeInvestorRuntimeUowFactory,
)


class TestApplicationFakes:
    def test_fake_clock_and_id_generator(self):
        clock = FakeClock("2026-10-08T12:00:00Z", 1791460800.0)
        assert clock.now_utc() == "2026-10-08T12:00:00Z"
        assert clock.now_epoch() == 1791460800.0
        clock.advance(60)
        assert clock.now_epoch() == 1791460860.0

        id_gen = FakeIdGenerator("fixed_id")
        assert id_gen.new_id() == "fixed_id-1"
        assert id_gen.new_id() == "fixed_id-2"

    def test_in_memory_session_repo_save_and_get(self):
        repo = InMemorySessionRepository()
        session = EssenceSession(session_id="sess_abc")
        repo.save(session)

        retrieved = repo.get("sess_abc")
        assert retrieved is not None
        assert retrieved.session_id == "sess_abc"

        active = repo.get_current("workspace")
        assert active is not None
        assert active.session_id == "sess_abc"

    def test_fake_uow_commit_and_rollback(self):
        factory = FakeInvestorRuntimeUowFactory()
        with factory.open() as uow:
            session = EssenceSession(session_id="sess_uow")
            uow.sessions.save(session)
            assert uow.sessions.get("sess_uow") is not None
            uow.commit()

        assert factory._uow.committed is True

    def test_fake_interview_generator(self):
        gen = FakeInterviewGenerator()
        evidence = EvidenceSnapshot(
            session_id="sess_gen",
            branch_id="main",
            revision=1,
            context_hash="hash123",
            qa_pairs=[],
            coverage_topics=[],
        )
        proposal = gen.generate_next_question(evidence, prompt_version="1.0")
        assert proposal.text is not None
        assert len(proposal.options) == 4
