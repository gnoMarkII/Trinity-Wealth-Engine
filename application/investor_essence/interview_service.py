"""Application service orchestrating adaptive interview sessions."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from core.investor_essence.models import (
    Answer,
    AnswerKind,
    ContentOption,
    EvidenceType,
    GeneratedQuestion,
    SessionStatus,
)
from core.investor_essence.interview import EssenceSession
from application.investor_essence.dto import (
    EvidenceSnapshot,
    SessionView,
)
from application.investor_essence.errors import (
    ResourceNotFoundError,
    RevisionConflictError,
    ValidationFailedError,
)
from application.investor_essence.ports import (
    ClockPort,
    IdGeneratorPort,
    InterviewGeneratorPort,
    InvestorRuntimeUowFactory,
)

logger = logging.getLogger(__name__)


def _to_session_view(session: EssenceSession) -> SessionView:
    active_qa = session.get_active_qa_pairs()
    qa_history = []
    for q, ans in active_qa:
        item: Dict[str, Any] = {
            "question_id": q.question_id,
            "sequence_no": q.sequence_no,
            "text": q.text,
            "options": [{"option_id": o.option_id, "option_key": o.option_key, "text": o.text} for o in q.options],
            "evidence_type": q.evidence_type.value,
            "coverage_topics": q.coverage_topics,
            "is_clarification": q.is_clarification,
            "answer": {
                "answer_id": ans.answer_id,
                "answer_kind": ans.answer_kind.value,
                "option_id": ans.option_id,
                "free_text": ans.free_text,
                "evidence_type": ans.evidence_type.value,
            } if ans else None,
        }
        qa_history.append(item)

    curr_unanswered = session.current_active_question()
    current_q_dict = None
    if curr_unanswered:
        current_q_dict = {
            "question_id": curr_unanswered.question_id,
            "sequence_no": curr_unanswered.sequence_no,
            "text": curr_unanswered.text,
            "options": [{"option_id": o.option_id, "option_key": o.option_key, "text": o.text} for o in curr_unanswered.options],
            "evidence_type": curr_unanswered.evidence_type.value,
            "coverage_topics": curr_unanswered.coverage_topics,
            "is_clarification": curr_unanswered.is_clarification,
        }

    return SessionView(
        session_id=session.session_id,
        status=session.status.value,
        revision=session.revision,
        active_branch_id=session.active_branch_id,
        questions_count=len(session.questions),
        answers_count=len(session.answers),
        is_complete=session.is_base_interview_complete(),
        current_question=current_q_dict,
        qa_history=qa_history,
    )


class InterviewService:
    """Use cases for managing the adaptive 10-question interview workflow."""

    def __init__(
        self,
        uow_factory: InvestorRuntimeUowFactory,
        generator: Optional[InterviewGeneratorPort] = None,
        clock: Optional[ClockPort] = None,
        id_gen: Optional[IdGeneratorPort] = None,
        *,
        interview_generator: Optional[InterviewGeneratorPort] = None,
    ) -> None:
        self._uow_factory = uow_factory
        gen = generator or interview_generator
        if gen is None:
            raise ValueError("An InterviewGeneratorPort instance is required")
        self._generator = gen
        self._clock = clock  # type: ignore[assignment]
        self._id_gen = id_gen  # type: ignore[assignment]

    def start_session(self, scope: str = "workspace", prompt_version: str = "1.0") -> SessionView:
        """Starts a new interview session and generates the first question."""
        session_id = self._id_gen.new_id()
        now_iso = self._clock.now_utc()

        session = EssenceSession(
            session_id=session_id,
            scope=scope,
            status=SessionStatus.INTERVIEWING,
            active_branch_id="main",
            revision=1,
            created_at_iso=now_iso,
            updated_at_iso=now_iso,
        )

        # Generate Q1
        evidence = EvidenceSnapshot(
            session_id=session_id,
            branch_id="main",
            revision=1,
            context_hash=session.compute_context_hash(),
            qa_pairs=[],
            coverage_topics=[],
        )
        proposal = self._generator.generate_next_question(evidence, prompt_version=prompt_version)

        options = [
            ContentOption(
                option_id=f"opt_1_{p.key.lower()}",
                option_key=p.key,
                text=p.text,
            )
            for p in proposal.options
        ]
        q1 = GeneratedQuestion(
            question_id=f"q_{session_id}_1",
            sequence_no=1,
            text=proposal.text,
            options=options,
            evidence_type=EvidenceType(proposal.evidence_type) if proposal.evidence_type in EvidenceType._value2member_map_ else EvidenceType.SELF_REPORT,
            coverage_topics=proposal.coverage_topics,
            created_at_iso=now_iso,
        )
        session.add_question(q1)

        with self._uow_factory.open() as uow:
            uow.sessions.save(session)
            uow.commit()

        return _to_session_view(session)

    def get_session(self, session_id: str) -> Optional[SessionView]:
        with self._uow_factory.open() as uow:
            session = uow.sessions.get(session_id)
            if not session:
                return None
            return _to_session_view(session)

    def get_current_session(self, scope: str = "workspace") -> Optional[SessionView]:
        with self._uow_factory.open() as uow:
            session = uow.sessions.get_current(scope)
            if not session:
                return None
            return _to_session_view(session)

    def record_answer(
        self,
        session_id: str,
        question_id: str,
        answer_kind: AnswerKind,
        option_id: Optional[str] = None,
        free_text: Optional[str] = None,
        expected_revision: Optional[int] = None,
    ) -> SessionView:
        with self._uow_factory.open() as uow:
            session = uow.sessions.get(session_id)
            if not session:
                raise ResourceNotFoundError(f"Interview session {session_id} not found")

            if expected_revision is not None and session.revision != expected_revision:
                raise RevisionConflictError(
                    f"Revision mismatch: expected {expected_revision}, current {session.revision}",
                    latest_revision=session.revision,
                )

            q = session.get_question(question_id)
            if not q:
                raise ResourceNotFoundError(f"Question {question_id} not found in session {session_id}")

            ans_id = self._id_gen.new_id()
            now_iso = self._clock.now_utc()
            ans = Answer(
                answer_id=ans_id,
                question_id=question_id,
                answer_kind=answer_kind,
                option_id=option_id,
                free_text=free_text,
                evidence_type=q.evidence_type,
                created_at_iso=now_iso,
            )
            session.record_answer(ans)
            session.revision += 1
            session.updated_at_iso = now_iso

            uow.sessions.save(session)
            uow.commit()

            return _to_session_view(session)

    def advance_next_question(
        self,
        session_id: str,
        expected_revision: Optional[int] = None,
        prompt_version: str = "1.0",
    ) -> SessionView:
        """Generates and appends the next question in the active branch."""
        with self._uow_factory.open() as uow:
            session = uow.sessions.get(session_id)
            if not session:
                raise ResourceNotFoundError(f"Interview session {session_id} not found")

            if expected_revision is not None and session.revision != expected_revision:
                raise RevisionConflictError(
                    f"Revision mismatch: expected {expected_revision}, current {session.revision}",
                    latest_revision=session.revision,
                )

            if session.is_base_interview_complete():
                raise ValidationFailedError("Base interview already completed (10 questions answered)")

            curr_unanswered = session.current_active_question()
            if curr_unanswered is not None:
                # Question already waiting to be answered
                return _to_session_view(session)

            active_qa = session.get_active_qa_pairs()
            next_seq = len(active_qa) + 1
            if next_seq > 10:
                raise ValidationFailedError("Cannot exceed 10 base interview questions")

            qa_history_dict = []
            coverage_topics: List[str] = []
            for q, a in active_qa:
                coverage_topics.extend(q.coverage_topics)
                qa_history_dict.append({
                    "question_id": q.question_id,
                    "sequence_no": q.sequence_no,
                    "text": q.text,
                    "answer": {
                        "kind": a.answer_kind.value if a else None,
                        "option_id": a.option_id if a else None,
                        "free_text": a.free_text if a else None,
                    } if a else None,
                })

            evidence = EvidenceSnapshot(
                session_id=session_id,
                branch_id=session.active_branch_id,
                revision=session.revision,
                context_hash=session.compute_context_hash(),
                qa_pairs=qa_history_dict,
                coverage_topics=list(set(coverage_topics)),
            )

            proposal = self._generator.generate_next_question(evidence, prompt_version=prompt_version)
            options = [
                ContentOption(
                    option_id=f"opt_{next_seq}_{p.key.lower()}",
                    option_key=p.key,
                    text=p.text,
                )
                for p in proposal.options
            ]

            ev_type = (
                EvidenceType(proposal.evidence_type)
                if proposal.evidence_type in EvidenceType._value2member_map_
                else EvidenceType.SELF_REPORT
            )
            now_iso = self._clock.now_utc()
            next_q = GeneratedQuestion(
                question_id=f"q_{session_id}_{next_seq}",
                sequence_no=next_seq,
                text=proposal.text,
                options=options,
                evidence_type=ev_type,
                coverage_topics=proposal.coverage_topics,
                created_at_iso=now_iso,
            )

            session.add_question(next_q)
            session.revision += 1
            session.updated_at_iso = now_iso

            uow.sessions.save(session)
            uow.commit()

            return _to_session_view(session)

    def revise_earlier_answer(
        self,
        session_id: str,
        question_id: str,
        new_answer_kind: AnswerKind,
        new_option_id: Optional[str] = None,
        new_free_text: Optional[str] = None,
        expected_revision: Optional[int] = None,
    ) -> SessionView:
        """Forks a new branch from an edited answer, archiving future questions in that branch."""
        with self._uow_factory.open() as uow:
            session = uow.sessions.get(session_id)
            if not session:
                raise ResourceNotFoundError(f"Interview session {session_id} not found")

            if expected_revision is not None and session.revision != expected_revision:
                raise RevisionConflictError(
                    f"Revision mismatch: expected {expected_revision}, current {session.revision}",
                    latest_revision=session.revision,
                )

            q = session.get_question(question_id)
            if not q:
                raise ResourceNotFoundError(f"Question {question_id} not found")

            new_branch_id = self._id_gen.new_id()
            ans_id = self._id_gen.new_id()
            now_iso = self._clock.now_utc()

            new_ans = Answer(
                answer_id=ans_id,
                question_id=question_id,
                answer_kind=new_answer_kind,
                option_id=new_option_id,
                free_text=new_free_text,
                evidence_type=q.evidence_type,
                created_at_iso=now_iso,
            )

            session.revise_earlier_answer(
                question_id=question_id,
                new_answer=new_ans,
                new_branch_id=new_branch_id,
            )
            session.revision += 1
            session.updated_at_iso = now_iso

            uow.sessions.save(session)
            uow.commit()

            return _to_session_view(session)
