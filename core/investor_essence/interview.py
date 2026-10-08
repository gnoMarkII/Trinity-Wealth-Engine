"""EssenceSession aggregate and pure interview branch rules.

Domain logic for:
- Maintaining sequence 1..10 with strictly 4 options
- Single unanswered active question per branch invariant
- Answer recording (choice, free text, unsure, skipped)
- Branch bifurcation when an earlier answer is revised
- Context hash computation for deterministic LLM prompt pinning
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.investor_essence.models import (
    SCHEMA_VERSION,
    Answer,
    AnswerKind,
    GeneratedQuestion,
    SessionStatus,
)


@dataclass
class EssenceSession:
    """Aggregate root managing the 10-question adaptive interview session."""
    session_id: str
    scope: str = "workspace"
    status: SessionStatus = SessionStatus.INTERVIEWING
    active_branch_id: str = "main"
    revision: int = 1
    schema_version: int = SCHEMA_VERSION
    interview_version: str = "1.0"
    questions: List[GeneratedQuestion] = field(default_factory=list)
    answers: Dict[str, Answer] = field(default_factory=dict)
    branch_history: Dict[str, List[str]] = field(default_factory=dict)
    clarifications: List[GeneratedQuestion] = field(default_factory=list)
    clarification_answers: Dict[str, Answer] = field(default_factory=dict)
    created_at_iso: str = ""
    updated_at_iso: str = ""

    def __post_init__(self) -> None:
        if self.active_branch_id not in self.branch_history:
            self.branch_history[self.active_branch_id] = [
                q.question_id for q in self.questions if not q.is_clarification
            ]

    @property
    def active_question_ids(self) -> List[str]:
        return list(self.branch_history.get(self.active_branch_id, []))

    def get_question(self, question_id: str) -> Optional[GeneratedQuestion]:
        for q in self.questions:
            if q.question_id == question_id:
                return q
        for c in self.clarifications:
            if c.question_id == question_id:
                return c
        return None

    def get_answer(self, question_id: str) -> Optional[Answer]:
        if question_id in self.answers:
            return self.answers[question_id]
        return self.clarification_answers.get(question_id)

    def get_active_qa_pairs(self) -> List[Tuple[GeneratedQuestion, Optional[Answer]]]:
        """Returns the chronological Q&A pairs for the current active branch."""
        active_ids = self.active_question_ids
        q_map = {q.question_id: q for q in self.questions}
        pairs: List[Tuple[GeneratedQuestion, Optional[Answer]]] = []
        for qid in active_ids:
            if qid in q_map:
                pairs.append((q_map[qid], self.answers.get(qid)))
        return pairs

    def is_base_interview_complete(self) -> bool:
        """True if exactly 10 questions have been generated and answered in active branch."""
        pairs = self.get_active_qa_pairs()
        if len(pairs) != 10:
            return False
        return all(ans is not None for _, ans in pairs)

    def current_active_question(self) -> Optional[GeneratedQuestion]:
        """Returns the current unanswered question in the active branch, if any."""
        pairs = self.get_active_qa_pairs()
        for q, ans in pairs:
            if ans is None:
                return q
        return None

    def add_question(self, question: GeneratedQuestion) -> None:
        """Appends a new question to the active branch."""
        if question.is_clarification:
            if any(c.question_id == question.question_id for c in self.clarifications):
                raise ValueError(f"Clarification question {question.question_id} already exists")
            self.clarifications.append(question)
            return

        active_pairs = self.get_active_qa_pairs()
        if len(active_pairs) >= 10:
            raise ValueError("Base interview cannot exceed 10 questions. Q11 is strictly forbidden.")

        current_unanswered = self.current_active_question()
        if current_unanswered is not None:
            raise ValueError(
                f"Cannot add question {question.question_id}; "
                f"question {current_unanswered.question_id} (seq {current_unanswered.sequence_no}) is still unanswered."
            )

        expected_seq = len(active_pairs) + 1
        if question.sequence_no != expected_seq:
            raise ValueError(f"Expected question sequence {expected_seq}, got {question.sequence_no}")

        self.questions.append(question)
        if self.active_branch_id not in self.branch_history:
            self.branch_history[self.active_branch_id] = []
        self.branch_history[self.active_branch_id].append(question.question_id)

    def record_answer(self, answer: Answer) -> None:
        """Records user answer to an active question or clarification."""
        q = self.get_question(answer.question_id)
        if q is None:
            raise ValueError(f"Question {answer.question_id} does not exist in session {self.session_id}")

        if q.is_clarification:
            self.clarification_answers[answer.question_id] = answer
            return

        if answer.question_id not in self.active_question_ids:
            raise ValueError(f"Question {answer.question_id} is not in the active branch {self.active_branch_id}")

        if answer.answer_kind == AnswerKind.CHOICE:
            if not answer.option_id:
                raise ValueError("Option ID must be provided when answer_kind is 'choice'")
            valid_option_ids = {opt.option_id for opt in q.options}
            if answer.option_id not in valid_option_ids:
                raise ValueError(f"Option ID {answer.option_id} does not belong to question {q.question_id}")

        self.answers[answer.question_id] = answer

        # Check if interview just completed
        if self.is_base_interview_complete():
            self.status = SessionStatus.SUMMARY_PENDING

    def revise_earlier_answer(
        self,
        question_id: str,
        new_answer: Answer,
        new_branch_id: str,
    ) -> None:
        """Creates a new branch when an earlier answer is changed, archiving descendants."""
        active_ids = self.active_question_ids
        if question_id not in active_ids:
            raise ValueError(f"Question {question_id} is not in active branch {self.active_branch_id}")

        idx = active_ids.index(question_id)
        retained_question_ids = active_ids[: idx + 1]

        # Branch off
        self.branch_history[new_branch_id] = list(retained_question_ids)
        self.active_branch_id = new_branch_id
        self.record_answer(new_answer)
        self.status = SessionStatus.INTERVIEWING

    def compute_context_hash(self) -> str:
        """Computes deterministic SHA256 of the active Q&A sequence."""
        pairs = self.get_active_qa_pairs()
        canonical_items = []
        for q, a in pairs:
            item = {
                "question_id": q.question_id,
                "sequence_no": q.sequence_no,
                "text": q.text,
                "options": [{"key": opt.option_key, "text": opt.text} for opt in q.options],
                "answer": {
                    "kind": a.answer_kind.value if a else None,
                    "option_id": a.option_id if a else None,
                    "free_text": a.free_text if a else None,
                } if a else None,
            }
            canonical_items.append(item)

        serialized = json.dumps(canonical_items, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
