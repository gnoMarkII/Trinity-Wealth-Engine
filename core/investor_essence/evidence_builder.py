"""Shared evidence builder for adaptive interview, summary, and clarification.

Standardizes Q&A history representation across all stages:
- Injects selected option key and text
- Preserves free text explanations
- Computes valid coverage topics (excluding unsure and skipped answers)
- Provides canonical dictionary format for LLM prompt context
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from core.investor_essence.models import (
    Answer,
    AnswerKind,
    EvidenceSnapshot,
    EvidenceType,
    GeneratedQuestion,
)
from core.investor_essence.interview import EssenceSession


def format_qa_item(question: GeneratedQuestion, answer: Optional[Answer]) -> Dict[str, Any]:
    """Formats a single Q&A pair with full question options and selected answer text."""
    selected_opt = question.get_option(answer.option_id) if (answer and answer.option_id) else None

    quote_parts = []
    if selected_opt:
        quote_parts.append(selected_opt.text)
    if answer and answer.free_text:
        quote_parts.append(f"คำอธิบายเพิ่มเติม: {answer.free_text}")
    quote = " | ".join(quote_parts)

    return {
        "question_id": question.question_id,
        "sequence_no": question.sequence_no,
        "text": question.text,
        "evidence_type": question.evidence_type.value if hasattr(question.evidence_type, "value") else str(question.evidence_type),
        "coverage_topics": list(question.coverage_topics),
        "is_clarification": question.is_clarification,
        "options": [
            {
                "option_id": opt.option_id,
                "key": opt.option_key,
                "text": opt.text,
            }
            for opt in question.options
        ],
        "answer": {
            "answer_id": answer.answer_id,
            "kind": answer.answer_kind.value,
            "option_id": answer.option_id,
            "selected_option_key": selected_opt.option_key if selected_opt else None,
            "selected_option_text": selected_opt.text if selected_opt else None,
            "free_text": answer.free_text,
            "quote": quote,
            "is_informative": answer.answer_kind in (AnswerKind.CHOICE, AnswerKind.FREE_TEXT),
        } if answer else None,
    }


def compute_informative_coverage_topics(pairs: List[Tuple[GeneratedQuestion, Optional[Answer]]]) -> List[str]:
    """Computes topics that have been answered with meaningful substance.

    Unsure, skipped, or unanswered questions do NOT count as covered.
    """
    covered: Set[str] = set()
    for q, a in pairs:
        if a and a.answer_kind in (AnswerKind.CHOICE, AnswerKind.FREE_TEXT):
            covered.update(q.coverage_topics)
    return sorted(covered)


def build_evidence_snapshot_from_session(session: EssenceSession) -> EvidenceSnapshot:
    """Builds a standardized EvidenceSnapshot from an active EssenceSession."""
    all_pairs: List[Tuple[GeneratedQuestion, Optional[Answer]]] = list(session.get_active_qa_pairs())
    for c in session.clarifications:
        ans = session.clarification_answers.get(c.question_id)
        all_pairs.append((c, ans))

    qa_items = [format_qa_item(q, a) for q, a in all_pairs]
    covered_topics = compute_informative_coverage_topics(all_pairs)

    return EvidenceSnapshot(
        session_id=session.session_id,
        branch_id=session.active_branch_id,
        revision=session.revision,
        context_hash=session.compute_context_hash(),
        qa_pairs=qa_items,
        coverage_topics=covered_topics,
    )
