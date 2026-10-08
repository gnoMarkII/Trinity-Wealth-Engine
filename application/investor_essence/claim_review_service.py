"""Application service orchestrating claim review, editing, and clarification."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.investor_essence.models import (
    Answer,
    AnswerKind,
    ContentOption,
    CoverageItem,
    EssenceClaim,
    EssenceSummaryDraft,
    EvidenceRef,
    EvidenceType,
    FitRating,
    GeneratedQuestion,
    SessionStatus,
    SourceKind,
)
from core.investor_essence.claims import (
    edit_claim_text,
    exclude_claim,
    rate_claim_fit,
)
from application.investor_essence.dto import (
    EvidenceSnapshot,
    SummaryView,
)
from application.investor_essence.errors import (
    ResourceNotFoundError,
    RevisionConflictError,
    ValidationFailedError,
)
from application.investor_essence.ports import (
    ClockPort,
    EssenceGeneratorPort,
    IdGeneratorPort,
    InterviewGeneratorPort,
    InvestorRuntimeUowFactory,
)

logger = logging.getLogger(__name__)


def _to_summary_view(summary: EssenceSummaryDraft) -> SummaryView:
    claims_list = []
    for c in summary.claims:
        claims_list.append({
            "claim_id": c.claim_id,
            "text": c.text,
            "effective_text": c.effective_text,
            "source_kind": c.source_kind.value,
            "fit_rating": c.fit_rating.value if c.fit_rating else None,
            "edited_text": c.edited_text,
            "text_revision": c.text_revision,
            "is_accepted": c.is_accepted,
            "evidence_refs": [
                {
                    "answer_id": er.answer_id,
                    "question_id": er.question_id,
                    "revision": er.revision,
                    "quote": er.quote,
                    "evidence_type": er.evidence_type.value,
                }
                for er in c.evidence_refs
            ],
        })

    coverage_list = [
        {
            "topic": ci.topic,
            "status": ci.status,
            "supporting_answer_ids": ci.supporting_answer_ids,
        }
        for ci in summary.coverage_report
    ]

    return SummaryView(
        summary_id=summary.summary_id,
        session_id=summary.session_id,
        statement=summary.statement,
        claims=claims_list,
        unresolved_topics=summary.unresolved_topics,
        coverage_report=coverage_list,
        revision=summary.revision,
    )


class ClaimReviewService:
    """Use cases for summarizing Q&A into claims, editing, fit-rating, and clarifications."""

    def __init__(
        self,
        uow_factory: InvestorRuntimeUowFactory,
        essence_generator: EssenceGeneratorPort,
        interview_generator: InterviewGeneratorPort,
        clock: ClockPort,
        id_gen: IdGeneratorPort,
    ) -> None:
        self._uow_factory = uow_factory
        self._essence_generator = essence_generator
        self._interview_generator = interview_generator
        self._clock = clock
        self._id_gen = id_gen

    def generate_summary(
        self,
        session_id: str,
        expected_revision: Optional[int] = None,
        prompt_version: str = "1.0",
    ) -> SummaryView:
        with self._uow_factory.open() as uow:
            session = uow.sessions.get(session_id)
            if not session:
                raise ResourceNotFoundError(f"Interview session {session_id} not found")

            if expected_revision is not None and session.revision != expected_revision:
                raise RevisionConflictError(
                    f"Revision conflict: expected {expected_revision}, current {session.revision}",
                    latest_revision=session.revision,
                )

            if not session.is_base_interview_complete():
                raise ValidationFailedError("Cannot summarize incomplete interview (needs all 10 questions answered)")

            active_qa = session.get_active_qa_pairs()
            qa_dict = []
            for q, a in active_qa:
                qa_dict.append({
                    "question_id": q.question_id,
                    "sequence_no": q.sequence_no,
                    "text": q.text,
                    "answer": {
                        "kind": a.answer_kind.value if a else None,
                        "option_id": a.option_id if a else None,
                        "free_text": a.free_text if a else None,
                        "quote": a.free_text or (q.get_option(a.option_id).text if a.option_id and q.get_option(a.option_id) else ""),
                    } if a else None,
                })

            evidence = EvidenceSnapshot(
                session_id=session_id,
                branch_id=session.active_branch_id,
                revision=session.revision,
                context_hash=session.compute_context_hash(),
                qa_pairs=qa_dict,
                coverage_topics=[],
            )

            proposal = self._essence_generator.generate_summary(evidence, prompt_version=prompt_version)

            claims: List[EssenceClaim] = []
            for idx, cp in enumerate(proposal.claims, start=1):
                ev_refs = []
                for qid in cp.supporting_question_ids:
                    q = session.get_question(qid)
                    ans = session.get_answer(qid)
                    if q and ans:
                        ev_refs.append(
                            EvidenceRef(
                                answer_id=ans.answer_id,
                                question_id=qid,
                                revision=1,
                                quote=cp.quote or ans.free_text or "",
                                evidence_type=ans.evidence_type,
                            )
                        )
                claims.append(
                    EssenceClaim(
                        claim_id=f"c_{session_id}_{idx}",
                        text=cp.text,
                        source_kind=SourceKind(cp.source_kind) if cp.source_kind in SourceKind._value2member_map_ else SourceKind.AI_INFERRED,
                        evidence_refs=ev_refs,
                        fit_rating=None,
                        is_accepted=False,
                    )
                )

            coverage_items = [
                CoverageItem(
                    topic=ci.get("topic", ""),
                    status=ci.get("status", "covered"),
                    supporting_answer_ids=ci.get("supporting_answer_ids", []),
                )
                for ci in proposal.coverage_report
            ]

            now_iso = self._clock.now_utc()
            summary = EssenceSummaryDraft(
                summary_id=f"summary_{session_id}",
                session_id=session_id,
                statement=proposal.statement,
                claims=claims,
                unresolved_topics=proposal.unresolved_topics,
                coverage_report=coverage_items,
                context_hash=evidence.context_hash,
                revision=1,
                created_at_iso=now_iso,
            )

            session.status = SessionStatus.REVIEW
            session.updated_at_iso = now_iso
            uow.sessions.save(session)
            uow.sessions.save_summary(summary)
            uow.commit()

            return _to_summary_view(summary)

    def get_summary(self, session_id: str) -> Optional[SummaryView]:
        with self._uow_factory.open() as uow:
            summary = uow.sessions.get_summary(session_id)
            if not summary:
                return None
            return _to_summary_view(summary)

    def rate_claim(
        self,
        session_id: str,
        claim_id: str,
        fit_rating: FitRating,
        expected_revision: Optional[int] = None,
    ) -> SummaryView:
        with self._uow_factory.open() as uow:
            summary = uow.sessions.get_summary(session_id)
            if not summary:
                raise ResourceNotFoundError(f"Summary draft for session {session_id} not found")

            if expected_revision is not None and summary.revision != expected_revision:
                raise RevisionConflictError(
                    f"Revision conflict: expected {expected_revision}, current {summary.revision}",
                    latest_revision=summary.revision,
                )

            target_idx = None
            for idx, c in enumerate(summary.claims):
                if c.claim_id == claim_id:
                    target_idx = idx
                    break

            if target_idx is None:
                raise ResourceNotFoundError(f"Claim {claim_id} not found in summary")

            updated_claim = rate_claim_fit(summary.claims[target_idx], fit_rating)
            new_claims = list(summary.claims)
            new_claims[target_idx] = updated_claim

            updated_summary = EssenceSummaryDraft(
                summary_id=summary.summary_id,
                session_id=summary.session_id,
                statement=summary.statement,
                claims=new_claims,
                unresolved_topics=summary.unresolved_topics,
                coverage_report=summary.coverage_report,
                context_hash=summary.context_hash,
                revision=summary.revision + 1,
                created_at_iso=summary.created_at_iso,
            )

            uow.sessions.save_summary(updated_summary)
            uow.commit()

            return _to_summary_view(updated_summary)

    def edit_claim(
        self,
        session_id: str,
        claim_id: str,
        new_text: str,
        expected_revision: Optional[int] = None,
    ) -> SummaryView:
        with self._uow_factory.open() as uow:
            summary = uow.sessions.get_summary(session_id)
            if not summary:
                raise ResourceNotFoundError(f"Summary draft for session {session_id} not found")

            if expected_revision is not None and summary.revision != expected_revision:
                raise RevisionConflictError(
                    f"Revision conflict: expected {expected_revision}, current {summary.revision}",
                    latest_revision=summary.revision,
                )

            target_idx = None
            for idx, c in enumerate(summary.claims):
                if c.claim_id == claim_id:
                    target_idx = idx
                    break

            if target_idx is None:
                raise ResourceNotFoundError(f"Claim {claim_id} not found in summary")

            updated_claim = edit_claim_text(summary.claims[target_idx], new_text)
            new_claims = list(summary.claims)
            new_claims[target_idx] = updated_claim

            updated_summary = EssenceSummaryDraft(
                summary_id=summary.summary_id,
                session_id=summary.session_id,
                statement=summary.statement,
                claims=new_claims,
                unresolved_topics=summary.unresolved_topics,
                coverage_report=summary.coverage_report,
                context_hash=summary.context_hash,
                revision=summary.revision + 1,
                created_at_iso=summary.created_at_iso,
            )

            uow.sessions.save_summary(updated_summary)
            uow.commit()

            return _to_summary_view(updated_summary)

    def exclude_claim(
        self,
        session_id: str,
        claim_id: str,
        expected_revision: Optional[int] = None,
    ) -> SummaryView:
        with self._uow_factory.open() as uow:
            summary = uow.sessions.get_summary(session_id)
            if not summary:
                raise ResourceNotFoundError(f"Summary draft for session {session_id} not found")

            if expected_revision is not None and summary.revision != expected_revision:
                raise RevisionConflictError(
                    f"Revision conflict: expected {expected_revision}, current {summary.revision}",
                    latest_revision=summary.revision,
                )

            target_idx = None
            for idx, c in enumerate(summary.claims):
                if c.claim_id == claim_id:
                    target_idx = idx
                    break

            if target_idx is None:
                raise ResourceNotFoundError(f"Claim {claim_id} not found in summary")

            updated_claim = exclude_claim(summary.claims[target_idx])
            new_claims = list(summary.claims)
            new_claims[target_idx] = updated_claim

            updated_summary = EssenceSummaryDraft(
                summary_id=summary.summary_id,
                session_id=summary.session_id,
                statement=summary.statement,
                claims=new_claims,
                unresolved_topics=summary.unresolved_topics,
                coverage_report=summary.coverage_report,
                context_hash=summary.context_hash,
                revision=summary.revision + 1,
                created_at_iso=summary.created_at_iso,
            )

            uow.sessions.save_summary(updated_summary)
            uow.commit()

            return _to_summary_view(updated_summary)
