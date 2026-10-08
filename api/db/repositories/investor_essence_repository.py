"""SQLite concrete repository adapters for Investor Essence runtime.

Implements:
- SessionRepositoryPort
- PlanningRepositoryPort
- OperationRepositoryPort
- IntentRepositoryPort
"""
from __future__ import annotations

import json
import sqlite3
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from core.investor_essence.models import (
    AllocationBasis,
    AllocationMappingCell,
    AllocationPlanRow,
    ArtifactRef,
    Answer,
    AnswerKind,
    BucketPlanDraft,
    BucketPlanStatus,
    ContentOption,
    CoverageItem,
    EssenceClaim,
    EssenceConfirmationSnapshot,
    EssenceSummaryDraft,
    EvidenceRef,
    EvidenceType,
    FinancialContextSnapshot,
    FinancialReadinessIssue,
    FitRating,
    GeneratedQuestion,
    InvestmentAxisDraft,
    NumericPolicyField,
    NumericPolicyOrigin,
    PerClaimConfirmationSnapshot,
    PurposeBucketDraft,
    SessionStatus,
    SourceKind,
)
from core.investor_essence.interview import EssenceSession
from application.investor_essence.dto import OperationView
from application.investor_essence.ports import (
    IntentRepositoryPort,
    OperationRepositoryPort,
    PlanningRepositoryPort,
    SessionRepositoryPort,
)


def _decimal_to_str(val: Any) -> Any:
    if isinstance(val, Decimal):
        return str(val)
    if isinstance(val, list):
        return [_decimal_to_str(x) for x in val]
    if isinstance(val, dict):
        return {k: _decimal_to_str(v) for k, v in val.items()}
    return val


def _opt_decimal(val: Any) -> Optional[Decimal]:
    if val is None:
        return None
    return Decimal(str(val))


class SqliteSessionRepository(SessionRepositoryPort):
    """Connection-bound implementation of SessionRepositoryPort."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get(self, session_id: str) -> Optional[EssenceSession]:
        row = self._conn.execute(
            "SELECT session_id, scope, status, active_branch_id, revision, schema_version, "
            "interview_version, created_at, updated_at "
            "FROM investor_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None

        # Fetch questions
        q_rows = self._conn.execute(
            "SELECT question_id, branch_id, sequence_no, text, options_json, "
            "evidence_type, coverage_topics_json, is_clarification, created_at "
            "FROM investor_questions WHERE session_id = ? ORDER BY sequence_no ASC",
            (session_id,),
        ).fetchall()

        questions: List[GeneratedQuestion] = []
        clarifications: List[GeneratedQuestion] = []
        branch_history: Dict[str, List[str]] = {}

        for qr in q_rows:
            raw_options = json.loads(qr[4])
            options = [
                ContentOption(
                    option_id=opt["option_id"],
                    option_key=opt["option_key"],
                    text=opt["text"],
                )
                for opt in raw_options
            ]
            gq = GeneratedQuestion(
                question_id=qr[0],
                sequence_no=qr[2],
                text=qr[3],
                options=options,
                evidence_type=EvidenceType(qr[5]),
                coverage_topics=json.loads(qr[6]),
                is_clarification=bool(qr[7]),
                created_at_iso=qr[8] or "",
            )
            branch = qr[1]
            if gq.is_clarification:
                clarifications.append(gq)
            else:
                questions.append(gq)
                if branch not in branch_history:
                    branch_history[branch] = []
                branch_history[branch].append(gq.question_id)

        # Fetch answers
        a_rows = self._conn.execute(
            "SELECT answer_id, question_id, branch_id, answer_kind, option_id, "
            "free_text, evidence_type, created_at "
            "FROM investor_answers WHERE session_id = ?",
            (session_id,),
        ).fetchall()

        answers: Dict[str, Answer] = {}
        clarification_answers: Dict[str, Answer] = {}
        clarification_ids = {c.question_id for c in clarifications}

        for ar in a_rows:
            ans = Answer(
                answer_id=ar[0],
                question_id=ar[1],
                answer_kind=AnswerKind(ar[3]),
                option_id=ar[4],
                free_text=ar[5],
                evidence_type=EvidenceType(ar[6]),
                created_at_iso=ar[7] or "",
            )
            if ans.question_id in clarification_ids:
                clarification_answers[ans.question_id] = ans
            else:
                answers[ans.question_id] = ans

        return EssenceSession(
            session_id=row[0],
            scope=row[1],
            status=SessionStatus(row[2]),
            active_branch_id=row[3],
            revision=row[4],
            schema_version=row[5],
            interview_version=row[6],
            questions=questions,
            answers=answers,
            branch_history=branch_history,
            clarifications=clarifications,
            clarification_answers=clarification_answers,
            created_at_iso=row[7] or "",
            updated_at_iso=row[8] or "",
        )

    def get_current(self, scope: str = "workspace") -> Optional[EssenceSession]:
        row = self._conn.execute(
            "SELECT session_id FROM investor_sessions WHERE scope = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (scope,),
        ).fetchone()
        if not row:
            return None
        return self.get(row[0])

    def save(self, session: EssenceSession) -> None:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        created_at = session.created_at_iso or now_iso
        updated_at = now_iso
        ctx_hash = session.compute_context_hash()

        self._conn.execute(
            "INSERT INTO investor_sessions (session_id, scope, status, active_branch_id, "
            "revision, schema_version, interview_version, context_hash, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "status = excluded.status, "
            "active_branch_id = excluded.active_branch_id, "
            "revision = excluded.revision, "
            "context_hash = excluded.context_hash, "
            "updated_at = excluded.updated_at",
            (
                session.session_id,
                session.scope,
                session.status.value,
                session.active_branch_id,
                session.revision,
                session.schema_version,
                session.interview_version,
                ctx_hash,
                created_at,
                updated_at,
            ),
        )

        # Upsert questions
        all_q: List[Tuple[GeneratedQuestion, str]] = []
        for branch_id, q_ids in session.branch_history.items():
            for q_id in q_ids:
                q = session.get_question(q_id)
                if q:
                    all_q.append((q, branch_id))

        for c in session.clarifications:
            all_q.append((c, session.active_branch_id))

        for q, branch in all_q:
            opts_json = json.dumps(
                [{"option_id": o.option_id, "option_key": o.option_key, "text": o.text} for o in q.options],
                ensure_ascii=False,
            )
            topics_json = json.dumps(q.coverage_topics, ensure_ascii=False)
            self._conn.execute(
                "INSERT INTO investor_questions (question_id, session_id, branch_id, sequence_no, "
                "text, options_json, evidence_type, coverage_topics_json, is_clarification, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(question_id) DO UPDATE SET "
                "branch_id = excluded.branch_id",
                (
                    q.question_id,
                    session.session_id,
                    branch,
                    q.sequence_no,
                    q.text,
                    opts_json,
                    q.evidence_type.value,
                    topics_json,
                    1 if q.is_clarification else 0,
                    q.created_at_iso or now_iso,
                ),
            )

        # Upsert answers
        all_answers = list(session.answers.values()) + list(session.clarification_answers.values())
        for a in all_answers:
            self._conn.execute(
                "INSERT INTO investor_answers (answer_id, session_id, question_id, branch_id, "
                "answer_kind, option_id, free_text, evidence_type, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(answer_id) DO UPDATE SET "
                "option_id = excluded.option_id, "
                "free_text = excluded.free_text, "
                "answer_kind = excluded.answer_kind",
                (
                    a.answer_id,
                    session.session_id,
                    a.question_id,
                    session.active_branch_id,
                    a.answer_kind.value,
                    a.option_id,
                    a.free_text,
                    a.evidence_type.value,
                    a.created_at_iso or now_iso,
                ),
            )

    def get_summary(self, session_id: str) -> Optional[EssenceSummaryDraft]:
        row = self._conn.execute(
            "SELECT statement, claims_json, unresolved_topics_json, coverage_report_json, "
            "evidence_snapshot_hash, revision, created_at, updated_at "
            "FROM investor_summary_drafts WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None

        raw_claims = json.loads(row[1])
        claims: List[EssenceClaim] = []
        for rc in raw_claims:
            ev_refs = [
                EvidenceRef(
                    answer_id=er["answer_id"],
                    question_id=er["question_id"],
                    revision=er["revision"],
                    quote=er["quote"],
                    evidence_type=EvidenceType(er["evidence_type"]),
                )
                for er in rc.get("evidence_refs", [])
            ]
            counter_refs = [
                EvidenceRef(
                    answer_id=er["answer_id"],
                    question_id=er["question_id"],
                    revision=er["revision"],
                    quote=er["quote"],
                    evidence_type=EvidenceType(er["evidence_type"]),
                )
                for er in rc.get("counter_evidence_refs", [])
            ]
            fit_val = rc.get("fit_rating")
            claim = EssenceClaim(
                claim_id=rc["claim_id"],
                text=rc["text"],
                source_kind=SourceKind(rc["source_kind"]),
                evidence_refs=ev_refs,
                counter_evidence_refs=counter_refs,
                fit_rating=FitRating(fit_val) if fit_val else None,
                edited_text=rc.get("edited_text"),
                text_revision=rc.get("text_revision", 1),
                is_accepted=rc.get("is_accepted", False),
            )
            claims.append(claim)

        raw_coverage = json.loads(row[3])
        coverage_items = [
            CoverageItem(
                topic=ci["topic"],
                status=ci["status"],
                supporting_answer_ids=ci.get("supporting_answer_ids", []),
            )
            for ci in raw_coverage
        ]

        return EssenceSummaryDraft(
            summary_id=f"summary_{session_id}",
            session_id=session_id,
            statement=row[0],
            claims=claims,
            unresolved_topics=json.loads(row[2]),
            coverage_report=coverage_items,
            context_hash=row[4],
            revision=row[5],
            created_at_iso=row[6] or "",
        )

    def save_summary(self, summary: EssenceSummaryDraft) -> None:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        claims_data = []
        for c in summary.claims:
            claims_data.append(
                {
                    "claim_id": c.claim_id,
                    "text": c.text,
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
                    "counter_evidence_refs": [
                        {
                            "answer_id": er.answer_id,
                            "question_id": er.question_id,
                            "revision": er.revision,
                            "quote": er.quote,
                            "evidence_type": er.evidence_type.value,
                        }
                        for er in c.counter_evidence_refs
                    ],
                }
            )

        coverage_data = [
            {
                "topic": ci.topic,
                "status": ci.status,
                "supporting_answer_ids": ci.supporting_answer_ids,
            }
            for ci in summary.coverage_report
        ]

        self._conn.execute(
            "INSERT INTO investor_summary_drafts (session_id, statement, claims_json, "
            "unresolved_topics_json, coverage_report_json, evidence_snapshot_hash, "
            "revision, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "statement = excluded.statement, "
            "claims_json = excluded.claims_json, "
            "unresolved_topics_json = excluded.unresolved_topics_json, "
            "coverage_report_json = excluded.coverage_report_json, "
            "evidence_snapshot_hash = excluded.evidence_snapshot_hash, "
            "revision = excluded.revision, "
            "updated_at = excluded.updated_at",
            (
                summary.session_id,
                summary.statement,
                json.dumps(claims_data, ensure_ascii=False),
                json.dumps(summary.unresolved_topics, ensure_ascii=False),
                json.dumps(coverage_data, ensure_ascii=False),
                summary.context_hash,
                summary.revision,
                summary.created_at_iso or now_iso,
                now_iso,
            ),
        )


class SqlitePlanningRepository(PlanningRepositoryPort):
    """Connection-bound implementation of PlanningRepositoryPort."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save_context_snapshot(self, snapshot: FinancialContextSnapshot) -> None:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        data = {
            "snapshot_id": snapshot.snapshot_id,
            "portfolio_id": snapshot.portfolio_id,
            "horizon_years": str(snapshot.horizon_years) if snapshot.horizon_years is not None else None,
            "target_use_amount": str(snapshot.target_use_amount) if snapshot.target_use_amount is not None else None,
            "target_use_range": snapshot.target_use_range,
            "target_use_timeline": snapshot.target_use_timeline,
            "emergency_reserves_amount": str(snapshot.emergency_reserves_amount) if snapshot.emergency_reserves_amount is not None else None,
            "emergency_reserves_months": str(snapshot.emergency_reserves_months) if snapshot.emergency_reserves_months is not None else None,
            "obligations_monthly": str(snapshot.obligations_monthly) if snapshot.obligations_monthly is not None else None,
            "obligations_description": snapshot.obligations_description,
            "withdrawal_frequency": snapshot.withdrawal_frequency,
            "withdrawal_amount": str(snapshot.withdrawal_amount) if snapshot.withdrawal_amount is not None else None,
            "experience_description": snapshot.experience_description,
            "unknown_fields": snapshot.unknown_fields,
            "as_of": snapshot.as_of,
            "source": snapshot.source,
            "portfolio_checkpoint_refs": snapshot.portfolio_checkpoint_refs,
            "readiness_issues": [
                {"code": i.code, "severity": i.severity, "message": i.message, "field_path": i.field_path}
                for i in snapshot.readiness_issues
            ],
            "is_ready_for_numeric_policy": snapshot.is_ready_for_numeric_policy,
        }
        self._conn.execute(
            "INSERT INTO investor_context_snapshots (snapshot_id, portfolio_id, data_json, is_ready, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(snapshot_id) DO UPDATE SET "
            "data_json = excluded.data_json, "
            "is_ready = excluded.is_ready",
            (
                snapshot.snapshot_id,
                snapshot.portfolio_id,
                json.dumps(data, ensure_ascii=False),
                1 if snapshot.is_ready_for_numeric_policy else 0,
                snapshot.as_of or now_iso,
            ),
        )

    def get_context_snapshot(self, portfolio_id: str) -> Optional[FinancialContextSnapshot]:
        row = self._conn.execute(
            "SELECT data_json FROM investor_context_snapshots WHERE portfolio_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (portfolio_id,),
        ).fetchone()
        if not row:
            return None

        d = json.loads(row[0])
        issues = [
            FinancialReadinessIssue(
                code=i["code"],
                severity=i["severity"],
                message=i["message"],
                field_path=i["field_path"],
            )
            for i in d.get("readiness_issues", [])
        ]
        return FinancialContextSnapshot(
            snapshot_id=d["snapshot_id"],
            portfolio_id=d["portfolio_id"],
            horizon_years=_opt_decimal(d.get("horizon_years")),
            target_use_amount=_opt_decimal(d.get("target_use_amount")),
            target_use_range=d.get("target_use_range"),
            target_use_timeline=d.get("target_use_timeline"),
            emergency_reserves_amount=_opt_decimal(d.get("emergency_reserves_amount")),
            emergency_reserves_months=_opt_decimal(d.get("emergency_reserves_months")),
            obligations_monthly=_opt_decimal(d.get("obligations_monthly")),
            obligations_description=d.get("obligations_description"),
            withdrawal_frequency=d.get("withdrawal_frequency"),
            withdrawal_amount=_opt_decimal(d.get("withdrawal_amount")),
            experience_description=d.get("experience_description"),
            unknown_fields=d.get("unknown_fields", []),
            as_of=d.get("as_of", ""),
            source=d.get("source", "user_reported"),
            portfolio_checkpoint_refs=d.get("portfolio_checkpoint_refs", {}),
            readiness_issues=issues,
            is_ready_for_numeric_policy=d.get("is_ready_for_numeric_policy", False),
        )

    def save_axis_draft(self, draft: InvestmentAxisDraft) -> None:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        data = {
            "draft_id": draft.draft_id,
            "portfolio_id": draft.portfolio_id,
            "essence_ref": {
                "document_key": draft.essence_ref.document_key,
                "note_id": draft.essence_ref.note_id,
                "revision_id": draft.essence_ref.revision_id,
                "content_hash": draft.essence_ref.content_hash,
                "artifact_set_hash": draft.essence_ref.artifact_set_hash,
            },
            "context_ref": draft.context_ref,
            "basic_policy": draft.basic_policy,
            "risk_limits": {
                k: {
                    "field_id": v.field_id,
                    "value": str(v.value) if v.value is not None else None,
                    "unit": v.unit,
                    "calculation_basis": v.calculation_basis,
                    "origin": v.origin.value,
                    "is_confirmed": v.is_confirmed,
                    "assumptions": v.assumptions,
                    "confirmed_text_revision": v.confirmed_text_revision,
                    "source_refs": v.source_refs,
                }
                for k, v in draft.risk_limits.items()
            },
            "invest_targets": draft.invest_targets,
            "exclude_targets": draft.exclude_targets,
            "primary_methods": draft.primary_methods,
            "secondary_methods": draft.secondary_methods,
            "investment_horizon": draft.investment_horizon,
            "rebalance_frequency": draft.rebalance_frequency,
            "allocation_basis": draft.allocation_basis.value,
            "allocation_rows": [
                {
                    "allocation_id": r.allocation_id,
                    "category_name": r.category_name,
                    "target_percent": str(r.target_percent),
                    "role_description": r.role_description,
                }
                for r in draft.allocation_rows
            ],
            "role_models": draft.role_models,
            "non_actions": draft.non_actions,
            "numeric_fields": {
                k: {
                    "field_id": v.field_id,
                    "value": str(v.value) if v.value is not None else None,
                    "unit": v.unit,
                    "calculation_basis": v.calculation_basis,
                    "origin": v.origin.value,
                    "is_confirmed": v.is_confirmed,
                }
                for k, v in draft.numeric_fields.items()
            },
            "assumptions": draft.assumptions,
            "clarifications": draft.clarifications,
            "revision": draft.revision,
        }

        self._conn.execute(
            "INSERT INTO investor_plan_drafts (draft_id, draft_type, portfolio_id, data_json, "
            "revision, created_at, updated_at) "
            "VALUES (?, 'axis', ?, ?, ?, ?, ?) "
            "ON CONFLICT(draft_id) DO UPDATE SET "
            "data_json = excluded.data_json, "
            "revision = excluded.revision, "
            "updated_at = excluded.updated_at",
            (
                draft.draft_id,
                draft.portfolio_id,
                json.dumps(data, ensure_ascii=False),
                draft.revision,
                draft.created_at_iso or now_iso,
                now_iso,
            ),
        )

    def get_axis_draft(self, draft_id: str) -> Optional[InvestmentAxisDraft]:
        row = self._conn.execute(
            "SELECT data_json, created_at FROM investor_plan_drafts WHERE draft_id = ? AND draft_type = 'axis'",
            (draft_id,),
        ).fetchone()
        if not row:
            return None

        d = json.loads(row[0])
        ref_d = d["essence_ref"]
        essence_ref = ArtifactRef(
            document_key=ref_d["document_key"],
            note_id=ref_d["note_id"],
            revision_id=ref_d["revision_id"],
            content_hash=ref_d["content_hash"],
            artifact_set_hash=ref_d["artifact_set_hash"],
        )

        def _parse_numeric_fields(raw: Dict[str, Any]) -> Dict[str, NumericPolicyField]:
            res = {}
            for k, v in raw.items():
                res[k] = NumericPolicyField(
                    field_id=v["field_id"],
                    value=_opt_decimal(v.get("value")),
                    unit=v["unit"],
                    calculation_basis=v["calculation_basis"],
                    origin=NumericPolicyOrigin(v["origin"]),
                    source_refs=v.get("source_refs", []),
                    assumptions=v.get("assumptions", ""),
                    confirmed_text_revision=v.get("confirmed_text_revision", 1),
                    is_confirmed=v.get("is_confirmed", False),
                )
            return res

        rows = [
            AllocationPlanRow(
                allocation_id=r["allocation_id"],
                category_name=r["category_name"],
                target_percent=Decimal(r["target_percent"]),
                role_description=r["role_description"],
            )
            for r in d.get("allocation_rows", [])
        ]

        return InvestmentAxisDraft(
            draft_id=d["draft_id"],
            portfolio_id=d["portfolio_id"],
            essence_ref=essence_ref,
            context_ref=d["context_ref"],
            basic_policy=d["basic_policy"],
            risk_limits=_parse_numeric_fields(d.get("risk_limits", {})),
            invest_targets=d.get("invest_targets", []),
            exclude_targets=d.get("exclude_targets", []),
            primary_methods=d.get("primary_methods", []),
            secondary_methods=d.get("secondary_methods", []),
            investment_horizon=d.get("investment_horizon", ""),
            allocation_basis=AllocationBasis(d["allocation_basis"]),
            allocation_rows=rows,
            rebalance_frequency=d.get("rebalance_frequency", ""),
            role_models=d.get("role_models", []),
            non_actions=d.get("non_actions", []),
            numeric_fields=_parse_numeric_fields(d.get("numeric_fields", {})),
            assumptions=d.get("assumptions", []),
            clarifications=d.get("clarifications", []),
            revision=d.get("revision", 1),
            created_at_iso=row[1] or "",
        )

    def save_bucket_draft(self, draft: BucketPlanDraft) -> None:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        data = {
            "draft_id": draft.draft_id,
            "portfolio_id": draft.portfolio_id,
            "axis_ref": {
                "document_key": draft.axis_ref.document_key,
                "note_id": draft.axis_ref.note_id,
                "revision_id": draft.axis_ref.revision_id,
                "content_hash": draft.axis_ref.content_hash,
                "artifact_set_hash": draft.axis_ref.artifact_set_hash,
            },
            "status": draft.status.value,
            "purpose_buckets": [
                {
                    "bucket_id": b.bucket_id,
                    "name": b.name,
                    "role": b.role,
                    "color_hex": b.color_hex,
                    "target_percent": str(b.target_percent),
                    "source_value_ids": b.source_value_ids,
                    "source_axis_allocation_ids": b.source_axis_allocation_ids,
                }
                for b in draft.purpose_buckets
            ],
            "allocation_mapping": [
                {
                    "axis_allocation_id": c.axis_allocation_id,
                    "bucket_id": c.bucket_id,
                    "portfolio_weight_percent": str(c.portfolio_weight_percent),
                }
                for c in draft.allocation_mapping
            ],
            "holding_remapping": draft.holding_remapping,
            "revision": draft.revision,
        }

        self._conn.execute(
            "INSERT INTO investor_plan_drafts (draft_id, draft_type, portfolio_id, data_json, "
            "revision, created_at, updated_at) "
            "VALUES (?, 'bucket', ?, ?, ?, ?, ?) "
            "ON CONFLICT(draft_id) DO UPDATE SET "
            "data_json = excluded.data_json, "
            "revision = excluded.revision, "
            "updated_at = excluded.updated_at",
            (
                draft.draft_id,
                draft.portfolio_id,
                json.dumps(data, ensure_ascii=False),
                draft.revision,
                draft.created_at_iso or now_iso,
                now_iso,
            ),
        )

    def get_bucket_draft(self, draft_id: str) -> Optional[BucketPlanDraft]:
        row = self._conn.execute(
            "SELECT data_json, created_at FROM investor_plan_drafts WHERE draft_id = ? AND draft_type = 'bucket'",
            (draft_id,),
        ).fetchone()
        if not row:
            return None

        d = json.loads(row[0])
        ref_d = d["axis_ref"]
        axis_ref = ArtifactRef(
            document_key=ref_d["document_key"],
            note_id=ref_d["note_id"],
            revision_id=ref_d["revision_id"],
            content_hash=ref_d["content_hash"],
            artifact_set_hash=ref_d["artifact_set_hash"],
        )
        buckets = [
            PurposeBucketDraft(
                bucket_id=b["bucket_id"],
                name=b["name"],
                role=b["role"],
                color_hex=b["color_hex"],
                target_percent=Decimal(b["target_percent"]),
                source_value_ids=b.get("source_value_ids", []),
                source_axis_allocation_ids=b.get("source_axis_allocation_ids", []),
            )
            for b in d.get("purpose_buckets", [])
        ]
        cells = [
            AllocationMappingCell(
                axis_allocation_id=c["axis_allocation_id"],
                bucket_id=c["bucket_id"],
                portfolio_weight_percent=Decimal(c["portfolio_weight_percent"]),
            )
            for c in d.get("allocation_mapping", [])
        ]

        return BucketPlanDraft(
            draft_id=d["draft_id"],
            portfolio_id=d["portfolio_id"],
            axis_ref=axis_ref,
            status=BucketPlanStatus(d["status"]),
            purpose_buckets=buckets,
            allocation_mapping=cells,
            holding_remapping=d.get("holding_remapping", {}),
            revision=d.get("revision", 1),
            created_at_iso=row[1] or "",
        )

    def get_confirmed_pointer(self, scope: str, kind: str) -> Optional[str]:
        pointer_key = f"{scope}:{kind}"
        row = self._conn.execute(
            "SELECT artifact_ref_json FROM investor_confirmed_refs WHERE pointer_key = ?",
            (pointer_key,),
        ).fetchone()
        if not row:
            return None
        return row[0]

    def set_confirmed_pointer(
        self, scope: str, kind: str, artifact_ref: str, expected_ref: Optional[str]
    ) -> bool:
        pointer_key = f"{scope}:{kind}"
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        current = self.get_confirmed_pointer(scope, kind)
        if expected_ref is None:
            if current is not None:
                return False
            self._conn.execute(
                "INSERT INTO investor_confirmed_refs (pointer_key, scope, ref_type, "
                "artifact_ref_json, snapshot_json, revision, confirmed_at) "
                "VALUES (?, ?, ?, ?, '{}', 1, ?)",
                (pointer_key, scope, kind, artifact_ref, now_iso),
            )
            return True
        else:
            if current != expected_ref:
                return False
            cursor = self._conn.execute(
                "UPDATE investor_confirmed_refs SET artifact_ref_json = ?, revision = revision + 1, "
                "confirmed_at = ? WHERE pointer_key = ? AND artifact_ref_json = ?",
                (artifact_ref, now_iso, pointer_key, expected_ref),
            )
            return cursor.rowcount == 1


class SqliteOperationRepository(OperationRepositoryPort):
    """Connection-bound implementation of OperationRepositoryPort."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def enqueue(
        self,
        operation_id: str,
        stage: str,
        resource_id: str,
        resource_revision: int,
        input_hash: str,
        prompt_version: str,
        frozen_input: Dict[str, Any],
    ) -> None:
        now = time.time()
        payload = {
            "stage": stage,
            "resource_id": resource_id,
            "resource_revision": resource_revision,
            "input_hash": input_hash,
            "prompt_version": prompt_version,
            "frozen_input": frozen_input,
        }
        self._conn.execute(
            "INSERT INTO investor_operations (operation_id, task_type, resource_id, status, "
            "payload_json, created_at, updated_at) "
            "VALUES (?, ?, ?, 'queued', ?, ?, ?)",
            (operation_id, stage, resource_id, json.dumps(payload, ensure_ascii=False), now, now),
        )

    def get(self, operation_id: str) -> Optional[OperationView]:
        row = self._conn.execute(
            "SELECT operation_id, task_type, resource_id, status, result_json, error_message, "
            "fencing_token, attempts, payload_json FROM investor_operations WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if not row:
            return None

        result_dict = json.loads(row[4]) if row[4] else None
        payload = json.loads(row[8]) if row[8] else {}
        return OperationView(
            operation_id=row[0],
            stage=row[1],
            resource_id=row[2],
            resource_revision=payload.get("resource_revision", 1),
            status=row[3],
            attempt=row[7],
            poll_url=f"/api/investor/operations/{row[0]}",
            error_message=row[5],
            result=result_dict,
        )

    def claim_lease(self, worker_id: str, lease_seconds: int) -> Optional[Dict[str, Any]]:
        now = time.time()
        expires = now + lease_seconds

        row = self._conn.execute(
            "SELECT operation_id, fencing_token FROM investor_operations "
            "WHERE status = 'queued' OR (status = 'running' AND lease_expires_at < ?) "
            "ORDER BY created_at ASC LIMIT 1",
            (now,),
        ).fetchone()
        if not row:
            return None

        op_id, current_fence = row[0], row[1]
        new_fence = current_fence + 1

        cursor = self._conn.execute(
            "UPDATE investor_operations SET status = 'running', lease_owner = ?, "
            "lease_expires_at = ?, fencing_token = ?, attempts = attempts + 1, updated_at = ? "
            "WHERE operation_id = ? AND fencing_token = ?",
            (worker_id, expires, new_fence, now, op_id, current_fence),
        )
        if cursor.rowcount != 1:
            return None

        op_row = self._conn.execute(
            "SELECT operation_id, task_type, resource_id, payload_json, fencing_token "
            "FROM investor_operations WHERE operation_id = ?",
            (op_id,),
        ).fetchone()
        return {
            "operation_id": op_row[0],
            "task_type": op_row[1],
            "resource_id": op_row[2],
            "payload": json.loads(op_row[3]),
            "fencing_token": op_row[4],
        }

    def complete_with_fence(self, operation_id: str, fence: int, result: Dict[str, Any]) -> bool:
        now = time.time()
        cursor = self._conn.execute(
            "UPDATE investor_operations SET status = 'succeeded', result_json = ?, updated_at = ? "
            "WHERE operation_id = ? AND fencing_token = ?",
            (json.dumps(result, ensure_ascii=False), now, operation_id, fence),
        )
        return cursor.rowcount == 1

    def fail_with_fence(
        self, operation_id: str, fence: int, error_code: str, error_message: str, retryable: bool
    ) -> bool:
        now = time.time()
        status = "retryable" if retryable else "failed"
        full_err = f"[{error_code}] {error_message}"
        cursor = self._conn.execute(
            "UPDATE investor_operations SET status = ?, error_message = ?, updated_at = ? "
            "WHERE operation_id = ? AND fencing_token = ?",
            (status, full_err, now, operation_id, fence),
        )
        return cursor.rowcount == 1


class SqliteIntentRepository(IntentRepositoryPort):
    """Connection-bound implementation of IntentRepositoryPort."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save_intent(
        self,
        intent_id: str,
        scope: str,
        kind: str,
        idempotency_key: str,
        request_hash: str,
        payload: Dict[str, Any],
    ) -> None:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        fingerprint = f"{scope}:{kind}:{idempotency_key}:{request_hash}"
        self._conn.execute(
            "INSERT INTO investor_intents (intent_id, intent_type, resource_id, status, "
            "request_fingerprint, receipt_json, created_at, updated_at) "
            "VALUES (?, ?, ?, 'pending', ?, ?, ?, ?) "
            "ON CONFLICT(intent_id) DO UPDATE SET updated_at = excluded.updated_at",
            (intent_id, kind, scope, fingerprint, json.dumps(payload, ensure_ascii=False), now_iso, now_iso),
        )

    def get_intent_by_idempotency(
        self, scope: str, kind: str, idempotency_key: str
    ) -> Optional[Dict[str, Any]]:
        prefix = f"{scope}:{kind}:{idempotency_key}:%"
        row = self._conn.execute(
            "SELECT intent_id, status, receipt_json, request_fingerprint "
            "FROM investor_intents WHERE request_fingerprint LIKE ? LIMIT 1",
            (prefix,),
        ).fetchone()
        if not row:
            return None
        return {
            "intent_id": row[0],
            "status": row[1],
            "receipt": json.loads(row[2]) if row[2] else None,
            "request_fingerprint": row[3],
        }

    def update_intent_receipt(self, intent_id: str, receipt: Dict[str, Any], status: str) -> None:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._conn.execute(
            "UPDATE investor_intents SET status = ?, receipt_json = ?, updated_at = ? "
            "WHERE intent_id = ?",
            (status, json.dumps(receipt, ensure_ascii=False), now_iso, intent_id),
        )

    def save_command_receipt(
        self, scope: str, use_case: str, idempotency_key: str, request_hash: str, result: Dict[str, Any]
    ) -> None:
        key = f"{scope}:{use_case}:{idempotency_key}"
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._conn.execute(
            "INSERT INTO investor_command_receipts (receipt_key, scope, use_case, idempotency_key, "
            "request_hash, result_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(receipt_key) DO UPDATE SET "
            "result_json = excluded.result_json",
            (key, scope, use_case, idempotency_key, request_hash, json.dumps(result, ensure_ascii=False), now_iso),
        )

    def get_command_receipt(
        self, scope: str, use_case: str, idempotency_key: str
    ) -> Optional[Dict[str, Any]]:
        key = f"{scope}:{use_case}:{idempotency_key}"
        row = self._conn.execute(
            "SELECT result_json, request_hash FROM investor_command_receipts WHERE receipt_key = ?",
            (key,),
        ).fetchone()
        if not row:
            return None
        return {
            "result": json.loads(row[0]),
            "request_hash": row[1],
        }
