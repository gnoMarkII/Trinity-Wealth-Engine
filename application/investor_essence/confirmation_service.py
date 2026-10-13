"""Application service for confirming investor essence with idempotent CAS pointers."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from core.investor_essence.claims import build_confirmation_snapshot
from core.investor_essence.models import (
    ConfirmedSnapshotRecord,
    SessionStatus,
)
from application.investor_essence.dto import ConfirmationView
from application.investor_essence.errors import (
    CurrentRefConflictError,
    IdempotencyConflictError,
    ResourceNotFoundError,
    RevisionConflictError,
    ValidationFailedError,
)
from application.investor_essence.ports import (
    ClockPort,
    IdGeneratorPort,
    InvestorRuntimeUowFactory,
    KnowledgeWritePort,
)

logger = logging.getLogger(__name__)


def _compute_request_hash(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ConfirmationService:
    """Use cases for finalizing and confirming investor essence."""

    def __init__(
        self,
        uow_factory: InvestorRuntimeUowFactory,
        clock: ClockPort,
        id_gen: IdGeneratorPort,
        knowledge_write: Optional[KnowledgeWritePort] = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._clock = clock
        self._id_gen = id_gen
        self._knowledge_write = knowledge_write

    def confirm_essence(
        self,
        session_id: str,
        accepted_claim_ids: Optional[List[str]] = None,
        expected_summary_revision: Optional[int] = None,
        idempotency_key: str = "",
    ) -> ConfirmationView:
        with self._uow_factory.open() as uow:
            req_payload = {
                "session_id": session_id,
                "accepted_claim_ids": sorted(accepted_claim_ids or []),
                "expected_summary_revision": expected_summary_revision,
            }
            req_hash = _compute_request_hash(req_payload)

            # 1. Idempotency Check
            if idempotency_key:
                cached = uow.intents.get_command_receipt("workspace", "confirm_essence", idempotency_key)
                if cached:
                    if cached.get("request_hash") != req_hash:
                        raise IdempotencyConflictError(idempotency_key)
                    res = cached["result"]
                    return ConfirmationView(
                        confirmation_id=res["confirmation_id"],
                        status=res["status"],
                        accepted_claims_count=res["accepted_claims_count"],
                        artifact_ref=res.get("artifact_ref"),
                        message=res.get("message", "Essence confirmed (idempotent replay)"),
                    )

            # 2. Retrieve Summary Draft
            summary = uow.sessions.get_summary(session_id)
            if not summary:
                raise ResourceNotFoundError("EssenceSummaryDraft", session_id)

            if expected_summary_revision is not None and summary.revision != expected_summary_revision:
                raise RevisionConflictError(
                    resource_id=summary.summary_id,
                    expected_revision=expected_summary_revision,
                    actual_revision=summary.revision,
                )

            # 3. Build snapshot using pure domain logic
            confirmation_id = self._id_gen.new_id()
            now_iso = self._clock.now_utc()
            try:
                snapshot = build_confirmation_snapshot(
                    confirmation_id=confirmation_id,
                    session_id=session_id,
                    claims=summary.claims,
                    unresolved_topics=summary.unresolved_topics,
                    evidence_snapshot_hash=summary.context_hash,
                    confirmed_at_iso=now_iso,
                    accepted_claim_ids=accepted_claim_ids,
                )
            except ValueError as ex:
                raise ValidationFailedError(str(ex))

            # 4. Compute Content Hash of confirmed snapshot
            serialized_claims = [
                {
                    "claim_id": c.claim_id,
                    "final_text": c.final_text,
                    "text_revision": c.text_revision,
                    "source_kind": c.source_kind.value,
                    "fit_rating": c.fit_rating.value,
                }
                for c in snapshot.per_claim_snapshot
            ]
            content_hash = hashlib.sha256(
                json.dumps(serialized_claims, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()

            doc_key = f"investor-essence-{confirmation_id}"
            artifact_ref_str = f"vault:investor_essence:{confirmation_id}:{content_hash[:16]}"
            artifact_ref_dict = {
                "document_key": doc_key,
                "note_id": confirmation_id,
                "revision_id": "1",
                "content_hash": content_hash,
                "artifact_set_hash": content_hash,
            }

            snapshot_dict = {
                "confirmation_id": confirmation_id,
                "session_id": session_id,
                "statement": summary.statement,
                "accepted_claims": [
                    {
                        "claim_id": c.claim_id,
                        "text": c.text,
                        "effective_text": c.effective_text,
                        "final_text": c.effective_text,
                        "source_kind": c.source_kind.value,
                        "fit_rating": c.fit_rating.value if c.fit_rating else "exact",
                        "text_revision": c.text_revision,
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
                    }
                    for c in snapshot.accepted_claims
                ],
                "unresolved_topics": list(snapshot.unresolved_topics),
                "coverage_report": [
                    {
                        "topic": ci.topic,
                        "status": ci.status,
                        "supporting_answer_ids": ci.supporting_answer_ids,
                    }
                    for ci in summary.coverage_report
                ],
                "content_hash": content_hash,
                "confirmed_at_iso": now_iso,
            }

            # 5. Persist immutable confirmed snapshot
            record = ConfirmedSnapshotRecord(
                snapshot_id=confirmation_id,
                scope="workspace",
                scope_id="workspace",
                kind="investor_essence",
                source_id=session_id,
                source_revision=summary.revision,
                content_hash=content_hash,
                snapshot_json=json.dumps(snapshot_dict, ensure_ascii=False),
                confirmed_at_iso=now_iso,
            )
            uow.planning.save_confirmed_snapshot(record)

            # 6. CAS Update on Planning Repository
            current_pointer = uow.planning.get_confirmed_pointer("workspace", "investor_essence")
            cas_ok = uow.planning.set_confirmed_pointer(
                scope="workspace",
                kind="investor_essence",
                artifact_ref=artifact_ref_str,
                expected_ref=current_pointer,
                snapshot=snapshot_dict,
            )
            if not cas_ok:
                actual_pointer = uow.planning.get_confirmed_pointer("workspace", "investor_essence")
                raise CurrentRefConflictError(
                    scope="workspace",
                    expected_ref=current_pointer,
                    actual_ref=actual_pointer,
                )

            # 6. Optional Knowledge Write Broker call
            if self._knowledge_write is not None:
                try:
                    from tools.investor_essence.adapters.knowledge_publication_adapter import (
                        KnowledgePublicationAdapter,
                    )
                    pub_adapter = KnowledgePublicationAdapter(self._knowledge_write)
                    pub_ref = pub_adapter.publish_essence(
                        confirmation_id=confirmation_id,
                        snapshot_dict=snapshot_dict,
                        idempotency_key=idempotency_key,
                    )
                    if pub_ref:
                        artifact_ref_dict = pub_ref
                except Exception as e:
                    logger.warning("Optional knowledge write broker submission failed: %s", e)

            # 7. Update Session Status
            session = uow.sessions.get(session_id)
            if session:
                session.status = SessionStatus.CONFIRMED
                session.updated_at_iso = now_iso
                uow.sessions.save(session)

            # 8. Create View & Record Idempotency Receipt
            view = ConfirmationView(
                confirmation_id=confirmation_id,
                status="committed",
                accepted_claims_count=len(snapshot.accepted_claims),
                artifact_ref=artifact_ref_dict,
                message="Investor essence confirmed and published successfully",
            )

            if idempotency_key:
                uow.intents.save_command_receipt(
                    scope="workspace",
                    use_case="confirm_essence",
                    idempotency_key=idempotency_key,
                    request_hash=req_hash,
                    result=asdict(view),
                )

            uow.commit()
            return view

    def get_current_confirmed_essence(self) -> Optional[Dict[str, Any]]:
        with self._uow_factory.open() as uow:
            pointer = uow.planning.get_confirmed_pointer("workspace", "investor_essence")
            if not pointer:
                return None
            snapshot = uow.planning.get_confirmed_snapshot("workspace", "investor_essence")
            if snapshot:
                res = dict(snapshot)
                res["scope"] = "workspace"
                res["kind"] = "investor_essence"
                res["pointer"] = pointer
                return res
            return {
                "scope": "workspace",
                "kind": "investor_essence",
                "pointer": pointer,
            }
