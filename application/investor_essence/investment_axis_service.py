"""Application service for drafting, revising, and confirming 8-section investment policies."""
from __future__ import annotations

import hashlib
import json
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from core.investor_essence.investment_axis import (
    build_confirmed_axis,
    validate_axis_completeness,
)
from core.investor_essence.models import (
    AllocationBasis,
    AllocationPlanRow,
    ArtifactRef,
    InvestmentAxisDraft,
    NumericPolicyField,
    NumericPolicyOrigin,
)
from application.investor_essence.dto import AxisDraftView, ConfirmationView
from application.investor_essence.errors import (
    AxisIncompleteError,
    CurrentRefConflictError,
    IdempotencyConflictError,
    ResourceNotFoundError,
    RevisionConflictError,
    ValidationFailedError,
)
from application.investor_essence.ports import (
    AxisGeneratorPort,
    ClockPort,
    ConfirmedKnowledgeReaderPort,
    IdGeneratorPort,
    InvestorRuntimeUowFactory,
    KnowledgeWritePort,
)

logger = logging.getLogger(__name__)


def _to_axis_view(draft: InvestmentAxisDraft) -> AxisDraftView:
    issues = validate_axis_completeness(draft)
    return AxisDraftView(
        draft_id=draft.draft_id,
        portfolio_id=draft.portfolio_id,
        essence_ref={
            "document_key": draft.essence_ref.document_key,
            "note_id": draft.essence_ref.note_id,
            "revision_id": draft.essence_ref.revision_id,
            "content_hash": draft.essence_ref.content_hash,
            "artifact_set_hash": draft.essence_ref.artifact_set_hash,
        },
        context_ref=draft.context_ref,
        basic_policy=draft.basic_policy,
        risk_limits={
            k: {
                "field_id": v.field_id,
                "value": str(v.value) if v.value is not None else None,
                "unit": v.unit,
                "calculation_basis": v.calculation_basis,
                "origin": v.origin.value,
                "assumptions": v.assumptions,
                "is_confirmed": v.is_confirmed,
            }
            for k, v in draft.risk_limits.items()
        },
        invest_targets=list(draft.invest_targets),
        exclude_targets=list(draft.exclude_targets),
        primary_methods=list(draft.primary_methods),
        secondary_methods=list(draft.secondary_methods),
        investment_horizon=draft.investment_horizon,
        allocation_basis=draft.allocation_basis.value,
        allocation_rows=[
            {
                "allocation_id": r.allocation_id,
                "category_name": r.category_name,
                "target_percent": str(r.target_percent),
                "role_description": r.role_description,
            }
            for r in draft.allocation_rows
        ],
        rebalance_frequency=draft.rebalance_frequency,
        role_models=list(draft.role_models),
        non_actions=list(draft.non_actions),
        assumptions=list(draft.assumptions),
        clarifications=list(draft.clarifications),
        revision=draft.revision,
        completeness_issues=issues,
        is_complete=len(issues) == 0,
    )


class InvestmentAxisService:
    """Use cases for drafting and confirming 8-section investment axes."""

    def __init__(
        self,
        uow_factory: InvestorRuntimeUowFactory,
        axis_generator: AxisGeneratorPort,
        clock: ClockPort,
        id_gen: IdGeneratorPort,
        reader_port: Optional[ConfirmedKnowledgeReaderPort] = None,
        knowledge_write: Optional[KnowledgeWritePort] = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._axis_generator = axis_generator
        self._clock = clock
        self._id_gen = id_gen
        self._reader_port = reader_port
        self._knowledge_write = knowledge_write

    def create_axis_draft(
        self,
        portfolio_id: str,
        prompt_version: str = "1.0",
    ) -> AxisDraftView:
        with self._uow_factory.open() as uow:
            essence_pointer = uow.planning.get_confirmed_pointer("workspace", "investor_essence")
            if not essence_pointer:
                raise ValidationFailedError(
                    "Cannot draft investment axis: no confirmed investor essence found in workspace"
                )

            # Retrieve context snapshot
            ctx = uow.planning.get_context_snapshot(portfolio_id)
            ctx_ref = ctx.snapshot_id if ctx else f"ctx_default_{portfolio_id}"
            ctx_dict = {
                "portfolio_id": portfolio_id,
                "horizon_years": str(ctx.horizon_years) if ctx and ctx.horizon_years is not None else None,
                "target_use_amount": str(ctx.target_use_amount) if ctx and ctx.target_use_amount is not None else None,
                "target_use_range": ctx.target_use_range if ctx else None,
                "emergency_reserves_amount": str(ctx.emergency_reserves_amount) if ctx and ctx.emergency_reserves_amount is not None else None,
                "obligations_monthly": str(ctx.obligations_monthly) if ctx and ctx.obligations_monthly is not None else None,
                "unknown_fields": ctx.unknown_fields if ctx else [],
            }

            accepted_claims: List[Dict[str, Any]] = []
            curr_session = uow.sessions.get_current("workspace")
            if curr_session:
                draft_summary = uow.sessions.get_summary(curr_session.session_id)
                if draft_summary:
                    accepted_claims = [
                        {"claim_id": c.claim_id, "text": c.effective_text}
                        for c in draft_summary.claims
                        if c.is_accepted
                    ]

            proposal = self._axis_generator.generate_axis(
                accepted_claims=accepted_claims,
                financial_context=ctx_dict,
                prompt_version=prompt_version,
            )

            risk_limits_dict: Dict[str, NumericPolicyField] = {}
            for k, rf in proposal.risk_limits.items():
                val_dec = Decimal(rf.value) if rf.value is not None else None
                risk_limits_dict[k] = NumericPolicyField(
                    field_id=rf.field_id,
                    value=val_dec,
                    unit=rf.unit,
                    calculation_basis=rf.calculation_basis,
                    origin=NumericPolicyOrigin(rf.origin) if rf.origin in NumericPolicyOrigin._value2member_map_ else NumericPolicyOrigin.AI_PROPOSAL,
                    source_refs=rf.source_refs,
                    assumptions=rf.assumptions,
                    is_confirmed=rf.is_confirmed,
                )

            alloc_rows = [
                AllocationPlanRow(
                    allocation_id=r.allocation_id,
                    category_name=r.category_name,
                    target_percent=Decimal(r.target_percent),
                    role_description=r.role_description,
                )
                for r in proposal.allocation_rows
            ]

            draft_id = f"axis_draft_{portfolio_id}_{self._id_gen.new_id()}"
            now_iso = self._clock.now_utc()
            parts = essence_pointer.split(":")
            note_id = parts[2] if len(parts) > 2 else "latest"
            c_hash = parts[3] if len(parts) > 3 else "hash"

            draft = InvestmentAxisDraft(
                draft_id=draft_id,
                portfolio_id=portfolio_id,
                essence_ref=ArtifactRef(
                    document_key=f"investor-essence-{note_id}",
                    note_id=note_id,
                    revision_id="1",
                    content_hash=c_hash,
                    artifact_set_hash=c_hash,
                ),
                context_ref=ctx_ref,
                basic_policy=proposal.basic_policy,
                risk_limits=risk_limits_dict,
                invest_targets=proposal.invest_targets,
                exclude_targets=proposal.exclude_targets,
                primary_methods=proposal.primary_methods,
                secondary_methods=proposal.secondary_methods,
                investment_horizon=proposal.investment_horizon,
                allocation_basis=AllocationBasis(proposal.allocation_basis) if proposal.allocation_basis in AllocationBasis._value2member_map_ else AllocationBasis.PURPOSE,
                allocation_rows=alloc_rows,
                rebalance_frequency=proposal.rebalance_frequency,
                role_models=proposal.role_models,
                non_actions=proposal.non_actions,
                numeric_fields=risk_limits_dict,
                assumptions=proposal.assumptions,
                clarifications=proposal.clarifications,
                revision=1,
                created_at_iso=now_iso,
            )

            uow.planning.save_axis_draft(draft)
            uow.commit()
            return _to_axis_view(draft)

    def get_axis_draft(self, draft_id: str) -> Optional[AxisDraftView]:
        with self._uow_factory.open() as uow:
            draft = uow.planning.get_axis_draft(draft_id)
            if not draft:
                return None
            return _to_axis_view(draft)

    def update_axis_draft(
        self,
        draft_id: str,
        updates: Dict[str, Any],
        expected_revision: Optional[int] = None,
    ) -> AxisDraftView:
        with self._uow_factory.open() as uow:
            draft = uow.planning.get_axis_draft(draft_id)
            if not draft:
                raise ResourceNotFoundError("InvestmentAxisDraft", draft_id)

            if expected_revision is not None and draft.revision != expected_revision:
                raise RevisionConflictError(
                    resource_id=draft.draft_id,
                    expected_revision=expected_revision,
                    actual_revision=draft.revision,
                )

            basic_pol = updates.get("basic_policy", draft.basic_policy)
            invest_t = updates.get("invest_targets", draft.invest_targets)
            exclude_t = updates.get("exclude_targets", draft.exclude_targets)
            prim_m = updates.get("primary_methods", draft.primary_methods)
            sec_m = updates.get("secondary_methods", draft.secondary_methods)
            inv_hor = updates.get("investment_horizon", draft.investment_horizon)
            reb_freq = updates.get("rebalance_frequency", draft.rebalance_frequency)
            role_m = updates.get("role_models", draft.role_models)
            non_act = updates.get("non_actions", draft.non_actions)

            # Update allocation rows if present
            alloc_rows = draft.allocation_rows
            if "allocation_rows" in updates:
                alloc_rows = [
                    AllocationPlanRow(
                        allocation_id=r.get("allocation_id", f"alloc_{idx}"),
                        category_name=r["category_name"],
                        target_percent=Decimal(str(r["target_percent"])),
                        role_description=r.get("role_description", ""),
                    )
                    for idx, r in enumerate(updates["allocation_rows"], start=1)
                ]

            # Update risk limits / numeric fields
            risk_lims = dict(draft.risk_limits)
            if "risk_limits" in updates:
                for k, v in updates["risk_limits"].items():
                    val = Decimal(str(v["value"])) if v.get("value") is not None else None
                    risk_lims[k] = NumericPolicyField(
                        field_id=k,
                        value=val,
                        unit=v.get("unit", "%"),
                        calculation_basis=v.get("calculation_basis", "NAV"),
                        origin=NumericPolicyOrigin.USER_INPUT,
                        assumptions=v.get("assumptions", ""),
                        is_confirmed=bool(v.get("is_confirmed", True)),
                    )

            updated_draft = InvestmentAxisDraft(
                draft_id=draft.draft_id,
                portfolio_id=draft.portfolio_id,
                essence_ref=draft.essence_ref,
                context_ref=draft.context_ref,
                basic_policy=basic_pol,
                risk_limits=risk_lims,
                invest_targets=invest_t,
                exclude_targets=exclude_t,
                primary_methods=prim_m,
                secondary_methods=sec_m,
                investment_horizon=inv_hor,
                allocation_basis=draft.allocation_basis,
                allocation_rows=alloc_rows,
                rebalance_frequency=reb_freq,
                role_models=role_m,
                non_actions=non_act,
                numeric_fields=risk_lims,
                assumptions=draft.assumptions,
                clarifications=draft.clarifications,
                revision=draft.revision + 1,
                created_at_iso=draft.created_at_iso,
            )

            uow.planning.save_axis_draft(updated_draft)
            uow.commit()
            return _to_axis_view(updated_draft)

    def confirm_axis(
        self,
        draft_id: str,
        idempotency_key: str = "",
    ) -> ConfirmationView:
        with self._uow_factory.open() as uow:
            draft = uow.planning.get_axis_draft(draft_id)
            if not draft:
                raise ResourceNotFoundError("InvestmentAxisDraft", draft_id)

            scope = f"portfolio:{draft.portfolio_id}"
            req_hash = hashlib.sha256(f"confirm_axis:{draft_id}".encode("utf-8")).hexdigest()

            # 1. Idempotency Check
            if idempotency_key:
                cached = uow.intents.get_command_receipt(scope, "confirm_axis", idempotency_key)
                if cached:
                    if cached.get("request_hash") != req_hash:
                        raise IdempotencyConflictError(idempotency_key)
                    res = cached["result"]
                    return ConfirmationView(
                        confirmation_id=res["confirmation_id"],
                        status=res["status"],
                        accepted_claims_count=res["accepted_claims_count"],
                        artifact_ref=res.get("artifact_ref"),
                        message=res.get("message", "Axis confirmed (idempotent replay)"),
                    )

            # 2. Completeness validation
            issues = validate_axis_completeness(draft)
            if issues:
                raise AxisIncompleteError(issues)

            # 3. Build confirmed axis
            conf_id = self._id_gen.new_id()
            now_iso = self._clock.now_utc()
            confirmed = build_confirmed_axis(
                confirmation_id=conf_id,
                draft=draft,
                confirmed_at_iso=now_iso,
            )

            # 4. CAS Pointer Update
            current_pointer = uow.planning.get_confirmed_pointer(scope, "investment_axis")
            doc_key = f"investment-axis-{draft.portfolio_id}"
            artifact_ref_str = f"vault:investment_axis:{conf_id}:{confirmed.content_hash[:16]}"
            cas_ok = uow.planning.set_confirmed_pointer(
                scope=scope,
                kind="investment_axis",
                artifact_ref=artifact_ref_str,
                expected_ref=current_pointer,
            )
            if not cas_ok:
                actual = uow.planning.get_confirmed_pointer(scope, "investment_axis")
                raise CurrentRefConflictError(
                    scope=scope,
                    expected_ref=current_pointer,
                    actual_ref=actual,
                )

            # 5. Optional Knowledge Broker write
            if self._knowledge_write is not None:
                try:
                    self._knowledge_write.submit({
                        "command_id": f"cmd-confirm-axis-{conf_id}",
                        "idempotency_key": idempotency_key or f"idem-axis-{conf_id}",
                        "operation": "upsert_note",
                        "document_key": doc_key,
                        "entity_type": "investment_axis",
                        "portfolio_id": draft.portfolio_id,
                        "content_hash": confirmed.content_hash,
                        "sections": confirmed.complete_sections,
                        "confirmed_at": now_iso,
                    })
                except Exception as ex:
                    logger.warning("Optional knowledge write broker axis submission failed: %s", ex)

            # 6. Create Confirmation View & Record Receipt
            view = ConfirmationView(
                confirmation_id=conf_id,
                status="committed",
                accepted_claims_count=len(confirmed.complete_sections),
                artifact_ref={
                    "document_key": doc_key,
                    "note_id": conf_id,
                    "revision_id": "1",
                    "content_hash": confirmed.content_hash,
                    "artifact_set_hash": confirmed.content_hash,
                },
                message=f"Investment axis confirmed for portfolio {draft.portfolio_id}",
            )

            if idempotency_key:
                from dataclasses import asdict
                uow.intents.save_command_receipt(
                    scope=scope,
                    use_case="confirm_axis",
                    idempotency_key=idempotency_key,
                    request_hash=req_hash,
                    result=asdict(view),
                )

            uow.commit()
            return view

    def get_current_confirmed_axis(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        with self._uow_factory.open() as uow:
            scope = f"portfolio:{portfolio_id}"
            pointer = uow.planning.get_confirmed_pointer(scope, "investment_axis")
            if not pointer:
                return None
            return {
                "scope": scope,
                "portfolio_id": portfolio_id,
                "kind": "investment_axis",
                "pointer": pointer,
            }
