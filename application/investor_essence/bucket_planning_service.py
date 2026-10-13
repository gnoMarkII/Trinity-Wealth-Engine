"""Application service for purpose bucket generation, editing, preview, and atomic apply."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from decimal import Decimal
from typing import Any, Dict, List, Optional

from core.investor_essence.models import (
    AllocationBasis,
    AllocationMappingCell,
    ArtifactRef,
    BucketPlanDraft,
    BucketPlanStatus,
    BucketRemappingItem,
    HUNDRED_PERCENT,
    PERCENT_TOLERANCE,
    PurposeBucketDraft,
)
from core.investor_essence.validation import (
    to_decimal_2dp,
    validate_allocation_matrix,
    validate_percent_sum,
)
from application.investor_essence.dto import (
    AllocationApplyReceipt,
    AllocationPreview,
    ApplyAllocationCommand,
    BucketPlanDraftView,
    PreviewAllocationCommand,
)
from application.investor_essence.errors import (
    IdempotencyConflictError,
    PortfolioConflictError,
    ResourceNotFoundError,
    RevisionConflictError,
    ValidationFailedError,
)
from application.investor_essence.ports import (
    BucketGeneratorPort,
    ClockPort,
    IdGeneratorPort,
    InvestorRuntimeUowFactory,
    PortfolioPlanningPort,
)

logger = logging.getLogger(__name__)


def _to_view(draft: BucketPlanDraft) -> BucketPlanDraftView:
    from core.investor_essence.bucket_validation import validate_bucket_plan_completeness

    issues = validate_bucket_plan_completeness(draft)

    return BucketPlanDraftView(
        draft_id=draft.draft_id,
        portfolio_id=draft.portfolio_id,
        essence_ref={
            "document_key": draft.essence_ref.document_key,
            "note_id": draft.essence_ref.note_id,
            "revision_id": draft.essence_ref.revision_id,
            "content_hash": draft.essence_ref.content_hash,
            "artifact_set_hash": draft.essence_ref.artifact_set_hash,
        },
        axis_ref={
            "document_key": draft.axis_ref.document_key,
            "note_id": draft.axis_ref.note_id,
            "revision_id": draft.axis_ref.revision_id,
            "content_hash": draft.axis_ref.content_hash,
            "artifact_set_hash": draft.axis_ref.artifact_set_hash,
        },
        context_ref=draft.context_ref,
        portfolio_checkpoint=draft.portfolio_checkpoint,
        purpose_buckets=[
            {
                "bucket_id": b.bucket_id,
                "name": b.name,
                "role": b.role,
                "color": b.color,
                "target_percent": str(b.target_percent),
                "source_value_ids": b.source_value_ids,
                "source_axis_allocation_ids": b.source_axis_allocation_ids,
                "linked_goal_name": b.linked_goal_name,
                "eligible_assets": b.eligible_assets,
                "cash_inflow_routing_pct": b.cash_inflow_routing_pct,
                "action_rules": b.action_rules,
                "target_amount": str(b.target_amount) if b.target_amount is not None else None,
                "minimum_amount": str(b.minimum_amount) if b.minimum_amount is not None else None,
                "need_by_date": b.need_by_date,
                "inflow_route_mode": b.inflow_route_mode,
                "inflow_priority": b.inflow_priority,
                "when_full_redirect_bucket_id": b.when_full_redirect_bucket_id,
                "goal_id": b.goal_id,
                "schema_version": b.schema_version,
            }
            for b in draft.purpose_buckets
        ],
        allocation_basis=draft.allocation_basis.value,
        mapping_weights=[
            {
                "axis_allocation_id": m.axis_allocation_id,
                "bucket_id": m.bucket_id,
                "portfolio_weight_percent": str(m.portfolio_weight_percent),
            }
            for m in draft.mapping_weights
        ],
        constraints=list(draft.constraints),
        remapping=[
            {
                "old_bucket_id": r.old_bucket_id,
                "target_bucket_id": r.target_bucket_id,
                "affected_holding_count": r.affected_holding_count,
            }
            for r in draft.remapping
        ],
        status=draft.status.value,
        revision=draft.revision,
        validation_issues=issues,
        is_valid=len(issues) == 0,
    )


class BucketPlanningService:
    """Use cases for purpose bucket generation, editing, previewing, and applying to portfolio."""

    def __init__(
        self,
        uow_factory: InvestorRuntimeUowFactory,
        bucket_generator: BucketGeneratorPort,
        portfolio_port: PortfolioPlanningPort,
        clock: ClockPort,
        id_gen: IdGeneratorPort,
    ) -> None:
        self._uow_factory = uow_factory
        self._bucket_generator = bucket_generator
        self._portfolio_port = portfolio_port
        self._clock = clock
        self._id_gen = id_gen

    def create_bucket_plan_draft(
        self,
        portfolio_id: str,
        prompt_version: str = "1.0",
    ) -> BucketPlanDraftView:
        with self._uow_factory.open() as uow:
            scope = f"portfolio:{portfolio_id}"
            axis_pointer = uow.planning.get_confirmed_pointer(scope, "investment_axis")
            if not axis_pointer:
                raise ValidationFailedError(
                    f"Cannot create bucket plan: portfolio '{portfolio_id}' does not have a confirmed investment axis"
                )

            essence_pointer = uow.planning.get_confirmed_pointer("workspace", "investor_essence")
            if not essence_pointer:
                raise ValidationFailedError("Cannot create bucket plan: no confirmed investor essence found")

            # Obtain consistent portfolio checkpoint & snapshot
            p_snap = self._portfolio_port.snapshot(portfolio_id)
            checkpoint_dict = {
                "sequence": p_snap.checkpoint_sequence,
                "state_hash": p_snap.checkpoint_state_hash,
                "as_of": p_snap.as_of,
                "nav_thb": str(p_snap.nav_thb),
            }

            axis_snapshot = uow.planning.get_confirmed_snapshot(scope, "investment_axis") or {}
            goals = []
            try:
                goals = self._portfolio_port.get_goals(portfolio_id)
            except Exception as e:
                logger.warning("Failed to fetch goals for bucket planning: %s", e)

            confirmed_axis_dict = {
                "portfolio_id": portfolio_id,
                "axis_pointer": axis_pointer,
                "basic_policy": axis_snapshot.get("basic_policy", ""),
                "risk_limits": axis_snapshot.get("risk_limits", {}),
                "invest_targets": axis_snapshot.get("invest_targets", []),
                "exclude_targets": axis_snapshot.get("exclude_targets", []),
                "primary_methods": axis_snapshot.get("primary_methods", []),
                "secondary_methods": axis_snapshot.get("secondary_methods", []),
                "investment_horizon": axis_snapshot.get("investment_horizon", ""),
                "allocation_rows": axis_snapshot.get("allocation_rows", []),
                "rebalance_frequency": axis_snapshot.get("rebalance_frequency", ""),
                "role_models": axis_snapshot.get("role_models", []),
                "non_actions": axis_snapshot.get("non_actions", []),
            }

            essence_snapshot = uow.planning.get_confirmed_snapshot("workspace", "investor_essence") or {}
            accepted_claims = essence_snapshot.get("accepted_claims", [])

            holdings_payload = []
            if p_snap.holdings:
                for h in p_snap.holdings:
                    holdings_payload.append({
                        "holding_id": h.holding_id,
                        "symbol": h.symbol,
                        "asset_type": h.asset_type,
                        "market_value_thb": str(h.market_value_thb),
                        "value_thb": str(h.market_value_thb),
                        "weight_percent": str(h.weight_percent),
                        "bucket_id": h.existing_bucket_id,
                        "currency": h.currency,
                    })
            else:
                for h in p_snap.holdings_summary:
                    val_str = str(h.get("market_value_thb") or h.get("value_thb") or "0")
                    holdings_payload.append({
                        "holding_id": h.get("holding_id", f"{portfolio_id}:{h.get('symbol')}"),
                        "symbol": h.get("symbol"),
                        "asset_type": h.get("asset_type", "Unknown"),
                        "market_value_thb": val_str,
                        "value_thb": val_str,
                        "weight_percent": str(h.get("weight_percent", "0")),
                        "bucket_id": h.get("bucket_id"),
                        "currency": h.get("currency", "THB"),
                    })

            portfolio_snapshot_dict = {
                "nav_thb": str(p_snap.nav_thb),
                "cash_thb": str(p_snap.cash_thb),
                "holdings_count": len(holdings_payload),
                "coverage_count": len(holdings_payload),
                "holdings": holdings_payload,
            }

            proposal = self._bucket_generator.generate_buckets(
                confirmed_axis=confirmed_axis_dict,
                prompt_version=prompt_version,
                portfolio_goals=goals,
                portfolio_snapshot=portfolio_snapshot_dict,
                accepted_claims=accepted_claims,
            )

            # Build purpose bucket domain objects
            purpose_buckets = [
                PurposeBucketDraft(
                    bucket_id=p.bucket_id or f"b_{self._id_gen.new_id()}",
                    name=p.name,
                    role=p.role,
                    color=p.color,
                    target_percent=Decimal(p.target_percent),
                    source_value_ids=p.source_value_ids,
                    source_axis_allocation_ids=p.source_axis_allocation_ids,
                    linked_goal_name=p.linked_goal_name,
                    eligible_assets=p.eligible_assets,
                    cash_inflow_routing_pct=p.cash_inflow_routing_pct,
                    action_rules=p.action_rules,
                    target_amount=Decimal(p.target_amount) if getattr(p, "target_amount", None) else None,
                    minimum_amount=Decimal(p.minimum_amount) if getattr(p, "minimum_amount", None) else None,
                    need_by_date=getattr(p, "need_by_date", None),
                    inflow_route_mode=getattr(p, "inflow_route_mode", "fixed_ratio") or "fixed_ratio",
                    inflow_priority=getattr(p, "inflow_priority", 1) or 1,
                    when_full_redirect_bucket_id=getattr(p, "when_full_redirect_bucket_id", None),
                    goal_id=getattr(p, "goal_id", None),
                    schema_version="2.0",
                )
                for p in proposal.purpose_buckets
            ]

            # Build mapping weights
            from core.investor_essence.bucket_validation import (
                reconcile_allocation_matrix,
                reconcile_bucket_remapping,
                validate_bucket_plan_completeness,
            )

            raw_mapping_weights = [
                AllocationMappingCell(
                    axis_allocation_id=mw["axis_allocation_id"],
                    bucket_id=mw["bucket_id"],
                    portfolio_weight_percent=Decimal(str(mw["portfolio_weight_percent"])),
                )
                for mw in proposal.mapping_weights
            ]
            mapping_weights = reconcile_allocation_matrix(
                axis_rows=axis_snapshot.get("allocation_rows", []),
                purpose_buckets=purpose_buckets,
                existing_mapping=raw_mapping_weights,
            )

            # Build initial remappings for existing buckets
            existing_target_ids = {t["bucket_id"] for t in p_snap.targets}
            remapping_items: List[BucketRemappingItem] = []
            for old_id in existing_target_ids:
                # Find matching purpose bucket by name or fallback to unassigned
                matched = next((b for b in purpose_buckets if b.name.lower() in old_id.lower() or old_id.lower() in b.name.lower()), None)
                target_id = matched.bucket_id if matched else None
                affected_count = sum(1 for h in p_snap.holdings_summary if h.get("bucket_id") == old_id)
                remapping_items.append(
                    BucketRemappingItem(
                        old_bucket_id=old_id,
                        target_bucket_id=target_id,
                        affected_holding_count=affected_count,
                    )
                )
            remapping_items = reconcile_bucket_remapping(remapping_items, purpose_buckets)

            draft_id = f"bplan_{portfolio_id}_{self._id_gen.new_id()}"
            now_iso = self._clock.now_utc()

            axis_c_hash = axis_snapshot.get("content_hash") or (axis_pointer.split(":")[-1] if ":" in axis_pointer else "hash")
            axis_conf_id = axis_snapshot.get("artifact_id") or (axis_pointer.split(":")[2] if len(axis_pointer.split(":")) > 2 else "latest")
            ref_ess = axis_snapshot.get("essence_ref") or {}
            if isinstance(ref_ess, dict):
                ess_doc_key = ref_ess.get("document_key", f"investor-essence-{portfolio_id}")
                ess_note_id = ref_ess.get("note_id", "latest")
                ess_c_hash = ref_ess.get("content_hash", "hash")
            else:
                ess_doc_key = str(ref_ess)
                ess_note_id = "latest"
                ess_c_hash = "hash"

            draft = BucketPlanDraft(
                draft_id=draft_id,
                portfolio_id=portfolio_id,
                essence_ref=ArtifactRef(
                    document_key=ess_doc_key,
                    note_id=ess_note_id,
                    revision_id="1",
                    content_hash=ess_c_hash,
                    artifact_set_hash=ess_c_hash,
                ),
                axis_ref=ArtifactRef(
                    document_key=f"investment-axis-{portfolio_id}",
                    note_id=axis_conf_id,
                    revision_id="1",
                    content_hash=axis_c_hash,
                    artifact_set_hash=axis_c_hash,
                ),
                context_ref=axis_snapshot.get("context_ref", f"ctx_{portfolio_id}"),
                portfolio_checkpoint=checkpoint_dict,
                purpose_buckets=purpose_buckets,
                allocation_basis=AllocationBasis(proposal.allocation_basis) if proposal.allocation_basis in AllocationBasis._value2member_map_ else AllocationBasis.PURPOSE,
                mapping_weights=mapping_weights,
                constraints=proposal.constraints,
                remapping=remapping_items,
                status=BucketPlanStatus.DRAFT,
                revision=1,
                created_at_iso=now_iso,
            )

            uow.planning.save_bucket_draft(draft)
            uow.commit()
            return _to_view(draft)

    def get_bucket_plan_draft(self, draft_id: str) -> Optional[BucketPlanDraftView]:
        with self._uow_factory.open() as uow:
            draft = uow.planning.get_bucket_draft(draft_id)
            if not draft:
                return None
            return _to_view(draft)

    def get_latest_bucket_plan_draft(self, portfolio_id: str) -> Optional[BucketPlanDraftView]:
        with self._uow_factory.open() as uow:
            draft = uow.planning.get_latest_bucket_draft(portfolio_id)
            if not draft:
                return None
            return _to_view(draft)

    def update_bucket_plan_draft(
        self,
        draft_id: str,
        updates: Dict[str, Any],
        expected_revision: Optional[int] = None,
    ) -> BucketPlanDraftView:
        with self._uow_factory.open() as uow:
            draft = uow.planning.get_bucket_draft(draft_id)
            if not draft:
                raise ResourceNotFoundError("BucketPlanDraft", draft_id)

            if expected_revision is not None and draft.revision != expected_revision:
                raise RevisionConflictError(
                    resource_id=draft.draft_id,
                    expected_revision=expected_revision,
                    actual_revision=draft.revision,
                )

            purpose_buckets = draft.purpose_buckets
            if "purpose_buckets" in updates:
                purpose_buckets = [
                    PurposeBucketDraft(
                        bucket_id=b["bucket_id"],
                        name=b["name"],
                        role=b.get("role", ""),
                        color=b.get("color", "#3B82F6"),
                        target_percent=Decimal(str(b["target_percent"])),
                        source_value_ids=b.get("source_value_ids", []),
                        source_axis_allocation_ids=b.get("source_axis_allocation_ids", []),
                        linked_goal_name=b.get("linked_goal_name"),
                        eligible_assets=b.get("eligible_assets", []),
                        cash_inflow_routing_pct=str(b.get("cash_inflow_routing_pct")) if b.get("cash_inflow_routing_pct") is not None else None,
                        action_rules=b.get("action_rules", []),
                        target_amount=Decimal(str(b["target_amount"])) if b.get("target_amount") is not None else None,
                        minimum_amount=Decimal(str(b["minimum_amount"])) if b.get("minimum_amount") is not None else None,
                        need_by_date=b.get("need_by_date"),
                        inflow_route_mode=b.get("inflow_route_mode", "fixed_ratio") or "fixed_ratio",
                        inflow_priority=b.get("inflow_priority", 1) or 1,
                        when_full_redirect_bucket_id=b.get("when_full_redirect_bucket_id"),
                        goal_id=b.get("goal_id"),
                        schema_version=b.get("schema_version", "2.0") or "2.0",
                    )
                    for b in updates["purpose_buckets"]
                ]

            mapping_weights = draft.mapping_weights
            if "mapping_weights" in updates:
                mapping_weights = [
                    AllocationMappingCell(
                        axis_allocation_id=m["axis_allocation_id"],
                        bucket_id=m["bucket_id"],
                        portfolio_weight_percent=Decimal(str(m["portfolio_weight_percent"])),
                    )
                    for m in updates["mapping_weights"]
                ]

            remapping = draft.remapping
            if "remapping" in updates:
                remapping = [
                    BucketRemappingItem(
                        old_bucket_id=r["old_bucket_id"],
                        target_bucket_id=r.get("target_bucket_id"),
                        affected_holding_count=r.get("affected_holding_count", 0),
                    )
                    for r in updates["remapping"]
                ]

            scope = f"portfolio:{draft.portfolio_id}"
            axis_snap = uow.planning.get_confirmed_snapshot(scope, "investment_axis") or {}
            axis_rows = axis_snap.get("allocation_rows", [])

            from core.investor_essence.bucket_validation import (
                reconcile_allocation_matrix,
                reconcile_bucket_remapping,
                validate_bucket_plan_completeness,
            )

            mapping_weights = reconcile_allocation_matrix(
                axis_rows=axis_rows,
                purpose_buckets=purpose_buckets,
                existing_mapping=mapping_weights if "mapping_weights" in updates else None,
            )
            remapping = reconcile_bucket_remapping(remapping, purpose_buckets)

            constraints = updates.get("constraints", draft.constraints)

            updated_draft = BucketPlanDraft(
                draft_id=draft.draft_id,
                portfolio_id=draft.portfolio_id,
                essence_ref=draft.essence_ref,
                axis_ref=draft.axis_ref,
                context_ref=draft.context_ref,
                portfolio_checkpoint=draft.portfolio_checkpoint,
                purpose_buckets=purpose_buckets,
                allocation_basis=draft.allocation_basis,
                mapping_weights=mapping_weights,
                constraints=constraints,
                remapping=remapping,
                status=draft.status,
                revision=draft.revision + 1,
                created_at_iso=draft.created_at_iso,
            )

            uow.planning.save_bucket_draft(updated_draft)
            uow.commit()
            return _to_view(updated_draft)

    def preview_bucket_plan(self, draft_id: str) -> AllocationPreview:
        with self._uow_factory.open() as uow:
            draft = uow.planning.get_bucket_draft(draft_id)
            if not draft:
                raise ResourceNotFoundError("BucketPlanDraft", draft_id)

            from core.investor_essence.bucket_validation import (
                reconcile_allocation_matrix,
                reconcile_bucket_remapping,
                validate_bucket_plan_completeness,
            )

            scope = f"portfolio:{draft.portfolio_id}"
            axis_snap = uow.planning.get_confirmed_snapshot(scope, "investment_axis") or {}
            axis_rows = axis_snap.get("allocation_rows", [])
            axis_targets = {}
            for r in axis_rows:
                try:
                    axis_targets[r["allocation_id"]] = Decimal(str(r["target_percent"]))
                except Exception:
                    pass

            domain_issues = validate_bucket_plan_completeness(
                draft, axis_allocation_targets=axis_targets if axis_targets else None
            )

            # Auto-heal draft if matrix or remapping contains orphan bucket IDs that do not exist
            seen_bids = {b.bucket_id for b in draft.purpose_buckets}
            has_orphan_bids = any(c.bucket_id not in seen_bids for c in draft.mapping_weights)
            has_orphan_remapping = any(r.target_bucket_id is not None and r.target_bucket_id not in seen_bids for r in draft.remapping)
            if has_orphan_bids or has_orphan_remapping:
                healed_mw = reconcile_allocation_matrix(
                    axis_rows=axis_rows,
                    purpose_buckets=draft.purpose_buckets,
                    existing_mapping=draft.mapping_weights,
                )
                healed_rem = reconcile_bucket_remapping(draft.remapping, draft.purpose_buckets)
                draft = BucketPlanDraft(
                    draft_id=draft.draft_id,
                    portfolio_id=draft.portfolio_id,
                    essence_ref=draft.essence_ref,
                    axis_ref=draft.axis_ref,
                    context_ref=draft.context_ref,
                    portfolio_checkpoint=draft.portfolio_checkpoint,
                    purpose_buckets=draft.purpose_buckets,
                    allocation_basis=draft.allocation_basis,
                    mapping_weights=healed_mw,
                    constraints=draft.constraints,
                    remapping=healed_rem,
                    status=draft.status,
                    revision=draft.revision,
                    created_at_iso=draft.created_at_iso,
                )
                uow.planning.save_bucket_draft(draft)
                uow.commit()
                domain_issues = validate_bucket_plan_completeness(
                    draft, axis_allocation_targets=axis_targets if axis_targets else None
                )

            target_rows = [
                {
                    "bucket_id": b.bucket_id,
                    "name": b.name,
                    "target_percent": float(b.target_percent),
                    "color": b.color,
                }
                for b in draft.purpose_buckets
            ]
            remapping_rows = [
                {
                    "old_bucket_id": r.old_bucket_id,
                    "target_bucket_id": r.target_bucket_id,
                }
                for r in draft.remapping
            ]
            mapping_weights = [
                {
                    "axis_allocation_id": m.axis_allocation_id,
                    "bucket_id": m.bucket_id,
                    "portfolio_weight_percent": float(m.portfolio_weight_percent),
                }
                for m in draft.mapping_weights
            ]

            cmd = PreviewAllocationCommand(
                portfolio_id=draft.portfolio_id,
                expected_checkpoint_sequence=draft.portfolio_checkpoint.get("sequence", 0),
                expected_checkpoint_state_hash=draft.portfolio_checkpoint.get("state_hash", ""),
                target_rows=target_rows,
                mapping_weights=mapping_weights,
                remapping=remapping_rows,
            )
            preview_res = self._portfolio_port.preview(cmd)

            all_issues = list(dict.fromkeys(preview_res.issues + domain_issues))
            preview_id = f"prev_{self._id_gen.new_id()}"

            return AllocationPreview(
                checkpoint_sequence=preview_res.checkpoint_sequence,
                checkpoint_state_hash=preview_res.checkpoint_state_hash,
                validated_targets=preview_res.validated_targets,
                affected_holdings=preview_res.affected_holdings,
                before_allocation=preview_res.before_allocation,
                after_allocation=preview_res.after_allocation,
                issues=all_issues,
                payload_hash=preview_res.payload_hash,
                preview_id=preview_id,
                draft_id=draft.draft_id,
                draft_revision=draft.revision,
                source_refs={
                    "essence_ref": draft.essence_ref.content_hash,
                    "axis_ref": draft.axis_ref.content_hash,
                },
            )

    def apply_bucket_plan(
        self,
        draft_id: str,
        idempotency_key: str = "",
        expected_draft_revision: Optional[int] = None,
        expected_checkpoint_sequence: Optional[int] = None,
        expected_checkpoint_state_hash: Optional[str] = None,
        preview_id: Optional[str] = None,
    ) -> AllocationApplyReceipt:
        with self._uow_factory.open() as uow:
            draft = uow.planning.get_bucket_draft(draft_id)
            if not draft:
                raise ResourceNotFoundError("BucketPlanDraft", draft_id)

            scope = f"portfolio:{draft.portfolio_id}"
            if expected_draft_revision is not None:
                target_rev = expected_draft_revision
            elif draft.status == BucketPlanStatus.APPLIED:
                target_rev = draft.revision - 1
            else:
                target_rev = draft.revision

            req_hash = hashlib.sha256(
                f"apply_bucket_plan:{draft_id}:{target_rev}:{expected_checkpoint_sequence}:{preview_id}".encode("utf-8")
            ).hexdigest()

            # 1. Idempotency Check (AC19, AC20 - check before conflicts so replay succeeds)
            if idempotency_key:
                cached = uow.intents.get_command_receipt(scope, "apply_bucket_plan", idempotency_key)
                if cached:
                    if cached.get("request_hash") != req_hash:
                        raise IdempotencyConflictError(idempotency_key)
                    res = cached["result"]
                    return AllocationApplyReceipt(
                        command_id=res["command_id"],
                        portfolio_id=res["portfolio_id"],
                        request_hash=res["request_hash"],
                        applied_sequence=res["applied_sequence"],
                        applied_state_hash=res["applied_state_hash"],
                        applied_at_iso=res["applied_at_iso"],
                        canonical_status=res["canonical_status"],
                        projection_status=res["projection_status"],
                        warnings=res.get("warnings", []),
                    )

                # Reconciler: if prior intent exists and portfolio committed (WP12 task 7)
                prior_intent = uow.intents.get_intent_by_idempotency(scope, "apply_bucket_plan", idempotency_key)
                if prior_intent:
                    reconciled = self._portfolio_port.get_apply_receipt(draft.portfolio_id, prior_intent["intent_id"])
                    if reconciled:
                        applied_draft = BucketPlanDraft(
                            draft_id=draft.draft_id,
                            portfolio_id=draft.portfolio_id,
                            essence_ref=draft.essence_ref,
                            axis_ref=draft.axis_ref,
                            context_ref=draft.context_ref,
                            portfolio_checkpoint=draft.portfolio_checkpoint,
                            purpose_buckets=draft.purpose_buckets,
                            allocation_basis=draft.allocation_basis,
                            mapping_weights=draft.mapping_weights,
                            constraints=draft.constraints,
                            remapping=draft.remapping,
                            status=BucketPlanStatus.APPLIED,
                            revision=draft.revision + 1 if draft.status != BucketPlanStatus.APPLIED else draft.revision,
                            created_at_iso=draft.created_at_iso,
                        )
                        uow.planning.save_bucket_draft(applied_draft)
                        uow.intents.save_command_receipt(
                            scope=scope,
                            use_case="apply_bucket_plan",
                            idempotency_key=idempotency_key,
                            request_hash=req_hash,
                            result=asdict(reconciled),
                        )
                        uow.intents.update_intent_receipt(prior_intent["intent_id"], asdict(reconciled), "committed")
                        uow.commit()
                        return reconciled

            # 2. Check draft revision (AC16, AC17)
            if expected_draft_revision is not None and draft.revision != expected_draft_revision:
                raise RevisionConflictError(
                    resource_id=draft.draft_id,
                    expected_revision=expected_draft_revision,
                    actual_revision=draft.revision,
                )

            # 3. Check checkpoint staleness against current portfolio state (AC16)
            p_snap = self._portfolio_port.snapshot(draft.portfolio_id)
            if expected_checkpoint_sequence is not None and p_snap.checkpoint_sequence != expected_checkpoint_sequence:
                raise PortfolioConflictError(
                    portfolio_id=draft.portfolio_id,
                    expected_seq=expected_checkpoint_sequence,
                    actual_seq=p_snap.checkpoint_sequence,
                )

            # 4. Validation (AC12, AC13)
            from core.investor_essence.bucket_validation import (
                reconcile_allocation_matrix,
                reconcile_bucket_remapping,
                validate_bucket_plan_completeness,
            )

            axis_snap = uow.planning.get_confirmed_snapshot(scope, "investment_axis") or {}
            axis_rows = axis_snap.get("allocation_rows", [])
            axis_targets = {}
            for r in axis_rows:
                try:
                    axis_targets[r["allocation_id"]] = Decimal(str(r["target_percent"]))
                except Exception:
                    pass

            issues = validate_bucket_plan_completeness(
                draft, axis_allocation_targets=axis_targets if axis_targets else None
            )
            seen_bids = {b.bucket_id for b in draft.purpose_buckets}
            has_orphan_bids = any(c.bucket_id not in seen_bids for c in draft.mapping_weights)
            has_orphan_remapping = any(r.target_bucket_id is not None and r.target_bucket_id not in seen_bids for r in draft.remapping)
            if has_orphan_bids or has_orphan_remapping:
                healed_mw = reconcile_allocation_matrix(
                    axis_rows=axis_rows,
                    purpose_buckets=draft.purpose_buckets,
                    existing_mapping=draft.mapping_weights,
                )
                healed_rem = reconcile_bucket_remapping(draft.remapping, draft.purpose_buckets)
                draft = BucketPlanDraft(
                    draft_id=draft.draft_id,
                    portfolio_id=draft.portfolio_id,
                    essence_ref=draft.essence_ref,
                    axis_ref=draft.axis_ref,
                    context_ref=draft.context_ref,
                    portfolio_checkpoint=draft.portfolio_checkpoint,
                    purpose_buckets=draft.purpose_buckets,
                    allocation_basis=draft.allocation_basis,
                    mapping_weights=healed_mw,
                    constraints=draft.constraints,
                    remapping=healed_rem,
                    status=draft.status,
                    revision=draft.revision,
                    created_at_iso=draft.created_at_iso,
                )
                uow.planning.save_bucket_draft(draft)
                uow.commit()
                issues = validate_bucket_plan_completeness(
                    draft, axis_allocation_targets=axis_targets if axis_targets else None
                )

            if issues:
                raise ValidationFailedError(
                    f"Cannot apply bucket plan with validation issues:\n" + "\n".join(issues)
                )

            # 5. Compile portfolio targets & apply
            target_rows = [
                {
                    "bucket_id": b.bucket_id,
                    "name": b.name,
                    "target_percent": float(b.target_percent),
                    "color": b.color,
                }
                for b in draft.purpose_buckets
            ]
            remapping_rows = [
                {
                    "old_bucket_id": r.old_bucket_id,
                    "target_bucket_id": r.target_bucket_id,
                }
                for r in draft.remapping
            ]
            mapping_weights = [
                {
                    "axis_allocation_id": m.axis_allocation_id,
                    "bucket_id": m.bucket_id,
                    "portfolio_weight_percent": float(m.portfolio_weight_percent),
                }
                for m in draft.mapping_weights
            ]

            cmd_id = self._id_gen.new_id()
            if idempotency_key:
                uow.intents.save_intent(
                    intent_id=cmd_id,
                    scope=scope,
                    kind="apply_bucket_plan",
                    idempotency_key=idempotency_key,
                    request_hash=req_hash,
                    payload={"draft_id": draft.draft_id, "command_id": cmd_id, "revision": draft.revision},
                )

            cmd = ApplyAllocationCommand(
                command_id=cmd_id,
                request_hash=req_hash,
                portfolio_id=draft.portfolio_id,
                expected_checkpoint_sequence=expected_checkpoint_sequence if expected_checkpoint_sequence is not None else draft.portfolio_checkpoint.get("sequence", 0),
                expected_checkpoint_state_hash=expected_checkpoint_state_hash if expected_checkpoint_state_hash is not None else draft.portfolio_checkpoint.get("state_hash", ""),
                target_rows=target_rows,
                mapping_weights=mapping_weights,
                remapping=remapping_rows,
                accepted_policy_snapshot={
                    "draft_id": draft.draft_id,
                    "allocation_basis": draft.allocation_basis.value,
                    "purpose_buckets": [
                        {
                            "bucket_id": b.bucket_id,
                            "name": b.name,
                            "role": b.role,
                            "target_percent": str(b.target_percent),
                            "color": b.color,
                            "eligible_assets": b.eligible_assets,
                            "action_rules": b.action_rules,
                            "cash_inflow_routing_pct": b.cash_inflow_routing_pct,
                        }
                        for b in draft.purpose_buckets
                    ],
                },
                source_refs={
                    "essence_ref": draft.essence_ref.document_key,
                    "axis_ref": draft.axis_ref.document_key,
                    "draft_id": draft.draft_id,
                    "draft_revision": str(draft.revision),
                },
                expected_draft_revision=expected_draft_revision,
                preview_id=preview_id,
                idempotency_key=idempotency_key,
            )

            receipt = self._portfolio_port.apply(cmd)

            # 6. Mark draft as applied
            applied_draft = BucketPlanDraft(
                draft_id=draft.draft_id,
                portfolio_id=draft.portfolio_id,
                essence_ref=draft.essence_ref,
                axis_ref=draft.axis_ref,
                context_ref=draft.context_ref,
                portfolio_checkpoint=draft.portfolio_checkpoint,
                purpose_buckets=draft.purpose_buckets,
                allocation_basis=draft.allocation_basis,
                mapping_weights=draft.mapping_weights,
                constraints=draft.constraints,
                remapping=draft.remapping,
                status=BucketPlanStatus.APPLIED,
                revision=draft.revision + 1,
                created_at_iso=draft.created_at_iso,
            )
            uow.planning.save_bucket_draft(applied_draft)

            # 7. Save command receipt for idempotency & update intent
            if idempotency_key:
                uow.intents.save_command_receipt(
                    scope=scope,
                    use_case="apply_bucket_plan",
                    idempotency_key=idempotency_key,
                    request_hash=req_hash,
                    result=asdict(receipt),
                )
                uow.intents.update_intent_receipt(cmd_id, asdict(receipt), "committed")

            uow.commit()
            return receipt
