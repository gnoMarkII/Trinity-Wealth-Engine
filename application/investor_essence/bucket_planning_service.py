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
    issues: List[str] = []

    # Validate purpose buckets total == 100%
    bucket_pcts = [b.target_percent for b in draft.purpose_buckets]
    total_b_pct = sum(bucket_pcts, Decimal("0.00"))
    if abs(total_b_pct - HUNDRED_PERCENT) > PERCENT_TOLERANCE:
        issues.append(f"ผลรวมสัดส่วน Buckets ({total_b_pct}%) ต้องเท่ากับ 100%")

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

            confirmed_axis_dict = {
                "portfolio_id": portfolio_id,
                "axis_pointer": axis_pointer,
            }

            proposal = self._bucket_generator.generate_buckets(
                confirmed_axis=confirmed_axis_dict,
                prompt_version=prompt_version,
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
                )
                for p in proposal.purpose_buckets
            ]

            # Build mapping weights
            mapping_weights = [
                AllocationMappingCell(
                    axis_allocation_id=mw["axis_allocation_id"],
                    bucket_id=mw["bucket_id"],
                    portfolio_weight_percent=Decimal(str(mw["portfolio_weight_percent"])),
                )
                for mw in proposal.mapping_weights
            ]

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

            draft_id = f"bplan_{portfolio_id}_{self._id_gen.new_id()}"
            now_iso = self._clock.now_utc()

            draft = BucketPlanDraft(
                draft_id=draft_id,
                portfolio_id=portfolio_id,
                essence_ref=ArtifactRef(
                    document_key=f"investor-essence-{portfolio_id}",
                    note_id="latest",
                    revision_id="1",
                    content_hash="hash",
                    artifact_set_hash="hash",
                ),
                axis_ref=ArtifactRef(
                    document_key=f"investment-axis-{portfolio_id}",
                    note_id="latest",
                    revision_id="1",
                    content_hash="hash",
                    artifact_set_hash="hash",
                ),
                context_ref=f"ctx_{portfolio_id}",
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
            return self._portfolio_port.preview(cmd)

    def apply_bucket_plan(
        self,
        draft_id: str,
        idempotency_key: str = "",
    ) -> AllocationApplyReceipt:
        with self._uow_factory.open() as uow:
            draft = uow.planning.get_bucket_draft(draft_id)
            if not draft:
                raise ResourceNotFoundError("BucketPlanDraft", draft_id)

            scope = f"portfolio:{draft.portfolio_id}"
            req_hash = hashlib.sha256(f"apply_bucket_plan:{draft_id}".encode("utf-8")).hexdigest()

            # 1. Idempotency Check
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

            # 2. Validation
            total_pct = sum((b.target_percent for b in draft.purpose_buckets), Decimal("0.00"))
            if abs(total_pct - HUNDRED_PERCENT) > PERCENT_TOLERANCE:
                raise ValidationFailedError(
                    f"Cannot apply bucket plan: total bucket percent ({total_pct}%) must equal 100%"
                )

            # 3. Compile portfolio targets & apply
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
            cmd = ApplyAllocationCommand(
                command_id=cmd_id,
                request_hash=req_hash,
                portfolio_id=draft.portfolio_id,
                expected_checkpoint_sequence=draft.portfolio_checkpoint.get("sequence", 0),
                expected_checkpoint_state_hash=draft.portfolio_checkpoint.get("state_hash", ""),
                target_rows=target_rows,
                mapping_weights=mapping_weights,
                remapping=remapping_rows,
                accepted_policy_snapshot={
                    "draft_id": draft.draft_id,
                    "allocation_basis": draft.allocation_basis.value,
                },
                source_refs={
                    "essence_ref": draft.essence_ref.document_key,
                    "axis_ref": draft.axis_ref.document_key,
                },
            )

            receipt = self._portfolio_port.apply(cmd)

            # 4. Mark draft as applied
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

            # 5. Save command receipt for idempotency
            if idempotency_key:
                uow.intents.save_command_receipt(
                    scope=scope,
                    use_case="apply_bucket_plan",
                    idempotency_key=idempotency_key,
                    request_hash=req_hash,
                    result=asdict(receipt),
                )

            uow.commit()
            return receipt
