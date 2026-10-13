"""Portfolio planning adapter implementing PortfolioPlanningPort for Investor Essence."""
from __future__ import annotations

import hashlib
import json
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from application.investor_essence.dto import (
    AllocationApplyReceipt,
    AllocationPreview,
    ApplyAllocationCommand,
    PortfolioPlanningSnapshot,
    PreviewAllocationCommand,
)
from application.investor_essence.errors import (
    IdempotencyConflictError,
    PortfolioConflictError,
    ResourceNotFoundError,
    ValidationFailedError,
)
from application.investor_essence.ports import PortfolioPlanningPort
from tools.portfolio.bootstrap import build_default_portfolio_dependencies
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.models import AllocationTarget

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class PortfolioPlanningAdapter(PortfolioPlanningPort):
    """Adapter bridging Investor Essence bucket planning use cases to the transactional Portfolio repository."""

    def __init__(self, repo=None, repository=None) -> None:
        chosen = repo if repo is not None else repository
        if chosen is None:
            deps = build_default_portfolio_dependencies()
            self._repo = getattr(deps, "repo", None) or getattr(deps, "repository", None)
        else:
            self._repo = chosen

    def snapshot(self, portfolio_id: str) -> PortfolioPlanningSnapshot:
        from core.investor_essence.models import HoldingPlanningSnapshot

        if hasattr(self._repo, "unit_of_work"):
            with self._repo.unit_of_work(portfolio_id) as uow:
                state = uow.load_state()
                chk = self._repo.store.checkpoint(portfolio_id)
        else:
            state = self._repo.load_state(portfolio_id)
            chk = self._repo.store.checkpoint(portfolio_id)

        nav_val = Decimal(str(state.summary.total_value_thb))
        cash_val = Decimal("0.00")
        holdings_snapshots: List[HoldingPlanningSnapshot] = []
        holdings_summary: List[Dict[str, Any]] = []

        for h in state.holdings:
            mv = Decimal(str(h.market_value_thb))
            if h.asset_type.lower() in ("cash", "currency"):
                cash_val += mv

            if nav_val > Decimal("0"):
                weight_pct = (mv / nav_val * Decimal("100")).quantize(Decimal("0.01"))
            else:
                weight_pct = Decimal("0.00")

            hid = getattr(h, "holding_id", None) or f"{portfolio_id}:{h.symbol}"
            curr = getattr(h, "currency", "THB") or "THB"

            snap = HoldingPlanningSnapshot(
                holding_id=hid,
                symbol=h.symbol,
                asset_type=h.asset_type,
                market_value_thb=mv,
                weight_percent=weight_pct,
                existing_bucket_id=h.bucket_id,
                currency=curr,
            )
            holdings_snapshots.append(snap)
            holdings_summary.append({
                "holding_id": hid,
                "symbol": snap.symbol,
                "asset_type": snap.asset_type,
                "market_value_thb": float(mv),
                "value_thb": float(mv),
                "weight_percent": float(weight_pct),
                "bucket_id": snap.existing_bucket_id,
                "currency": curr,
            })

        targets_list = [
            {
                "bucket_id": t.bucket_id,
                "name": t.name,
                "target_percent": float(t.target_percent),
                "color": t.color,
            }
            for t in state.allocation_targets
        ]

        return PortfolioPlanningSnapshot(
            portfolio_id=portfolio_id,
            name=portfolio_id,
            checkpoint_sequence=chk.sequence,
            checkpoint_state_hash=chk.state_hash,
            base_currency="THB",
            nav_thb=nav_val,
            cash_thb=cash_val,
            as_of=state.last_updated,
            targets=targets_list,
            holdings_summary=holdings_summary,
            holdings=holdings_snapshots,
        )

    def preview(self, command: PreviewAllocationCommand) -> AllocationPreview:
        state = self._repo.load_state(command.portfolio_id)
        chk = self._repo.store.checkpoint(command.portfolio_id)

        # Build remapping lookup
        remapping_map = {}
        for r in command.remapping:
            remapping_map[r["old_bucket_id"]] = r.get("target_bucket_id")

        affected = []
        for h in state.holdings:
            if h.bucket_id in remapping_map:
                new_bid = remapping_map[h.bucket_id]
                if new_bid != h.bucket_id:
                    affected.append({
                        "symbol": h.symbol,
                        "old_bucket_id": h.bucket_id,
                        "new_bucket_id": new_bid,
                        "market_value_thb": float(h.market_value_thb),
                    })

        total_pct = sum(float(t.get("target_percent", 0.0)) for t in command.target_rows)
        issues = []
        if abs(total_pct - 100.0) > 0.01:
            issues.append(f"ผลรวมสัดส่วนเป้าหมาย ({total_pct:.2f}%) ต้องเท่ากับ 100%")

        before_targets = {t.bucket_id: t.target_percent for t in state.allocation_targets}
        after_targets = {t["bucket_id"]: float(t["target_percent"]) for t in command.target_rows}

        payload_hash = hashlib.sha256(
            json.dumps(command.target_rows, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return AllocationPreview(
            checkpoint_sequence=chk.sequence,
            checkpoint_state_hash=chk.state_hash,
            validated_targets=command.target_rows,
            affected_holdings=affected,
            before_allocation=before_targets,
            after_allocation=after_targets,
            issues=issues,
            payload_hash=payload_hash,
        )

    def apply(self, command: ApplyAllocationCommand) -> AllocationApplyReceipt:
        pid = command.portfolio_id

        # 1. Check durable receipt before sequence check (AC19 / AC20 / WP12 task 5)
        if command.idempotency_key:
            cached = self._repo.store.get_command_receipt(pid, "apply_allocation", command.idempotency_key)
            if cached:
                if cached.get("request_hash") != command.request_hash:
                    raise IdempotencyConflictError(command.idempotency_key)
                res = cached["result"]
                return AllocationApplyReceipt(
                    command_id=res["command_id"],
                    portfolio_id=res["portfolio_id"],
                    request_hash=res["request_hash"],
                    applied_sequence=res["applied_sequence"],
                    applied_state_hash=res["applied_state_hash"],
                    applied_at_iso=res["applied_at_iso"],
                    canonical_status="duplicate_reused",
                    projection_status=res.get("projection_status", "ready"),
                    warnings=["Replayed from durable portfolio transaction store"],
                )

        with self._repo.unit_of_work(pid) as uow:
            if uow.sequence != command.expected_checkpoint_sequence:
                raise PortfolioConflictError(
                    portfolio_id=pid,
                    expected_seq=command.expected_checkpoint_sequence,
                    actual_seq=uow.sequence,
                )

            state = uow.load_state()

            # 2. Update allocation targets
            new_targets = []
            for tr in command.target_rows:
                new_targets.append(
                    AllocationTarget(
                        bucket_id=tr["bucket_id"],
                        name=tr["name"],
                        target_percent=float(tr["target_percent"]),
                        color=tr.get("color"),
                    )
                )
            state.allocation_targets = new_targets

            # 3. Remap holdings bucket IDs (units, costs, ledger remain strictly untouched per WP12 task 11)
            remapping_map = {}
            for r in command.remapping:
                remapping_map[r["old_bucket_id"]] = r.get("target_bucket_id")

            for h in state.holdings:
                if h.bucket_id in remapping_map:
                    h.bucket_id = remapping_map[h.bucket_id]

            # 4. Save full applied BucketPolicy, mapping, and source refs into PortfolioState (WP12 task 1)
            applied_at = _now_iso()
            targets_hash = hashlib.sha256(
                json.dumps(command.target_rows, sort_keys=True).encode("utf-8")
            ).hexdigest()

            state.allocation_policy = {
                "draft_id": command.accepted_policy_snapshot.get("draft_id"),
                "allocation_basis": command.accepted_policy_snapshot.get("allocation_basis"),
                "purpose_buckets": command.accepted_policy_snapshot.get("purpose_buckets", []),
                "mapping_weights": command.mapping_weights,
                "applied_at_iso": applied_at,
            }

            state.allocation_plan_origin = {
                "applied_draft_id": command.source_refs.get("draft_id", ""),
                "command_id": command.command_id,
                "essence_ref": command.source_refs.get("essence_ref", ""),
                "axis_ref": command.source_refs.get("axis_ref", ""),
                "context_ref": command.source_refs.get("context_ref", ""),
                "allocation_basis": command.accepted_policy_snapshot.get("allocation_basis"),
                "applied_mapping": command.mapping_weights,
                "applied_at_iso": applied_at,
                "targets_hash": targets_hash,
                "is_modified_since_apply": False,
            }

            # 5. Build receipt and commit atomically in single SQLite transaction (WP12 tasks 3 & 4)
            next_seq = uow.sequence + 1
            receipt_data = {
                "command_id": command.command_id,
                "portfolio_id": pid,
                "request_hash": command.request_hash,
                "applied_sequence": next_seq,
                "applied_state_hash": "",
                "applied_at_iso": applied_at,
                "canonical_status": "committed",
                "projection_status": "ready",
                "warnings": [],
            }
            cmd_receipt = {
                "use_case": "apply_allocation",
                "idempotency_key": command.idempotency_key or "",
                "request_hash": command.request_hash,
                "command_id": command.command_id,
                "result": receipt_data,
            }

            uow.commit(state, LedgerChange(kind="unchanged"), command_receipt=cmd_receipt)
            applied_chk = self._repo.store.checkpoint(pid)
            receipt_data["applied_state_hash"] = applied_chk.state_hash

            return AllocationApplyReceipt(**receipt_data)

    def get_apply_receipt(self, portfolio_id: str, command_id: str) -> Optional[AllocationApplyReceipt]:
        cached = self._repo.store.get_command_receipt_by_command_id(portfolio_id, command_id)
        if not cached:
            return None
        res = cached["result"]
        return AllocationApplyReceipt(
            command_id=res["command_id"],
            portfolio_id=res["portfolio_id"],
            request_hash=res["request_hash"],
            applied_sequence=res["applied_sequence"],
            applied_state_hash=res["applied_state_hash"],
            applied_at_iso=res["applied_at_iso"],
            canonical_status=res.get("canonical_status", "committed"),
            projection_status=res.get("projection_status", "ready"),
            warnings=res.get("warnings", []),
        )

    def repair_projection(self, portfolio_id: str, command_id: str) -> AllocationApplyReceipt:
        # Audit evidence: check command ID in store (WP12 task 9)
        cached = self._repo.store.get_command_receipt_by_command_id(portfolio_id, command_id)
        if not cached:
            raise ResourceNotFoundError("AllocationApplyReceipt", command_id)

        chk = self._repo.store.checkpoint(portfolio_id)
        st = self._repo.load_state(portfolio_id)
        with self._repo.unit_of_work(portfolio_id) as uow:
            uow.underlying.commit(st, LedgerChange(kind="unchanged"))

        res = cached["result"]
        return AllocationApplyReceipt(
            command_id=command_id,
            portfolio_id=portfolio_id,
            request_hash=res.get("request_hash", "repaired"),
            applied_sequence=chk.sequence,
            applied_state_hash=chk.state_hash,
            applied_at_iso=_now_iso(),
            canonical_status="committed",
            projection_status="ready",
            warnings=["Projection repaired from committed transaction store"],
        )

    def get_goals(self, portfolio_id: str) -> List[Dict[str, Any]]:
        """Retrieve structured goals for the investor across portfolios."""
        try:
            from tools.portfolio.goals import get_structured_goals
            all_goals = get_structured_goals(portfolio_id=None)
            if all_goals:
                return all_goals
            return get_structured_goals(portfolio_id=portfolio_id)
        except Exception as e:
            logger.warning("Failed to retrieve structured goals for %s: %s", portfolio_id, e)
            return []

