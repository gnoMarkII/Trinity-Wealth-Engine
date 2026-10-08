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
    PortfolioConflictError,
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

    def __init__(self, repo=None) -> None:
        if repo is None:
            deps = build_default_portfolio_dependencies()
            self._repo = getattr(deps, "repo", None) or getattr(deps, "repository", None)
        else:
            self._repo = repo

    def snapshot(self, portfolio_id: str) -> PortfolioPlanningSnapshot:
        state = self._repo.load_state(portfolio_id)
        chk = self._repo.store.checkpoint(portfolio_id)

        cash_val = Decimal("0.00")
        for h in state.holdings:
            if h.asset_type.lower() in ("cash", "currency"):
                cash_val += Decimal(str(h.market_value_thb))

        targets_list = [
            {
                "bucket_id": t.bucket_id,
                "name": t.name,
                "target_percent": float(t.target_percent),
                "color": t.color,
            }
            for t in state.allocation_targets
        ]

        holdings_list = [
            {
                "symbol": h.symbol,
                "asset_type": h.asset_type,
                "market_value_thb": float(h.market_value_thb),
                "bucket_id": h.bucket_id,
            }
            for h in state.holdings
        ]

        return PortfolioPlanningSnapshot(
            portfolio_id=portfolio_id,
            name=portfolio_id,
            checkpoint_sequence=chk.sequence,
            checkpoint_state_hash=chk.state_hash,
            base_currency="THB",
            nav_thb=Decimal(str(state.summary.total_value_thb)),
            cash_thb=cash_val,
            as_of=state.last_updated,
            targets=targets_list,
            holdings_summary=holdings_list,
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
        with self._repo.unit_of_work(pid) as uow:
            if uow.sequence != command.expected_checkpoint_sequence:
                raise PortfolioConflictError(
                    portfolio_id=pid,
                    expected_seq=command.expected_checkpoint_sequence,
                    actual_seq=uow.sequence,
                )

            state = uow.load_state()

            # 1. Update allocation targets
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

            # 2. Remap holdings bucket IDs
            remapping_map = {}
            for r in command.remapping:
                remapping_map[r["old_bucket_id"]] = r.get("target_bucket_id")

            for h in state.holdings:
                if h.bucket_id in remapping_map:
                    h.bucket_id = remapping_map[h.bucket_id]

            # 3. Commit atomic state update
            uow.commit(state, LedgerChange(kind="unchanged"))
            applied_chk = self._repo.store.checkpoint(pid)

            return AllocationApplyReceipt(
                command_id=command.command_id,
                portfolio_id=pid,
                request_hash=command.request_hash,
                applied_sequence=applied_chk.sequence,
                applied_state_hash=applied_chk.state_hash,
                applied_at_iso=_now_iso(),
                canonical_status="committed",
                projection_status="ready",
                warnings=[],
            )

    def get_apply_receipt(self, portfolio_id: str, command_id: str) -> Optional[AllocationApplyReceipt]:
        return None

    def repair_projection(self, portfolio_id: str, command_id: str) -> AllocationApplyReceipt:
        chk = self._repo.store.checkpoint(portfolio_id)
        return AllocationApplyReceipt(
            command_id=command_id,
            portfolio_id=portfolio_id,
            request_hash="repaired",
            applied_sequence=chk.sequence,
            applied_state_hash=chk.state_hash,
            applied_at_iso=_now_iso(),
            canonical_status="committed",
            projection_status="ready",
            warnings=["Projection repaired from committed transaction store"],
        )
