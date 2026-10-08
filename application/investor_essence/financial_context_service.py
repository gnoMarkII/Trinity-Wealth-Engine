"""Application service for managing financial context and readiness audits."""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, Optional

from core.investor_essence.financial_context import update_readiness
from core.investor_essence.models import FinancialContextSnapshot
from application.investor_essence.dto import FinancialContextView
from application.investor_essence.ports import (
    ClockPort,
    IdGeneratorPort,
    InvestorRuntimeUowFactory,
    PortfolioPlanningPort,
)

logger = logging.getLogger(__name__)


def _to_view(s: FinancialContextSnapshot) -> FinancialContextView:
    issues_list = [
        {
            "code": i.code,
            "severity": i.severity,
            "message": i.message,
            "field_path": i.field_path,
        }
        for i in s.readiness_issues
    ]
    return FinancialContextView(
        snapshot_id=s.snapshot_id,
        portfolio_id=s.portfolio_id,
        horizon_years=str(s.horizon_years) if s.horizon_years is not None else None,
        target_use_amount=str(s.target_use_amount) if s.target_use_amount is not None else None,
        target_use_range=s.target_use_range,
        target_use_timeline=s.target_use_timeline,
        emergency_reserves_amount=str(s.emergency_reserves_amount) if s.emergency_reserves_amount is not None else None,
        emergency_reserves_months=str(s.emergency_reserves_months) if s.emergency_reserves_months is not None else None,
        obligations_monthly=str(s.obligations_monthly) if s.obligations_monthly is not None else None,
        obligations_description=s.obligations_description,
        withdrawal_frequency=s.withdrawal_frequency,
        withdrawal_amount=str(s.withdrawal_amount) if s.withdrawal_amount is not None else None,
        experience_description=s.experience_description,
        unknown_fields=list(s.unknown_fields),
        as_of=s.as_of,
        source=s.source,
        readiness_issues=issues_list,
        is_ready_for_numeric_policy=s.is_ready_for_numeric_policy,
    )


def _to_optional_decimal(val: Any) -> Optional[Decimal]:
    if val is None:
        return None
    val_str = str(val).strip()
    if not val_str:
        return None
    return Decimal(val_str)


class FinancialContextService:
    """Use cases for querying and updating financial context facts and readiness."""

    def __init__(
        self,
        uow_factory: InvestorRuntimeUowFactory,
        clock: ClockPort,
        id_gen: IdGeneratorPort,
        portfolio_port: Optional[PortfolioPlanningPort] = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._clock = clock
        self._id_gen = id_gen
        self._portfolio_port = portfolio_port

    def get_financial_context(self, portfolio_id: str) -> FinancialContextView:
        with self._uow_factory.open() as uow:
            existing = uow.planning.get_context_snapshot(portfolio_id)
            if existing:
                return _to_view(existing)

            # Create default uninitialized snapshot
            now_iso = self._clock.now_utc()
            chk_refs = {}
            if self._portfolio_port:
                try:
                    p_snap = self._portfolio_port.snapshot(portfolio_id)
                    chk_refs = {
                        "checkpoint_sequence": p_snap.checkpoint_sequence,
                        "checkpoint_state_hash": p_snap.checkpoint_state_hash,
                        "nav_thb": str(p_snap.nav_thb),
                        "cash_thb": str(p_snap.cash_thb),
                    }
                except Exception as ex:
                    logger.warning("Failed to fetch initial portfolio snapshot for %s: %s", portfolio_id, ex)

            raw_snap = FinancialContextSnapshot(
                snapshot_id=f"ctx_{portfolio_id}_{self._id_gen.new_id()}",
                portfolio_id=portfolio_id,
                unknown_fields=["horizon_years", "emergency_reserves", "obligations", "withdrawal_plan"],
                as_of=now_iso,
                source="default_initial",
                portfolio_checkpoint_refs=chk_refs,
            )
            evaluated = update_readiness(raw_snap)
            uow.planning.save_context_snapshot(evaluated)
            uow.commit()
            return _to_view(evaluated)

    def update_financial_context(
        self,
        portfolio_id: str,
        updates: Dict[str, Any],
    ) -> FinancialContextView:
        with self._uow_factory.open() as uow:
            current = uow.planning.get_context_snapshot(portfolio_id)
            now_iso = self._clock.now_utc()

            horizon = current.horizon_years if current else None
            if "horizon_years" in updates:
                horizon = _to_optional_decimal(updates["horizon_years"])

            target_amount = current.target_use_amount if current else None
            if "target_use_amount" in updates:
                target_amount = _to_optional_decimal(updates["target_use_amount"])

            target_range = updates.get("target_use_range", current.target_use_range if current else None)
            target_timeline = updates.get("target_use_timeline", current.target_use_timeline if current else None)

            reserves_amount = current.emergency_reserves_amount if current else None
            if "emergency_reserves_amount" in updates:
                reserves_amount = _to_optional_decimal(updates["emergency_reserves_amount"])

            reserves_months = current.emergency_reserves_months if current else None
            if "emergency_reserves_months" in updates:
                reserves_months = _to_optional_decimal(updates["emergency_reserves_months"])

            obligations_m = current.obligations_monthly if current else None
            if "obligations_monthly" in updates:
                obligations_m = _to_optional_decimal(updates["obligations_monthly"])

            obligations_desc = updates.get("obligations_description", current.obligations_description if current else None)
            withdrawal_freq = updates.get("withdrawal_frequency", current.withdrawal_frequency if current else None)

            withdrawal_amt = current.withdrawal_amount if current else None
            if "withdrawal_amount" in updates:
                withdrawal_amt = _to_optional_decimal(updates["withdrawal_amount"])

            exp_desc = updates.get("experience_description", current.experience_description if current else None)

            unknown_f = current.unknown_fields if current else []
            if "unknown_fields" in updates:
                unknown_f = list(updates["unknown_fields"])

            raw_snap = FinancialContextSnapshot(
                snapshot_id=f"ctx_{portfolio_id}_{self._id_gen.new_id()}",
                portfolio_id=portfolio_id,
                horizon_years=horizon,
                target_use_amount=target_amount,
                target_use_range=target_range,
                target_use_timeline=target_timeline,
                emergency_reserves_amount=reserves_amount,
                emergency_reserves_months=reserves_months,
                obligations_monthly=obligations_m,
                obligations_description=obligations_desc,
                withdrawal_frequency=withdrawal_freq,
                withdrawal_amount=withdrawal_amt,
                experience_description=exp_desc,
                unknown_fields=unknown_f,
                as_of=now_iso,
                source="user_reported",
                portfolio_checkpoint_refs=current.portfolio_checkpoint_refs if current else {},
            )
            evaluated = update_readiness(raw_snap)
            uow.planning.save_context_snapshot(evaluated)
            uow.commit()
            return _to_view(evaluated)
