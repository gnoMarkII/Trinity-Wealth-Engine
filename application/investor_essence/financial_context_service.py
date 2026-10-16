"""Application service for managing financial context and readiness audits."""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, Optional

from core.investor_essence.financial_context import parse_numeric_fact, update_readiness
from core.investor_essence.models import FinancialContextSnapshot
from application.investor_essence.dto import FinancialContextView
from application.investor_essence.errors import ValidationFailedError
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
    facts_dict = {}
    for k, f in s.facts.items():
        facts_dict[k] = {
            "field_key": f.field_key,
            "status": f.status,
            "value_numeric": str(f.value_numeric) if f.value_numeric is not None else None,
            "value_text": f.value_text,
            "unit": f.unit,
            "source": f.source,
            "as_of": f.as_of,
        }

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
        monthly_investable_contribution=str(s.monthly_investable_contribution) if s.monthly_investable_contribution is not None else None,
        essential_monthly_expense=str(s.essential_monthly_expense) if s.essential_monthly_expense is not None else None,
        reserves_outside_portfolio=str(s.reserves_outside_portfolio) if s.reserves_outside_portfolio is not None else None,
        facts=facts_dict,
        unknown_fields=list(s.unknown_fields),
        as_of=s.as_of,
        source=s.source,
        readiness_issues=issues_list,
        is_ready_for_numeric_policy=s.is_ready_for_numeric_policy,
    )


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

    def _derive_context_from_goals(self, portfolio_id: str) -> Dict[str, Any]:
        derived: Dict[str, Any] = {}
        goals: List[Dict[str, Any]] = []
        chk_refs: Dict[str, Any] = {}
        cash_balance: Optional[Decimal] = None

        if self._portfolio_port:
            try:
                p_snap = self._portfolio_port.snapshot(portfolio_id)
                chk_refs = {
                    "checkpoint_sequence": p_snap.checkpoint_sequence,
                    "checkpoint_state_hash": p_snap.checkpoint_state_hash,
                    "nav_thb": str(p_snap.nav_thb),
                    "cash_thb": str(p_snap.cash_thb),
                }
                cash_balance = p_snap.cash_thb
            except Exception as ex:
                logger.warning("Failed to fetch initial portfolio snapshot for %s: %s", portfolio_id, ex)

            try:
                goals = self._portfolio_port.get_goals(portfolio_id)
            except Exception as ex:
                logger.warning("Failed to fetch goals for %s: %s", portfolio_id, ex)

        # 1. Main wealth / NAV goal
        nav_goals = [g for g in goals if g.get("goal_type") in ("nav_target", "bucket_target")]
        main_goal = None
        if nav_goals:
            main_goal = max(nav_goals, key=lambda g: float(g.get("target_amount_thb") or 0))
        elif goals:
            main_goal = max(goals, key=lambda g: float(g.get("target_amount_thb") or 0))

        if main_goal:
            target_amt = main_goal.get("target_amount_thb")
            if target_amt is not None and float(target_amt) > 0:
                int_val = int(float(target_amt))
                derived["target_use_amount"] = Decimal(str(int_val if int_val == float(target_amt) else target_amt))

            goal_name = main_goal.get("name") or ""
            goal_notes = main_goal.get("notes") or ""
            timeline_desc = f"{goal_name} ({goal_notes})".strip(" ()") if goal_notes else goal_name
            if timeline_desc:
                derived["target_use_timeline"] = timeline_desc

            # Calculate horizon_years from deadline
            deadline_str = main_goal.get("deadline")
            deadline_days = main_goal.get("deadline_days_left")
            if deadline_days is not None and deadline_days > 0:
                years = max(1, round(deadline_days / 365.25))
                derived["horizon_years"] = Decimal(str(years))
            elif deadline_str:
                try:
                    deadline_year = int(deadline_str.split("-")[0])
                    now_year = int(self._clock.now_utc().split("-")[0])
                    years = max(1, deadline_year - now_year)
                    derived["horizon_years"] = Decimal(str(years))
                except Exception:
                    derived["horizon_years"] = Decimal("10")
            else:
                derived["horizon_years"] = Decimal("10")
            # 2. Emergency reserves
            cash_goal = next(
                (g for g in goals if g.get("goal_type") == "cash_target" or "สำรอง" in (g.get("name") or "") or "ฉุกเฉิน" in (g.get("name") or "")),
                None,
            )
            if cash_goal and cash_goal.get("target_amount_thb"):
                amt = float(cash_goal["target_amount_thb"])
                int_val = int(amt)
                derived["emergency_reserves_amount"] = Decimal(str(int_val if int_val == amt else amt))
                derived["emergency_reserves_months"] = Decimal("6")
            elif cash_balance is not None and cash_balance > 0:
                derived["emergency_reserves_amount"] = cash_balance
                derived["emergency_reserves_months"] = Decimal("3")
            else:
                derived["emergency_reserves_amount"] = Decimal("0")
                derived["emergency_reserves_months"] = Decimal("0")
                derived["reserves_outside_portfolio"] = "จัดสรรเงินสำรองฉุกเฉินแยกไว้ภายนอกหรือพอร์ตอื่น"

            # 3. Withdrawal plan & obligations defaults
            derived["withdrawal_frequency"] = "ไม่มีการถอน (เน้นสะสมมูลค่าระยะยาว)"
            derived["withdrawal_amount"] = Decimal("0")
            derived["obligations_monthly"] = Decimal("0")
            derived["obligations_description"] = "ไม่มีภาระผูกพันกระทบพอร์ตนี้"

            # 4. Experience description
            derived["experience_description"] = "หุ้น กองทุนรวม ทองคำ"
            derived["unknown_fields"] = []
        else:
            # No goals found for this portfolio: keep fields unknown
            derived["unknown_fields"] = ["horizon_years", "emergency_reserves", "obligations", "withdrawal_plan"]

        derived["chk_refs"] = chk_refs
        return derived

    def get_financial_context(self, portfolio_id: str) -> FinancialContextView:
        with self._uow_factory.open() as uow:
            existing = uow.planning.get_context_snapshot(portfolio_id)
            if existing:
                return _to_view(existing)

            # Create default initial snapshot with unknown fields
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

    def sync_from_goals(self, portfolio_id: str) -> FinancialContextView:
        with self._uow_factory.open() as uow:
            derived = self._derive_context_from_goals(portfolio_id)
            now_iso = self._clock.now_utc()
            existing = uow.planning.get_context_snapshot(portfolio_id)
            snapshot_id = existing.snapshot_id if existing else f"ctx_{portfolio_id}_{self._id_gen.new_id()}"
            snap = FinancialContextSnapshot(
                snapshot_id=snapshot_id,
                portfolio_id=portfolio_id,
                horizon_years=derived.get("horizon_years"),
                target_use_amount=derived.get("target_use_amount"),
                target_use_timeline=derived.get("target_use_timeline"),
                emergency_reserves_amount=derived.get("emergency_reserves_amount"),
                emergency_reserves_months=derived.get("emergency_reserves_months"),
                obligations_monthly=derived.get("obligations_monthly"),
                obligations_description=derived.get("obligations_description"),
                withdrawal_frequency=derived.get("withdrawal_frequency"),
                withdrawal_amount=derived.get("withdrawal_amount"),
                experience_description=derived.get("experience_description"),
                reserves_outside_portfolio=derived.get("reserves_outside_portfolio"),
                monthly_investable_contribution=derived.get("monthly_investable_contribution"),
                essential_monthly_expense=derived.get("essential_monthly_expense"),
                unknown_fields=derived.get("unknown_fields", []),
                as_of=now_iso,
                source="auto_synced_from_goals",
                portfolio_checkpoint_refs=derived.get("chk_refs", {}),
            )
            evaluated = update_readiness(snap)
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

            def safe_parse(field_name: str, current_val: Optional[Decimal]) -> Optional[Decimal]:
                if field_name in updates:
                    try:
                        return parse_numeric_fact(field_name, updates[field_name])
                    except ValueError as err:
                        raise ValidationFailedError(str(err)) from err
                return current_val

            horizon = safe_parse("horizon_years", current.horizon_years if current else None)
            target_amount = safe_parse("target_use_amount", current.target_use_amount if current else None)
            target_range = updates.get("target_use_range", current.target_use_range if current else None)
            target_timeline = updates.get("target_use_timeline", current.target_use_timeline if current else None)

            reserves_amount = safe_parse("emergency_reserves_amount", current.emergency_reserves_amount if current else None)
            reserves_months = safe_parse("emergency_reserves_months", current.emergency_reserves_months if current else None)
            obligations_m = safe_parse("obligations_monthly", current.obligations_monthly if current else None)
            obligations_desc = updates.get("obligations_description", current.obligations_description if current else None)
            withdrawal_freq = updates.get("withdrawal_frequency", current.withdrawal_frequency if current else None)
            withdrawal_amt = safe_parse("withdrawal_amount", current.withdrawal_amount if current else None)
            exp_desc = updates.get("experience_description", current.experience_description if current else None)

            monthly_contrib = safe_parse("monthly_investable_contribution", current.monthly_investable_contribution if current else None)
            essential_expense = safe_parse("essential_monthly_expense", current.essential_monthly_expense if current else None)
            reserves_outside = safe_parse("reserves_outside_portfolio", current.reserves_outside_portfolio if current else None)

            unknown_f = list(current.unknown_fields if current else [])
            if "unknown_fields" in updates and updates["unknown_fields"] is not None:
                unknown_f = list(updates["unknown_fields"])

            # Auto-remove fields from unknown_fields if valid values are provided
            if horizon is not None and "horizon_years" in unknown_f:
                unknown_f.remove("horizon_years")
            if (reserves_amount is not None or reserves_months is not None) and "emergency_reserves" in unknown_f:
                unknown_f.remove("emergency_reserves")
            if obligations_m is not None and "obligations" in unknown_f:
                unknown_f.remove("obligations")

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
                monthly_investable_contribution=monthly_contrib,
                essential_monthly_expense=essential_expense,
                reserves_outside_portfolio=reserves_outside,
                unknown_fields=unknown_f,
                as_of=now_iso,
                source="user_reported",
                portfolio_checkpoint_refs=current.portfolio_checkpoint_refs if current else {},
            )
            evaluated = update_readiness(raw_snap)
            uow.planning.save_context_snapshot(evaluated)
            uow.commit()
            return _to_view(evaluated)
