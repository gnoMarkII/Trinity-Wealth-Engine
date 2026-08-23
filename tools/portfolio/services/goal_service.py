"""PortfolioGoalService — Goal CRUD (set, remove, read progress)."""
import json
from typing import Optional, List, Dict, Literal

from tools.portfolio.domain.models import GoalsState, GoalItem, _now_iso
from tools.portfolio.ports.goals_port import GoalsRepositoryPort


class PortfolioGoalService:
    """Handles all Goal set/remove/read operations."""

    def __init__(self, goals_repo: GoalsRepositoryPort) -> None:
        self.goals_repo = goals_repo

    def set_goal(
        self,
        name: str,
        goal_type: Literal["nav_target", "cash_target", "passive_income_ytd", "bucket_target"],
        target_amount_thb: float,
        deadline: Optional[str] = None,
        years_from_now: Optional[int] = None,
        notes: Optional[str] = None,
        portfolio_id: str = "default",
        bucket_id: Optional[str] = None,
    ) -> str:
        try:
            self.structured_upsert_goal(
                name=name,
                goal_type=goal_type,
                target_amount_thb=target_amount_thb,
                deadline=deadline,
                years_from_now=years_from_now,
                notes=notes,
                portfolio_id=portfolio_id,
                bucket_id=bucket_id,
            )
            return f"[GOAL] บันทึกเป้าหมาย '{name}' {target_amount_thb:,.2f} THB สำเร็จ"
        except Exception as e:
            return f"Error: {e}"

    def remove_goal(self, name: str) -> str:
        try:
            self.structured_remove_goal(name=name)
            return f"[GOAL] ลบเป้าหมาย '{name}' สำเร็จ"
        except Exception as e:
            return f"Error: {e}"

    def get_goals_progress(self, portfolio_id: str = "default") -> str:
        goals = self.get_structured_goals(portfolio_id=portfolio_id)
        return json.dumps(goals, ensure_ascii=False, indent=2)

    def get_structured_goals(self, portfolio_id: Optional[str] = None) -> List[Dict]:
        state = self.goals_repo.load_goals(portfolio_id=portfolio_id)
        return [g.model_dump(exclude_none=True) for g in state.goals]

    def structured_upsert_goal(
        self,
        name: str,
        goal_type: Literal["nav_target", "cash_target", "passive_income_ytd", "bucket_target"],
        target_amount_thb: float,
        deadline: Optional[str] = None,
        years_from_now: Optional[int] = None,
        notes: Optional[str] = None,
        portfolio_id: str = "default",
        bucket_id: Optional[str] = None,
    ) -> List[Dict]:
        state = self.goals_repo.load_goals(portfolio_id=None)
        clean_name = name.strip()
        existing = next(
            (g for g in state.goals if g.name == clean_name and g.portfolio_id == portfolio_id), None
        )
        if existing:
            existing.goal_type = goal_type
            existing.target_amount_thb = target_amount_thb
            existing.deadline = deadline
            existing.years_from_now = years_from_now
            existing.notes = notes
            existing.bucket_id = bucket_id
        else:
            state.goals.append(
                GoalItem(
                    name=clean_name,
                    goal_type=goal_type,
                    target_amount_thb=target_amount_thb,
                    deadline=deadline,
                    years_from_now=years_from_now,
                    notes=notes,
                    created_date=_now_iso()[:10],
                    portfolio_id=portfolio_id,
                    bucket_id=bucket_id,
                )
            )
        self.goals_repo.save_goals(state)
        return self.get_structured_goals(portfolio_id=portfolio_id)

    def structured_remove_goal(self, name: str, portfolio_id: Optional[str] = None) -> List[Dict]:
        clean_name = name.strip()
        state = self.goals_repo.load_goals(portfolio_id=None)
        state.goals = [
            g
            for g in state.goals
            if not (g.name == clean_name and (portfolio_id is None or g.portfolio_id == portfolio_id))
        ]
        self.goals_repo.save_goals(state)
        return self.get_structured_goals(portfolio_id=portfolio_id)
