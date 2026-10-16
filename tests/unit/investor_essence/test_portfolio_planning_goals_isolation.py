from unittest.mock import patch
from tools.investor_essence.adapters.portfolio_planning_adapter import PortfolioPlanningAdapter


def test_portfolio_planning_adapter_get_goals_strictly_isolated_to_portfolio_id():
    adapter = PortfolioPlanningAdapter(repo=object())

    dummy_goals_port_a = [
        {"name": "เกษียณสุขใจ", "goal_type": "nav_target", "portfolio_id": "port_a", "target_amount_thb": 10000000.0}
    ]

    with patch("tools.portfolio.goals.get_structured_goals") as mock_get_structured:
        def fake_get_structured(portfolio_id=None):
            if portfolio_id == "port_a":
                return dummy_goals_port_a
            elif portfolio_id == "port_b":
                return []
            return [
                {"name": "Goal All", "portfolio_id": "other"}
            ]

        mock_get_structured.side_effect = fake_get_structured

        # 1. When querying port_a, must only return port_a goals
        goals_a = adapter.get_goals("port_a")
        assert len(goals_a) == 1
        assert goals_a[0]["name"] == "เกษียณสุขใจ"
        mock_get_structured.assert_called_with(portfolio_id="port_a")

        # 2. When querying port_b which has no goals, must return empty list (no leak from other portfolios)
        goals_b = adapter.get_goals("port_b")
        assert goals_b == []
        mock_get_structured.assert_called_with(portfolio_id="port_b")


def test_portfolio_planning_adapter_get_goals_handles_exceptions_gracefully():
    adapter = PortfolioPlanningAdapter(repo=object())
    with patch("tools.portfolio.goals.get_structured_goals", side_effect=RuntimeError("Storage error")):
        goals = adapter.get_goals("port_error")
        assert goals == []
