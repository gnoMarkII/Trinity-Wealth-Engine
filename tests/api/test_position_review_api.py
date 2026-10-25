import pytest
from fastapi.testclient import TestClient

from api.auth import require_session
from api.dependencies import get_position_review_service
from api.main import app
from core.equity_analytics.models import (
    PositionReviewReport,
    PriceMoveDecomposition,
    ThenVsNowSnapshot,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def override_session():
    app.dependency_overrides[require_session] = lambda: {"user_id": "test_user"}
    yield
    app.dependency_overrides.pop(require_session, None)


class MockPositionReviewService:
    def run_position_review(self, ticker: str, market: str = "US"):
        return PositionReviewReport(
            ticker=ticker,
            as_of_date="2026-10-23",
            holding_summary={"is_held": True, "units": 10, "avg_cost": 100.0, "current_weight_pct": 5.0},
            decompositions=[
                PriceMoveDecomposition(
                    horizon_label="1Y",
                    start_date="2025-10-23",
                    end_date="2026-10-23",
                    start_price=100.0,
                    end_price=150.0,
                    total_return_pct=50.0,
                    start_eps=5.0,
                    end_eps=6.0,
                    eps_growth_pct=20.0,
                    start_pe=20.0,
                    end_pe=25.0,
                    pe_expansion_pct=25.0,
                    primary_driver="PE_EXPANSION",
                    diagnosis_tier="SPECULATIVE_EXPANSION",
                    diagnosis_summary="Multiple expansion dominated return",
                )
            ],
            then_vs_now=ThenVsNowSnapshot(
                is_held=True,
                period_label="since_entry",
                then_date="2025-10-23",
                now_date="2026-10-23",
                then_price=100.0,
                now_price=150.0,
                what_was_true_then="Thesis at entry",
                what_is_true_now="Current reality",
                thesis_reflection_question="Would you buy today?",
            ),
            triggers=[],
            honest_gaps=["Test gap"],
        )


def test_get_position_review_success():
    app.dependency_overrides[get_position_review_service] = lambda: MockPositionReviewService()
    try:
        response = client.get("/api/equity/AAPL/position-review")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert len(data["decompositions"]) == 1
        assert data["decompositions"][0]["horizon_label"] == "1Y"
        assert data["then_vs_now"]["what_was_true_then"] == "Thesis at entry"
    finally:
        app.dependency_overrides.clear()


def test_get_position_review_invalid_ticker():
    response = client.get("/api/equity/%20%20%20/position-review")
    assert response.status_code == 400
