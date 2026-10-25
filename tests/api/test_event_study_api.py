"""Integration tests for Equity Event Study API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from api.main import app
from api.dependencies import get_event_study_service
from api.auth import require_session
from core.event_study.models import (
    DailyPriceBar,
    EventInstanceResult,
    EventPathPoint,
    EventStudyReport,
    HorizonSummaryStats,
)


@pytest.fixture(autouse=True)
def override_require_session():
    app.dependency_overrides[require_session] = lambda: {"user_id": "test_user"}
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_service():
    service = MagicMock()
    service.get_available_presets.return_value = [
        {
            "preset_id": "earnings",
            "name": "Earnings Announcements",
            "category": "earnings",
            "description": "Quarterly earnings releases",
            "recommended_ticker": "",
        },
        {
            "preset_id": "apple_iphone",
            "name": "Apple iPhone Launch Events",
            "category": "product_launch",
            "description": "Apple Keynotes",
            "recommended_ticker": "AAPL",
        },
    ]

    mock_report = EventStudyReport(
        ticker="AAPL",
        event_name="Apple iPhone Launch Events",
        horizons=[5, 10, 20, 60],
        events=[
            EventInstanceResult(
                event_id="evt_1_2023-09-12",
                event_date="2023-09-12",
                entry_date="2023-09-13",
                entry_price=175.5,
                description="iPhone 15 Series",
                forward_returns={5: 2.5, 10: 4.1, 20: -1.2, 60: 8.5},
                daily_trajectory=[
                    EventPathPoint(day_offset=1, date="2023-09-13", close=176.0, cumulative_return_pct=0.28),
                    EventPathPoint(day_offset=5, date="2023-09-19", close=179.88, cumulative_return_pct=2.5),
                ],
                is_win_by_horizon={5: True, 10: True, 20: False, 60: True},
            )
        ],
        horizon_stats={
            5: HorizonSummaryStats(
                horizon_days=5,
                sample_size=1,
                hit_rate_pct=100.0,
                avg_return_pct=2.5,
                median_return_pct=2.5,
                best_return_pct=2.5,
                worst_return_pct=2.5,
                avg_win_pct=2.5,
                avg_loss_pct=None,
                risk_reward_ratio=None,
                win_count=1,
                loss_count=0,
            )
        },
        average_trajectory=[],
        warnings=[],
    )
    service.run_event_study.return_value = mock_report
    return service


def test_get_event_study_presets(mock_service):
    app.dependency_overrides[get_event_study_service] = lambda: mock_service
    client = TestClient(app)
    try:
        response = client.get("/api/equity/event-study/presets")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["preset_id"] == "earnings"
        assert data[1]["preset_id"] == "apple_iphone"
    finally:
        app.dependency_overrides.pop(get_event_study_service, None)


def test_execute_event_study_success(mock_service):
    app.dependency_overrides[get_event_study_service] = lambda: mock_service
    client = TestClient(app)
    try:
        payload = {
            "preset_id": "apple_iphone",
            "horizons": [5, 10, 20, 60],
        }
        response = client.post("/api/equity/AAPL/event-study", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["event_name"] == "Apple iPhone Launch Events"
        assert len(data["events"]) == 1
        assert data["events"][0]["entry_price"] == 175.5
        assert "5" in data["horizon_stats"]
        assert data["horizon_stats"]["5"]["hit_rate_pct"] == 100.0
    finally:
        app.dependency_overrides.pop(get_event_study_service, None)


def test_execute_event_study_invalid_ticker():
    client = TestClient(app)
    payload = {"preset_id": "earnings"}
    response = client.post("/api/equity/INVALID..TICKER/event-study", json=payload)
    assert response.status_code == 400
