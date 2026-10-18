"""Unit tests for PortfolioDailySnapshotWorker and batch snapshot endpoint."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from api.auth import require_session
from api.dependencies import get_portfolio_service
from api.main import app
from api.workers.portfolio_snapshot_worker import PortfolioDailySnapshotWorker
from tools.portfolio.domain.models import PortfolioMeta, _now_iso
from tools.portfolio.service import PortfolioService


@pytest.fixture(autouse=True)
def bypass_auth():
    """Bypass auth session dependency in router tests."""
    app.dependency_overrides[require_session] = lambda: None
    yield
    app.dependency_overrides.pop(require_session, None)


def test_snapshot_all_skips_already_snapshotted_today():
    """Test that snapshot_all_portfolios skips portfolios with a snapshot recorded today."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    mock_service = MagicMock(spec=PortfolioService)
    mock_service.list_portfolios.return_value = [
        PortfolioMeta(id="default", name="Default Portfolio", is_default=True, created_at=_now_iso()),
        PortfolioMeta(id="retirement", name="Retirement", is_default=False, created_at=_now_iso()),
    ]

    # 'default' has snapshot today, 'retirement' only has yesterday
    def mock_get_history(days: int = 2, portfolio_id: str = "default"):
        if portfolio_id == "default":
            return [{"Date": today_str, "NAV": 100000.0}]
        return [{"Date": "2026-01-01", "NAV": 50000.0}]

    mock_service.get_structured_performance_history.side_effect = mock_get_history
    mock_service.record_performance_snapshot.return_value = "Snapshot recorded successfully"

    worker = PortfolioDailySnapshotWorker(service=mock_service)
    results = worker.snapshot_all_portfolios(refresh_prices=False, force=False)

    assert "default" in results
    assert "Already snapshotted" in results["default"]
    assert "retirement" in results
    assert results["retirement"] == "Snapshot recorded successfully"

    # Verify record_performance_snapshot was called only for 'retirement'
    mock_service.record_performance_snapshot.assert_called_once_with(
        refresh_prices=False,
        portfolio_id="retirement",
    )


def test_snapshot_all_force_records_even_if_already_snapshotted():
    """Test that force=True bypasses the daily check."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    mock_service = MagicMock(spec=PortfolioService)
    mock_service.list_portfolios.return_value = [
        PortfolioMeta(id="default", name="Default Portfolio", is_default=True, created_at=_now_iso())
    ]
    mock_service.record_performance_snapshot.return_value = "Forced snapshot recorded"

    worker = PortfolioDailySnapshotWorker(service=mock_service)
    results = worker.snapshot_all_portfolios(refresh_prices=True, force=True)

    assert results["default"] == "Forced snapshot recorded"
    mock_service.record_performance_snapshot.assert_called_once_with(
        refresh_prices=True,
        portfolio_id="default",
    )


@pytest.mark.asyncio
async def test_worker_lifecycle():
    """Test starting and stopping worker lifecycle cleanly."""
    mock_service = MagicMock(spec=PortfolioService)
    mock_service.list_portfolios.return_value = []

    worker = PortfolioDailySnapshotWorker(service=mock_service, check_interval_seconds=0.1)
    assert not worker._running

    worker.start()
    assert worker._running
    assert worker._task is not None

    await worker.stop()
    assert not worker._running


def test_router_performance_snapshot_all_endpoint():
    """Test POST /api/portfolio/actual/performance/snapshot-all endpoint."""
    mock_service = MagicMock(spec=PortfolioService)
    mock_service.list_portfolios.return_value = [
        PortfolioMeta(id="default", name="Default Portfolio", is_default=True, created_at=_now_iso())
    ]
    mock_service.get_structured_performance_history.return_value = []
    mock_service.record_performance_snapshot.return_value = "Recorded snapshot"

    app.dependency_overrides[get_portfolio_service] = lambda: mock_service
    client = TestClient(app)

    try:
        response = client.post("/api/portfolio/actual/performance/snapshot-all?refresh_prices=false&force=true")
        assert response.status_code == 200
        data = response.json()
        assert "default" in data
        assert data["default"] == "Recorded snapshot"
    finally:
        app.dependency_overrides.pop(get_portfolio_service, None)
