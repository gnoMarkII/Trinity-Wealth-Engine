"""Unit tests for Portfolio Router Dependency Injection and Error Mapping."""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from api.main import app
from api.auth import require_session
from api.dependencies import get_portfolio_service
from tools.portfolio.service import PortfolioService
from tools.portfolio.domain.errors import (
    PortfolioNotFoundError,
    HoldingNotFoundError,
    InsufficientCashError,
    InvalidTradeError,
)
from tools.portfolio.domain.models import PortfolioMeta, _now_iso


@pytest.fixture(autouse=True)
def bypass_auth():
    """Bypass auth session dependency in router tests."""
    app.dependency_overrides[require_session] = lambda: None
    yield
    app.dependency_overrides.pop(require_session, None)


def test_router_dependency_injection_override():
    """Test that app.dependency_overrides[get_portfolio_service] intercepts endpoint requests."""
    mock_service = MagicMock(spec=PortfolioService)
    mock_service.list_portfolios.return_value = [
        PortfolioMeta(id="mock_port", name="Mock Portfolio", is_default=True, created_at=_now_iso())
    ]

    app.dependency_overrides[get_portfolio_service] = lambda: mock_service
    client = TestClient(app)

    try:
        response = client.get("/api/portfolio/list")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "mock_port"
        mock_service.list_portfolios.assert_called_once()
    finally:
        app.dependency_overrides.pop(get_portfolio_service, None)


def test_router_error_mapping_portfolio_not_found():
    """Test that PortfolioNotFoundError maps to 404 Not Found."""
    mock_service = MagicMock(spec=PortfolioService)
    mock_service.delete_portfolio.side_effect = PortfolioNotFoundError("พอร์ตไม่พบในระบบ")

    app.dependency_overrides[get_portfolio_service] = lambda: mock_service
    client = TestClient(app)

    try:
        response = client.delete("/api/portfolio/non_existent")
        assert response.status_code == 404
        assert "พอร์ตไม่พบในระบบ" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_portfolio_service, None)


def test_router_error_mapping_insufficient_cash():
    """Test that InsufficientCashError maps to 400 Bad Request."""
    mock_service = MagicMock(spec=PortfolioService)
    mock_service.structured_execute_trade.side_effect = InsufficientCashError("เงินสดไม่เพียงพอสำหรับซื้อหุ้น")

    app.dependency_overrides[get_portfolio_service] = lambda: mock_service
    client = TestClient(app)

    payload = {
        "symbol": "AAPL",
        "asset_type": "US_Equity",
        "action": "buy",
        "units": 10.0,
        "price": 150.0,
        "currency": "USD",
    }

    try:
        response = client.post("/api/portfolio/actual/trade", json=payload)
        assert response.status_code == 400
        assert "เงินสดไม่เพียงพอ" in response.json()["detail"]
    finally:
        app.dependency_overrides.pop(get_portfolio_service, None)
