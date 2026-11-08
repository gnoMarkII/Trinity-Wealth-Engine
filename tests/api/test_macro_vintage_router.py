"""API tests for Macro Vintage Walk & Portfolio Factor Exposure Endpoints."""
from decimal import Decimal
from fastapi.testclient import TestClient
import pytest

from api.auth import require_session
from api.main import app


@pytest.fixture
def authed_client():
    app.dependency_overrides[require_session] = lambda: {"user": "test_pm"}
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_macro_vintage_walk_endpoint_cutoff_june_10(authed_client):
    """Verifies GET /api/macro/vintage-walk reproduces Chapter 6 benchmark."""
    response = authed_client.get("/api/macro/vintage-walk?cutoff_date=2026-06-10")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["cutoff_date"] == "2026-06-10"
    assert data["status"] == "REVIEW_REQUIRED"
    assert data["total_series_count"] >= 2

    # Check April Payroll row
    payroll_rows = [r for r in data["rows"] if r["series_id"] == "SYN_PAYROLL" and r["observation_date"] == "2026-04-01"]
    assert len(payroll_rows) == 1
    row = payroll_rows[0]
    assert row["selected_vintage_date"] == "2026-06-05"
    assert row["asof_value"] == 148.0
    assert row["latest_vintage_date"] == "2026-07-03"
    assert row["latest_value"] == 151.0
    assert row["revision_after_cutoff"] == 3.0
    assert row["status"] == "REVIEW_REQUIRED"


def test_portfolio_factor_exposure_endpoint(authed_client):
    """Verifies GET /api/portfolio/factor-exposure returns Chapter 8 snapshot contract."""
    response = authed_client.get("/api/portfolio/factor-exposure?portfolio_id=default")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert "gross_market_value" in data
    assert "net_market_value" in data
    assert "long_exposure_pct" in data
    assert "short_exposure_pct" in data
    assert "net_exposure_pct" in data
    assert data["status"] == "REVIEW_REQUIRED"

