from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_components_endpoint():
    response = client.get("/api/health/components")
    assert response.status_code == 200
    data = response.json()
    assert "api" in data
    assert "database" in data
    assert "job_worker" in data
    assert "market_provider" in data
    assert "llm" in data
    assert data["api"] == "ok"


def test_harness_telemetry_endpoint():
    response = client.get("/api/health/harness-telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "total_events" in data
    assert "resolved_rate" in data
    assert "issue_counts" in data
    assert "top_recurring_issues" in data
