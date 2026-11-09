"""Tests for the 4 Institutional Hedge Fund REST APIs (Chapters 1, 2, 5, 7, 10)."""
from contextlib import closing
import pytest
from fastapi.testclient import TestClient

from api.auth import require_session
from api.db import get_connection, init_schema
from api.db.adapters import SqliteInstitutionalLedgerRepositoryAdapter
from api.dependencies import (
    get_catalyst_service,
    get_estimate_ledger_service,
    get_routine_monitor_service,
    get_task_routing_service,
)
from api.main import app
from application.equity.estimate_service import EstimateLedgerApplicationService
from application.governance.routing_service import TaskRoutingGovernanceService
from application.resilience.routine_monitor_service import RoutineMonitorApplicationService
from application.thesis.catalyst_service import CatalystApplicationService


@pytest.fixture
def auth_client(tmp_path):
    db_path = str(tmp_path / "institutional.sqlite")
    with closing(get_connection(db_path)) as conn:
        init_schema(conn)
    repo = SqliteInstitutionalLedgerRepositoryAdapter(
        db_path=db_path,
        artifact_root=tmp_path / "source-artifacts",
    )
    catalyst_svc = CatalystApplicationService(ledger_repository=repo)
    estimate_svc = EstimateLedgerApplicationService(ledger_repository=repo)
    routing_svc = TaskRoutingGovernanceService(routing_audit_dir=tmp_path / "routing")
    routine_svc = RoutineMonitorApplicationService(ledger_repository=repo)

    app.dependency_overrides[require_session] = lambda: {"user": "test_pm", "role": "PORTFOLIO_MANAGER_HERMES"}
    app.dependency_overrides[get_catalyst_service] = lambda: catalyst_svc
    app.dependency_overrides[get_estimate_ledger_service] = lambda: estimate_svc
    app.dependency_overrides[get_task_routing_service] = lambda: routing_svc
    app.dependency_overrides[get_routine_monitor_service] = lambda: routine_svc
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_catalyst_exceptions_flow(auth_client):
    """Test Gem 1: GET catalyst exceptions and POST human PM review."""
    # 1. GET catalyst exceptions
    resp = auth_client.get("/api/thesis/catalyst-exceptions")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 2
    assert any(item["catalyst_id"] == "CAT-001" for item in data)

    # 2. Review exception
    target_ex = next(item for item in data if item["catalyst_id"] == "CAT-001")
    target_id = target_ex["exception_id"]
    rev_resp = auth_client.post(
        f"/api/thesis/catalyst-exceptions/{target_id}/review",
        json={"human_assessment": "Assessed non-material 1-day earnings shift"},
    )
    assert rev_resp.status_code == 200
    rev_data = rev_resp.json()
    assert rev_data["status"] == "ACKNOWLEDGED"
    assert "Assessed non-material" in rev_data["human_assessment"]


def test_variant_perception_and_estimates_ledger(auth_client):
    """Test Gem 2: GET variant perception 3-way surprise and estimates ledger."""
    # 1. GET variant perception
    resp = auth_client.get("/api/equity/GOOGL/variant-perception?period=2026Q2&metric_name=revenue&reported=88268.00")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "GOOGL"
    assert data["street_surprise"] == "1868.00"
    assert data["variant_spread"] == "1100.00"
    assert data["variant_realization"] == "768.00"
    assert data["alpha_verdict"] == "ALPHA_CONFIRMED"

    # 2. GET estimates ledger
    hist_resp = auth_client.get("/api/equity/GOOGL/estimates-ledger")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert isinstance(hist_data, list)
    assert len(hist_data) >= 1

    # 3. Missing reported -> 422
    resp_missing = auth_client.get("/api/equity/GOOGL/variant-perception?period=2026Q2&metric_name=revenue")
    assert resp_missing.status_code == 422

    # 4. Invalid/NaN/Infinity reported -> 422
    resp_nan = auth_client.get("/api/equity/GOOGL/variant-perception?period=2026Q2&metric_name=revenue&reported=NaN")
    assert resp_nan.status_code == 422
    resp_bad = auth_client.get("/api/equity/GOOGL/variant-perception?period=2026Q2&metric_name=revenue&reported=not_a_number")
    assert resp_bad.status_code == 422

    # 5. Unknown ticker/period -> 404
    resp_404 = auth_client.get("/api/equity/UNKNOWN_TICKER/variant-perception?period=2026Q2&metric_name=revenue&reported=100.00")
    assert resp_404.status_code == 404



def test_task_routing_governance(auth_client):
    """Test Gem 3: POST route-task and GET routing-ledger."""
    payload = {
        "task_id": "TASK-TEST-01",
        "task_description": "Calculate factor beta and portfolio exposure with Decimal",
        "work_shape": "LIVE_FINANCIAL_CALCULATION",
        "has_authoritative_calculator": True,
        "requires_reconstructability": True,
    }
    resp = auth_client.post("/api/governance/route-task", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["selected_surface"] == "DETERMINISTIC_TOOL_OR_SCRIPT"
    assert "Python Core" in data["authoritative_system"]

    # Ledger audit
    led_resp = auth_client.get("/api/governance/routing-ledger")
    assert led_resp.status_code == 200
    led_data = led_resp.json()
    assert any(r["task_id"] == "TASK-TEST-01" for r in led_data)


def test_routine_resilience_and_heartbeat(auth_client):
    """Test Gem 4: GET routine freshness and POST heartbeat."""
    # 1. GET routine status
    resp = auth_client.get("/api/resilience/routine-status?target_date=2026-08-06")
    assert resp.status_code == 200
    data = resp.json()
    assert "routine_id" in data
    assert "status" in data

    # 2. POST heartbeat
    hb_payload = {
        "execution_id": "RUN-TEST-01",
        "routine_id": "ROUTINE-AM-REVIEW",
        "scheduled_utc": "2026-08-06T06:00:00Z",
        "actual_start_utc": "2026-08-06T06:10:00Z",
        "completed_utc": "2026-08-06T06:12:00Z",
    }
    hb_resp = auth_client.post("/api/resilience/heartbeat", json=hb_payload)
    assert hb_resp.status_code == 200
    hb_data = hb_resp.json()
    assert hb_data["status"] == "ON_TIME"
    assert hb_data["execution_id"] == "RUN-TEST-01"

