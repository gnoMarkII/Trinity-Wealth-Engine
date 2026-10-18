"""Integration and route tests for Equity NotebookLM endpoints:
- POST /api/equity/{ticker}/notebooklm/exports
- GET  /api/equity/{ticker}/notebooklm/exports/latest
- GET  /api/equity/notebooklm/exports/{export_id}
- POST /api/equity/notebooklm/exports/{export_id}/retry
"""
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from api.auth import require_session
from api.dependencies import get_equity_notebooklm_export_service
from api.main import app
from application.equity.notebooklm_export_ports import (
    EquityExportBlockedError,
    EquityExportRecord,
)


@pytest.fixture(autouse=True)
def override_auth():
    app.dependency_overrides[require_session] = lambda: {"user_id": "test_user"}
    yield
    app.dependency_overrides.pop(require_session, None)


@pytest.fixture
def mock_equity_export_record():
    return EquityExportRecord(
        export_id="export_nflx_test_123",
        request_key="req_nflx_123",
        ticker="NFLX",
        market="US",
        content_hash="abc123nflxhash",
        job_id="job_nflx_456",
        state="queued",
        stage="initialized",
        snapshot_at="2026-10-16T12:00:00Z",
        notebook_id="nb_nflx_123",
        notebook_url="https://notebooklm.google.com/notebook/nb_nflx_123",
        manifest_path=None,
        inventory={
            "total_sources": 10,
            "sources": [
                {
                    "file_name": "00-research-guide.md",
                    "title": "Research Guide",
                    "status": "ready",
                }
            ],
            "counts": {
                "historical_analyses": 1,
                "earnings_calls": 2,
                "news_items": 5,
                "user_notes": 1,
            },
            "has_sec_filings": True,
            "has_insider_trades": True,
            "has_options_data": True,
            "has_portfolio_holding": True,
        },
        warnings=[],
        error_code=None,
        error_message=None,
        created_at=1760000000.0,
        updated_at=1760000000.0,
    )


@pytest.fixture
def mock_service(mock_equity_export_record):
    service = MagicMock()
    service.request_export.return_value = mock_equity_export_record
    service.get_latest_export.return_value = mock_equity_export_record
    service.get_export.return_value = mock_equity_export_record
    service.retry_export.return_value = mock_equity_export_record
    return service


def test_export_equity_to_notebooklm_success(mock_service):
    app.dependency_overrides[get_equity_notebooklm_export_service] = lambda: mock_service
    client = TestClient(app)

    try:
        response = client.post(
            "/api/equity/NFLX/notebooklm/exports",
            json={"market": "US", "mode": "all_retained"},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["export_id"] == "export_nflx_test_123"
        assert data["ticker"] == "NFLX"
        assert data["state"] == "queued"
        mock_service.request_export.assert_called_once_with(ticker="NFLX", market="US", mode="all_retained")
    finally:
        app.dependency_overrides.pop(get_equity_notebooklm_export_service, None)


def test_export_equity_to_notebooklm_blocked_when_no_analysis(mock_service):
    mock_service.request_export.side_effect = EquityExportBlockedError("No baseline analysis found")
    app.dependency_overrides[get_equity_notebooklm_export_service] = lambda: mock_service
    client = TestClient(app)

    try:
        response = client.post(
            "/api/equity/UNKNOWN/notebooklm/exports",
            json={"market": "US", "mode": "all_retained"},
        )
        assert response.status_code == 409
        data = response.json()
        assert data["detail"]["code"] == "equity_export_blocked"
    finally:
        app.dependency_overrides.pop(get_equity_notebooklm_export_service, None)


def test_get_latest_equity_notebooklm_export(mock_service):
    app.dependency_overrides[get_equity_notebooklm_export_service] = lambda: mock_service
    client = TestClient(app)

    try:
        response = client.get("/api/equity/NFLX/notebooklm/exports/latest?market=US")
        assert response.status_code == 200
        data = response.json()
        assert data["export_id"] == "export_nflx_test_123"
        assert data["ticker"] == "NFLX"
        assert len(data["notebooks"]) == 1
        assert data["coverage"]["has_sec_filings"] is True
        assert data["coverage"]["has_insider_trades"] is True
        assert data["coverage"]["has_options_derivatives"] is True
        assert data["coverage"]["has_portfolio_holding"] is True
    finally:
        app.dependency_overrides.pop(get_equity_notebooklm_export_service, None)


def test_get_equity_notebooklm_export_by_id_not_found(mock_service):
    mock_service.get_export.return_value = None
    app.dependency_overrides[get_equity_notebooklm_export_service] = lambda: mock_service
    client = TestClient(app)

    try:
        response = client.get("/api/equity/notebooklm/exports/non_existent_id")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["code"] == "equity_export_not_found"
    finally:
        app.dependency_overrides.pop(get_equity_notebooklm_export_service, None)


def test_retry_equity_notebooklm_export(mock_service):
    app.dependency_overrides[get_equity_notebooklm_export_service] = lambda: mock_service
    client = TestClient(app)

    try:
        response = client.post("/api/equity/notebooklm/exports/export_nflx_test_123/retry")
        assert response.status_code == 202
        data = response.json()
        assert data["export_id"] == "export_nflx_test_123"
        mock_service.retry_export.assert_called_once_with("export_nflx_test_123")
    finally:
        app.dependency_overrides.pop(get_equity_notebooklm_export_service, None)
