"""Integration and route tests for Macro NotebookLM endpoints:
- POST /api/macro/notebooklm/exports
- GET  /api/macro/notebooklm/exports/latest
- GET  /api/macro/notebooklm/exports/{export_id}
- POST /api/macro/notebooklm/exports/{export_id}/retry
"""
import pytest
from unittest.mock import MagicMock

from api.main import app
from api.dependencies import get_macro_notebooklm_export_service
from application.macro.notebooklm_export_ports import MacroExportRecord


@pytest.fixture
def mock_export_record():
    return MacroExportRecord(
        export_id="export_macro_test_123",
        request_key="req_key_123",
        content_hash="abc123hash",
        job_id="job_macro_456",
        state="queued",
        stage="initialized",
        snapshot_at="2026-10-05T12:00:00Z",
        strategy_report_id="macro_strategy_2026-10-05",
        notebook_id="nb_123",
        notebook_url="https://notebooklm.google.com/notebook/nb_123",
        manifest_path=None,
        inventory={
            "total_sources": 9,
            "sources": [
                {
                    "file_name": "00-research-guide.md",
                    "title": "Research Guide",
                    "status": "ready",
                }
            ],
            "counts": {
                "historical_reports": 1,
                "catalog_notes": 5,
                "indicator_series": 2,
                "market_observables_cached": 13,
                "market_observables_total": 13,
            },
        },
        warnings=[],
        error_code=None,
        error_message=None,
        created_at=1760000000.0,
        updated_at=1760000000.0,
    )


@pytest.fixture
def mock_service(mock_export_record):
    service = MagicMock()
    service.request_export.return_value = mock_export_record
    service.get_latest_export.return_value = mock_export_record
    service.get_export.return_value = mock_export_record
    service.retry_export.return_value = mock_export_record
    return service


def test_post_export_requires_auth(client):
    r = client.post("/api/macro/notebooklm/exports", json={})
    assert r.status_code == 401


def test_post_export_success(authed_client, mock_service):
    app.dependency_overrides[get_macro_notebooklm_export_service] = lambda: mock_service
    try:
        r = authed_client.post(
            "/api/macro/notebooklm/exports",
            json={"mode": "all_retained"},
        )
        assert r.status_code == 202
        data = r.json()
        assert data["export_id"] == "export_macro_test_123"
        assert data["state"] == "queued"
        assert data["stage"] == "initialized"
        mock_service.request_export.assert_called_once_with(mode="all_retained")
    finally:
        app.dependency_overrides.pop(get_macro_notebooklm_export_service, None)


def test_get_latest_export_success(authed_client, mock_service):
    app.dependency_overrides[get_macro_notebooklm_export_service] = lambda: mock_service
    try:
        r = authed_client.get("/api/macro/notebooklm/exports/latest")
        assert r.status_code == 200
        data = r.json()
        assert data["export_id"] == "export_macro_test_123"
        assert data["strategy_report_id"] == "macro_strategy_2026-10-05"
        assert len(data["notebooks"]) >= 1
    finally:
        app.dependency_overrides.pop(get_macro_notebooklm_export_service, None)


def test_get_latest_export_empty_returns_null(authed_client, mock_service):
    mock_service.get_latest_export.return_value = None
    app.dependency_overrides[get_macro_notebooklm_export_service] = lambda: mock_service
    try:
        r = authed_client.get("/api/macro/notebooklm/exports/latest")
        assert r.status_code == 200
        assert r.json() is None
    finally:
        app.dependency_overrides.pop(get_macro_notebooklm_export_service, None)


def test_get_export_by_id_success(authed_client, mock_service):
    app.dependency_overrides[get_macro_notebooklm_export_service] = lambda: mock_service
    try:
        r = authed_client.get("/api/macro/notebooklm/exports/export_macro_test_123")
        assert r.status_code == 200
        data = r.json()
        assert data["export_id"] == "export_macro_test_123"
        mock_service.get_export.assert_called_once_with("export_macro_test_123")
    finally:
        app.dependency_overrides.pop(get_macro_notebooklm_export_service, None)


def test_get_export_by_id_not_found(authed_client, mock_service):
    mock_service.get_export.return_value = None
    app.dependency_overrides[get_macro_notebooklm_export_service] = lambda: mock_service
    try:
        r = authed_client.get("/api/macro/notebooklm/exports/non_existent_id")
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "macro_export_not_found"
    finally:
        app.dependency_overrides.pop(get_macro_notebooklm_export_service, None)


def test_retry_export_success(authed_client, mock_service):
    app.dependency_overrides[get_macro_notebooklm_export_service] = lambda: mock_service
    try:
        r = authed_client.post("/api/macro/notebooklm/exports/export_macro_test_123/retry")
        assert r.status_code == 202
        data = r.json()
        assert data["export_id"] == "export_macro_test_123"
        mock_service.retry_export.assert_called_once_with("export_macro_test_123")
    finally:
        app.dependency_overrides.pop(get_macro_notebooklm_export_service, None)


def test_retry_export_not_found_returns_404(authed_client, mock_service):
    mock_service.retry_export.side_effect = LookupError("Export not found")
    app.dependency_overrides[get_macro_notebooklm_export_service] = lambda: mock_service
    try:
        r = authed_client.post("/api/macro/notebooklm/exports/export_macro_test_123/retry")
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "macro_export_not_found"
    finally:
        app.dependency_overrides.pop(get_macro_notebooklm_export_service, None)
