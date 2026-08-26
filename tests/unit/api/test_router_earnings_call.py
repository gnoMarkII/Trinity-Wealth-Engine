"""Unit tests for /api/equity/{ticker}/earnings-call endpoints."""
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.auth import require_session
from api.dependencies import get_earnings_call_service
from application.earnings_call.dto import EarningsCallRunDTO
from application.earnings_call.errors import (
    EarningsCallProviderUnavailableError,
    EarningsCallRunNotFoundError,
    EarningsCallTickerMismatchError,
    EarningsCallValidationError,
)
from application.earnings_call.workflow import (
    EarningsCallKanbanStatus,
    EarningsCallRunStatus,
)


@pytest.fixture
def client():
    app.dependency_overrides[require_session] = lambda: {"user_id": "test_user"}
    yield TestClient(app)
    app.dependency_overrides.pop(require_session, None)
    app.dependency_overrides.pop(get_earnings_call_service, None)


def _make_run(
    run_id="run-1",
    ticker="TSM",
    period="Q4 2024",
    status=EarningsCallRunStatus.COMPLETED,
    kanban_status=EarningsCallKanbanStatus.CREATED,
    highlights="### 1. Financial Highlights\nRevenue up 25%.",
    vault_path="30_Knowledge_Base/Earnings_Calls/TSM/Q4_2024_TSM_Earnings_Call.md",
    kanban_card_id="card-999",
    reused_existing_run=False,
    last_error_code=None,
):
    return EarningsCallRunDTO(
        run_id=run_id,
        source_key="source-key-1",
        ticker=ticker,
        period=period,
        transcript_hash="hash-1",
        prompt_version="v1",
        status=status,
        kanban_status=kanban_status,
        highlights=highlights,
        vault_path=vault_path,
        kanban_card_id=kanban_card_id,
        reused_existing_run=reused_existing_run,
        last_error_code=last_error_code,
        created_at=100.0,
        updated_at=100.0,
    )


def test_summarize_earnings_call_success_200(client):
    mock_service = MagicMock()
    mock_service.summarize_and_store.return_value = _make_run(
        status=EarningsCallRunStatus.COMPLETED,
        kanban_status=EarningsCallKanbanStatus.CREATED,
    )
    app.dependency_overrides[get_earnings_call_service] = lambda: mock_service

    payload = {
        "period": "Q4 2024",
        "transcript": "Good morning and welcome to TSMC Fourth Quarter 2024 Earnings Call.",
    }
    response = client.post("/api/equity/TSM/earnings-call/summarize", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "run-1"
    assert data["ticker"] == "TSM"
    assert data["status"] == "completed"
    assert data["kanban_status"] == "created"
    assert data["kanban_card_id"] == "card-999"


def test_summarize_earnings_call_pending_202(client):
    mock_service = MagicMock()
    mock_service.summarize_and_store.return_value = _make_run(
        status=EarningsCallRunStatus.KANBAN_PENDING,
        kanban_status=EarningsCallKanbanStatus.PENDING,
        kanban_card_id=None,
    )
    app.dependency_overrides[get_earnings_call_service] = lambda: mock_service

    payload = {
        "period": "Q4 2024",
        "transcript": "Good morning and welcome to TSMC Fourth Quarter 2024 Earnings Call.",
    }
    response = client.post("/api/equity/TSM/earnings-call/summarize", json=payload)

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "kanban_pending"
    assert data["kanban_status"] == "pending"


def test_summarize_earnings_call_validation_error_422(client):
    mock_service = MagicMock()
    mock_service.summarize_and_store.side_effect = EarningsCallValidationError("Invalid ticker symbol")
    app.dependency_overrides[get_earnings_call_service] = lambda: mock_service

    payload = {
        "period": "Q4 2024",
        "transcript": "Valid transcript content for testing validation error flow.",
    }
    response = client.post("/api/equity/INVALID/earnings-call/summarize", json=payload)

    assert response.status_code == 422
    assert "Invalid ticker symbol" in response.json()["detail"]


def test_summarize_earnings_call_provider_unavailable_503(client):
    mock_service = MagicMock()
    mock_service.summarize_and_store.side_effect = EarningsCallProviderUnavailableError("Timeout from Gemini API")
    app.dependency_overrides[get_earnings_call_service] = lambda: mock_service

    payload = {
        "period": "Q4 2024",
        "transcript": "Valid transcript content for testing provider error flow.",
    }
    response = client.post("/api/equity/TSM/earnings-call/summarize", json=payload)

    assert response.status_code == 503
    assert "LLM provider is currently unavailable" in response.json()["detail"]


def test_get_earnings_call_run_success(client):
    mock_service = MagicMock()
    mock_service.get_run_for_ticker.return_value = _make_run(run_id="run-123", ticker="TSM")
    app.dependency_overrides[get_earnings_call_service] = lambda: mock_service

    response = client.get("/api/equity/TSM/earnings-call/runs/run-123")
    assert response.status_code == 200
    data = response.json()
    assert data["run_id"] == "run-123"
    assert data["ticker"] == "TSM"


def test_get_earnings_call_run_not_found_404(client):
    mock_service = MagicMock()
    mock_service.get_run_for_ticker.side_effect = EarningsCallRunNotFoundError("Run not found")
    app.dependency_overrides[get_earnings_call_service] = lambda: mock_service

    response = client.get("/api/equity/TSM/earnings-call/runs/run-nonexistent")
    assert response.status_code == 404


def test_retry_earnings_call_run_200(client):
    mock_service = MagicMock()
    mock_service.retry_run_for_ticker.return_value = _make_run(
        run_id="run-123", ticker="TSM", status=EarningsCallRunStatus.COMPLETED
    )
    app.dependency_overrides[get_earnings_call_service] = lambda: mock_service

    response = client.post("/api/equity/TSM/earnings-call/runs/run-123/retry")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"


def test_get_earnings_calls_success(client):
    from application.earnings_call.dto import EarningsCallNoteDTO

    mock_service = MagicMock()
    mock_service.list_earnings_calls.return_value = [
        EarningsCallNoteDTO(
            title="TSM Earnings Call Q4 2024",
            ticker="TSM",
            period="Q4 2024",
            vault_path="30_Knowledge_Base/Earnings_Calls/TSM/2024-Q4_TSM_Earnings_Call.md",
            highlights="Highlights text",
            date="2026-08-26",
            last_updated="2026-08-26 23:01:06",
            has_full_transcript=True,
        )
    ]
    app.dependency_overrides[get_earnings_call_service] = lambda: mock_service

    response = client.get("/api/equity/TSM/earnings-calls")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "TSM"
    assert data["total_count"] == 1
    assert data["items"][0]["period"] == "Q4 2024"
    assert data["items"][0]["highlights"] == "Highlights text"
