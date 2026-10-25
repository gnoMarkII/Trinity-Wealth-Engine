"""Integration and API tests for Promise Check endpoints."""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app
from schemas.promise_schemas import (
    PromiseScoresDTO,
    PromiseLedgerDTO,
    PromiseSyncResponseDTO,
    TranscriptCallDTO,
    PromiseItemDTO,
)
from tools.market.adapters.transcript_vault_adapter import (
    save_promise_ledger,
    save_transcript_text,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def override_require_session():
    from api.auth import require_session
    app.dependency_overrides[require_session] = lambda: {"user_id": "test_user"}
    yield
    app.dependency_overrides = {}


def test_get_promise_ledger_not_found(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "tools.market.adapters.transcript_vault_adapter.DEFAULT_TRANSCRIPT_CACHE_DIR",
        tmp_path / "cache" / "transcripts",
    )
    response = client.get("/api/equity/UNKNOWN/earnings-call/ledger")
    assert response.status_code == 404
    assert "No promise ledger found" in response.json()["detail"]


def test_get_promise_ledger_success(tmp_path: Path, monkeypatch):
    cache_dir = tmp_path / "cache" / "transcripts"
    monkeypatch.setattr(
        "tools.market.adapters.transcript_vault_adapter.DEFAULT_TRANSCRIPT_CACHE_DIR",
        cache_dir,
    )
    
    # Pre-populate sample ledger with 8 calls
    ledger_data = {
        "framework_version": "0.1",
        "ticker": "NVDA",
        "company": "NVIDIA Corporation",
        "run_date": "2025-02-26",
        "calls": [
            {"quarter": f"FY2024Q{i}", "date": f"2023-0{i*3}-15", "status": "ok", "source": "stockanalysis", "words": 7000}
            for i in range(1, 5)
        ] + [
            {"quarter": f"FY2025Q{i}", "date": f"2024-0{i*3}-15", "status": "ok", "source": "stockanalysis", "words": 7000}
            for i in range(1, 5)
        ],
        "promises": [
            {
                "id": "P-NVDA-FY2024Q1-01",
                "made_in": "FY2024Q1",
                "speaker": "Colette Kress, CFO",
                "section": "prepared",
                "type": "A_numeric",
                "category": "revenue",
                "scope": "company",
                "quote": "We expect Q2 revenue to be $11.00 billion.",
                "metric": "Q2 FY24 Revenue",
                "target_kind": "point",
                "target_point": 11.0,
                "due": "FY2024Q2",
                "due_precision": "explicit",
                "status": "Delivered",
                "resolved_in": "FY2024Q2",
                "evidence_quote": "Revenue for the second quarter was $13.51 billion.",
                "actual_numeric": 13.51,
                "notes": "Delivered ahead of guidance.",
            }
        ],
    }
    save_promise_ledger("NVDA", ledger_data, base_dir=cache_dir)

    response = client.get("/api/equity/NVDA/earnings-call/ledger")
    assert response.status_code == 200
    data = response.json()
    assert data["from_cache"] is True
    assert data["scores"]["credibility"] == 100.0
    assert data["scores"]["grade"] == "A"
    assert len(data["ledger"]["calls"]) == 8
    assert len(data["ledger"]["promises"]) == 1


def test_sync_promise_ledger_endpoint(monkeypatch):
    class MockPromiseService:
        def sync_ledger(self, ticker: str, force_refresh: bool = False, limit_calls: int = 8):
            return PromiseSyncResponseDTO(
                ledger=PromiseLedgerDTO(
                    ticker=ticker.upper(),
                    company="Mock Corp",
                    run_date="2026-10-20",
                    calls=[],
                    promises=[],
                ),
                scores=PromiseScoresDTO(
                    ticker=ticker.upper(),
                    company="Mock Corp",
                    run_date="2026-10-20",
                    window={"from": "FY24 Q1", "to": "FY25 Q4", "calls": 8, "missing": []},
                    credibility=95.0,
                    grade="A",
                    confidence="High",
                    delivery=95.0,
                    transparency=100.0,
                    n_promises=10,
                    n_resolved=8,
                    counts={"Delivered": 8},
                    bias={
                        "label": "Straight shooter",
                        "code": "straight",
                        "message": "Balanced guidance",
                        "median_delta_pct": 0.5,
                        "beat_rate": 60.0,
                        "n": 6,
                        "min_n": 5,
                    },
                    milestone_slip_quarters=0.2,
                    milestone_slip_n=3,
                    trend={"verdict": "improving"},
                ),
                from_cache=False,
            )

    from api.dependencies import get_promise_service
    app.dependency_overrides[get_promise_service] = lambda: MockPromiseService()

    response = client.post("/api/equity/MSFT/earnings-call/sync?force_refresh=true")
    assert response.status_code == 200
    data = response.json()
    assert data["scores"]["credibility"] == 95.0
    assert data["scores"]["bias"]["label"] == "Straight shooter"
    assert data["ledger"]["ticker"] == "MSFT"
