"""FastAPI lifespan tests verifying InvestorWorkflowWorker starts and stops cleanly."""
from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient

from api import main as api_main


def test_lifespan_starts_and_stops_investor_workflow_worker(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "lifespan_state.sqlite"
    monkeypatch.setenv("WEBUI_STATE_DB_PATH", str(db_path))
    monkeypatch.setenv("ENABLE_BACKGROUND_WORKERS", "true")
    monkeypatch.setenv("ENABLE_JOB_WORKERS", "false")
    monkeypatch.setenv("SCHEDULER_ENABLED", "false")

    events: list[str] = []

    class _StubWorker:
        def __init__(self, name: str, **kwargs) -> None:
            self.name = name
            self.kwargs = kwargs

        def start(self) -> None:
            events.append(f"start:{self.name}")

        async def stop(self) -> None:
            events.append(f"stop:{self.name}")

    monkeypatch.setattr(
        "api.workers.earnings_call_outbox_worker.EarningsCallOutboxWorker",
        lambda **kw: _StubWorker("earnings_outbox", **kw),
    )
    monkeypatch.setattr(
        "api.workers.portfolio_snapshot_worker.PortfolioDailySnapshotWorker",
        lambda **kw: _StubWorker("portfolio_snapshot", **kw),
    )
    monkeypatch.setattr(
        "api.workers.investor_workflow_worker.InvestorWorkflowWorker",
        lambda **kw: _StubWorker("investor_workflow", **kw),
    )
    monkeypatch.setattr("api.dependencies.get_earnings_call_service", lambda: object())
    monkeypatch.setattr("api.dependencies.get_portfolio_service", lambda: object())
    monkeypatch.setattr("api.dependencies.get_investor_uow_factory", lambda: object())
    monkeypatch.setattr("api.dependencies.get_interview_service", lambda: object())
    monkeypatch.setattr("api.dependencies.get_claim_review_service", lambda: object())
    monkeypatch.setattr("api.dependencies.get_investment_axis_service", lambda: object())
    monkeypatch.setattr("api.dependencies.get_bucket_planning_service", lambda: object())

    with TestClient(api_main.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert api_main.app.state.investor_workflow_worker is not None
        assert "start:investor_workflow" in events

    assert "stop:investor_workflow" in events
    assert events.index("start:investor_workflow") < events.index("stop:investor_workflow")

