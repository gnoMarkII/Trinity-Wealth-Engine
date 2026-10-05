"""
FastAPI HTTP Contract and Performance Tests for Sector Rotation Endpoints.
Validates AC-18, AC-19, AC-20, AC-21, AC-23, AC-31 and RC-03, RC-04, RC-12, RC-23, RC-24.
"""
from datetime import date, datetime, timezone
import json
import time
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_sector_rotation_service
from application.macro.sector_rotation_service import SectorRotationApplicationService
from schemas.sector_rotation_schemas import SectorRotationSnapshot
from tools.macro.adapters.sector_snapshot_adapter import SectorSnapshotStore
from tools.macro.adapters.sector_run_binding_adapter import SectorRunBindingStore
from tools.macro.sector_rotation.domain.calculations import (
    CALENDAR_VERSION,
    FORMULA_CONFIG,
    TRANSITION_RULE_VERSION,
)
from tools.macro.sector_rotation.domain.universe import BENCHMARK, SECTOR_TICKERS


def _create_mock_snapshot(as_of: str = "2026-10-02") -> SectorRotationSnapshot:
    rows = []
    for ticker in SECTOR_TICKERS:
        rows.append({
            "ticker": ticker,
            "name": ticker,
            "status": "available",
            "returns_pct": {"1W_absolute_pct": 1.5, "1W_excess_pp": 0.5, "1W_relative_pct": 0.49},
            "return_metrics": {},
            "relative_trend": 102.0,
            "relative_momentum": 101.0,
            "quadrant": "Leading",
            "momentum_direction": "rising",
            "history": {"weekly": [{"as_of": as_of, "relative_trend": 102.0, "relative_momentum": 101.0, "quadrant": "Leading", "status": "available"}]},
            "quadrant_transitions": [],
        })
    return SectorRotationSnapshot(
        input_digest="http_test_digest_123",
        snapshot_id="sr_http_test_v2",
        as_of_date=as_of,
        calendar_version=CALENDAR_VERSION,
        transition_rule_version=TRANSITION_RULE_VERSION,
        formula_config=FORMULA_CONFIG,
        expected_session=as_of,
        expected_weekly_session=as_of,
        input_start_date="2025-01-01",
        coverage={"XLK": 252},
        available_sectors=11,
        benchmark_status="available",
        rows=rows,
    )


# ==============================================================================
# Authentication Enforcement Tests (401 without valid session cookie)
# ==============================================================================

def test_all_sector_rotation_endpoints_require_session(client: TestClient):
    """Every sector-rotation HTTP endpoint must reject unauthenticated requests with 401."""
    endpoints = [
        ("GET", "/api/macro/sector-rotation/latest"),
        ("POST", "/api/macro/sector-rotation/refresh"),
        ("GET", "/api/macro/sector-rotation/snapshots/sr_12345"),
        ("GET", "/api/macro/sector-rotation/history?snapshot_id=sr_12345"),
    ]
    for method, path in endpoints:
        res = client.request(method, path)
        assert res.status_code == 401, f"{method} {path} returned {res.status_code}, expected 401"
        assert "login" in res.json().get("detail", "").lower() or "session" in res.json().get("detail", "").lower()


# ==============================================================================
# HTTP Query Parameter Validation (422 for invalid parameters)
# ==============================================================================

def test_latest_and_history_parameter_validation(authed_client: TestClient):
    """Endpoints validate parameter ranges, enums, and required fields."""
    # Invalid timeframe
    r1 = authed_client.get("/api/macro/sector-rotation/latest?timeframe=monthly")
    assert r1.status_code == 422

    # Invalid tail (must be 1 <= tail <= 60)
    r2 = authed_client.get("/api/macro/sector-rotation/latest?tail=0")
    assert r2.status_code == 422
    r3 = authed_client.get("/api/macro/sector-rotation/latest?tail=61")
    assert r3.status_code == 422

    # History missing required snapshot_id
    r4 = authed_client.get("/api/macro/sector-rotation/history")
    assert r4.status_code == 422

    # History invalid range
    r5 = authed_client.get("/api/macro/sector-rotation/history?snapshot_id=sr_123&range=5y")
    assert r5.status_code == 422


# ==============================================================================
# Latest Endpoint: Cold 202, Warm 200, and Flag Disabled Behavior
# ==============================================================================

def test_latest_cold_store_returns_202_accepted(authed_client: TestClient, tmp_path, monkeypatch):
    """When cache is cold and background refresh is running, endpoint returns 202."""
    runtime = tmp_path / "runtime"
    store = SectorSnapshotStore(runtime / "sector_cache")
    # Simulate cold running state
    store.update_state(refresh_state="running", last_attempt_at="2026-10-02T10:00:00Z")

    service = SectorRotationApplicationService(
        store=store,
        evidence=None,
        history=None,
        calendar=lambda: date(2026, 10, 2),
        data_enabled=True,
    )
    from api.main import app
    app.dependency_overrides[get_sector_rotation_service] = lambda: service
    try:
        res = authed_client.get("/api/macro/sector-rotation/latest?timeframe=weekly")
        assert res.status_code == 202
        body = res.json()
        assert body["refresh_state"] == "running"
        assert body["retry_after_seconds"] == 2
        assert body["snapshot"] is None
    finally:
        app.dependency_overrides.pop(get_sector_rotation_service, None)


def test_latest_warm_store_returns_200_with_full_dto(authed_client: TestClient, tmp_path):
    """When snapshot is cached, latest returns 200 with full DTO and summary."""
    snapshot = _create_mock_snapshot()
    receipt = {"status": "committed"}

    runtime = tmp_path / "runtime"
    store = SectorSnapshotStore(runtime / "sector_cache")
    service = SectorRotationApplicationService(
        store=store,
        evidence=None,
        history=None,
        calendar=lambda: date.fromisoformat("2026-10-02"),
        data_enabled=True,
    )
    service._load_latest = lambda: (snapshot, receipt)

    from api.main import app
    app.dependency_overrides[get_sector_rotation_service] = lambda: service
    try:
        res = authed_client.get("/api/macro/sector-rotation/latest?timeframe=weekly&tail=12")
        assert res.status_code == 200
        data = res.json()
        assert data["capability_status"] == "enabled"
        assert data["refresh_state"] == "idle"
        assert data["snapshot"]["snapshot_id"] == snapshot.snapshot_id
        assert len(data["snapshot"]["rows"]) == 11
    finally:
        app.dependency_overrides.pop(get_sector_rotation_service, None)


def test_latest_returns_disabled_capability_when_data_flag_off(authed_client: TestClient):
    """When SECTOR_ROTATION_DATA_ENABLED is off, latest returns capability_status=disabled."""
    class _DisabledService:
        data_enabled = False

        def latest(self, *, timeframe="weekly", tail=12):
            return {
                "capability_status": "disabled",
                "refresh_state": "idle",
                "served_at": "2026-10-02T10:15:00Z",
                "timeframe": timeframe,
                "tail": tail,
                "snapshot": None,
                "summary": None,
            }

    from api.main import app
    app.dependency_overrides[get_sector_rotation_service] = lambda: _DisabledService()
    try:
        res = authed_client.get("/api/macro/sector-rotation/latest")
        assert res.status_code == 200
        data = res.json()
        assert data["capability_status"] == "disabled"
        assert data["snapshot"] is None
    finally:
        app.dependency_overrides.pop(get_sector_rotation_service, None)


# ==============================================================================
# Snapshot ID and History Lookups: 404, 503 Integrity Handling
# ==============================================================================

def test_snapshot_by_id_not_found_returns_404(authed_client: TestClient):
    """When requested snapshot_id does not exist, endpoint returns 404 with error code."""
    class _EmptyService:
        def get_snapshot(self, snapshot_id):
            return None

    from api.main import app
    app.dependency_overrides[get_sector_rotation_service] = lambda: _EmptyService()
    try:
        res = authed_client.get("/api/macro/sector-rotation/snapshots/sr_non_existent")
        assert res.status_code == 404
        assert res.json()["detail"]["code"] == "sector_snapshot_not_found"
    finally:
        app.dependency_overrides.pop(get_sector_rotation_service, None)


def test_snapshot_by_id_integrity_failure_returns_503(authed_client: TestClient):
    """When snapshot fails cryptographic verification, endpoint returns 503."""
    class _TamperedService:
        def get_snapshot(self, snapshot_id):
            raise RuntimeError("sector_snapshot_runtime_archive_mismatch")

    from api.main import app
    app.dependency_overrides[get_sector_rotation_service] = lambda: _TamperedService()
    try:
        res = authed_client.get("/api/macro/sector-rotation/snapshots/sr_tampered")
        assert res.status_code == 503
        assert res.json()["detail"]["code"] == "sector_snapshot_integrity_failure"
    finally:
        app.dependency_overrides.pop(get_sector_rotation_service, None)


# ==============================================================================
# Performance Benchmarks: Warm P95 < 500ms, Cold 202 < 1s, Payload < 200 KiB
# ==============================================================================

def test_latest_performance_benchmarks_and_payload_targets(authed_client: TestClient, tmp_path):
    """Measure warm authenticated latest latency and payload size against engineering targets."""
    snapshot = _create_mock_snapshot()
    receipt = {"status": "committed"}

    runtime = tmp_path / "runtime"
    store = SectorSnapshotStore(runtime / "sector_cache")
    service = SectorRotationApplicationService(
        store=store,
        evidence=None,
        history=None,
        calendar=lambda: date.fromisoformat("2026-10-02"),
        data_enabled=True,
    )
    service._load_latest = lambda: (snapshot, receipt)

    from api.main import app
    app.dependency_overrides[get_sector_rotation_service] = lambda: service
    try:
        # Warm-up (3 requests)
        for _ in range(3):
            authed_client.get("/api/macro/sector-rotation/latest?timeframe=weekly&tail=12")

        # 100 requests benchmark
        latencies_ms = []
        payload_bytes = 0
        for _ in range(100):
            t0 = time.perf_counter()
            r = authed_client.get("/api/macro/sector-rotation/latest?timeframe=weekly&tail=12")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            payload_bytes = len(r.content)

        latencies_ms.sort()
        p50 = latencies_ms[int(len(latencies_ms) * 0.50)]
        p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
        max_lat = latencies_ms[-1]

        # Measure 5 cold 202 requests
        store.update_state(refresh_state="running", last_attempt_at="2026-10-02T10:00:00Z")
        service._load_latest = lambda: None
        cold_latencies_ms = []
        for _ in range(5):
            t0 = time.perf_counter()
            r_cold = authed_client.get("/api/macro/sector-rotation/latest?timeframe=weekly&tail=12")
            assert r_cold.status_code == 202
            cold_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        cold_p95 = sorted(cold_latencies_ms)[int(len(cold_latencies_ms) * 0.95)]

        # Engineering targets from plan:
        # Warm p95 < 500 ms, Cold 202 < 1s (1000 ms), Payload < 200 KiB (204,800 bytes)
        assert p95 < 500.0, f"Warm p95 {p95:.2f}ms exceeded 500ms target"
        assert cold_p95 < 1000.0, f"Cold p95 {cold_p95:.2f}ms exceeded 1000ms target"
        assert payload_bytes < 204800, f"Payload {payload_bytes} bytes exceeded 200 KiB limit"

        # Record results in artifact directory
        from pathlib import Path
        perf_dir = Path("scratch/sector-rotation-verification-20261003-144100")
        perf_dir.mkdir(parents=True, exist_ok=True)
        perf_file = perf_dir / "performance.json"
        perf_data = {
            "run_id": "20261003-144100",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "warm_latest": {
                "request_count": 100,
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "max_ms": round(max_lat, 3),
                "target_p95_ms": 500.0,
                "status": "PASS" if p95 < 500.0 else "FAIL",
                "payload_bytes": payload_bytes,
                "target_payload_bytes": 204800,
                "payload_status": "PASS" if payload_bytes < 204800 else "FAIL",
            },
            "cold_latest_202": {
                "request_count": 5,
                "p95_ms": round(cold_p95, 3),
                "target_p95_ms": 1000.0,
                "status": "PASS" if cold_p95 < 1000.0 else "FAIL",
            },
        }
        perf_file.write_text(json.dumps(perf_data, indent=2), encoding="utf-8")

    finally:
        app.dependency_overrides.pop(get_sector_rotation_service, None)

