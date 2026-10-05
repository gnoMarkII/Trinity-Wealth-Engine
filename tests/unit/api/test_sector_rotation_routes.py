from datetime import datetime, timezone

import pytest
from fastapi import HTTPException, Response

from api.routers.portfolio.router_macro import (
    get_sector_rotation_history,
    get_sector_rotation_latest,
    refresh_sector_rotation,
)


def _cold_payload(timeframe: str = "weekly", tail: int = 12):
    return {
        "capability_status": "enabled",
        "refresh_state": "running",
        "retry_after_seconds": 2,
        "error_code": None,
        "last_attempt_at": None,
        "expected_session": "2026-10-01",
        "freshness": "unknown",
        "missing_sessions": 0,
        "served_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "timeframe": timeframe,
        "tail": tail,
        "summary": None,
        "snapshot": None,
    }


def test_latest_returns_accepted_while_cold_refresh_runs():
    class Service:
        def latest(self, *, timeframe, tail):
            return _cold_payload(timeframe, tail)

    response = Response()
    result = get_sector_rotation_latest(response, timeframe="daily", tail=20, service=Service())

    assert response.status_code == 202
    assert result.refresh_state == "running"
    assert result.retry_after_seconds == 2
    assert result.snapshot is None


def test_refresh_requests_work_and_returns_accepted_state():
    class Service:
        refresh_requested = False

        def request_refresh(self, *, force=False):
            self.refresh_requested = force

        def latest(self, *, timeframe, tail):
            return _cold_payload(timeframe, tail)

    service = Service()
    response = Response()
    result = refresh_sector_rotation(response, timeframe="weekly", tail=12, service=service)

    assert service.refresh_requested is True
    assert response.status_code == 202
    assert result.refresh_state == "running"


def test_history_maps_missing_archive_to_404():
    class Service:
        def history(self, snapshot_id, *, timeframe, range_name):
            return None

    with pytest.raises(HTTPException) as raised:
        get_sector_rotation_history("missing", timeframe="weekly", range_name="1y", service=Service())

    assert raised.value.status_code == 404
    assert raised.value.detail == {"code": "sector_snapshot_not_found"}


def test_history_maps_integrity_failure_to_503():
    class Service:
        def history(self, snapshot_id, *, timeframe, range_name):
            raise RuntimeError("archive hash mismatch")

    with pytest.raises(HTTPException) as raised:
        get_sector_rotation_history("broken", timeframe="daily", range_name="3m", service=Service())

    assert raised.value.status_code == 503
    assert raised.value.detail == {"code": "sector_snapshot_integrity_failure"}
