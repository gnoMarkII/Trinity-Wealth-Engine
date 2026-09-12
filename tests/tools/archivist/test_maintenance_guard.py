from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.archivist.core import _atomic_write_text
from tools.archivist.maintenance_guard import (
    MaintenanceLeaseConflictError,
    acquire_maintenance_lease,
    assert_write_allowed,
    release_maintenance_lease,
)


def test_active_lease_blocks_unowned_atomic_write_before_temp_creation(tmp_path: Path, monkeypatch) -> None:
    lease = acquire_maintenance_lease(
        tmp_path,
        owner="migration-owner",
        purpose="test",
        ttl_seconds=60,
        lease_id="lease-test",
    )
    target = tmp_path / "30_Knowledge_Base" / "Concepts" / "blocked.md"

    with pytest.raises(MaintenanceLeaseConflictError, match="lease-test"):
        _atomic_write_text(target, "blocked")
    assert not target.exists()
    assert not list(target.parent.glob(".*.tmp"))

    monkeypatch.setenv("VAULT_MAINTENANCE_OWNER", lease.owner)
    _atomic_write_text(target, "allowed")
    assert target.read_text(encoding="utf-8") == "allowed"


def test_expired_active_lease_fails_closed_until_explicit_release(tmp_path: Path) -> None:
    path = tmp_path / ".system" / "maintenance.json"
    path.parent.mkdir(parents=True)
    now = datetime.now(timezone.utc)
    path.write_text(
        json.dumps(
            {
                "lease_version": 1,
                "lease_id": "expired-test",
                "owner": "owner-a",
                "purpose": "test",
                "status": "active",
                "started_at": (now - timedelta(minutes=2)).isoformat(),
                "expires_at": (now - timedelta(minutes=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MaintenanceLeaseConflictError, match="expired"):
        assert_write_allowed(tmp_path / "note.md")

    released = release_maintenance_lease(tmp_path, owner="owner-a", reason="test-complete")
    assert released.status == "released"
    assert_write_allowed(tmp_path / "note.md")
