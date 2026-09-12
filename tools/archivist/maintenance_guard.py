"""Fail-closed maintenance lease checks for Vault writers.

The vault is also an Obsidian Sync tree, so a migration needs a small,
durable coordination point that every managed writer can consult before it
creates a directory, lock, temporary file, or projection.  Query code does
not call this module and therefore never changes maintenance state.

An ``active`` lease blocks writers whose owner does not match the explicit
``VAULT_MAINTENANCE_OWNER`` environment variable.  Expired active leases are
also treated as blocking: an operator must explicitly release or replace the
lease rather than allowing a stale process to race the remediation.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional


class MaintenanceLeaseError(RuntimeError):
    """Base error for malformed or unsafe maintenance lease operations."""


class MaintenanceLeaseConflictError(MaintenanceLeaseError):
    """Raised when a writer is not the owner of an active lease."""


class MaintenanceLeaseOwnershipError(MaintenanceLeaseError):
    """Raised when release/renew is attempted by another owner."""


@dataclass(frozen=True)
class MaintenanceLease:
    lease_id: str
    owner: str
    purpose: str
    status: str
    started_at: str
    expires_at: str
    baseline_tree_fingerprint: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MaintenanceLease":
        required = ("lease_id", "owner", "status", "started_at", "expires_at")
        missing = [key for key in required if not str(payload.get(key) or "").strip()]
        if missing:
            raise MaintenanceLeaseError(
                f"maintenance lease is missing required fields: {', '.join(missing)}"
            )
        return cls(
            lease_id=str(payload["lease_id"]),
            owner=str(payload["owner"]),
            purpose=str(payload.get("purpose") or ""),
            status=str(payload["status"]).lower(),
            started_at=str(payload["started_at"]),
            expires_at=str(payload["expires_at"]),
            baseline_tree_fingerprint=(
                str(payload["baseline_tree_fingerprint"])
                if payload.get("baseline_tree_fingerprint")
                else None
            ),
        )

    def is_active(self) -> bool:
        return self.status == "active"

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        try:
            expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise MaintenanceLeaseError(
                f"maintenance lease has invalid expires_at: {self.expires_at!r}"
            ) from exc
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return expiry <= (now or datetime.now(timezone.utc))


def lease_path(vault_root: str | Path) -> Path:
    root = Path(vault_root).resolve()
    return root / ".system" / "maintenance.json"


def load_maintenance_lease(vault_root: str | Path) -> Optional[MaintenanceLease]:
    path = lease_path(vault_root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaintenanceLeaseError(f"cannot read maintenance lease {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MaintenanceLeaseError(f"maintenance lease must be a JSON object: {path}")
    return MaintenanceLease.from_dict(payload)


def _candidate_vault_root(path: str | Path) -> Optional[Path]:
    """Find the nearest vault root without treating an arbitrary parent as a vault."""
    resolved = Path(path).resolve()
    candidates = [resolved, *resolved.parents]
    for candidate in candidates:
        if (candidate / ".system" / "maintenance.json").is_file():
            return candidate
    return None


def assert_write_allowed(path: str | Path, *, owner: Optional[str] = None) -> None:
    """Raise before a managed writer mutates a vault under another lease.

    Paths outside a vault, including scratch evidence and external vector
    runtime state, are deliberately ignored.  A caller may opt into the
    current lease using ``owner`` or ``VAULT_MAINTENANCE_OWNER``.
    """
    vault_root = _candidate_vault_root(path)
    if vault_root is None:
        return
    target = Path(path).resolve()
    # Updating the control document is handled by acquire/release helpers and
    # must not recurse through this guard.
    if target == lease_path(vault_root):
        return
    lease = load_maintenance_lease(vault_root)
    if lease is None or not lease.is_active():
        return
    effective_owner = owner or os.getenv("VAULT_MAINTENANCE_OWNER", "").strip()
    if effective_owner != lease.owner:
        expiry_note = "expired" if lease.is_expired() else "active"
        raise MaintenanceLeaseConflictError(
            f"Vault writer blocked by {expiry_note} maintenance lease "
            f"{lease.lease_id!r} owned by {lease.owner!r}; target={target}"
        )


def _write_lease_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def acquire_maintenance_lease(
    vault_root: str | Path,
    *,
    owner: str,
    purpose: str,
    ttl_seconds: int,
    baseline_tree_fingerprint: Optional[str] = None,
    lease_id: Optional[str] = None,
) -> MaintenanceLease:
    """Create a lease, allowing takeover only when the prior lease expired."""
    if not owner.strip():
        raise ValueError("maintenance lease owner is required")
    path = lease_path(vault_root)
    existing = load_maintenance_lease(vault_root)
    if existing and existing.is_active() and not existing.is_expired():
        raise MaintenanceLeaseConflictError(
            f"maintenance lease {existing.lease_id!r} is owned by {existing.owner!r}"
        )
    now = datetime.now(timezone.utc)
    payload = {
        "lease_version": 1,
        "lease_id": lease_id or f"lease_{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}",
        "owner": owner,
        "purpose": purpose,
        "status": "active",
        "started_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": (now + timedelta(seconds=max(1, int(ttl_seconds)))).isoformat().replace("+00:00", "Z"),
        "baseline_tree_fingerprint": baseline_tree_fingerprint,
    }
    _write_lease_payload(path, payload)
    return MaintenanceLease.from_dict(payload)


def release_maintenance_lease(
    vault_root: str | Path,
    *,
    owner: str,
    reason: str = "completed",
) -> MaintenanceLease:
    """Explicitly release the current lease; ownership is checked first."""
    path = lease_path(vault_root)
    current = load_maintenance_lease(vault_root)
    if current is None:
        raise MaintenanceLeaseError(f"no maintenance lease exists at {path}")
    if current.owner != owner:
        raise MaintenanceLeaseOwnershipError(
            f"cannot release lease owned by {current.owner!r} as {owner!r}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "released"
    payload["released_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload["release_reason"] = reason
    _write_lease_payload(path, payload)
    return MaintenanceLease.from_dict(payload)
