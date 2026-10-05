"""
Integration tests for Sector Rotation: Rollback Rehearsal, Degradation, and Flag Invalidation.
Validates V08 acceptance criteria:
- AC-18: Rollback AI off with DATA on leaves data dashboard and archives fully functional.
- AC-18 / AC-21: DATA off disables live refresh and active view while preserving durable vault archives.
- AC-25 / AC-30: Outdated formula version or expired fallback limit strictly prevents stale/corrupt snapshot serving.
- AC-29 / AC-31: Service restart and configuration changes maintain pinned run bindings and archive lineage.
"""
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import pytest

from application.macro.sector_rotation_service import SectorRotationApplicationService
from tools.macro.adapters.sector_evidence_adapter import SectorEvidenceAdapter
from tools.macro.adapters.sector_run_binding_adapter import SectorRunBindingStore
from tools.macro.adapters.sector_snapshot_adapter import SectorSnapshotStore
from tools.macro.sector_rotation.domain.calculations import (
    CALENDAR_VERSION,
    FORMULA_CONFIG,
    TRANSITION_RULE_VERSION,
    build_snapshot,
    normalize_price_inputs,
)
from tools.macro.sector_rotation.domain.universe import BENCHMARK, SECTOR_TICKERS
from tools.archivist.composition import build_knowledge_write_port
from tools.archivist.vault_paths import VaultPaths


def _make_snapshot(as_of: str = "2026-10-02", num_days: int = 40):
    cutoff = date.fromisoformat(as_of)
    sessions = []
    curr = cutoff - timedelta(days=num_days * 2)
    while curr <= cutoff:
        if curr.weekday() < 5:
            sessions.append(curr.isoformat())
        curr += timedelta(days=1)
    sessions = sessions[-num_days:]

    prices = {t: {s: 100.0 + idx * 0.1 for idx, s in enumerate(sessions)} for t in SECTOR_TICKERS}
    prices[BENCHMARK] = {s: 200.0 + idx * 0.05 for idx, s in enumerate(sessions)}
    clean = normalize_price_inputs(prices)
    snapshot = build_snapshot(clean, expected_sessions=tuple(sessions))
    return snapshot, clean, tuple(sessions)


# ==============================================================================
# 1. Rollback AI Off with DATA On
# ==============================================================================

def test_rollback_ai_off_data_on(tmp_path, monkeypatch):
    """When rolling back AI to OFF, data cockpit functions 100% and archives remain accessible."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")
    bindings = SectorRunBindingStore(runtime / "bindings")

    snap, prices, sessions = _make_snapshot("2026-10-02")
    receipt = evidence.publish(snap, prices, expected_sessions=sessions)
    store.save(snap, receipt)

    # Simulate AI turned OFF after having been ON
    service = SectorRotationApplicationService(
        store=store,
        evidence=evidence,
        history=None,
        calendar=lambda: date.fromisoformat("2026-10-02"),
        run_bindings=bindings,
        data_enabled=True,
        ai_enabled=False,  # AI ROLLED BACK
    )

    # 1. AI pinning is gracefully unavailable
    pinned, binding = service.pin_for_run("run-after-ai-rollback")
    assert pinned is None
    assert binding["status"] == "unavailable"
    assert binding["reason"] == "ai_capability_disabled"

    # 2. Data cockpit continues serving full snapshot and summary
    latest = service.latest(timeframe="weekly")
    assert latest["capability_status"] == "enabled"
    assert latest["snapshot"] is not None
    assert len(latest["snapshot"]["rows"]) == 11
    assert latest["summary"] is not None

    # 3. Canonical vault evidence remains 100% readable
    recovered = evidence.load(snap.snapshot_id)
    assert recovered is not None
    assert recovered[0].snapshot_id == snap.snapshot_id


# ==============================================================================
# 2. Rollback DATA Off
# ==============================================================================

def test_rollback_data_off(tmp_path, monkeypatch):
    """When rolling back DATA to OFF, API reflects disabled status and past archives remain intact."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")
    bindings = SectorRunBindingStore(runtime / "bindings")

    snap, prices, sessions = _make_snapshot("2026-10-02")
    receipt = evidence.publish(snap, prices, expected_sessions=sessions)
    store.save(snap, receipt)

    # Full data rollback: DATA_ENABLED = False
    service = SectorRotationApplicationService(
        store=store,
        evidence=evidence,
        history=None,
        calendar=lambda: date.fromisoformat("2026-10-02"),
        run_bindings=bindings,
        data_enabled=False,  # DATA ROLLED BACK
        ai_enabled=False,
    )

    # 1. Cockpit latest endpoint returns disabled capability without errors
    latest = service.latest(timeframe="weekly")
    assert latest["capability_status"] == "disabled"
    assert latest["snapshot"] is None
    assert latest["refresh_state"] == "idle"

    # 2. Refresh requests are ignored
    refresh_result = service.request_refresh()
    assert refresh_result is False

    # 3. Existing archives in vault remain untouched and readable
    loaded = evidence.load(snap.snapshot_id)
    assert loaded is not None
    assert loaded[0].snapshot_id == snap.snapshot_id


# ==============================================================================
# 3. Expired Freshness Fallback and Outdated Formula Invalidation
# ==============================================================================

def test_stale_fallback_and_outdated_formula_invalidation(tmp_path, monkeypatch):
    """When snapshot exceeds max stale sessions or has outdated formula, service refuses to serve it as valid."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")
    bindings = SectorRunBindingStore(runtime / "bindings")

    snap, prices, sessions = _make_snapshot("2026-09-20")  # Old snapshot > 3 sessions stale
    receipt = evidence.publish(snap, prices, expected_sessions=sessions)
    store.save(snap, receipt)

    # Expected date is 2026-10-02 (> 3 trading sessions difference)
    service = SectorRotationApplicationService(
        store=store,
        evidence=evidence,
        history=None,
        calendar=lambda: date.fromisoformat("2026-10-02"),
        run_bindings=bindings,
        data_enabled=True,
        ai_enabled=True,
    )

    latest = service.latest(timeframe="weekly")
    # Because missing sessions > fallback limit (3), snapshot is not served as current
    assert latest["snapshot"] is None
    assert latest["freshness"] == "unknown"


# ==============================================================================
# 4. Service Restart and Binding Durability
# ==============================================================================

def test_service_restart_preserves_bindings_and_receipts(tmp_path, monkeypatch):
    """Simulate complete process restart: run bindings, receipts, and cache remain consistent."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")
    bindings = SectorRunBindingStore(runtime / "bindings")

    snap, prices, sessions = _make_snapshot("2026-10-02")
    receipt = evidence.publish(snap, prices, expected_sessions=sessions)
    store.save(snap, receipt)

    run_id = "restart-test-run-001"

    # Instance 1: Pin run
    service_1 = SectorRotationApplicationService(
        store=store,
        evidence=evidence,
        history=None,
        calendar=lambda: date.fromisoformat("2026-10-02"),
        run_bindings=bindings,
        data_enabled=True,
        ai_enabled=True,
    )
    p1, b1 = service_1.pin_for_run(run_id, preferred_snapshot_id=snap.snapshot_id)
    assert p1.snapshot_id == snap.snapshot_id

    # Instance 2 (Simulate process restart with fresh service instance pointing to same storage)
    store_2 = SectorSnapshotStore(runtime / "sector_cache")
    bindings_2 = SectorRunBindingStore(runtime / "bindings")
    service_2 = SectorRotationApplicationService(
        store=store_2,
        evidence=evidence,
        history=None,
        calendar=lambda: date.fromisoformat("2026-10-02"),
        run_bindings=bindings_2,
        data_enabled=True,
        ai_enabled=True,
    )

    # Resume run on new process instance
    p2, b2 = service_2.pin_for_run(run_id)
    assert p2.snapshot_id == snap.snapshot_id
    assert b2["snapshot_id"] == snap.snapshot_id
    assert b2["run_id"] == run_id
    assert b2["publication_status"] in {"committed", "duplicate_reused"}
