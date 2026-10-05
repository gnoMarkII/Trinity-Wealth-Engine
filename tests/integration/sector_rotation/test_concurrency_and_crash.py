"""
Integration tests for Sector Rotation: Concurrency, Coalescing, Crash Recovery, and Run Binding.
Validates AC-10, AC-11, AC-12, AC-27, AC-29, AC-31 and RC-04, RC-16, RC-17, RC-18, RC-19, RC-21.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import json
import time
import pytest

from application.macro.sector_rotation_service import SectorRotationApplicationService
from application.macro.sector_rotation_ports import SectorHistoryBatch
from schemas.sector_rotation_schemas import SectorRotationSnapshot
from tools.archivist.composition import build_knowledge_write_port
from tools.archivist.vault_paths import VaultPaths
from tools.macro.adapters.sector_evidence_adapter import SectorEvidenceAdapter
from tools.macro.adapters.sector_snapshot_adapter import SectorSnapshotStore
from tools.macro.adapters.sector_run_binding_adapter import SectorRunBindingStore
from tools.macro.adapters.sector_history_adapter import SectorHistoryAdapter
from tools.macro.sector_rotation.domain.calculations import (
    build_snapshot,
    normalize_price_inputs,
)
from tools.macro.sector_rotation.domain.universe import BENCHMARK, SECTOR_TICKERS


def _create_fixture_prices(as_of: str = "2026-10-02"):
    end_date = date.fromisoformat(as_of)
    dates = [(end_date - timedelta(days=20 - i)).isoformat() for i in range(21)]
    prices = {t: {d: 100.0 + i for i, d in enumerate(dates)} for t in SECTOR_TICKERS}
    prices[BENCHMARK] = {d: 100.0 + i * 0.5 for i, d in enumerate(dates)}
    return normalize_price_inputs(prices), dates


# ==============================================================================
# AC-10, RC-04: Concurrent Request Coalescing Across Workers
# ==============================================================================

def test_concurrent_refresh_coalesces_to_single_provider_fetch(tmp_path):
    """Multiple concurrent refresh requests coalesce per key and do not double-fetch."""
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")
    bindings = SectorRunBindingStore(runtime / "bindings")

    prices, sessions = _create_fixture_prices()
    as_of = date.fromisoformat("2026-10-02")

    fetch_call_count = 0

    class _CountingHistory:
        def fetch(self):
            nonlocal fetch_call_count
            fetch_call_count += 1
            time.sleep(0.05)  # Simulate provider latency
            return SectorHistoryBatch(prices, {}, as_of, tuple(sessions))

    service = SectorRotationApplicationService(
        store=store,
        evidence=evidence,
        history=_CountingHistory(),
        calendar=lambda: as_of,
        run_bindings=bindings,
        data_enabled=True,
        ai_enabled=True,
    )

    # Launch two concurrent refreshes simultaneously using ThreadPool
    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(service.refresh_now)
        f2 = executor.submit(service.refresh_now)
        snap1 = f1.result()
        snap2 = f2.result()

    # Both must receive the exact same snapshot
    assert snap1.snapshot_id == snap2.snapshot_id
    assert snap1.input_digest == snap2.input_digest
    # Provider fetch must have been coalesced (only 1 fetch call)
    assert fetch_call_count == 1


# ==============================================================================
# AC-11, AC-12, AC-29, RC-16, RC-17, RC-18, RC-21: Crash Recovery & Pending Replay
# ==============================================================================

def test_crash_during_pending_publication_replays_exact_payload_on_retry(tmp_path):
    """When a run crashes while binding is in 'pending' status, restart replays exact payload."""
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")
    bindings = SectorRunBindingStore(runtime / "bindings")

    prices, sessions = _create_fixture_prices()
    snapshot = build_snapshot(prices, expected_sessions=sessions)
    run_id = "job-42:turn-1:1"

    # Simulate CRASH: Stage binding with status="pending" (selected but uncommitted)
    pending_record = {
        "run_id": run_id,
        "job_id": "job-42",
        "logical_task_id": "task-1",
        "snapshot_id": snapshot.snapshot_id,
        "data_revision": snapshot.input_digest,
        "formula_version": snapshot.formula_version,
        "calendar_version": snapshot.calendar_version,
        "transition_rule_version": snapshot.transition_rule_version,
        "formula_config": snapshot.formula_config,
        "selected_at": "2026-10-02T10:00:00Z",
        "evaluation_as_of": snapshot.as_of_date,
        "eligibility_at_selection": {},
        "publication_status": "pending",
        "publication_receipt": None,
        "pending_snapshot": snapshot.model_dump(mode="json"),
        "pending_prices": prices,
        "pending_expected_sessions": sessions,
    }
    bindings.create(run_id, pending_record)

    service = SectorRotationApplicationService(
        store=store,
        evidence=evidence,
        history=None,
        calendar=lambda: date.fromisoformat(snapshot.as_of_date),
        run_bindings=bindings,
        data_enabled=True,
        ai_enabled=True,
    )

    # RESTART / RETRY with the same run_id:
    resumed_snapshot, binding = service.pin_for_run(run_id)

    assert resumed_snapshot is not None
    assert resumed_snapshot.snapshot_id == snapshot.snapshot_id
    assert binding["publication_status"] in ("committed", "duplicate_reused")
    # Verify that pending payloads were cleaned up after successful commit
    assert "pending_snapshot" not in binding
    assert "pending_prices" not in binding

    # Verify that canonical evidence in vault is now committed and recoverable
    archived = evidence.load(snapshot.snapshot_id)
    assert archived is not None
    assert archived[0].snapshot_id == snapshot.snapshot_id


# ==============================================================================
# AC-12, AC-27, RC-18, RC-19: Run Binding Immutability Across Later Updates
# ==============================================================================

def test_run_binding_remains_pinned_to_initial_snapshot_after_cache_update(tmp_path):
    """A pinned run binding retains its original snapshot even if global cache is refreshed."""
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")
    bindings = SectorRunBindingStore(runtime / "bindings")

    prices_v1, sessions_v1 = _create_fixture_prices("2026-10-01")
    prices_v2, sessions_v2 = _create_fixture_prices("2026-10-02")

    current_prices = prices_v1
    current_as_of = date(2026, 10, 1)

    class _MutableHistory:
        def fetch(self):
            return SectorHistoryBatch(current_prices, {}, current_as_of, tuple(sessions_v1))

    service = SectorRotationApplicationService(
        store=store,
        evidence=evidence,
        history=_MutableHistory(),
        calendar=lambda: current_as_of,
        run_bindings=bindings,
        data_enabled=True,
        ai_enabled=True,
    )

    # 1. First run R1 pins snapshot S1
    snap_r1_init, binding_r1 = service.pin_for_run("run-alpha-1")
    assert snap_r1_init is not None
    s1_id = snap_r1_init.snapshot_id

    # 2. Later, market advances to 2026-10-02:
    current_prices = prices_v2
    current_as_of = date(2026, 10, 2)
    # Refresh global cache with new day
    snap_v2 = service.refresh_now(allow_stale_provider_data=True)
    s2_id = snap_v2.snapshot_id
    assert s1_id != s2_id

    # 3. Old run R1 resumes: MUST REMAIN PINNED to S1!
    snap_r1_resumed, binding_r1_resumed = service.pin_for_run("run-alpha-1")
    assert snap_r1_resumed.snapshot_id == s1_id
    assert snap_r1_resumed.snapshot_id != s2_id

    # 4. A new task R2 receives the new snapshot S2:
    snap_r2, binding_r2 = service.pin_for_run("run-beta-2")
    assert snap_r2.snapshot_id == s2_id


# ==============================================================================
# RC-18, RC-19: Concurrent Contention on Same Run ID
# ==============================================================================

def test_concurrent_pinning_same_run_id_returns_identical_binding(tmp_path):
    """Two concurrent workers trying to pin the same run_id return identical binding without collision."""
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")
    bindings = SectorRunBindingStore(runtime / "bindings")

    prices, sessions = _create_fixture_prices()
    as_of = date.fromisoformat("2026-10-02")

    class _History:
        def fetch(self):
            return SectorHistoryBatch(prices, {}, as_of, tuple(sessions))

    service = SectorRotationApplicationService(
        store=store,
        evidence=evidence,
        history=_History(),
        calendar=lambda: as_of,
        run_bindings=bindings,
        data_enabled=True,
        ai_enabled=True,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(service.pin_for_run, "shared-run-id")
        f2 = executor.submit(service.pin_for_run, "shared-run-id")
        res1, bind1 = f1.result()
        res2, bind2 = f2.result()

    assert res1.snapshot_id == res2.snapshot_id
    assert bind1["snapshot_id"] == bind2["snapshot_id"]
    assert bind1["run_id"] == "shared-run-id"


# ==============================================================================
# RC-04: Stale Lease Recovery
# ==============================================================================

def test_stale_refresh_lease_is_recovered(tmp_path, monkeypatch):
    """When a refresh lease is older than lease_seconds, a new refresh request is allowed."""
    runtime = tmp_path / "runtime"
    store = SectorSnapshotStore(runtime / "sector_cache")

    # Set state as 'running' with a timestamp 300 seconds ago
    store.update_state(
        refresh_state="running",
        last_attempt_at="2026-10-02T10:00:00Z",
    )

    # Set lease seconds to 60s
    monkeypatch.setenv("SECTOR_ROTATION_REFRESH_LEASE_SECONDS", "60")

    service = SectorRotationApplicationService(
        store=store,
        evidence=None,
        history=None,
        calendar=lambda: date(2026, 10, 2),
        data_enabled=True,
    )

    # request_refresh must detect expired lease and accept new refresh
    # (returns True as it schedules the background execution)
    accepted = service.request_refresh(force=False)
    assert accepted is True
