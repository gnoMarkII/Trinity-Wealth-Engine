"""
Comprehensive tests for Sector Rotation: Canonical Evidence, Integrity, Recovery, and Feature Flags.
Validates AC-09, AC-11, AC-20, AC-21, AC-23, AC-29, AC-30 and RC-01, RC-02, RC-03, RC-14, RC-15, RC-17, RC-24.
"""
from datetime import date, timedelta
import json
import os
import pytest

from application.knowledge.write_models import KnowledgeWriteCommand
from application.macro.sector_rotation_service import SectorRotationApplicationService
from application.macro.sector_rotation_ports import SectorHistoryBatch
from schemas.sector_rotation_schemas import SectorRotationSnapshot
from tools.archivist.composition import build_knowledge_write_port
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.artifact_store import DurableArtifactStore
from tools.macro.adapters.sector_evidence_adapter import SectorEvidenceAdapter, SectorEvidenceError
from tools.macro.adapters.sector_snapshot_adapter import SectorSnapshotStore
from tools.macro.adapters.sector_run_binding_adapter import SectorRunBindingStore
from tools.macro.sector_rotation.domain.calculations import (
    build_snapshot,
    normalize_price_inputs,
)
from tools.macro.sector_rotation.domain.universe import BENCHMARK, SECTOR_TICKERS


def _create_sample_snapshot():
    dates = [f"2026-09-{i:02d}" for i in range(1, 26)]
    prices = {t: {d: 100.0 + i for i, d in enumerate(dates)} for t in SECTOR_TICKERS}
    prices[BENCHMARK] = {d: 100.0 + i * 0.5 for i, d in enumerate(dates)}
    clean = normalize_price_inputs(prices)
    return build_snapshot(clean, expected_sessions=dates), clean, dates


# ==============================================================================
# AC-23, AC-29, RC-14, RC-15, RC-17: Canonical Evidence Publishing & Verification
# ==============================================================================

def test_evidence_publish_committed_and_duplicate_reuse(tmp_path):
    """Verify first publish is committed and second identical publish is duplicate_reused."""
    snapshot, prices, sessions = _create_sample_snapshot()
    vault = tmp_path / "vault"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=tmp_path / "runtime")
    adapter = SectorEvidenceAdapter(write_port=port, vault_paths=paths)

    receipt1 = adapter.publish(snapshot, prices, expected_sessions=sessions)
    assert receipt1["status"] == "committed"
    assert receipt1["note_id"] is not None and receipt1["note_id"].startswith("note_")
    assert receipt1["idempotency_key"] == f"sector-rotation:evidence-v3:{snapshot.snapshot_id}"

    receipt2 = adapter.publish(snapshot, prices, expected_sessions=sessions)
    assert receipt2["status"] == "duplicate_reused"
    assert receipt2["revision_id"] == receipt1["revision_id"]


def test_oversize_archive_rejection(tmp_path):
    """Verify that an archive payload exceeding MAX_ARCHIVE_BODY_BYTES is rejected."""
    snapshot, prices, sessions = _create_sample_snapshot()
    vault = tmp_path / "vault"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=tmp_path / "runtime")
    adapter = SectorEvidenceAdapter(write_port=port, vault_paths=paths)

    # Monkeypatch MAX_ARCHIVE_BODY_BYTES to a tiny limit (e.g. 500 bytes)
    orig_max = adapter.MAX_ARCHIVE_BODY_BYTES
    adapter.MAX_ARCHIVE_BODY_BYTES = 500
    try:
        with pytest.raises(SectorEvidenceError, match="sector_snapshot_evidence_exceeds_safe_payload_limit"):
            adapter.publish(snapshot, prices, expected_sessions=sessions)
    finally:
        adapter.MAX_ARCHIVE_BODY_BYTES = orig_max


# ==============================================================================
# AC-30, RC-01, RC-02, RC-24: Cache Loss and Canonical Vault Recovery
# ==============================================================================

def test_cache_loss_recovers_identically_from_vault_evidence(tmp_path):
    """Verify that when local snapshot cache is cleared, get_snapshot falls back to vault archive."""
    snapshot, prices, sessions = _create_sample_snapshot()
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")

    # Commit snapshot to vault and store in cache
    receipt = evidence.publish(snapshot, prices, expected_sessions=sessions)
    store.save(snapshot, receipt)

    service = SectorRotationApplicationService(
        store=store,
        evidence=evidence,
        history=None,
        calendar=lambda: date.fromisoformat(snapshot.as_of_date),
    )

    # Sanity check: loaded from cache
    cached = service.get_snapshot(snapshot.snapshot_id)
    assert cached is not None
    assert cached.snapshot_id == snapshot.snapshot_id

    # SIMULATE CACHE LOSS: clear the snapshot directory in store
    snapshots_dir = store.root / "snapshots"
    for f in snapshots_dir.glob("*.json"):
        f.unlink()

    # Re-fetch through service -> must successfully recover from canonical vault evidence!
    recovered = service.get_snapshot(snapshot.snapshot_id)
    assert recovered is not None
    assert recovered.snapshot_id == snapshot.snapshot_id
    assert recovered.input_digest == snapshot.input_digest
    assert recovered.model_dump(mode="json") == snapshot.model_dump(mode="json")


def test_runtime_tampering_is_detected_and_rejected(tmp_path):
    """Verify that tampering with local cache is detected against canonical vault fingerprint."""
    snapshot, prices, sessions = _create_sample_snapshot()
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")

    receipt = evidence.publish(snapshot, prices, expected_sessions=sessions)
    store.save(snapshot, receipt)

    service = SectorRotationApplicationService(
        store=store,
        evidence=evidence,
        history=None,
        calendar=lambda: date.fromisoformat(snapshot.as_of_date),
    )

    # Tamper with snapshot in local cache
    cache_file = store.root / "snapshots" / f"{snapshot.snapshot_id}.json"
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    # Tamper input_digest
    data["snapshot"]["input_digest"] = "tampered_digest_value_12345"
    cache_file.write_text(json.dumps(data), encoding="utf-8")

    # Calling get_snapshot must detect mismatch and raise RuntimeError
    with pytest.raises(RuntimeError, match="sector_snapshot_runtime_archive_mismatch"):
        service.get_snapshot(snapshot.snapshot_id)


# ==============================================================================
# RC-24: Feature Flags Matrix (4 combinations)
# ==============================================================================

def test_feature_flags_all_four_combinations(tmp_path):
    """Verify serving behavior for all 4 DATA/AI flag permutations."""
    snapshot, prices, sessions = _create_sample_snapshot()
    vault = tmp_path / "vault"
    runtime = tmp_path / "runtime"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=runtime)
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    store = SectorSnapshotStore(runtime / "sector_cache")
    receipt = evidence.publish(snapshot, prices, expected_sessions=sessions)
    store.save(snapshot, receipt)

    bindings = SectorRunBindingStore(runtime / "bindings")

    class _MockHistory:
        def fetch(self):
            return SectorHistoryBatch(prices, {}, date.fromisoformat(snapshot.as_of_date), tuple(sessions))

    # Permutation 1: DATA=True, AI=False (Default production setting)
    svc_1 = SectorRotationApplicationService(
        store=store, evidence=evidence, history=_MockHistory(),
        calendar=lambda: date.fromisoformat(snapshot.as_of_date),
        run_bindings=bindings, data_enabled=True, ai_enabled=False,
    )
    latest_1 = svc_1.latest()
    assert latest_1["capability_status"] == "enabled"
    assert latest_1["snapshot"] is not None
    _, prep_1 = svc_1.pin_for_run("r1", job_id="j1", task_id="t1")
    assert prep_1["status"] == "unavailable"
    assert prep_1["reason"] == "ai_capability_disabled"

    # Permutation 2: DATA=True, AI=True (Shadow environment setting)
    svc_2 = SectorRotationApplicationService(
        store=store, evidence=evidence, history=_MockHistory(),
        calendar=lambda: date.fromisoformat(snapshot.as_of_date),
        run_bindings=bindings, data_enabled=True, ai_enabled=True,
    )
    latest_2 = svc_2.latest()
    assert latest_2["capability_status"] == "enabled"
    snap_2, prep_2 = svc_2.pin_for_run("r2", job_id="j2", task_id="t2")
    assert snap_2 is not None
    assert prep_2["publication_status"] in ("committed", "duplicate_reused")

    # Permutation 3: DATA=False, AI=False (Completely disabled)
    svc_3 = SectorRotationApplicationService(
        store=store, evidence=evidence, history=_MockHistory(),
        calendar=lambda: date.fromisoformat(snapshot.as_of_date),
        run_bindings=bindings, data_enabled=False, ai_enabled=False,
    )
    latest_3 = svc_3.latest()
    assert latest_3["capability_status"] == "disabled"
    assert latest_3["snapshot"] is None
    assert svc_3.request_refresh() is False
    # Archived reads MUST still work!
    archived_3 = svc_3.get_snapshot(snapshot.snapshot_id)
    assert archived_3 is not None
    assert archived_3.snapshot_id == snapshot.snapshot_id

    # Permutation 4: DATA=False, AI=True (Invalid combination: AI cannot bypass DATA)
    svc_4 = SectorRotationApplicationService(
        store=store, evidence=evidence, history=_MockHistory(),
        calendar=lambda: date.fromisoformat(snapshot.as_of_date),
        run_bindings=bindings, data_enabled=False, ai_enabled=True,
    )
    _, prep_4 = svc_4.pin_for_run("r4", job_id="j4", task_id="t4")
    assert prep_4["status"] == "unavailable"
    assert prep_4["reason"] == "data_capability_disabled"
