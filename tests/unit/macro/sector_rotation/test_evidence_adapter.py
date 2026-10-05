from datetime import date, timedelta
import json

from application.knowledge.write_models import KnowledgeWriteCommand
from tools.archivist.composition import build_knowledge_write_port
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.artifact_store import DurableArtifactStore
from tools.macro.adapters.sector_evidence_adapter import SectorEvidenceAdapter
from tools.macro.adapters.sector_evidence_adapter import SectorEvidenceError
from tools.macro.sector_rotation.domain.calculations import build_snapshot, normalize_price_inputs
from tools.macro.sector_rotation.domain.universe import BENCHMARK, SECTOR_TICKERS


def test_canonical_evidence_is_idempotent_and_recoverable_from_broker_receipt(tmp_path):
    dates = []
    current = date(2025, 1, 1)
    while len(dates) < 320:
        if current.weekday() < 5:
            dates.append(current.isoformat())
        current += timedelta(days=1)
    prices = {
        ticker: {day: 100.0 + i * (0.12 + 0.005 * position) for position, day in enumerate(dates)}
        for i, ticker in enumerate(SECTOR_TICKERS)
    }
    prices[BENCHMARK] = {day: 100.0 + i * 0.1 for i, day in enumerate(dates)}
    snapshot = build_snapshot(prices)
    vault = tmp_path / "vault"
    paths = VaultPaths(vault)
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=tmp_path / "runtime")
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)

    first = evidence.publish(snapshot, prices)
    recovered = evidence.load(snapshot.snapshot_id)
    second = evidence.publish(snapshot, prices)
    artifact = DurableArtifactStore(paths).get_revision_artifact(first["note_id"], first["revision_id"])
    archived = json.loads(artifact.body)

    assert first["status"] == "committed"
    assert second["status"] == "duplicate_reused"
    assert first["idempotency_key"] == f"sector-rotation:evidence-v3:{snapshot.snapshot_id}"
    assert archived["archive_schema"] == "sector-rotation-evidence-v3"
    assert "snapshot" not in archived
    assert recovered is not None
    assert recovered[0].snapshot_id == snapshot.snapshot_id
    assert recovered[0].input_digest == snapshot.input_digest
    assert recovered[1] == normalize_price_inputs(prices)


def test_long_history_archive_keeps_only_rebuildable_inputs_and_identity(tmp_path):
    sessions = []
    current = date(2021, 1, 1)
    # A no-holiday weekday grid stresses the payload ceiling beyond a normal
    # five-year US trading calendar.
    while len(sessions) < 1400:
        if current.weekday() < 5:
            sessions.append(current.isoformat())
        current += timedelta(days=1)
    prices = {
        ticker: {day: 100.0 + index * (0.08 + 0.003 * position)
                 for position, day in enumerate(sessions)}
        for index, ticker in enumerate(SECTOR_TICKERS)
    }
    prices[BENCHMARK] = {day: 100.0 + index * 0.07 for index, day in enumerate(sessions)}
    canonical = normalize_price_inputs(prices)
    snapshot = build_snapshot(canonical, expected_sessions=sessions)
    expanded_archive_size = len(json.dumps({
        "snapshot": snapshot.model_dump(mode="json"),
        "normalized_prices": canonical,
        "expected_sessions": sessions,
    }, separators=(",", ":")).encode("utf-8"))
    assert expanded_archive_size > SectorEvidenceAdapter.MAX_ARCHIVE_BODY_BYTES

    paths = VaultPaths(tmp_path / "vault")
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=tmp_path / "runtime")
    evidence = SectorEvidenceAdapter(write_port=port, vault_paths=paths)
    receipt = evidence.publish(snapshot, canonical, expected_sessions=sessions)
    recovered = evidence.load(snapshot.snapshot_id)

    assert receipt["status"] == "committed"
    assert recovered is not None
    assert recovered[0].model_dump(mode="json") == snapshot.model_dump(mode="json")
    assert recovered[1] == canonical


def test_v2_archive_rebuild_rejects_mutated_snapshot_facts():
    sessions = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    prices = {
        ticker: {day: 100.0 + index * (0.12 + 0.005 * position)
                 for position, day in enumerate(sessions)}
        for index, ticker in enumerate(SECTOR_TICKERS)
    }
    prices[BENCHMARK] = {day: 100.0 + index * 0.1 for index, day in enumerate(sessions)}
    snapshot = build_snapshot(prices, expected_sessions=sessions)
    archived_snapshot = snapshot.model_dump(mode="json")
    archived_snapshot["rows"][0]["relative_strength"] = 999.0
    archive = {
        "archive_schema": "sector-rotation-evidence-v2",
        "snapshot": archived_snapshot,
        "normalized_prices": normalize_price_inputs(prices),
        "expected_sessions": sessions,
    }

    try:
        SectorEvidenceAdapter._validate_archive(snapshot.snapshot_id, snapshot.input_digest, archive)
    except SectorEvidenceError as exc:
        assert str(exc) == "sector_snapshot_archive_rebuild_mismatch"
    else:
        raise AssertionError("mutated archive facts must fail canonical rebuild validation")


def test_legacy_v2_receipt_remains_readable_after_v3_migration(tmp_path):
    sessions = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    prices = {
        ticker: {day: 100.0 + index * (0.12 + 0.005 * position)
                 for position, day in enumerate(sessions)}
        for index, ticker in enumerate(SECTOR_TICKERS)
    }
    prices[BENCHMARK] = {day: 100.0 + index * 0.1 for index, day in enumerate(sessions)}
    canonical = normalize_price_inputs(prices)
    snapshot = build_snapshot(canonical, expected_sessions=sessions)
    document_key = f"macro:sector_rotation:{snapshot.snapshot_id}"
    body = json.dumps({
        "archive_schema": "sector-rotation-evidence-v2",
        "snapshot": snapshot.model_dump(mode="json"),
        "normalized_prices": canonical,
        "expected_sessions": sessions,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    paths = VaultPaths(tmp_path / "vault")
    port = build_knowledge_write_port(vault_paths=paths, runtime_base=tmp_path / "runtime")
    command = KnowledgeWriteCommand(
        operation="upsert_note",
        idempotency_key=f"sector-rotation:{snapshot.snapshot_id}",
        document_key=document_key,
        entity_type="macro_snapshot",
        producer="sector-rotation",
        producer_version=snapshot.formula_version,
        actor="macro-sector-rotation",
        payload={
            "metadata": {
                "schema_version": 2,
                "document_key": document_key,
                "entity_type": "macro_snapshot",
                "title": f"US Sector Rotation {snapshot.as_of_date or 'unavailable'}",
                "as_of_date": snapshot.as_of_date,
                "snapshot_id": snapshot.snapshot_id,
                "source": "Yahoo Finance via OHLCV adapter",
            },
            "body": body,
            "filename": f"Sector_Rotation_{snapshot.as_of_date or 'unavailable'}.md",
            "profile_id": "published",
        },
    )
    assert port.submit(command).is_success

    recovered = SectorEvidenceAdapter(write_port=port, vault_paths=paths).load(snapshot.snapshot_id)

    assert recovered is not None
    assert recovered[0].model_dump(mode="json") == snapshot.model_dump(mode="json")
    assert recovered[1] == canonical
