"""Unit tests for Content-Addressed Store (CAS) & Evidence Manifest Engine (Phase 0 & v3.1)."""
import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from schemas.micro_quant_schemas import (
    AnalysisEvidenceSnapshot,
    CorporateActionsEvidence,
    EvidenceItemMetadata,
    EvidenceManifestItem,
    SnapshotMetadata,
)
from tools.market.evidence_store import (
    canonical_json_dumps,
    compute_payload_sha256,
    save_evidence_payload,
    load_evidence_payload,
    build_evidence_manifest_item,
)


def test_canonical_json_and_sha256_deterministic():
    data1 = {"b": 2, "a": 1, "nested": {"z": 10, "y": 20}}
    data2 = {"a": 1, "nested": {"y": 20, "z": 10}, "b": 2}
    
    dump1 = canonical_json_dumps(data1)
    dump2 = canonical_json_dumps(data2)
    assert dump1 == dump2
    
    hash1 = compute_payload_sha256(data1)
    hash2 = compute_payload_sha256(data2)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_cas_save_and_load_integrity():
    with TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)
        payload = {
            "ticker": "AAPL",
            "income_statement": [
                {"period": "2024-Q3", "revenue": 94930000000.0, "net_income": 14736000000.0}
            ]
        }

        storage_ref, sha256_hash = save_evidence_payload(payload, base_dir=base_path)
        assert storage_ref == f".evidence_cache/{sha256_hash}.json"
        
        # Load back
        loaded = load_evidence_payload(storage_ref, expected_hash=sha256_hash, base_dir=base_path)
        assert loaded["ticker"] == "AAPL"
        assert len(loaded["income_statement"]) == 1


def test_build_evidence_manifest_item():
    with TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)
        payload = {"current_price": 210.0, "atr_14": 4.5}
        
        item = build_evidence_manifest_item(
            item_id="market_ohlcv",
            payload=payload,
            source_uri="yfinance:///AAPL/history",
            retrieved_at="2026-08-28T12:00:00Z",
            source_as_of="2026-08-28T12:00:00Z",
            provider_tier="primary_best_effort",
            currency="USD",
            query_slice={"history_window": "5Y"},
            base_dir=base_path,
        )

        assert item.item_id == "market_ohlcv"
        assert item.metadata.currency == "USD"
        assert item.metadata.source_uri == "yfinance:///AAPL/history"
        assert item.metadata.status == "available"
        assert item.storage_ref is not None
        assert item.query_slice == {"history_window": "5Y"}
        
        # Verify loaded payload matches
        loaded = load_evidence_payload(item.storage_ref, expected_hash=item.metadata.payload_hash, base_dir=base_path)
        assert loaded["current_price"] == 210.0


def test_analysis_evidence_snapshot_serialization():
    now_iso = "2026-08-28T12:00:00Z"
    meta = SnapshotMetadata(
        analysis_run_id="run_AAPL_20260828_120000_test",
        schema_version="3.1",
        as_of_date="2026-08-28",
        generated_at=now_iso,
        snapshot_sha256="abc123def456",
        data_quality_flags=[],
        coverage_pct=100.0,
    )
    
    item_meta = EvidenceItemMetadata(
        source_as_of=now_iso,
        retrieved_at=now_iso,
        source_uri="yfinance:///AAPL/info",
        payload_hash="dummyhash",
        provider_tier="primary_best_effort",
        status="available",
    )
    manifest_item = EvidenceManifestItem(
        item_id="info",
        metadata=item_meta,
        storage_ref=".evidence_cache/dummyhash.json",
    )
    
    corp_act = CorporateActionsEvidence(
        metadata=item_meta,
        chart_price_basis="split_and_dividend_adjusted",
        valuation_price_basis="unadjusted_close",
    )
    
    snapshot = AnalysisEvidenceSnapshot(
        metadata=meta,
        manifest_items={"info": manifest_item},
        corporate_actions=corp_act,
        derived_features={"atr_14": 4.5},
    )
    
    raw_json = snapshot.model_dump_json()
    reloaded = AnalysisEvidenceSnapshot.model_validate_json(raw_json)
    assert reloaded.metadata.analysis_run_id == "run_AAPL_20260828_120000_test"
    assert reloaded.corporate_actions.chart_price_basis == "split_and_dividend_adjusted"
    assert reloaded.corporate_actions.valuation_price_basis == "unadjusted_close"
    assert reloaded.derived_features["atr_14"] == 4.5
