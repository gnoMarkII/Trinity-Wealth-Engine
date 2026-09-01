"""Unit tests for Snapshot Hash Verification & CAS Immutability (Phase 0 & v3.1)."""
import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.market.evidence_store import (
    save_evidence_payload,
    load_evidence_payload,
    compute_payload_sha256,
    get_cas_storage_path,
)


def test_cas_detects_file_tampering():
    with TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)
        original_payload = {"ticker": "MSFT", "target_price": 450.0}
        
        storage_ref, expected_hash = save_evidence_payload(original_payload, base_dir=base_path)
        
        # Verify valid read
        loaded = load_evidence_payload(storage_ref, expected_hash=expected_hash, base_dir=base_path)
        assert loaded["target_price"] == 450.0
        
        # Tamper with file on disk directly
        file_path = get_cas_storage_path(expected_hash, base_dir=base_path)
        tampered_data = {"ticker": "MSFT", "target_price": 9999.0}
        file_path.write_text(json.dumps(tampered_data), encoding="utf-8")
        
        # Attempting to load with original expected hash MUST raise ValueError
        with pytest.raises(ValueError, match="CAS integrity check failed"):
            load_evidence_payload(storage_ref, expected_hash=expected_hash, base_dir=base_path)


def test_snapshot_checksum_changes_on_manifest_mutation():
    manifest_state_1 = {
        "financials": {"item_id": "financials", "payload_hash": "hash_a"},
        "macro": {"item_id": "macro", "payload_hash": "hash_b"},
    }
    checksum_1 = compute_payload_sha256(manifest_state_1)

    # Mutate a single payload hash
    manifest_state_2 = {
        "financials": {"item_id": "financials", "payload_hash": "hash_a_mutated"},
        "macro": {"item_id": "macro", "payload_hash": "hash_b"},
    }
    checksum_2 = compute_payload_sha256(manifest_state_2)

    assert checksum_1 != checksum_2
