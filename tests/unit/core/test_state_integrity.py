"""Unit tests for Frozen Checkpoint & State Hash Integrity Verification.

Hexagonal Architecture Rule: Pure domain testing using standard library fixtures and tmp_path.
"""
import hashlib
from pathlib import Path
import pytest

from core.portfolio_advisory.state_integrity import (
    compute_bytes_sha256,
    compute_file_sha256,
    create_state_manifest,
    verify_state_integrity,
    StateManifest,
)


def test_compute_file_sha256(tmp_path: Path):
    """File SHA-256 matches exact hash of content."""
    test_file = tmp_path / "sample.csv"
    content = b"symbol,date,close\nNVDA,2026-11-01,135.50\n"
    test_file.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert compute_file_sha256(test_file) == expected


def test_create_state_manifest_and_verify_healthy(tmp_path: Path):
    """Creating a manifest and verifying against identical files passes."""
    state_dir = tmp_path / "as_of_2026_11_04"
    state_dir.mkdir()

    f1 = state_dir / "prices_history.csv"
    f1.write_text("ticker,price\nNVDA,135.50\nAAPL,220.00\n", encoding="utf-8")

    f2 = state_dir / "theses_ledger.json"
    f2.write_text('{"records": [{"claim_id": "TH-01"}]}', encoding="utf-8")

    manifest = create_state_manifest(
        state_id="as_of_2026_11_04",
        base_dir=state_dir,
        file_rel_paths=["prices_history.csv", "theses_ledger.json"],
        cutoff_utc="2026-11-04T23:59:59Z",
    )

    assert manifest.state_id == "as_of_2026_11_04"
    assert len(manifest.immutable_files) == 2
    assert "prices_history.csv" in manifest.checksums

    report = verify_state_integrity(manifest, state_dir)
    assert report.is_valid is True
    assert report.summary_status == "HEALTHY"
    assert report.checked_files_count == 2
    assert len(report.mismatched_files) == 0


def test_tamper_detection_on_retroactive_modification(tmp_path: Path):
    """Modifying even 1 byte in a frozen input triggers CRITICAL INTEGRITY_TAMPERED."""
    state_dir = tmp_path / "frozen_state"
    state_dir.mkdir()

    p_file = state_dir / "prices_history.csv"
    p_file.write_text("NVDA,135.50\n", encoding="utf-8")

    manifest = create_state_manifest(
        state_id="state_nov04",
        base_dir=state_dir,
        file_rel_paths=["prices_history.csv"],
        cutoff_utc="2026-11-04T23:59:59Z",
    )

    # Simulate silent retroactive data tampering (changing 135.50 to 135.51)
    p_file.write_text("NVDA,135.51\n", encoding="utf-8")

    report = verify_state_integrity(manifest, state_dir)
    assert report.is_valid is False
    assert report.summary_status == "INTEGRITY_TAMPERED"
    assert "prices_history.csv" in report.mismatched_files
    assert "Byte-for-byte SHA-256 mismatch" in report.tamper_details[0]["reason"]
    assert "CRITICAL: Historical state 'state_nov04' tampered!" in report.message


def test_missing_frozen_file_detection(tmp_path: Path):
    """Missing frozen input file triggers MISSING_INPUTS."""
    state_dir = tmp_path / "frozen_state"
    state_dir.mkdir()

    f1 = state_dir / "prices.csv"
    f1.write_text("NVDA,135.50\n", encoding="utf-8")

    manifest = create_state_manifest(
        state_id="state_missing_test",
        base_dir=state_dir,
        file_rel_paths=["prices.csv"],
        cutoff_utc="2026-11-04T23:59:59Z",
    )

    # Delete the file
    f1.unlink()

    report = verify_state_integrity(manifest, state_dir)
    assert report.is_valid is False
    assert report.summary_status == "MISSING_INPUTS"
    assert "prices.csv" in report.missing_files


def test_sha256sums_export_format(tmp_path: Path):
    """Manifest exports in standard POSIX sha256sum format."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    f = state_dir / "data.txt"
    f.write_text("hello", encoding="utf-8")

    manifest = create_state_manifest(
        state_id="state_posix",
        base_dir=state_dir,
        file_rel_paths=["data.txt"],
        cutoff_utc="2026-11-04T00:00:00Z",
    )

    export_text = manifest.to_sha256sums_text()
    expected_hash = hashlib.sha256(b"hello").hexdigest()
    assert export_text.strip() == f"{expected_hash}  data.txt"


def test_manifest_dict_roundtrip():
    """StateManifest serializes and deserializes cleanly."""
    manifest = StateManifest(
        state_id="s1",
        cutoff_utc="2026-11-04T00:00:00Z",
        immutable_files=["a.csv", "b.json"],
        checksums={"a.csv": "abc", "b.json": "def"},
    )
    d = manifest.to_dict()
    restored = StateManifest.from_dict(d)
    assert restored.state_id == "s1"
    assert restored.immutable_files == ["a.csv", "b.json"]
    assert restored.checksums == {"a.csv": "abc", "b.json": "def"}

