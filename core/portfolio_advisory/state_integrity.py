"""Frozen Checkpoint & State Hash Integrity Verification.

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies (no FastAPI, Pydantic,
SQLite, or network I/O) per Hexagonal Architecture AST Domain Layer Isolation rules.

Reference:
Claude Code for Hedge Funds: Build a Reproducible As-Of Research State
Chapter 3 (The As-Of State Checkpoint) & Appendix A (State Manifests & sha256sums.json)
"Historical inputs are immutable after state freeze."
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def compute_bytes_sha256(content: bytes) -> str:
    """Computes standard hexadecimal SHA-256 digest of bytes."""
    return hashlib.sha256(content).hexdigest()


def compute_file_sha256(file_path: Path) -> str:
    """Reads a file in 64KB blocks and returns hexadecimal SHA-256 digest."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


@dataclass(frozen=True)
class StateManifest:
    """Immutable state snapshot manifest defining frozen inputs and their exact digests."""
    state_id: str
    cutoff_utc: str
    immutable_files: List[str]
    checksums: Dict[str, str]
    manifest_version: str = "v1"
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "manifest_version": self.manifest_version,
            "cutoff_utc": self.cutoff_utc,
            "created_at_utc": self.created_at_utc,
            "immutable_files": list(self.immutable_files),
            "checksums": dict(self.checksums),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateManifest":
        return cls(
            state_id=data["state_id"],
            cutoff_utc=data["cutoff_utc"],
            immutable_files=list(data.get("immutable_files", [])),
            checksums=dict(data.get("checksums", {})),
            manifest_version=data.get("manifest_version", "v1"),
            created_at_utc=data.get("created_at_utc", ""),
        )

    def to_sha256sums_text(self) -> str:
        """Exports in standard POSIX sha256sum format: '<hash>  <filename>'."""
        lines = []
        for file_rel in sorted(self.immutable_files):
            digest = self.checksums.get(file_rel, "")
            lines.append(f"{digest}  {file_rel}")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class StateIntegrityReport:
    """Report detailing byte-for-byte state integrity status."""
    is_valid: bool
    state_id: str
    checked_files_count: int
    mismatched_files: List[str]
    missing_files: List[str]
    tamper_details: List[Dict[str, str]]
    summary_status: str  # "HEALTHY", "INTEGRITY_TAMPERED", "MISSING_INPUTS"
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "state_id": self.state_id,
            "checked_files_count": self.checked_files_count,
            "mismatched_files": self.mismatched_files,
            "missing_files": self.missing_files,
            "tamper_details": self.tamper_details,
            "summary_status": self.summary_status,
            "message": self.message,
        }


def create_state_manifest(
    state_id: str,
    base_dir: Path,
    file_rel_paths: Sequence[str],
    cutoff_utc: str,
) -> StateManifest:
    """Constructs a StateManifest by computing SHA-256 for all existing target files."""
    checksums: Dict[str, str] = {}
    valid_files: List[str] = []

    for rel_str in file_rel_paths:
        target_path = base_dir / rel_str
        if not target_path.is_file():
            raise FileNotFoundError(f"Cannot freeze state: immutable input file not found: {target_path}")
        digest = compute_file_sha256(target_path)
        checksums[rel_str] = digest
        valid_files.append(rel_str)

    return StateManifest(
        state_id=state_id,
        cutoff_utc=cutoff_utc,
        immutable_files=valid_files,
        checksums=checksums,
    )


def verify_state_integrity(
    manifest: StateManifest,
    base_dir: Path,
) -> StateIntegrityReport:
    """Asserts that historical frozen state files have not been retroactively altered.
    
    Hard-fails if even a single byte differs from the frozen checkpoint digest.
    """
    mismatched: List[str] = []
    missing: List[str] = []
    tamper_details: List[Dict[str, str]] = []

    for file_rel in manifest.immutable_files:
        target_path = base_dir / file_rel
        expected_hash = manifest.checksums.get(file_rel, "")

        if not target_path.exists() or not target_path.is_file():
            missing.append(file_rel)
            tamper_details.append({
                "file": file_rel,
                "expected_hash": expected_hash,
                "computed_hash": "FILE_MISSING",
                "reason": "Immutable input file was deleted or moved.",
            })
            continue

        actual_hash = compute_file_sha256(target_path)
        if actual_hash != expected_hash:
            mismatched.append(file_rel)
            tamper_details.append({
                "file": file_rel,
                "expected_hash": expected_hash,
                "computed_hash": actual_hash,
                "reason": f"Byte-for-byte SHA-256 mismatch! Expected {expected_hash[:12]}..., got {actual_hash[:12]}...",
            })

    if not manifest.immutable_files:
        return StateIntegrityReport(
            is_valid=False,
            state_id=manifest.state_id,
            checked_files_count=0,
            mismatched_files=[],
            missing_files=[],
            tamper_details=[],
            summary_status="EMPTY_MANIFEST",
            message=f"CRITICAL: Historical state '{manifest.state_id}' manifest contains zero immutable files.",
        )

    if mismatched:
        return StateIntegrityReport(
            is_valid=False,
            state_id=manifest.state_id,
            checked_files_count=len(manifest.immutable_files),
            mismatched_files=mismatched,
            missing_files=missing,
            tamper_details=tamper_details,
            summary_status="INTEGRITY_TAMPERED",
            message=(
                f"CRITICAL: Historical state '{manifest.state_id}' tampered! "
                f"{len(mismatched)} file(s) modified after freeze: {', '.join(mismatched)}."
            ),
        )

    if missing:
        return StateIntegrityReport(
            is_valid=False,
            state_id=manifest.state_id,
            checked_files_count=len(manifest.immutable_files),
            mismatched_files=[],
            missing_files=missing,
            tamper_details=tamper_details,
            summary_status="MISSING_INPUTS",
            message=(
                f"CRITICAL: Historical state '{manifest.state_id}' incomplete! "
                f"{len(missing)} file(s) missing: {', '.join(missing)}."
            ),
        )

    return StateIntegrityReport(
        is_valid=True,
        state_id=manifest.state_id,
        checked_files_count=len(manifest.immutable_files),
        mismatched_files=[],
        missing_files=[],
        tamper_details=[],
        summary_status="HEALTHY",
        message=f"State '{manifest.state_id}' verified: 100% byte integrity across {len(manifest.immutable_files)} frozen inputs.",
    )

