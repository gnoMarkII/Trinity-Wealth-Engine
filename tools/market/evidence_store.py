"""Content-Addressed Store (CAS) & Evidence Manifest Engine (Phase 0 & v3.1).

Provides tamper-evident, SHA256-verified caching for raw payloads and builds
immutable Evidence Manifest items for the AnalysisEvidenceSnapshot.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from core.logger import get_logger
from schemas.micro_quant_schemas import (
    DataStatus,
    EvidenceItemMetadata,
    EvidenceManifestItem,
    SnapshotMetadata,
    AnalysisEvidenceSnapshot,
    CorporateActionsEvidence,
)
from tools._atomic_io import _atomic_write_to
from tools.archivist.core import VAULT_PATH

log = get_logger(__name__)

_DEFAULT_CAS_DIR = VAULT_PATH / "30_Knowledge_Base" / ".evidence_cache"


def canonical_json_dumps(obj: Any) -> str:
    """Serializes object to deterministic canonical JSON with sorted keys."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


def compute_payload_sha256(payload: Any) -> str:
    """Computes SHA256 checksum of canonical JSON representation of payload."""
    canonical_str = canonical_json_dumps(payload)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def get_cas_storage_path(sha256_hash: str, base_dir: Optional[Path] = None) -> Path:
    """Returns absolute path to CAS file for given hash."""
    target_dir = base_dir or _DEFAULT_CAS_DIR
    return target_dir / f"{sha256_hash}.json"


def save_evidence_payload(
    payload: Any,
    base_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[str, str]:
    """Saves raw payload into Content-Addressed Store by SHA256 checksum.

    Returns:
        tuple[str, str]: (relative_storage_ref, sha256_hash)
    """
    sha256_hash = compute_payload_sha256(payload)
    target_path = get_cas_storage_path(sha256_hash, base_dir=base_dir)
    
    canonical_content = canonical_json_dumps(payload)
    _atomic_write_to(target_path, canonical_content)
    
    relative_ref = f".evidence_cache/{sha256_hash}.json"
    return relative_ref, sha256_hash


def load_evidence_payload(
    storage_ref_or_hash: str,
    expected_hash: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Any:
    """Loads payload from CAS and verifies SHA256 integrity on read.

    Raises:
        ValueError: If SHA256 checksum does not match expected_hash (tampering detected).
        FileNotFoundError: If CAS file does not exist.
    """
    clean_ref = storage_ref_or_hash.replace(".json", "")
    if "/" in clean_ref or "\\" in clean_ref:
        clean_ref = Path(clean_ref).stem
    
    file_path = get_cas_storage_path(clean_ref, base_dir=base_dir)
    if not file_path.exists():
        raise FileNotFoundError(f"CAS payload not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    # Also verify parsed canonical hash if raw formatting differed
    parsed_data = json.loads(content)
    canonical_hash = compute_payload_sha256(parsed_data)

    target_expected = expected_hash or clean_ref
    if actual_hash != target_expected and canonical_hash != target_expected:
        raise ValueError(
            f"CAS integrity check failed! Expected hash {target_expected}, "
            f"but actual content hash is {actual_hash} (canonical: {canonical_hash})"
        )

    return parsed_data


def build_evidence_manifest_item(
    item_id: str,
    payload: Any,
    source_uri: str,
    retrieved_at: str,
    source_as_of: str,
    provider_tier: str = "primary_best_effort",
    status: DataStatus = "available",
    fiscal_period_end: Optional[str] = None,
    reported_at: Optional[str] = None,
    currency: str = "USD",
    unit: str = "units",
    query_slice: Optional[dict] = None,
    stale_reason: Optional[str] = None,
    store_payload: bool = True,
    base_dir: Optional[Path] = None,
) -> EvidenceManifestItem:
    """Convenience helper to hash, store in CAS, and return EvidenceManifestItem."""
    if store_payload and payload is not None:
        storage_ref, payload_hash = save_evidence_payload(payload, base_dir=base_dir)
    else:
        storage_ref = None
        payload_hash = compute_payload_sha256(payload if payload is not None else {})

    metadata = EvidenceItemMetadata(
        source_as_of=source_as_of,
        retrieved_at=retrieved_at,
        fiscal_period_end=fiscal_period_end,
        reported_at=reported_at,
        currency=currency,
        unit=unit,
        source_uri=source_uri,
        payload_hash=payload_hash,
        provider_tier=provider_tier,  # type: ignore
        status=status,
        stale_reason=stale_reason,
    )

    return EvidenceManifestItem(
        item_id=item_id,
        metadata=metadata,
        storage_ref=storage_ref,
        query_slice=query_slice,
    )
