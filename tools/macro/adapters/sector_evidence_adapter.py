"""Canonical sector snapshot evidence publication and integrity-checked reads."""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Mapping, Optional

from application.knowledge.write_models import KnowledgeWriteCommand
from application.knowledge.write_ports import KnowledgeWritePort
from schemas.sector_rotation_schemas import SectorRotationSnapshot
from tools.archivist.artifact_store import ArtifactError, DurableArtifactStore
from tools.archivist.composition import build_knowledge_write_port
from tools.archivist.vault_paths import VaultPaths
from tools.macro.sector_rotation.domain.calculations import (
    FORMULA_VERSION,
    build_snapshot,
    normalize_price_inputs,
    normalized_input_digest,
)


class SectorEvidenceError(RuntimeError):
    """Evidence was not committed or failed immutable artifact verification."""


class SectorEvidenceAdapter:
    ARCHIVE_SCHEMA = "sector-rotation-evidence-v3"
    MAX_ARCHIVE_BODY_BYTES = 4_500_000

    def __init__(self, write_port: Optional[KnowledgeWritePort] = None, vault_paths: Optional[VaultPaths] = None) -> None:
        self._paths = vault_paths or VaultPaths()
        self._port = write_port or build_knowledge_write_port(vault_paths=self._paths)
        self._artifacts = DurableArtifactStore(self._paths)

    @staticmethod
    def idempotency_key(snapshot_id: str) -> str:
        # A new key avoids reusing a v2 receipt whose immutable payload contains
        # the duplicated, multi-megabyte derived snapshot.
        return f"sector-rotation:evidence-v3:{snapshot_id}"

    @staticmethod
    def _legacy_idempotency_key(snapshot_id: str) -> str:
        return f"sector-rotation:{snapshot_id}"

    @staticmethod
    def _snapshot_identity(snapshot: SectorRotationSnapshot) -> dict[str, Any]:
        """Compact metadata needed to verify a snapshot rebuilt from source inputs."""
        return {
            "snapshot_id": snapshot.snapshot_id,
            "input_digest": snapshot.input_digest,
            "schema_version": snapshot.schema_version,
            "formula_version": snapshot.formula_version,
            "calendar_version": snapshot.calendar_version,
            "transition_rule_version": snapshot.transition_rule_version,
            "universe_version": snapshot.universe_version,
            "benchmark": snapshot.benchmark,
            "price_basis": snapshot.price_basis,
            "formula_config": snapshot.formula_config,
            "as_of_date": snapshot.as_of_date,
            "expected_session": snapshot.expected_session,
            "expected_weekly_session": snapshot.expected_weekly_session,
            "input_start_date": snapshot.input_start_date,
            "coverage": snapshot.coverage,
            "expected_sectors": snapshot.expected_sectors,
            "available_sectors": snapshot.available_sectors,
            "benchmark_status": snapshot.benchmark_status,
        }

    def publish(
        self,
        snapshot: SectorRotationSnapshot,
        normalized_prices: Mapping[str, Mapping[str, float]],
        *,
        expected_sessions: Optional[tuple[str, ...] | list[str]] = None,
    ) -> dict[str, Any]:
        canonical_prices = normalize_price_inputs(normalized_prices)
        observed_sessions = sorted({day for prices in canonical_prices.values() for day in prices})
        session_grid = sorted(set(expected_sessions if expected_sessions is not None else observed_sessions))
        rebuilt = build_snapshot(canonical_prices, expected_sessions=session_grid)
        if snapshot.formula_version != FORMULA_VERSION or rebuilt.model_dump(mode="json") != snapshot.model_dump(mode="json"):
            raise SectorEvidenceError("sector_snapshot_does_not_match_canonical_inputs")
        body_data = {
            "archive_schema": self.ARCHIVE_SCHEMA,
            # Rotation histories are fully derivable from prices and sessions.
            # Keeping them here doubled the archive and exceeded the write
            # broker's 5 MiB command limit for a normal five-year history.
            "snapshot_identity": self._snapshot_identity(snapshot),
            "normalized_prices": canonical_prices,
            "expected_sessions": session_grid,
        }
        body = json.dumps(body_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(body.encode("utf-8")) > self.MAX_ARCHIVE_BODY_BYTES:
            raise SectorEvidenceError("sector_snapshot_evidence_exceeds_safe_payload_limit")
        document_key = f"macro:sector_rotation:{snapshot.snapshot_id}"
        command = KnowledgeWriteCommand(
            operation="upsert_note",
            idempotency_key=self.idempotency_key(snapshot.snapshot_id),
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
                "filename": f"Sector_Rotation_{snapshot.as_of_date or 'unavailable'}_{snapshot.snapshot_id}.md",
                "profile_id": "published",
            },
        )
        receipt = self._port.submit(command)
        if not receipt.is_success:
            raise SectorEvidenceError(f"sector_snapshot_not_committed:{receipt.status}:{receipt.error_code or 'unknown'}")
        if not receipt.note_id or not receipt.revision_id:
            raise SectorEvidenceError("sector_snapshot_commit_missing_revision_identity")
        try:
            artifact = self._artifacts.get_revision_artifact(receipt.note_id, receipt.revision_id)
        except ArtifactError as exc:
            raise SectorEvidenceError("sector_snapshot_evidence_integrity_failure") from exc
        archived = json.loads(artifact.body)
        self._validate_archive(snapshot.snapshot_id, snapshot.input_digest, archived)
        return {
            "status": receipt.status,
            "note_id": receipt.note_id,
            "revision_id": receipt.revision_id,
            "relative_path": receipt.relative_path,
            "content_hash": receipt.content_hash,
            "artifact_set_hash": receipt.artifact_set_hash,
            "idempotency_key": receipt.idempotency_key,
        }

    def load(self, snapshot_id: str) -> Optional[tuple[SectorRotationSnapshot, dict[str, dict[str, float]]]]:
        receipt = self._port.get_receipt(idempotency_key=self.idempotency_key(snapshot_id))
        if receipt is None:
            # Preserve read access to snapshots written before the compact v3
            # archive. New publications always use the versioned v3 key.
            receipt = self._port.get_receipt(idempotency_key=self._legacy_idempotency_key(snapshot_id))
        if receipt is None or not receipt.is_success or not receipt.note_id or not receipt.revision_id:
            return None
        try:
            artifact = self._artifacts.get_revision_artifact(receipt.note_id, receipt.revision_id)
        except ArtifactError as exc:
            raise SectorEvidenceError("sector_snapshot_evidence_integrity_failure") from exc
        archived = json.loads(artifact.body)
        archive_schema = archived.get("archive_schema")
        if archive_schema == self.ARCHIVE_SCHEMA:
            identity = archived.get("snapshot_identity")
            prices_raw = archived.get("normalized_prices")
            if not isinstance(identity, dict) or not isinstance(prices_raw, dict):
                raise SectorEvidenceError("sector_snapshot_archive_shape_invalid")
            expected_digest = identity.get("input_digest")
            if not isinstance(expected_digest, str):
                raise SectorEvidenceError("sector_snapshot_archive_shape_invalid")
            snapshot = self._validate_archive(snapshot_id, expected_digest, archived)
            prices = normalize_price_inputs(prices_raw)
            if snapshot is None:
                raise SectorEvidenceError("sector_snapshot_archive_rebuild_failed")
            return snapshot, prices
        snapshot_raw = archived.get("snapshot")
        prices_raw = archived.get("normalized_prices")
        if not isinstance(snapshot_raw, dict) or not isinstance(prices_raw, dict):
            raise SectorEvidenceError("sector_snapshot_archive_shape_invalid")
        snapshot = SectorRotationSnapshot.model_validate(snapshot_raw)
        self._validate_archive(snapshot_id, snapshot.input_digest, archived)
        prices = {
            str(symbol): {str(day): float(value) for day, value in values.items()}
            for symbol, values in prices_raw.items() if isinstance(values, dict)
        }
        return snapshot, prices

    @staticmethod
    def _validate_archive(
        snapshot_id: str, expected_digest: str, archived: Mapping[str, Any],
    ) -> Optional[SectorRotationSnapshot]:
        archive_schema = archived.get("archive_schema")
        if archive_schema == SectorEvidenceAdapter.ARCHIVE_SCHEMA:
            identity = archived.get("snapshot_identity")
            prices = archived.get("normalized_prices")
            sessions = archived.get("expected_sessions")
            if not isinstance(identity, dict) or not isinstance(prices, dict):
                raise SectorEvidenceError("sector_snapshot_archive_shape_invalid")
            if identity.get("snapshot_id") != snapshot_id:
                raise SectorEvidenceError("sector_snapshot_archive_id_mismatch")
            if identity.get("input_digest") != expected_digest:
                raise SectorEvidenceError("sector_snapshot_archive_input_digest_mismatch")
            if not isinstance(sessions, list) or any(not isinstance(day, str) for day in sessions):
                raise SectorEvidenceError("sector_snapshot_archive_session_grid_invalid")
            try:
                canonical_sessions = [date.fromisoformat(day).isoformat() for day in sessions]
            except ValueError as exc:
                raise SectorEvidenceError("sector_snapshot_archive_session_grid_invalid") from exc
            if canonical_sessions != sorted(set(canonical_sessions)):
                raise SectorEvidenceError("sector_snapshot_archive_session_grid_not_canonical")
            try:
                canonical_prices = normalize_price_inputs(prices)
                if normalized_input_digest(canonical_prices) != expected_digest:
                    raise SectorEvidenceError("sector_snapshot_archive_input_digest_mismatch")
                rebuilt = build_snapshot(canonical_prices, expected_sessions=canonical_sessions)
            except SectorEvidenceError:
                raise
            except Exception as exc:
                raise SectorEvidenceError("sector_snapshot_archive_rebuild_failed") from exc
            if SectorEvidenceAdapter._snapshot_identity(rebuilt) != identity:
                raise SectorEvidenceError("sector_snapshot_archive_rebuild_mismatch")
            if rebuilt.snapshot_id != snapshot_id:
                raise SectorEvidenceError("sector_snapshot_archive_id_mismatch")
            return rebuilt

        snapshot = archived.get("snapshot")
        prices = archived.get("normalized_prices")
        if not isinstance(snapshot, dict) or not isinstance(prices, dict):
            raise SectorEvidenceError("sector_snapshot_archive_shape_invalid")
        if snapshot.get("snapshot_id") != snapshot_id:
            raise SectorEvidenceError("sector_snapshot_archive_id_mismatch")
        if normalized_input_digest(prices) != expected_digest:
            raise SectorEvidenceError("sector_snapshot_archive_input_digest_mismatch")
        if archive_schema not in {"sector-rotation-evidence-v1", "sector-rotation-evidence-v2"}:
            raise SectorEvidenceError("sector_snapshot_archive_version_unsupported")
        if archive_schema == "sector-rotation-evidence-v1":
            if snapshot.get("formula_version") == FORMULA_VERSION:
                raise SectorEvidenceError("current_snapshot_requires_rebuildable_archive")
            return None
        if archive_schema == "sector-rotation-evidence-v2":
            sessions = archived.get("expected_sessions")
            if not isinstance(sessions, list) or any(not isinstance(day, str) for day in sessions):
                raise SectorEvidenceError("sector_snapshot_archive_session_grid_invalid")
            try:
                canonical_sessions = [date.fromisoformat(day).isoformat() for day in sessions]
            except ValueError as exc:
                raise SectorEvidenceError("sector_snapshot_archive_session_grid_invalid") from exc
            if canonical_sessions != sorted(set(canonical_sessions)):
                raise SectorEvidenceError("sector_snapshot_archive_session_grid_not_canonical")
            try:
                canonical_prices = normalize_price_inputs(prices)
                rebuilt = build_snapshot(canonical_prices, expected_sessions=sessions)
            except Exception as exc:
                raise SectorEvidenceError("sector_snapshot_archive_rebuild_failed") from exc
            if rebuilt.model_dump(mode="json") != snapshot:
                raise SectorEvidenceError("sector_snapshot_archive_rebuild_mismatch")
            return SectorRotationSnapshot.model_validate(snapshot)
