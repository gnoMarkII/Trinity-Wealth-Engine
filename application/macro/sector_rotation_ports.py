"""Ports for sector-history ingestion, durable cache and canonical evidence."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Optional, Protocol

from schemas.sector_rotation_schemas import SectorRotationSnapshot


@dataclass(frozen=True)
class SectorHistoryBatch:
    prices: dict[str, dict[str, float]]
    reasons: dict[str, str]
    cutoff: date
    expected_sessions: tuple[str, ...]


class SectorHistoryPort(Protocol):
    def fetch(self) -> SectorHistoryBatch: ...


class SectorSnapshotStorePort(Protocol):
    @property
    def refresh_lock_path(self): ...
    def read_state(self) -> dict[str, Any]: ...
    def update_state(self, **changes: Any) -> dict[str, Any]: ...
    def save(self, snapshot: SectorRotationSnapshot, evidence_ref: dict[str, Any]) -> None: ...
    def load(self, snapshot_id: str) -> Optional[tuple[SectorRotationSnapshot, dict[str, Any]]]: ...
    def latest(self) -> Optional[tuple[SectorRotationSnapshot, dict[str, Any]]]: ...


class SectorEvidencePort(Protocol):
    def publish(
        self, snapshot: SectorRotationSnapshot, normalized_prices: Mapping[str, Mapping[str, float]],
        *, expected_sessions: Optional[tuple[str, ...] | list[str]] = None,
    ) -> dict[str, Any]: ...
    def load(self, snapshot_id: str) -> Optional[tuple[SectorRotationSnapshot, dict[str, dict[str, float]]]]: ...


class SectorRunBindingPort(Protocol):
    def lock_path(self, run_id: str): ...
    def load(self, run_id: str) -> Optional[dict[str, Any]]: ...
    def create(self, run_id: str, record: dict[str, Any]) -> None: ...
    def update_publication(self, run_id: str, status: str, receipt: Optional[dict[str, Any]] = None) -> dict[str, Any]: ...
