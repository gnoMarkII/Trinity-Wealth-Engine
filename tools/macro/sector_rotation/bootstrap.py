"""Composition root shared by API and Macro jobs."""
from functools import lru_cache

from application.macro.sector_rotation_service import SectorRotationApplicationService
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.runtime_layout import runtime_root_for
from tools.macro.adapters.sector_evidence_adapter import SectorEvidenceAdapter
from tools.macro.adapters.sector_history_adapter import SectorHistoryAdapter
from tools.macro.adapters.sector_run_binding_adapter import SectorRunBindingStore
from tools.macro.adapters.sector_snapshot_adapter import SectorSnapshotStore


@lru_cache(maxsize=1)
def get_sector_rotation_service() -> SectorRotationApplicationService:
    paths = VaultPaths()
    runtime_root = runtime_root_for(paths.root, create=True) / "sector_rotation"
    return SectorRotationApplicationService(
        history=SectorHistoryAdapter(),
        store=SectorSnapshotStore(runtime_root),
        evidence=SectorEvidenceAdapter(vault_paths=paths),
        run_bindings=SectorRunBindingStore(runtime_root),
    )
