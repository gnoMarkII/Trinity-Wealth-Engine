"""Vault-backed Macro strategy and indicator adapters."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tools.macro.dashboard import load_indicator_series


def _vault_path() -> Path:
    configured = os.getenv("OBSIDIAN_VAULT_PATH")
    if configured:
        return Path(configured).resolve()
    from tools.archivist import core as archivist_core

    return Path(archivist_core.VAULT_PATH).resolve()


class StrategyVaultAdapter:
    def __init__(self, vault_path: Path | None = None) -> None:
        self._vault_path = vault_path

    @property
    def vault_path(self) -> Path:
        return self._vault_path.resolve() if self._vault_path else _vault_path()

    def latest(self) -> dict[str, Any]:
        candidates: list[Path] = []
        v2_dir = self.vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Strategies"
        if v2_dir.exists():
            candidates.extend(p for p in v2_dir.rglob("Macro_Strategy_Direction_*.json") if "Revisions" not in p.parts)
        v1_dir = self.vault_path / "30_Knowledge_Base" / "Strategies"
        if v1_dir.exists():
            candidates.extend(p for p in v1_dir.glob("Macro_Strategy_Direction_*.json") if "Revisions" not in p.parts)
        if not candidates:
            raise FileNotFoundError("No Macro Strategy sidecar is available")
        candidates.sort(key=lambda p: p.name)
        return json.loads(candidates[-1].read_text(encoding="utf-8"))


class IndicatorSeriesAdapter:
    def __init__(self, strategy_adapter: StrategyVaultAdapter) -> None:
        self._strategy_adapter = strategy_adapter

    def load(self, series_key: str, range_name: str) -> list[dict[str, Any]]:
        return load_indicator_series(self._strategy_adapter.vault_path, series_key, range_name)
