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
        manifest_candidates = [
            self.vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Strategies" / "Macro_Strategy_Latest.json",
            self.vault_path / "30_Knowledge_Base" / "Strategies" / "Macro_Strategy_Latest.json",
        ]
        for manifest_path in manifest_candidates:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                report_id = str(manifest.get("strategy_report_id") or "")
                if report_id:
                    return self.report_by_id(report_id)
            except FileNotFoundError:
                continue
            except (OSError, json.JSONDecodeError, AttributeError):
                continue

        candidates: list[Path] = []
        v2_dir = self.vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Strategies"
        if v2_dir.exists():
            candidates.extend(p for p in v2_dir.rglob("Macro_Strategy_Direction_*.json") if "Revisions" not in p.parts)
        v1_dir = self.vault_path / "30_Knowledge_Base" / "Strategies"
        if v1_dir.exists():
            candidates.extend(p for p in v1_dir.glob("Macro_Strategy_Direction_*.json") if "Revisions" not in p.parts)
        if not candidates:
            raise FileNotFoundError("No Macro Strategy sidecar is available")
        def _sort_key(p: Path) -> tuple[str, str, str]:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                ev_at = str(data.get("evaluated_at", ""))
                run_started = str(data.get("run_started_at", ""))
                return (ev_at[:10], run_started, p.name)
            except Exception:
                return ("", "", p.name)

        candidates.sort(key=_sort_key)
        legacy_projection = None
        for path in reversed(candidates):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            report_id = str(data.get("strategy_report_id") or "")
            if report_id:
                try:
                    return self.report_by_id(report_id)
                except FileNotFoundError:
                    continue
            legacy_projection = legacy_projection or data
        if legacy_projection is not None:
            return legacy_projection
        raise FileNotFoundError("No committed Macro Strategy report is available")

    def report_by_id(self, strategy_report_id: str) -> dict[str, Any]:
        """Read one immutable broker-committed report by its stable report ID."""
        import re
        from tools.archivist.artifact_store import ArtifactError, DurableArtifactStore
        from tools.archivist.composition import build_knowledge_write_port
        from tools.archivist.vault_paths import VaultPaths

        report_id = str(strategy_report_id).strip()
        if not re.fullmatch(r"macro_report_[A-Za-z0-9_-]{1,96}", report_id):
            raise FileNotFoundError("Macro report not found")
        paths = VaultPaths(self.vault_path)
        port = build_knowledge_write_port(vault_paths=paths)
        receipt = port.get_receipt(idempotency_key=f"macro-strategy-report:{report_id}")
        if receipt is None or not receipt.is_success or not receipt.note_id or not receipt.revision_id:
            raise FileNotFoundError("Macro report not found")
        try:
            artifact = DurableArtifactStore(paths).get_revision_artifact(receipt.note_id, receipt.revision_id)
        except ArtifactError as exc:
            raise RuntimeError("Macro report evidence failed integrity verification") from exc
        data = json.loads(artifact.body)
        if data.get("strategy_report_id") != report_id:
            raise FileNotFoundError("Macro report not found")
        data["strategy_report_evidence"] = {
            "note_id": receipt.note_id,
            "revision_id": receipt.revision_id,
            "relative_path": receipt.relative_path,
            "content_hash": receipt.content_hash,
            "artifact_set_hash": receipt.artifact_set_hash,
        }
        return data


class IndicatorSeriesAdapter:
    def __init__(self, strategy_adapter: StrategyVaultAdapter) -> None:
        self._strategy_adapter = strategy_adapter

    def load(self, series_key: str, range_name: str) -> list[dict[str, Any]]:
        return load_indicator_series(self._strategy_adapter.vault_path, series_key, range_name)
