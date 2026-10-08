"""Macro NotebookLM Bundle Builder — Assembles frozen research bundle on disk.

Hexagonal Architecture Invariant:
Implements MacroExportBundlePort. Operates as a driven adapter in tools/ compiling
a MacroCorpusSnapshot into atomic markdown files, inventory, and corpus snapshot.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from application.macro.notebooklm_export_ports import (
    MacroCorpusSnapshot,
    MacroExportBundlePort,
)
from tools.macro.notebooklm_bundle_formatter import (
    format_current_macro_report,
    format_global_and_catalog_notes,
    format_historical_reports,
    format_news_and_references,
    format_research_guide,
    format_sector_rotation,
    format_structured_appendix,
    format_thailand_market_observables,
    format_us_market_observables,
)


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class MacroExportBundleBuilder(MacroExportBundlePort):
    """Builds and seals a deterministic frozen export bundle on disk."""

    def __init__(self, export_root: Optional[Path] = None) -> None:
        configured = export_root or os.getenv("MACRO_NOTEBOOKLM_EXPORT_DIR") or "data/notebooklm_macro_exports"
        self._export_root = Path(configured).resolve()

    @property
    def export_root(self) -> Path:
        return self._export_root

    def build_bundle(
        self, export_id: str, snapshot: MacroCorpusSnapshot
    ) -> Tuple[Path, str, Dict[str, Any]]:
        bundle_dir = self._export_root / export_id
        sources_dir = bundle_dir / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate text for each part
        parts: List[Tuple[str, str, str]] = [
            ("00-research-guide.md", "Research Guide & Inventory", format_research_guide(snapshot)),
            ("01-current-macro-report.md", "Current Macro Strategy Report", format_current_macro_report(snapshot)),
            ("reports-history-001.md", "Historical Macro Reports", format_historical_reports(snapshot)),
            ("observables-us-001.md", "US Macro Telemetry & Yields", format_us_market_observables(snapshot)),
            ("observables-th-001.md", "Thailand Macro & Official Hard Data", format_thailand_market_observables(snapshot)),
            ("observables-global-001.md", "Global Rates & Regional Notes", format_global_and_catalog_notes(snapshot)),
            ("sector-rotation-001.md", "Sector Rotation Analytics", format_sector_rotation(snapshot)),
            ("news-and-references-001.md", "Macro News Funnel & Evidence", format_news_and_references(snapshot)),
            ("structured-appendix-001.md", "Structured Mathematical Appendix", format_structured_appendix(snapshot)),
        ]

        source_inventory: List[Dict[str, Any]] = []

        # 2. Write markdown files and compute cumulative rendered files hash
        files_hasher = hashlib.sha256()
        for filename, title, content in parts:
            file_path = sources_dir / filename
            content_bytes = content.encode("utf-8")
            file_path.write_bytes(content_bytes)

            file_hash = _compute_sha256(content_bytes)
            files_hasher.update(filename.encode("utf-8"))
            files_hasher.update(file_hash.encode("utf-8"))

            source_inventory.append({
                "file_name": filename,
                "relative_path": f"sources/{filename}",
                "title": title,
                "size_bytes": len(content_bytes),
                "sha256": file_hash,
            })

        bundle_files_hash = files_hasher.hexdigest()

        # C13: Semantic data hash independent of fleeting snapshot timestamp
        semantic_payload = {
            "strategy_report_id": snapshot.strategy_report_id,
            "latest_report": snapshot.latest_report,
            "historical_reports": snapshot.historical_reports,
            "indicator_series": snapshot.indicator_series,
            "market_observables": snapshot.market_observables,
            "thailand_hard_data": snapshot.thailand_hard_data,
            "sector_rotation": snapshot.sector_rotation,
            "news_funnel": snapshot.news_funnel,
        }
        content_hash = hashlib.sha256(
            json.dumps(semantic_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        # 3. Write corpus.json
        corpus_data = {
            "snapshot_at": snapshot.snapshot_at,
            "content_hash": content_hash,
            "strategy_report_id": snapshot.strategy_report_id,
            "latest_report": snapshot.latest_report,
            "historical_reports": snapshot.historical_reports,
            "catalog_notes": snapshot.catalog_notes,
            "indicator_series": snapshot.indicator_series,
            "market_observables": snapshot.market_observables,
            "thailand_hard_data": snapshot.thailand_hard_data,
            "sector_rotation": snapshot.sector_rotation,
            "news_funnel": snapshot.news_funnel,
            "metadata": snapshot.metadata,
        }
        corpus_path = bundle_dir / "corpus.json"
        corpus_path.write_text(json.dumps(corpus_data, indent=2, default=str), encoding="utf-8")

        # 4. Write inventory.json
        inventory: Dict[str, Any] = {
            "export_id": export_id,
            "snapshot_at": snapshot.snapshot_at,
            "content_hash": content_hash,
            "bundle_files_hash": bundle_files_hash,
            "total_sources": len(source_inventory),
            "sources": source_inventory,
            "counts": snapshot.metadata.get("counts", {}),
            "warnings": snapshot.metadata.get("warnings", []),
        }
        inventory_path = bundle_dir / "inventory.json"
        inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

        return bundle_dir, content_hash, inventory
