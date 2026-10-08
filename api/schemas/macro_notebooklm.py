"""Pydantic DTOs for Macro NotebookLM Research Export endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from application.macro.notebooklm_export_ports import MacroExportRecord


class MacroNotebookLMNotebookDTO(BaseModel):
    notebook_id: str
    title: str
    url: str
    status: str
    source_count: int = 0


class MacroNotebookLMSourceResultDTO(BaseModel):
    file_name: str
    title: str
    status: str
    source_id: Optional[str] = None
    error: Optional[str] = None


class MacroNotebookLMCoverageDTO(BaseModel):
    strategy_report_present: bool = False
    historical_reports_count: int = 0
    catalog_notes_count: int = 0
    indicator_series_count: int = 0
    market_observables_cached: int = 0
    market_observables_total: int = 13
    thailand_hard_data_present: bool = False
    sector_rotation_present: bool = False
    news_events_count: int = 0


class MacroNotebookLMExportRequestDTO(BaseModel):
    mode: Literal["all_retained"] = "all_retained"


class MacroNotebookLMExportResponseDTO(BaseModel):
    export_id: str
    job_id: Optional[str] = None
    state: str
    stage: str
    message: str


class MacroNotebookLMExportStatusDTO(BaseModel):
    export_id: str
    job_id: Optional[str] = None
    mode: str = "all_retained"
    state: str
    stage: str
    snapshot_at: str
    bundle_hash: str
    strategy_report_id: Optional[str] = None
    notebooks: List[MacroNotebookLMNotebookDTO] = Field(default_factory=list)
    counts: Dict[str, Any] = Field(default_factory=dict)
    source_results: List[MacroNotebookLMSourceResultDTO] = Field(default_factory=list)
    coverage: MacroNotebookLMCoverageDTO
    warnings: List[str] = Field(default_factory=list)
    error_code: Optional[str] = None
    error: Optional[str] = None
    can_retry: bool = False


def macro_export_record_to_dto(record: MacroExportRecord) -> MacroNotebookLMExportStatusDTO:
    inv = record.inventory or {}
    counts = inv.get("counts", {})
    sources_meta = inv.get("sources", [])

    # Load on-disk manifest if available to get per-source upload results
    manifest_sources = {}
    if record.manifest_path and Path(record.manifest_path).is_file():
        try:
            m_data = json.loads(Path(record.manifest_path).read_text(encoding="utf-8"))
            manifest_sources = m_data.get("sources", {})
        except Exception:
            manifest_sources = {}

    source_results: List[MacroNotebookLMSourceResultDTO] = []
    for src in sources_meta:
        fname = src.get("file_name", "")
        m_src = manifest_sources.get(fname, {})
        m_status = m_src.get("status")
        # C14: Source status strictly reflects verified manifest state and source_id
        if m_status == "success" and m_src.get("source_id"):
            status = "ready"
        elif m_status:
            status = m_status
        elif record.state in ("failed", "blocked"):
            status = "failed"
        else:
            status = "pending"

        source_results.append(
            MacroNotebookLMSourceResultDTO(
                file_name=fname,
                title=src.get("title", fname),
                status=status,
                source_id=m_src.get("source_id"),
                error=m_src.get("error"),
            )
        )

    notebooks: List[MacroNotebookLMNotebookDTO] = []
    if record.notebook_id and record.notebook_url:
        notebooks.append(
            MacroNotebookLMNotebookDTO(
                notebook_id=record.notebook_id,
                title=f"Macro Research — {record.snapshot_at[:16]}",
                url=record.notebook_url,
                status=record.state,
                source_count=len(source_results),
            )
        )

    coverage = MacroNotebookLMCoverageDTO(
        strategy_report_present=bool(counts.get("latest_report_present")),
        historical_reports_count=counts.get("historical_reports", 0),
        catalog_notes_count=counts.get("catalog_notes", 0),
        indicator_series_count=counts.get("indicator_series", 0),
        market_observables_cached=counts.get("market_observables_cached", 0),
        market_observables_total=counts.get("market_observables_total", 13),
        thailand_hard_data_present=bool(counts.get("thailand_hard_data_present")),
        sector_rotation_present=bool(counts.get("sector_rotation_present")),
        news_events_count=counts.get("news_pending", 0) + counts.get("news_filtered", 0),
    )

    can_retry = record.state in ("failed", "partial", "blocked")

    return MacroNotebookLMExportStatusDTO(
        export_id=record.export_id,
        job_id=record.job_id,
        mode="all_retained",
        state=record.state,
        stage=record.stage,
        snapshot_at=record.snapshot_at,
        bundle_hash=record.content_hash,
        strategy_report_id=record.strategy_report_id,
        notebooks=notebooks,
        counts=counts,
        source_results=source_results,
        coverage=coverage,
        warnings=record.warnings,
        error_code=record.error_code,
        error=record.error_message,
        can_retry=can_retry,
    )
