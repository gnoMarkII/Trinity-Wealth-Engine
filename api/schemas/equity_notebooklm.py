"""Pydantic DTOs for Equity NotebookLM Research Export endpoints."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from application.equity.notebooklm_export_ports import EquityExportRecord


class EquityNotebookLMNotebookDTO(BaseModel):
    notebook_id: str
    title: str
    url: str
    status: str
    source_count: int = 0


class EquityNotebookLMSourceResultDTO(BaseModel):
    file_name: str
    title: str
    status: str
    source_id: Optional[str] = None
    error: Optional[str] = None


class EquityNotebookLMCoverageDTO(BaseModel):
    has_latest_analysis: bool = False
    historical_analyses_count: int = 0
    has_financials: bool = False
    has_sec_filings: bool = False
    has_insider_trades: bool = False
    earnings_calls_count: int = 0
    news_count: int = 0
    has_options_derivatives: bool = False
    has_portfolio_holding: bool = False
    user_notes_count: int = 0


class EquityNotebookLMExportRequestDTO(BaseModel):
    market: Optional[str] = "US"
    mode: Literal["all_retained"] = "all_retained"


class EquityNotebookLMExportResponseDTO(BaseModel):
    export_id: str
    ticker: str
    market: str
    job_id: Optional[str] = None
    state: str
    stage: str
    message: str


class EquityNotebookLMExportStatusDTO(BaseModel):
    export_id: str
    ticker: str
    market: str
    job_id: Optional[str] = None
    mode: str = "all_retained"
    state: str
    stage: str
    snapshot_at: str
    bundle_hash: str
    notebooks: List[EquityNotebookLMNotebookDTO] = Field(default_factory=list)
    counts: Dict[str, Any] = Field(default_factory=dict)
    source_results: List[EquityNotebookLMSourceResultDTO] = Field(default_factory=list)
    coverage: EquityNotebookLMCoverageDTO
    warnings: List[str] = Field(default_factory=list)
    error_code: Optional[str] = None
    error: Optional[str] = None
    can_retry: bool = False


def equity_export_record_to_dto(record: EquityExportRecord) -> EquityNotebookLMExportStatusDTO:
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

    source_results: List[EquityNotebookLMSourceResultDTO] = []
    for src in sources_meta:
        fname = src.get("file_name", "")
        m_src = manifest_sources.get(fname, {})
        m_status = m_src.get("status")
        if m_status == "success" and m_src.get("source_id"):
            status = "ready"
        elif m_status:
            status = m_status
        elif record.state in ("failed", "blocked"):
            status = "failed"
        else:
            status = "pending"

        source_results.append(
            EquityNotebookLMSourceResultDTO(
                file_name=fname,
                title=src.get("title", fname),
                status=status,
                source_id=m_src.get("source_id"),
                error=m_src.get("error"),
            )
        )

    notebooks: List[EquityNotebookLMNotebookDTO] = []
    if record.notebook_id and record.notebook_url:
        notebooks.append(
            EquityNotebookLMNotebookDTO(
                notebook_id=record.notebook_id,
                title=f"{record.ticker} Equity Research — {record.snapshot_at[:16]}",
                url=record.notebook_url,
                status=record.state,
                source_count=len(source_results),
            )
        )

    coverage = EquityNotebookLMCoverageDTO(
        has_latest_analysis=bool(inv.get("has_latest_analysis", True)),
        historical_analyses_count=counts.get("historical_analyses", 0),
        has_financials=bool(inv.get("has_financials", True)),
        has_sec_filings=bool(inv.get("has_sec_filings", inv.get("has_sec_data", False))),
        has_insider_trades=bool(inv.get("has_insider_trades", inv.get("has_sec_data", False))),
        earnings_calls_count=counts.get("earnings_calls", 0),
        news_count=counts.get("news_items", 0),
        has_options_derivatives=bool(inv.get("has_options_data", False)),
        has_portfolio_holding=bool(inv.get("has_portfolio_holding", False)),
        user_notes_count=counts.get("user_notes", 0),
    )

    can_retry = record.state in ("failed", "partial", "blocked")

    return EquityNotebookLMExportStatusDTO(
        export_id=record.export_id,
        ticker=record.ticker,
        market=record.market,
        job_id=record.job_id,
        mode="all_retained",
        state=record.state,
        stage=record.stage,
        snapshot_at=record.snapshot_at,
        bundle_hash=record.content_hash,
        notebooks=notebooks,
        counts=counts,
        source_results=source_results,
        coverage=coverage,
        warnings=record.warnings,
        error_code=record.error_code,
        error=record.error_message,
        can_retry=can_retry,
    )
