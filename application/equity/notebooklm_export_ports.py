"""Outbound Ports and Data Models for Equity NotebookLM Research Export.

Hexagonal Architecture Invariant:
This module defines application-level abstractions and contracts for exporting
equity research to Google NotebookLM. It MUST NOT import concrete adapters,
database drivers, web frameworks, or CLI tools directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable


class EquityExportBlockedError(Exception):
    """Raised when an equity export is attempted on a stock without baseline analysis."""


@dataclass
class EquityCorpusSnapshot:
    """Immutable in-memory snapshot of all retained Equity knowledge and market data."""

    ticker: str
    snapshot_at: str
    market: str = "US"
    latest_analysis: Optional[Dict[str, Any]] = None
    historical_analyses: List[Dict[str, Any]] = field(default_factory=list)
    financials: Optional[Dict[str, Any]] = None
    sec_financials: Optional[Dict[str, Any]] = None
    sec_insider_trades: Optional[Dict[str, Any]] = None
    insider_filings: Optional[Dict[str, Any]] = None
    earnings_calls: List[Dict[str, Any]] = field(default_factory=list)
    news: List[Dict[str, Any]] = field(default_factory=list)
    options_chain: Optional[Dict[str, Any]] = None
    max_pain: Optional[Dict[str, Any]] = None
    consensus: Optional[Dict[str, Any]] = None
    analyst_context: Optional[Dict[str, Any]] = None
    valuation_targets: Optional[Dict[str, Any]] = None
    ohlcv_summary: Optional[Dict[str, Any]] = None
    notes: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_holding: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EquityExportRecord:
    """Durable record of an Equity NotebookLM export job."""

    export_id: str
    request_key: str
    ticker: str
    market: str
    content_hash: str
    job_id: Optional[str] = None
    state: str = "queued"
    stage: str = "initialized"
    snapshot_at: str = ""
    notebook_id: Optional[str] = None
    notebook_url: Optional[str] = None
    manifest_path: Optional[str] = None
    inventory: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0


@runtime_checkable
class EquityCorpusReaderPort(Protocol):
    """Port for capturing the full retained Equity corpus for a ticker."""

    def capture_snapshot(self, ticker: str, market: str = "US") -> EquityCorpusSnapshot:
        """Capture retained equity analysis, financials, SEC, insider, news, derivatives, and notes."""
        ...


@runtime_checkable
class EquityExportBundlePort(Protocol):
    """Port for compiling an EquityCorpusSnapshot into a frozen markdown bundle with manifest."""

    def build_bundle(
        self, export_id: str, snapshot: EquityCorpusSnapshot
    ) -> Tuple[Path, str, Dict[str, Any]]:
        """Construct deterministic markdown source parts and return (bundle_path, content_hash, inventory)."""
        ...


@runtime_checkable
class EquityExportRepositoryPort(Protocol):
    """Port for persistent storage of Equity export metadata."""

    def create(self, record: EquityExportRecord) -> EquityExportRecord:
        ...

    def get_by_id(self, export_id: str) -> Optional[EquityExportRecord]:
        ...

    def get_by_request_key(self, request_key: str) -> Optional[EquityExportRecord]:
        ...

    def get_by_content_hash(
        self, ticker: str, market: str, content_hash: str
    ) -> Optional[EquityExportRecord]:
        ...

    def get_latest(self, ticker: str, market: str = "US") -> Optional[EquityExportRecord]:
        ...

    def update_state(
        self,
        export_id: str,
        state: str,
        stage: str,
        *,
        job_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
        notebook_url: Optional[str] = None,
        manifest_path: Optional[str] = None,
        inventory: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[EquityExportRecord]:
        ...


@runtime_checkable
class EquityExportDispatchPort(Protocol):
    """Port for queueing equity export tasks into the background job worker."""

    def dispatch(
        self,
        instruction: str,
        card_id: Optional[str] = None,
        flow: str = "equity_notebooklm",
        scope: str = "both",
    ) -> str:
        ...


@runtime_checkable
class EquityExportPipelinePort(Protocol):
    """Port for uploading markdown source parts to NotebookLM without Audio or Discord mutations."""

    async def execute_export(
        self,
        export_id: str,
        bundle_dir: Path,
        title: Optional[str] = None,
        on_step: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        ...
