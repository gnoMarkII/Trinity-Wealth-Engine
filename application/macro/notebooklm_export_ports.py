"""Outbound Ports and Data Models for Macro NotebookLM Research Export.

Hexagonal Architecture Invariant:
This module defines application-level abstractions and contracts. It MUST NOT
import concrete adapters, database drivers, web frameworks, or CLI tools directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@dataclass
class MacroCorpusSnapshot:
    """Immutable in-memory snapshot of all retained Macro knowledge and market observables."""

    snapshot_at: str
    strategy_report_id: Optional[str] = None
    latest_report: Optional[Dict[str, Any]] = None
    historical_reports: List[Dict[str, Any]] = field(default_factory=list)
    catalog_notes: List[Dict[str, Any]] = field(default_factory=list)
    indicator_series: List[Dict[str, Any]] = field(default_factory=list)
    market_observables: Dict[str, Any] = field(default_factory=dict)
    thailand_hard_data: Optional[Dict[str, Any]] = None
    sector_rotation: Optional[Dict[str, Any]] = None
    news_funnel: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroExportRecord:
    """Durable record of a Macro NotebookLM export job."""

    export_id: str
    request_key: str
    content_hash: str
    job_id: Optional[str] = None
    state: str = "queued"
    stage: str = "initialized"
    snapshot_at: str = ""
    strategy_report_id: Optional[str] = None
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
class MacroCorpusReaderPort(Protocol):
    """Port for capturing the full retained Macro corpus."""

    def capture_snapshot(self) -> MacroCorpusSnapshot:
        """Capture all retained macro reports, regional snapshots, indicators, and market observables."""
        ...


@runtime_checkable
class MacroExportBundlePort(Protocol):
    """Port for compiling a MacroCorpusSnapshot into a frozen markdown bundle with manifest."""

    def build_bundle(
        self, export_id: str, snapshot: MacroCorpusSnapshot
    ) -> Tuple[Path, str, Dict[str, Any]]:
        """Construct deterministic markdown source parts and return (bundle_path, content_hash, inventory)."""
        ...


@runtime_checkable
class MacroExportRepositoryPort(Protocol):
    """Port for persistent storage of Macro export metadata."""

    def create(self, record: MacroExportRecord) -> MacroExportRecord:
        ...

    def get_by_id(self, export_id: str) -> Optional[MacroExportRecord]:
        ...

    def get_by_request_key(self, request_key: str) -> Optional[MacroExportRecord]:
        ...

    def get_by_content_hash(self, content_hash: str) -> Optional[MacroExportRecord]:
        ...

    def get_latest(self) -> Optional[MacroExportRecord]:
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
    ) -> Optional[MacroExportRecord]:
        ...


@runtime_checkable
class MacroExportDispatchPort(Protocol):
    """Port for queueing macro export tasks into the background job worker."""

    def dispatch(
        self,
        instruction: str,
        card_id: Optional[str] = None,
        flow: str = "macro_notebooklm",
        scope: str = "both",
    ) -> str:
        ...


@runtime_checkable
class MacroExportPipelinePort(Protocol):
    """Port for uploading markdown source parts to NotebookLM without Audio or Discord mutations."""

    async def execute_export(
        self,
        export_id: str,
        bundle_dir: Path,
        title: Optional[str] = None,
        on_step: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        ...
