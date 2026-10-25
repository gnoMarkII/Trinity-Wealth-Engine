"""Protocol Ports for Portfolio Advisory Application Boundary.

Architecture Layer: application/portfolio/ (Ports)
Decouples application services from concrete database adapters, vault storage,
and LLM API client implementations.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from core.portfolio_advisory.models import PortfolioAdvisoryQuantReport
from schemas.portfolio_advisory_schemas import (
    AdvisorySidecarRecord,
    PortfolioAdvisoryNarrativeDTO,
)


class PortfolioStateReaderPort(Protocol):
    """Port for retrieving current portfolio composition and asset valuations."""
    def get_current_portfolio_state(self, refresh_prices: bool = False, portfolio_id: str = "default") -> Any:
        ...


class MacroContextReaderPort(Protocol):
    """Port for reading latest Macro Strategy and Sector Rotation sidecars."""
    def get_latest_macro_strategy(self) -> Optional[Any]:
        ...

    def get_latest_sector_rotation(self) -> Optional[Any]:
        ...


class EquitySidecarReaderPort(Protocol):
    """Port for scanning active MicroQuantOutput equity analysis files from Obsidian Vault."""
    def get_all_equity_sidecars(self) -> Dict[str, Any]:
        ...

    def get_watchlist_sidecars(self) -> Dict[str, Any]:
        ...


class AdvisoryVaultPort(Protocol):
    """Port for persisting and retrieving Portfolio Advisory Sidecars."""
    def save_advisory_sidecar(self, record: AdvisorySidecarRecord, portfolio_id: str = "default") -> None:
        ...

    def load_latest_advisory_sidecar(self, portfolio_id: str = "default") -> Optional[AdvisorySidecarRecord]:
        ...


class AdvisorySynthesizerPort(Protocol):
    """Port for invoking the Tier 2 LLM Portfolio Synthesizer."""
    def synthesize_playbook(
        self,
        payload_json: str,
        quant_report: PortfolioAdvisoryQuantReport,
    ) -> PortfolioAdvisoryNarrativeDTO:
        ...
