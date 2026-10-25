"""Concrete Adapters for Portfolio Advisory Driven Ports.

Architecture Layer: tools/portfolio/adapters/ (Infrastructure / Driven Adapters)
Implements:
- AdvisoryVaultPort (Atomic persistence to Obsidian Vault)
- PortfolioStateReaderPort (Structured portfolio state)
- MacroContextReaderPort (StrategyVaultAdapter & Sector Rotation)
- EquitySidecarReaderPort (Fast O(1) sidecar scan)
- AdvisorySynthesizerPort (LLM Portfolio Synthesizer invocation)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from core.llm_factory import get_chat_model
from core.portfolio_advisory.models import PortfolioAdvisoryQuantReport
from schemas.portfolio_advisory_schemas import (
    AdvisorySidecarRecord,
    PortfolioAdvisoryNarrativeDTO,
)
from tools._atomic_io import _atomic_write_to

log = logging.getLogger(__name__)


class AdvisoryVaultAdapter:
    """Persists and retrieves Portfolio Advisory sidecars from Obsidian Vault."""

    def __init__(self, vault_path: Optional[Path] = None) -> None:
        self.vault_path = vault_path or Path(os.getenv("OBSIDIAN_VAULT_PATH", "memories"))
        self.advisory_dir = self.vault_path / "30_Knowledge_Base" / "Portfolio" / "Advisory"

    def save_advisory_sidecar(self, record: AdvisorySidecarRecord, portfolio_id: str = "default") -> None:
        self.advisory_dir.mkdir(parents=True, exist_ok=True)
        content_json = record.model_dump_json(indent=2)
        suffix = "" if portfolio_id == "default" else f"_{portfolio_id}"

        # 1. Write latest.json
        latest_path = self.advisory_dir / f"Portfolio_Advisory{suffix}_latest.json"
        _atomic_write_to(latest_path, content_json)

        # 2. Write dated revision
        date_str = record.evaluated_at[:10].replace("-", "")
        dated_path = self.advisory_dir / f"Portfolio_Advisory{suffix}_{date_str}.json"
        try:
            _atomic_write_to(dated_path, content_json)
        except Exception as e:
            log.warning("Could not write dated advisory sidecar: %s", e)

    def load_latest_advisory_sidecar(self, portfolio_id: str = "default") -> Optional[AdvisorySidecarRecord]:
        suffix = "" if portfolio_id == "default" else f"_{portfolio_id}"
        latest_path = self.advisory_dir / f"Portfolio_Advisory{suffix}_latest.json"
        if not latest_path.exists():
            return None
        try:
            data = json.loads(latest_path.read_text(encoding="utf-8"))
            return AdvisorySidecarRecord.model_validate(data)
        except Exception as e:
            log.warning("Failed to load latest advisory sidecar: %s", e)
            return None


class PortfolioStateAdapter:
    """Reads structured portfolio state and valuations."""

    def get_current_portfolio_state(self, refresh_prices: bool = False, portfolio_id: str = "default") -> Any:
        from tools.portfolio.core import get_structured_portfolio_state
        return get_structured_portfolio_state(refresh_prices=refresh_prices, portfolio_id=portfolio_id)


class MacroContextAdapter:
    """Reads Macro Strategy Direction and Sector Rotation Snapshots (fail-open)."""

    def __init__(self, vault_path: Optional[Path] = None) -> None:
        self.vault_path = vault_path or Path(os.getenv("OBSIDIAN_VAULT_PATH", "memories"))

    def get_latest_macro_strategy(self) -> Optional[Any]:
        try:
            from tools.macro.adapters.strategy_vault_adapter import StrategyVaultAdapter
            adapter = StrategyVaultAdapter(self.vault_path)
            return adapter.latest()
        except Exception as e:
            log.debug("Macro strategy not available in vault: %s", e)
            return None

    def get_latest_sector_rotation(self) -> Optional[Any]:
        try:
            from tools.macro.sector_rotation.bootstrap import get_sector_rotation_service
            svc = get_sector_rotation_service()
            res = svc.latest(timeframe="weekly")
            if res and res.get("snapshot"):
                return res["snapshot"]
        except Exception as e:
            log.debug("Sector rotation service unavailable: %s", e)

        # Fallback to direct file read if exists
        sr_dir = self.vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Sector_Rotation"
        if sr_dir.exists():
            latest_f = sr_dir / "latest.json"
            if latest_f.exists():
                try:
                    return json.loads(latest_f.read_text(encoding="utf-8"))
                except Exception:
                    pass
        return None


class EquitySidecarAdapter:
    """Scans and loads active MicroQuantOutput sidecars from the Obsidian Vault."""

    def __init__(self, vault_path: Optional[Path] = None) -> None:
        self.vault_path = vault_path or Path(os.getenv("OBSIDIAN_VAULT_PATH", "memories"))

    def get_all_equity_sidecars(self) -> Dict[str, Any]:
        sidecars: dict[str, Any] = {}
        # 1. Primary: Use EquityVaultQueryAdapter to discover latest dated sidecars (.system/sidecars & 30_Knowledge_Base/Stocks)
        try:
            from tools.market.adapters.equity_vault_query_adapter import EquityVaultQueryAdapter
            query_adapter = EquityVaultQueryAdapter(vault_path=self.vault_path)
            for item in query_adapter.list_latest():
                ticker = str(item.get("ticker", "")).strip().upper()
                market = str(item.get("market", "")).strip().upper()
                if ticker:
                    sidecars[ticker] = item
                    clean_t = ticker.replace(".BK", "")
                    sidecars[clean_t] = item
                    if (market == "TH" or ticker.endswith(".BK")) and not ticker.endswith(".BK"):
                        sidecars[f"{ticker}.BK"] = item
        except Exception as e:
            log.debug("EquityVaultQueryAdapter scan fallback: %s", e)

        # 2. Fallback / supplement: Direct scan of 30_Knowledge_Base/Stocks/*/latest.json
        stocks_dir = self.vault_path / "30_Knowledge_Base" / "Stocks"
        if not stocks_dir.exists():
            return sidecars

        for ticker_dir in stocks_dir.iterdir():
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name.upper()
            if ticker in sidecars:
                continue
            latest_f = ticker_dir / "Analysis" / f"{ticker} Equity Analysis latest.json"
            if not latest_f.exists():
                latest_f = ticker_dir / f"{ticker} Equity Analysis latest.json"
            if not latest_f.exists():
                latest_f = ticker_dir / "latest.json"

            if latest_f.exists():
                try:
                    data = json.loads(latest_f.read_text(encoding="utf-8"))
                    market = str(data.get("market", "")).strip().upper()
                    sidecars[ticker] = data
                    if (market == "TH" or ticker.endswith(".BK")) and not ticker.endswith(".BK"):
                        sidecars[f"{ticker}.BK"] = data
                except Exception:
                    continue
        return sidecars

    def get_watchlist_sidecars(self) -> Dict[str, Any]:
        # Currently reuses all known sidecars
        return self.get_all_equity_sidecars()


class AdvisorySynthesizerAdapter:
    """Delegates playbook synthesis to the portfolio_synthesizer agent."""

    def synthesize_playbook(
        self,
        payload_json: str,
        quant_report: PortfolioAdvisoryQuantReport,
    ) -> PortfolioAdvisoryNarrativeDTO:
        from agents.portfolio_synthesizer import invoke_portfolio_synthesizer

        model = get_chat_model("PORTFOLIO_SYNTHESIZER_MODEL", default="gemini-3.1-flash-lite-preview")
        return invoke_portfolio_synthesizer(model, payload_json, quant_report)
