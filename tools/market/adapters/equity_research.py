"""Concrete outbound adapters for Equity application services."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yfinance as yf
from pydantic import ValidationError

from schemas.micro_quant_schemas import MicroQuantOutput
from tools.archivist.core import VAULT_PATH
from tools.market.asset_resolver import resolve_asset
from tools.market.calendar import get_asset_calendar
from tools.market.earnings import fetch_earnings_dates


class MarketAssetResolverAdapter:
    def resolve(self, ticker: str) -> Any:
        return resolve_asset(ticker)


class YFinanceAnalystProviderAdapter:
    """Wrap yfinance/calendar/earnings behind one analyst provider port."""

    def __init__(self, *, ticker_factory=None, calendar_fetcher=None, earnings_fetcher=None) -> None:
        self._ticker_factory = ticker_factory or yf.Ticker
        self._calendar_fetcher = calendar_fetcher or get_asset_calendar
        self._earnings_fetcher = earnings_fetcher or fetch_earnings_dates

    def price_targets(self, provider_symbol: str) -> Dict[str, Any]:
        ticker = self._ticker_factory(provider_symbol)
        targets = ticker.get_analyst_price_targets() or {}
        info = ticker.info or {}
        return {
            **targets,
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
            "targetMeanPrice": targets.get("targetMeanPrice") or info.get("targetMeanPrice"),
            "targetHighPrice": targets.get("targetHighPrice") or info.get("targetHighPrice"),
            "targetLowPrice": targets.get("targetLowPrice") or info.get("targetLowPrice"),
        }

    def calendar(self, provider_symbol: str) -> Dict[str, Any]:
        result = self._calendar_fetcher(provider_symbol)
        return result if isinstance(result, dict) else {}

    def earnings_history(self, provider_symbol: str, exchange_tz: str) -> Any:
        return self._earnings_fetcher(provider_symbol, exchange_tz)


class EquitySidecarValuationAdapter:
    """Read the latest validated equity-analysis sidecar without HTTP concerns."""

    def __init__(self, vault_path: Optional[Path] = None) -> None:
        self._vault_path = vault_path or VAULT_PATH

    def latest(self, ticker: str) -> Any:
        sys_pattern = f".system/sidecars/{ticker}/* Equity Analysis *.json"
        v2_pattern = f"30_Knowledge_Base/Stocks/{ticker}/Analysis/* Equity Analysis *.json"
        v1_pattern = f"30_Knowledge_Base/Stocks/{ticker}/{ticker} Equity Analysis *.json"
        all_files = list(self._vault_path.glob(sys_pattern)) + list(self._vault_path.glob(v2_pattern)) + list(self._vault_path.glob(v1_pattern))
        non_latest = [f for f in all_files if not f.name.endswith("latest.json")]
        candidate_files = non_latest if non_latest else all_files
        files = sorted(candidate_files, key=self._date_key, reverse=True)
        if not files:
            return None
        try:
            data = json.loads(files[0].read_text(encoding="utf-8"))
            model = MicroQuantOutput.model_validate(data)
            if model.ticker.upper() != ticker.upper() or model.quant_signals.ticker.upper() != ticker.upper():
                return None
            datetime.strptime(model.analysis_date, "%Y-%m-%d")
            return model
        except (OSError, ValueError, TypeError, ValidationError, KeyError):
            return None

    @staticmethod
    def _date_key(path: Path) -> tuple[str, str]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return (str(data.get("quant_signals", {}).get("evaluated_at", "")), path.name)
        except (OSError, ValueError, TypeError):
            return ("", path.name)
