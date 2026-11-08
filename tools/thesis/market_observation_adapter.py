"""Adapter implementing MarketObservationPort by querying Vault sidecars and market data."""
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

from application.thesis.ports import MarketObservationPort
from tools.portfolio.adapters.advisory_vault_adapter import EquitySidecarAdapter

log = logging.getLogger(__name__)


class VaultMarketObservationAdapter(MarketObservationPort):
    @staticmethod
    def _first_not_none(*values: Any) -> Any:
        """Return the first present value; zero is a valid financial observation."""
        return next((value for value in values if value is not None), None)

    """Adapts Obsidian Vault equity analysis sidecars into the MarketObservationPort."""

    def __init__(self, sidecar_adapter: Optional[EquitySidecarAdapter] = None) -> None:
        self._sidecar_adapter = sidecar_adapter or EquitySidecarAdapter()

    def get_financial_metric(self, ticker: str, metric_name: str) -> Optional[Decimal]:
        clean = ticker.strip().upper()
        try:
            sidecars = self._sidecar_adapter.get_all_equity_sidecars()
        except Exception as e:
            log.warning("Could not read equity sidecars for %s: %s", clean, e)
            return None

        item = (
            sidecars.get(clean)
            or sidecars.get(clean.replace(".BK", ""))
            or sidecars.get(f"{clean}.BK")
        )
        if not item or not isinstance(item, dict):
            return None

        raw_quant = item.get("quant_signals")
        if raw_quant is None:
            raw_quant = item.get("quantitative_signals")  # legacy sidecar alias
        if hasattr(raw_quant, "model_dump"):
            quant = raw_quant.model_dump()
        elif isinstance(raw_quant, dict):
            quant = raw_quant
        else:
            quant = {}

        val = None
        if metric_name in ("revenue_growth_yoy", "revenue_growth"):
            val = self._first_not_none(
                quant.get("revenue_growth_yoy_pct"), quant.get("revenue_growth_yoy"),
                quant.get("revenue_growth"), item.get("revenue_growth_yoy_pct"),
                item.get("revenue_growth_yoy"),
            )
        elif metric_name in ("operating_margin", "operating_margin_pct"):
            val = self._first_not_none(quant.get("operating_margin_pct"),
                                       quant.get("operating_margin"), item.get("operating_margin_pct"))
        elif metric_name in ("fcf_margin", "fcf_margin_pct"):
            val = self._first_not_none(quant.get("fcf_margin_pct"),
                                       quant.get("fcf_margin"), item.get("fcf_margin_pct"))
        elif metric_name in ("gross_margin", "gross_margin_pct"):
            val = self._first_not_none(quant.get("gross_margin_pct"),
                                       quant.get("gross_margin"), item.get("gross_margin_pct"))
        elif metric_name in ("current_price", "price"):
            val = self._first_not_none(item.get("current_price"),
                                       quant.get("raw_analysis_price_str"),
                                       quant.get("raw_analysis_price"))
        elif metric_name in ("pe_ratio", "pe"):
            val = self._first_not_none(quant.get("pe_ratio"), item.get("pe_ratio"))
        elif metric_name in quant:
            val = quant.get(metric_name)
        elif metric_name in item:
            val = item.get(metric_name)

        if val is not None:
            try:
                parsed = Decimal(str(val))
                return parsed if parsed.is_finite() else None
            except Exception:
                return None
        return None

    def get_upcoming_catalyst_date(self, ticker: str, catalyst_name: str) -> Optional[str]:
        clean = ticker.strip().upper()
        try:
            sidecars = self._sidecar_adapter.get_all_equity_sidecars()
        except Exception:
            return None

        item = (
            sidecars.get(clean)
            or sidecars.get(clean.replace(".BK", ""))
            or sidecars.get(f"{clean}.BK")
        )
        if not item or not isinstance(item, dict):
            return None

        for cat in item.get("catalysts", []):
            if isinstance(cat, dict) and cat.get("name") == catalyst_name:
                return str(cat.get("date") or "")
        return None

