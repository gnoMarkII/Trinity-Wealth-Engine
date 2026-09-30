"""SoSoValue Keyless Adapter for US Spot Bitcoin & Ethereum ETF Flows.

Fetches aggregate ETF net inflow history and issuer-level metrics from SoSoValue openapi.
Strict Rule: Best-effort gateway with feature flag; missing values are None, not 0.
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import (
    SpotEtfFlowSnapshot,
    SpotEtfIssuerFlow,
)
from tools.market.terminal_v2.ports.driven_ports import SpotEtfFlowsPort

logger = logging.getLogger(__name__)

BASE_URL = "https://api.sosovalue.xyz/openapi/v2/etf"
ETF_FLOWS_TTL_SECONDS = 21600.0        # 6 hours
ETF_FLOWS_MAX_STALE_SECONDS = 2 * 86400.0  # 2 days ceiling

TYPE_MAP = {
    "BTC": "us-btc-spot",
    "ETH": "us-eth-spot",
}


def _val(v: Any) -> Optional[float]:
    """Extract numeric value from SoSoValue nested {value: ...} object."""
    if v is None:
        return None
    raw = v.get("value") if isinstance(v, dict) else v
    if raw is None:
        return None
    try:
        val = float(raw)
        return val
    except (ValueError, TypeError):
        return None


class SoSoValueEtfFlowsAdapter(SpotEtfFlowsPort):
    """Adapter reading US Spot ETF flow statistics from SoSoValue."""

    def __init__(
        self,
        cache: Optional[ThreadSafeTTLCache] = None,
        enabled: Optional[bool] = None,
    ):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=ETF_FLOWS_TTL_SECONDS)
        if enabled is not None:
            self._enabled = enabled
        else:
            env_val = os.getenv("ENABLE_SOSOVALUE_ETF_FLOWS", "true").strip().lower()
            self._enabled = env_val not in ("0", "false", "no", "off")

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{BASE_URL}/{path}"
        headers = {**BROWSER_HEADERS, "Content-Type": "application/json"}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0 and data.get("code") != 200:
                raise ProviderError(f"SoSoValue returned non-zero code {data.get('code')}: {data.get('msg')}", source="SoSoValue")
            return data
        except Exception as exc:
            raise ProviderError(f"Failed to fetch {path} from SoSoValue: {exc}", source="SoSoValue") from exc

    def _fetch_snapshot(self, asset: str) -> SpotEtfFlowSnapshot:
        if not self._enabled:
            raise DataUnavailableError("SoSoValue ETF flows capability is currently disabled by feature flag", capability="etf-flows", source="SoSoValue")

        norm_asset = asset.strip().upper()
        type_str = TYPE_MAP.get(norm_asset)
        if not type_str:
            raise ValueError(f"Unsupported ETF asset '{asset}'. Supported assets are BTC and ETH.")

        payload = {"type": type_str}
        hist_data = self._post_json("historicalInflowChart", payload)
        metrics_data = self._post_json("currentEtfDataMetrics", payload)

        hist_rows = hist_data.get("data", [])
        latest_date = ""
        daily_total: Optional[float] = None
        cum_total: Optional[float] = None

        if isinstance(hist_rows, list) and hist_rows:
            last_hist = hist_rows[-1]
            latest_date = last_hist.get("date", "")
            raw_net = last_hist.get("totalNetInflow")
            if raw_net is not None:
                try:
                    daily_total = float(raw_net)
                except (ValueError, TypeError):
                    daily_total = None

        metric_obj = metrics_data.get("data", {})
        if isinstance(metric_obj, dict):
            cum_total = _val(metric_obj.get("cumNetInflow"))
            if daily_total is None:
                daily_total = _val(metric_obj.get("dailyNetInflow"))

        issuer_list = metric_obj.get("list", []) if isinstance(metric_obj, dict) else []
        issuers: List[SpotEtfIssuerFlow] = []

        if isinstance(issuer_list, list):
            for row in issuer_list:
                ticker = row.get("ticker", "").strip()
                if not ticker:
                    continue
                issuers.append(
                    SpotEtfIssuerFlow(
                        ticker=ticker,
                        institute=row.get("institute", ""),
                        daily_net_inflow_usd=_val(row.get("dailyNetInflow")),
                        cumulative_net_inflow_usd=_val(row.get("cumNetInflow")),
                        total_net_assets_usd=_val(row.get("netAssets")),
                    )
                )

        return SpotEtfFlowSnapshot(
            asset=norm_asset,
            report_date=latest_date,
            daily_total_usd=daily_total,
            cumulative_total_usd=cum_total,
            issuers=tuple(issuers),
            is_partial=(not latest_date or daily_total is None),
            completeness_notes="Aggregated issuer flows; values are USD." if latest_date else "Partial data",
            fetched_at=time.time(),
            source="SoSoValue",
        )

    def get_spot_etf_flows(self, asset: str) -> SpotEtfFlowSnapshot:
        norm = asset.strip().upper()
        cache_key = f"sosovalue:etf:{norm}"
        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._fetch_snapshot(norm),
            ttl_seconds=ETF_FLOWS_TTL_SECONDS,
            max_stale_seconds=ETF_FLOWS_MAX_STALE_SECONDS,
        )
