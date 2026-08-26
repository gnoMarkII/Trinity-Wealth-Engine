"""OHLCV Service Composition Root."""
from typing import Optional

from tools.market.ohlcv.ports.ohlcv_port import OhlcvProviderPort, CorporateActionProviderPort, AssetResolverPort
from tools.market.ohlcv.adapters.yfinance_adapter import YFinanceOhlcvAdapter
from tools.market.adapters.equity_research import MarketAssetResolverAdapter
from tools.market.ohlcv.application.query_service import OHLCVQueryService, CachedOhlcvQueryService


def build_ohlcv_service(
    ohlcv_provider: Optional[OhlcvProviderPort] = None,
    action_provider: Optional[CorporateActionProviderPort] = None,
    resolver: Optional[AssetResolverPort] = None,
    enable_cache: bool = True,
    cache_ttl: float = 300.0,
) -> OHLCVQueryService:
    """Build and wire OHLCV Query Service."""
    default_adapter = None
    if ohlcv_provider is None or action_provider is None:
        default_adapter = YFinanceOhlcvAdapter()
    resolved_ohlcv = ohlcv_provider or default_adapter
    resolved_action = action_provider or default_adapter
    resolved_resolver = resolver or MarketAssetResolverAdapter()

    if enable_cache:
        return CachedOhlcvQueryService(
            ohlcv_provider=resolved_ohlcv,
            action_provider=resolved_action,
            resolver=resolved_resolver,
            cache_ttl=cache_ttl,
        )
    return OHLCVQueryService(
        ohlcv_provider=resolved_ohlcv,
        action_provider=resolved_action,
        resolver=resolved_resolver,
    )
