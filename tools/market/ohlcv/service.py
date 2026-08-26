"""Application Service for OHLCV Market Data, Technicals, Warmup and Corporate Actions."""
import logging
from typing import Optional

from tools.market.ohlcv.domain.models import (
    CorporateActionEventDTO,
    CorporateActionsMetadataDTO,
    IndicatorBurnInPolicyDTO,
    IndicatorWarmupDetailDTO,
    OHLCVCandleDTO,
    OHLCVResponseDTO,
    PivotLevelsDTO,
)
from tools.market.ohlcv.domain.validation import (
    TIMEFRAME_CAPABILITIES,
    ALLOWED_RANGES,
    ALLOWED_INTERVALS,
    validate_ticker,
    validate_interval_and_range,
)
from tools.market.ohlcv.domain.calculations import (
    get_fetch_period as _get_fetch_period,
    calculate_indicator_burn_in as _calculate_indicator_burn_in,
    calculate_warmup_metadata as _calculate_warmup_metadata,
    calculate_pivot_levels as _calculate_pivot_levels,
    calculate_52w as _calculate_52w,
    map_corporate_actions as _map_corporate_actions,
)
from tools.market.ohlcv.ports.ohlcv_port import OhlcvProviderPort, CorporateActionProviderPort, AssetResolverPort
from tools.market.adapters.equity_research import MarketAssetResolverAdapter
from tools.market.ohlcv.application.query_service import OHLCVQueryService, CachedOhlcvQueryService

log = logging.getLogger(__name__)

EARNINGS_CACHE_TTL = 6 * 3600            # 6 Hours
DIVIDENDS_SPLITS_CACHE_TTL = 24 * 3600   # 24 Hours


class OhlcvService(CachedOhlcvQueryService):
    """Facade for OHLCV Service (maintained for backward compatibility)."""

    def __init__(
        self,
        ohlcv_provider: Optional[OhlcvProviderPort] = None,
        action_provider: Optional[CorporateActionProviderPort] = None,
        resolver: Optional[AssetResolverPort] = None,
        cache_ttl: float = 300.0,
    ):
        resolved_resolver = resolver or MarketAssetResolverAdapter()
        super().__init__(
            ohlcv_provider=ohlcv_provider,
            action_provider=action_provider,
            resolver=resolved_resolver,
            cache_ttl=cache_ttl,
        )


__all__ = [
    "OhlcvService",
    "OHLCVQueryService",
    "CachedOhlcvQueryService",
    "TIMEFRAME_CAPABILITIES",
    "ALLOWED_RANGES",
    "ALLOWED_INTERVALS",
    "OHLCVCandleDTO",
    "PivotLevelsDTO",
    "CorporateActionEventDTO",
    "CorporateActionsMetadataDTO",
    "IndicatorBurnInPolicyDTO",
    "IndicatorWarmupDetailDTO",
    "OHLCVResponseDTO",
    "_get_fetch_period",
    "_calculate_indicator_burn_in",
    "_calculate_warmup_metadata",
    "_calculate_pivot_levels",
    "_calculate_52w",
    "_map_corporate_actions",
]
