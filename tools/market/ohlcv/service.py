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
from tools.market.ohlcv.application.query_service import OHLCVQueryService, CachedOhlcvQueryService

log = logging.getLogger(__name__)

EARNINGS_CACHE_TTL = 6 * 3600            # 6 Hours
DIVIDENDS_SPLITS_CACHE_TTL = 24 * 3600   # 24 Hours


class OhlcvService:
    """Legacy facade delegating construction to the OHLCV composition root.

    New application code should inject ``OHLCVQueryService`` from
    ``build_ohlcv_service``.  This compatibility type retains the historical
    constructor and query method while keeping concrete adapter construction
    outside the application-facing service module.
    """

    def __init__(
        self,
        ohlcv_provider: Optional[OhlcvProviderPort] = None,
        action_provider: Optional[CorporateActionProviderPort] = None,
        resolver: Optional[AssetResolverPort] = None,
        cache_ttl: float = 300.0,
    ):
        from tools.market.ohlcv.bootstrap import build_ohlcv_service

        self._delegate = build_ohlcv_service(
            ohlcv_provider=ohlcv_provider,
            action_provider=action_provider,
            resolver=resolver,
            cache_ttl=cache_ttl,
        )

    def get_ohlcv(self, ticker: str, range_str: str = "6mo", interval_str: str = "1d") -> OHLCVResponseDTO:
        """Forward the legacy query API to the composed query service."""
        return self._delegate.get_ohlcv(ticker=ticker, range_str=range_str, interval_str=interval_str)


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
