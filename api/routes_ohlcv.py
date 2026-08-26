"""FastAPI Sub-router for OHLCV Market Data & Candlestick Charts (Compatibility Shell)."""
import yfinance as yf

from api.routers.equity.router_ohlcv import router, get_equity_ohlcv
from api.dependencies import _OHLCV_SERVICE
from tools.market.ohlcv.service import (
    OhlcvService,
    TIMEFRAME_CAPABILITIES,
    ALLOWED_RANGES,
    ALLOWED_INTERVALS,
    _get_fetch_period,
    _calculate_indicator_burn_in,
    _calculate_warmup_metadata,
    _calculate_pivot_levels,
    _calculate_52w,
    _map_corporate_actions,
    validate_ticker as _validate_ticker,
)

# Global service instance for backward-compatible test fixtures
_DEFAULT_SERVICE = _OHLCV_SERVICE
_CACHE = _DEFAULT_SERVICE._cache
_CACHE_LOCK = _DEFAULT_SERVICE._cache_lock
_ACTION_CACHE = _DEFAULT_SERVICE._action_cache
_ACTION_LOCK = _DEFAULT_SERVICE._action_lock
_KEY_LOCKS = _DEFAULT_SERVICE._key_locks
_ACTION_KEY_LOCKS = _DEFAULT_SERVICE._action_key_locks
_CACHE_TTL_SECONDS = _DEFAULT_SERVICE._cache_ttl

__all__ = [
    "router",
    "get_equity_ohlcv",
    "yf",
    "_DEFAULT_SERVICE",
    "_CACHE",
    "_CACHE_LOCK",
    "_ACTION_CACHE",
    "_ACTION_LOCK",
    "_KEY_LOCKS",
    "_ACTION_KEY_LOCKS",
    "_CACHE_TTL_SECONDS",
    "_validate_ticker",
    "TIMEFRAME_CAPABILITIES",
    "ALLOWED_RANGES",
    "ALLOWED_INTERVALS",
    "_get_fetch_period",
    "_calculate_indicator_burn_in",
    "_calculate_warmup_metadata",
    "_calculate_pivot_levels",
    "_calculate_52w",
    "_map_corporate_actions",
]
