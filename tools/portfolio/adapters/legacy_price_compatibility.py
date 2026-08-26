"""Compatibility bridge for legacy portfolio tool patch points.

The application services depend on :class:`MarketPricePort`; a small number of
older agent-tool tests and integrations still patch ``tools.portfolio.prices``
or ``tools.portfolio.trading``.  This adapter keeps those patch points working
at the composition boundary without leaking legacy imports into services.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, Literal, Optional, Tuple

from tools.portfolio.domain.models import PortfolioState
from tools.portfolio.ports.price_port import MarketPricePort


class LegacyPriceCompatibilityAdapter(MarketPricePort):
    """Delegate market I/O while honoring legacy module-level hooks."""

    def __init__(self, delegate: MarketPricePort) -> None:
        self._delegate = delegate

    @staticmethod
    def _legacy_prices():
        import tools.portfolio.prices as prices

        return prices

    @staticmethod
    def _legacy_trading():
        import tools.portfolio.trading as trading

        return trading

    def fetch_price(self, symbol: str, currency: Literal["THB", "USD"]) -> Optional[float]:
        trading = self._legacy_trading()
        legacy = getattr(trading, "fetch_latest_price", None)
        if callable(legacy):
            timeout = float(getattr(trading, "_PRICE_FETCH_TIMEOUT", 6.0) or 6.0)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(legacy, symbol, currency)
                try:
                    return future.result(timeout=timeout)
                except TimeoutError:
                    future.cancel()
                    return None
        return self._delegate.fetch_price(symbol, currency)

    def fetch_fx_rate(
        self,
        date_str: Optional[str] = None,
        fallback_rate: Optional[float] = None,
    ) -> Tuple[float, Literal["historical", "live", "fallback"]]:
        trading = self._legacy_trading()
        # The legacy update-FX tool historically exposed this lower-level hook
        # directly, and integrations still patch it.  Honor it for live FX;
        # historical dates continue through the normalized port method.
        fx_hook = getattr(trading, "_fetch_fx_rate", None)
        if not date_str and callable(fx_hook) and getattr(fx_hook, "__module__", "") != "tools.portfolio.trading":
            # A patched hook is authoritative: ``None`` must remain a failure
            # so the application service returns its historical error string.
            raw_rate = fx_hook()
            if isinstance(raw_rate, tuple):
                raw_rate = raw_rate[0]
            if raw_rate is None:
                return None, "fallback"  # type: ignore[return-value]
            return float(raw_rate), "live"

        legacy = getattr(trading, "fetch_fx_rate", None)
        if callable(legacy):
            return legacy(date_str=date_str, fallback_rate=fallback_rate)
        return self._delegate.fetch_fx_rate(date_str=date_str, fallback_rate=fallback_rate)

    def refresh_portfolio_prices(self, state: PortfolioState) -> Dict[str, str]:
        legacy = getattr(self._legacy_prices(), "_refresh_prices", None)
        if callable(legacy):
            return legacy(state)
        return self._delegate.refresh_portfolio_prices(state)

    def fetch_fundamentals(self, state: PortfolioState, force: bool = False) -> Dict[str, str]:
        # The legacy helper delegates back into the application service, so use
        # the real adapter here to avoid a recursive compatibility loop.
        return self._delegate.fetch_fundamentals(state, force=force)
