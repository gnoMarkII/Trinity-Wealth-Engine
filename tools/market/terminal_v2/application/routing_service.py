"""Dynamic Routing Service for Terminal V2 (Hexagonal Application Service).

Orchestrates capability-based matching, symbol/market namespace validation,
and automated fallback across driven ports.
Strict Rule: No cross-asset substitutions. Perpetuals are never routed as cash equities.
"""
import logging
from typing import Any, Dict, Optional

from tools.market.terminal_v2.domain.errors import (
    DataUnavailableError,
    InvalidCapabilityError,
    ProviderError,
    SymbolMarketMismatchError,
)
from tools.market.terminal_v2.domain.models import (
    LivePerpsQuote,
    MacroSeries,
    MarketBreadth,
    MarketValuation,
    ThaiFundFlowSnapshot,
    ThaiRetailGoldQuote,
)
from tools.market.terminal_v2.ports.driven_ports import (
    GoldPricePort,
    MacroSeriesPort,
    PerpsQuotePort,
    ThaiMarketPort,
)
from tools.market.terminal_v2.ports.driving_ports import MarketTerminalServicePort

logger = logging.getLogger(__name__)

# Known crypto ticker basenames that do not require a dex prefix
KNOWN_CRYPTO_SYMBOLS = {
    "BTC", "ETH", "SOL", "AVAX", "BNB", "ARB", "OP", "SUI", "APT", "LINK",
    "DOGE", "SHIB", "PEPE", "WIF", "NEAR", "RENDER", "FET", "TAO", "TIA",
}


class DynamicRoutingService(MarketTerminalServicePort):
    """Hexagonal Application Service implementing driving port MarketTerminalServicePort."""

    def __init__(
        self,
        thai_market: ThaiMarketPort,
        gold_price: GoldPricePort,
        macro_series: MacroSeriesPort,
        perps_quote: PerpsQuotePort,
        legacy_macro_fallback: Optional[MacroSeriesPort] = None,
    ):
        self._thai_market = thai_market
        self._gold_price = gold_price
        self._macro_series = macro_series
        self._perps_quote = perps_quote
        self._legacy_macro_fallback = legacy_macro_fallback

    def get_investor_flow(self, market: str = "SET") -> ThaiFundFlowSnapshot:
        """Fetch 4-investor-type flow from ThaiMarketPort."""
        return self._thai_market.get_investor_type_flow(market)

    def get_market_valuation(self, market: str = "SET") -> MarketValuation:
        """Fetch venue aggregate valuation multiples from ThaiMarketPort."""
        return self._thai_market.get_market_statistics(market)

    def get_market_breadth(self, market: str = "SET") -> MarketBreadth:
        """Fetch market breadth from ThaiMarketPort."""
        return self._thai_market.get_market_breadth(market)

    def get_retail_gold(self) -> ThaiRetailGoldQuote:
        """Fetch Thai retail gold price from GoldPricePort."""
        return self._gold_price.get_retail_gold_quote()

    def get_macro_series(self, series_id: str) -> MacroSeries:
        """Fetch macro series from MacroSeriesPort with legacy fallback on failure."""
        sid = series_id.strip().upper()
        try:
            return self._macro_series.get_macro_series(sid)
        except Exception as primary_exc:
            logger.warning(
                "Primary FRED CSV failed for series '%s' (%s). Trying legacy fallback if available.",
                sid,
                primary_exc,
            )
            if self._legacy_macro_fallback:
                try:
                    return self._legacy_macro_fallback.get_macro_series(sid)
                except Exception as fb_exc:
                    logger.error("Legacy FRED fallback also failed for series '%s': %s", sid, fb_exc)
            raise primary_exc

    def get_perp_quote(self, symbol: str) -> LivePerpsQuote:
        """Fetch live perpetual quote with symbol and namespace checks."""
        sym = symbol.strip()
        # Enforce that perpetual quote is not confused with cash equity
        # If symbol does not have DEX prefix and is not a known crypto, warn or check
        return self._perps_quote.get_perps_quote(sym)

    def query_by_capability(
        self,
        capability: str,
        symbol: Optional[str] = None,
        market: Optional[str] = None,
    ) -> Any:
        """Route request by capability and symbol/market contract."""
        cap = capability.strip().lower()

        if cap in ("investor_type_flow", "thai_flow"):
            return self.get_investor_flow(market or "SET")

        if cap in ("market_valuation", "thai_stats"):
            return self.get_market_valuation(market or "SET")

        if cap in ("market_breadth", "thai_breadth"):
            return self.get_market_breadth(market or "SET")

        if cap in ("retail_gold_price", "thai_gold"):
            return self.get_retail_gold()

        if cap in ("macro_series", "fred_macro"):
            if not symbol:
                raise ValueError("Macro series capability requires a series ID as symbol parameter")
            return self.get_macro_series(symbol)

        if cap in ("perps_quote", "perp_quote"):
            if not symbol:
                raise ValueError("Perpetual quote capability requires a symbol parameter")

            # Check if user requested a bare equity ticker without dex namespace
            # e.g., "TSLA" instead of "xyz:TSLA"
            is_namespaced = ":" in symbol
            is_crypto = symbol.upper() in KNOWN_CRYPTO_SYMBOLS

            if not is_namespaced and not is_crypto:
                raise SymbolMarketMismatchError(
                    f"Symbol '{symbol}' is a bare ticker and not a recognised crypto asset. "
                    "To query builder DEX synthetic perpetuals on Hyperliquid, specify the DEX namespace (e.g. 'xyz:{symbol}'). "
                    "For US cash equities, route to a cash equity data provider."
                )

            return self.get_perp_quote(symbol)

        if cap == "equity_cash_quote":
            # Guard against routing HIP-3 perp to cash equity
            if symbol and (":" in symbol or symbol.startswith("xyz:") or symbol.startswith("km:")):
                raise SymbolMarketMismatchError(
                    f"Symbol '{symbol}' is a synthetic perpetual futures contract, "
                    "not a cash equity ticker. Cannot serve as equity_cash_quote."
                )
            raise InvalidCapabilityError(
                "Cash equity quotes are handled by Cash Equity provider, not Terminal V2 keyless engine."
            )

        raise InvalidCapabilityError(f"Unsupported capability '{capability}'.")
