from abc import ABC, abstractmethod
from typing import Optional, Tuple, Literal, Dict
from tools.portfolio.domain.models import PortfolioState


class MarketPricePort(ABC):
    """Port interface for market prices, foreign exchange rates, and company fundamentals."""

    @abstractmethod
    def fetch_price(self, symbol: str, currency: Literal["THB", "USD"]) -> Optional[float]:
        """Fetch latest price for a single symbol."""
        ...

    @abstractmethod
    def fetch_fx_rate(
        self, date_str: Optional[str] = None, fallback_rate: Optional[float] = None
    ) -> Tuple[float, Literal["historical", "live", "fallback"]]:
        """Fetch FX rate for USDTHB with status."""
        ...

    @abstractmethod
    def refresh_portfolio_prices(self, state: PortfolioState) -> Dict[str, str]:
        """Batch refresh latest prices for all holdings in a portfolio state."""
        ...

    @abstractmethod
    def fetch_fundamentals(self, state: PortfolioState, force: bool = False) -> Dict[str, str]:
        """Fetch fundamental data (market cap, P/E, EPS, dividend yield) for equity holdings."""
        ...
