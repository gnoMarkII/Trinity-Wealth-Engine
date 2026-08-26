"""Ports for OHLCV Market Data and Corporate Actions."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class AssetResolverPort(ABC):
    """Port for resolving a user ticker to a provider symbol/market."""

    @abstractmethod
    def resolve(self, ticker: str) -> Any:
        pass


class OhlcvProviderPort(ABC):
    """Port for fetching historical OHLCV candles."""

    @abstractmethod
    def fetch_history(
        self,
        symbol: str,
        period: str,
        interval: str,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV historical dataframe for period and interval."""
        pass


class CorporateActionProviderPort(ABC):
    """Port for fetching corporate actions (dividends, splits, earnings)."""

    @abstractmethod
    def fetch_dividends(self, symbol: str) -> Tuple[List[Dict[str, Any]], str]:
        """Fetch raw dividend records and status."""
        pass

    @abstractmethod
    def fetch_splits(self, symbol: str) -> Tuple[List[Dict[str, Any]], str]:
        """Fetch raw stock split records and status."""
        pass

    @abstractmethod
    def fetch_earnings(self, symbol: str, tz_name: str) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
        """Fetch earnings rows, status, and source_as_of timestamp."""
        pass
