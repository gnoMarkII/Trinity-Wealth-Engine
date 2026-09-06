from abc import ABC, abstractmethod
from typing import Optional, Dict
from pydantic import BaseModel, Field


class FundNavData(BaseModel):
    """Normalized NAV data for a mutual fund."""
    symbol: str = Field(description="Fund ticker / short code, e.g. 'PRINCIPAL VNEQ-A'")
    nav: float = Field(description="Net Asset Value per unit")
    nav_date: str = Field(description="Date of the NAV (YYYY-MM-DD)")
    percent_change: Optional[float] = Field(default=None, description="Daily NAV change percentage")
    currency: str = Field(default="THB", description="Currency of the NAV")


class ThaiFundPricePort(ABC):
    """Port interface for Thai mutual fund prices and fund directory lookups."""

    @abstractmethod
    def has_fund(self, symbol: str) -> bool:
        """Check if fund symbol exists in the fund catalog."""
        ...

    @abstractmethod
    def fetch_nav(self, symbol: str) -> Optional[FundNavData]:
        """Fetch the latest NAV data for a given Thai mutual fund symbol."""
        ...

    @abstractmethod
    def refresh_catalog(self, force: bool = False) -> int:
        """Refresh the fund catalog index. Returns number of funds indexed."""
        ...

    @abstractmethod
    def fetch_historical_nav(self, symbol: str, target_date: str) -> Optional[float]:
        """Fetch historical NAV for a fund on a specific date (YYYY-MM-DD)."""
        ...
