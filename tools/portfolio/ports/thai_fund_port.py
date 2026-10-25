from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional, Dict, List
from pydantic import BaseModel, Field


class FundNavData(BaseModel):
    """Normalized NAV data for a mutual fund."""
    symbol: str = Field(description="Fund ticker / short code, e.g. 'PRINCIPAL VNEQ-A'")
    nav: float = Field(description="Net Asset Value per unit")
    nav_date: str = Field(description="Date of the NAV (YYYY-MM-DD)")
    percent_change: Optional[float] = Field(default=None, description="Daily NAV change percentage")
    currency: str = Field(default="THB", description="Currency of the NAV")


class FundHoldingItem(BaseModel):
    """Individual holding item within a fund's portfolio."""
    name: str = Field(description="Holding security or asset name")
    percent: Decimal = Field(description="Weight percentage in the fund (0-100)")
    ticker: Optional[str] = Field(default=None, description="Resolved stock ticker symbol if available")
    asset_type: str = Field(default="EQUITY", description="Asset type classification (EQUITY, CASH, BOND, etc.)")


class FundSectorItem(BaseModel):
    """Sector allocation breakdown item within a fund."""
    sector_name_th: str = Field(description="Original sector name in Thai or report currency")
    sector_name_en: str = Field(description="Mapped GICS English sector name")
    percent: Decimal = Field(description="Weight percentage in the fund (0-100)")


class FundConstituentsData(BaseModel):
    """Detailed look-through constituent breakdown of a mutual fund."""
    fund_id: str = Field(description="Internal or provider fund ID")
    symbol: str = Field(description="Fund symbol / short code")
    as_of_date: str = Field(description="Portfolio date (YYYY-MM-DD)")
    master_fund_name: Optional[str] = Field(default=None, description="Underlying master fund name if feeder fund")
    master_fund_ticker: Optional[str] = Field(default=None, description="Master fund ticker (e.g. IVV, JEPQ) if identified")
    top_holdings: List[FundHoldingItem] = Field(default_factory=list, description="Top holdings in the fund")
    sector_breakdown: List[FundSectorItem] = Field(default_factory=list, description="Sector breakdown")
    asset_allocation: Dict[str, Decimal] = Field(default_factory=dict, description="Asset class allocation weights")
    is_curated: bool = Field(default=False, description="True if constituents originate from curated master catalog")
    fetched_at: Optional[str] = Field(default=None, description="ISO timestamp of when data was fetched")


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

    def fetch_constituents(self, symbol: str) -> Optional[FundConstituentsData]:
        """Fetch constituent holdings, sector breakdown, and asset allocation for a fund."""
        return None

