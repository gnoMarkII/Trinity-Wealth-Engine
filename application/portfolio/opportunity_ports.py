"""Protocol Ports for Opportunity Finder Application Boundary.

Architecture Layer: application/portfolio/ (Ports)
Defines abstract typed protocols for snapshot retrieval, candidate discovery,
macro alignment, confirmed investment principles, and watchlist tracking.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class PortfolioSnapshotReaderPort(Protocol):
    """Port for retrieving consistent portfolio holdings, cash balances, and NAV."""
    def get_current_portfolio_state(self, refresh_prices: bool = False, portfolio_id: str = "default") -> Any:
        ...


class FundConstituentsReaderPort(Protocol):
    """Port for fetching mutual fund constituents and look-through metadata."""
    def fetch_constituents(self, symbol: str) -> Optional[Any]:
        ...


class EquityResearchReaderPort(Protocol):
    """Port for discovering and retrieving quantitative equity analysis sidecars."""
    def get_all_equity_sidecars(self) -> Dict[str, Any]:
        ...


class ConfirmedPolicyReaderPort(Protocol):
    """Port for reading confirmed Investor Axis constraints and Applied Bucket Policies."""
    def get_confirmed_investment_axis(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        ...

    def get_applied_bucket_policy(self, portfolio_id: str = "default") -> List[Dict[str, Any]]:
        ...


class MacroSnapshotReaderPort(Protocol):
    """Port for retrieving latest Macro Strategy and Sector Rotation analysis."""
    def get_latest_macro_strategy(self) -> Optional[Any]:
        ...

    def get_latest_sector_rotation(self) -> Optional[Any]:
        ...


class WatchlistReaderPort(Protocol):
    """Port for retrieving confirmed watchlist items for the portfolio."""
    def load_watchlist(self, portfolio_id: str = "default") -> Any:
        ...
