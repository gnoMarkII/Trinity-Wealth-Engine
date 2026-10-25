"""Look-Through Domain Models.

Architecture Layer: core/portfolio/ (Pure Domain)
Data classes representing true underlying stock exposures, sector look-throughs,
and concentration analysis across direct and indirect holdings.
Adheres strictly to Pure Domain isolation: no frameworks, no Pydantic, no tools imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Literal, Optional, Tuple


@dataclass
class FundConstituentHolding:
    """Individual holding item within a fund's portfolio (pure domain)."""
    name: str
    percent: Decimal
    ticker: Optional[str] = None
    asset_type: str = "EQUITY"


@dataclass
class FundConstituentSector:
    """Sector allocation breakdown item within a fund (pure domain)."""
    sector_name_th: str
    sector_name_en: str
    percent: Decimal


@dataclass
class FundConstituentsDomain:
    """Detailed look-through constituent breakdown of a mutual fund (pure domain)."""
    fund_id: str
    symbol: str
    as_of_date: str
    master_fund_name: Optional[str] = None
    master_fund_ticker: Optional[str] = None
    top_holdings: List[FundConstituentHolding] = field(default_factory=list)
    sector_breakdown: List[FundConstituentSector] = field(default_factory=list)
    asset_allocation: Dict[str, Decimal] = field(default_factory=dict)
    is_curated: bool = False
    source_date: Optional[str] = None
    weight_basis: str = "NAV"


@dataclass
class LookThroughCoverage:
    """Coverage metrics across fund and sector look-throughs."""
    fund_fetch_coverage: Decimal = Decimal("100.0")
    security_look_through_coverage: Decimal = Decimal("0.0")
    sector_coverage: Decimal = Decimal("0.0")
    unresolved_value_thb: Decimal = Decimal("0.0")
    total_fund_value_thb: Decimal = Decimal("0.0")
    resolved_fund_value_thb: Decimal = Decimal("0.0")


@dataclass
class LookThroughHoldingExposure:
    """True single-stock economic exposure combining direct holding and mutual funds."""
    ticker: str
    company_name: str
    market: Literal["US", "TH", "GLOBAL"]
    direct_value_thb: Decimal
    direct_weight_pct: Decimal
    indirect_value_thb: Decimal
    indirect_weight_pct: Decimal
    total_value_thb: Decimal
    total_weight_pct: Decimal
    source_funds: List[Tuple[str, Decimal]] = field(default_factory=list)  # [(fund_symbol, value_thb)]
    is_cap_breached: bool = False  # True if unrounded total_weight > cap
    is_hidden_concentration: bool = False  # True if direct <= cap and total > cap
    issuer_id: Optional[str] = None
    security_id: Optional[str] = None
    asset_type: str = "EQUITY"
    raw_total_weight_pct: Optional[Decimal] = None


@dataclass
class TrueSectorExposure:
    """Consolidated sector allocation across direct stocks and mutual funds."""
    sector_name: str  # GICS English
    sector_name_th: str
    total_value_thb: Decimal
    weight_pct: Decimal
    direct_stocks_pct: Decimal
    funds_pct: Decimal
    is_crowded: bool = False  # True if weight_pct > 30.0%


@dataclass
class PortfolioLookThroughReport:
    """Comprehensive portfolio look-through analysis report."""
    portfolio_id: str
    evaluated_at: str
    total_nav_thb: Decimal
    holdings_exposure: List[LookThroughHoldingExposure] = field(default_factory=list)
    top_overlapped_stocks: List[LookThroughHoldingExposure] = field(default_factory=list)
    sector_exposure: List[TrueSectorExposure] = field(default_factory=list)
    hidden_concentration_warnings: List[str] = field(default_factory=list)
    cap_breached_count: int = 0
    coverage: Optional[LookThroughCoverage] = None
    unresolved_value_thb: Decimal = Decimal("0.0")
