"""Inbound / Driving Ports for Terminal V2 (Interface Segregation Principle).

Defines the contracts exposed to application clients (FastAPI routers, AI agents, CLI).
Follows ISP by grouping capabilities into focused domain-specific sub-interfaces:
- MarketTerminalServicePort: Real-time price quotes & low-latency order routing
- MacroDataDrivingPort: Sovereign reference rates, yield curves, debt, systemic stress, policy rates
- FixedIncomeDrivingPort: Treasury auctions, auction demand moving averages, Thai debt & corporate bonds
- EquityDataDrivingPort: FINRA short volume, Nasdaq consensus, SEC financial facts, Form 4 insider trades, RSS news
- DerivativesDrivingPort: Cboe options chain, max pain, put/call ratios, commodity vol (GVZ, VXSLV, OVX), prediction markets
- FundFlowDrivingPort: SET investor flows, Thai mutual fund allocations, US Spot Bitcoin/Ethereum ETF flows

Composite Interface:
- TerminalDataServicePort: Combines all institutional data capabilities into a unified service contract
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

from tools.market.terminal_v2.domain.models import (
    AuctionDemandSnapshot,
    CommodityVolSnapshot,
    FinancialStressSnapshot,
    FinraShortVolumeSnapshot,
    GlobalPolicyRateSnapshot,
    LivePerpsQuote,
    MacroSeries,
    MarketBreadth,
    MarketValuation,
    MetalsCotPositioningSnapshot,
    NasdaqEarningsConsensusSnapshot,
    NewsDiscoverySnapshot,
    OptionsChainSnapshot,
    OptionsMaxPainResult,
    OptionsPutCallRatios,
    PredictionMarketItem,
    ReferenceRateSnapshot,
    SecCompanyFactsSnapshot,
    SecInsiderTradeSnapshot,
    SpotEtfFlowSnapshot,
    ThaiBondMarketStats,
    ThaiCorporateBondIssuance,
    ThaiFundAssetAllocationSnapshot,
    ThaiFundFlowSnapshot,
    ThaiPublicDebtSnapshot,
    ThaiRetailGoldQuote,
    TreasuryAuctionResult,
    TreasuryYieldCurveSnapshot,
    UsNationalDebtSnapshot,
)


# ============================================================================
# 1. Real-time Market Routing Driving Port
# ============================================================================

class MarketTerminalServicePort(ABC):
    """Driving interface for real-time market routing and dynamic capability lookup."""

    @abstractmethod
    def get_investor_flow(self, market: str = "SET") -> ThaiFundFlowSnapshot:
        """Fetch 4-investor-type flow on SET/mai."""
        pass

    @abstractmethod
    def get_market_valuation(self, market: str = "SET") -> MarketValuation:
        """Fetch venue aggregate valuation multiples."""
        pass

    @abstractmethod
    def get_market_breadth(self, market: str = "SET") -> MarketBreadth:
        """Fetch market breadth (gainers, losers, unchanged)."""
        pass

    @abstractmethod
    def get_retail_gold(self) -> ThaiRetailGoldQuote:
        """Fetch Thai retail gold prices from Gold Traders Association."""
        pass

    @abstractmethod
    def get_macro_series(self, series_id: str) -> MacroSeries:
        """Fetch macro series from FRED keyless provider."""
        pass

    @abstractmethod
    def get_perp_quote(self, symbol: str) -> LivePerpsQuote:
        """Fetch live perpetual quote from Hyperliquid."""
        pass

    @abstractmethod
    def query_by_capability(
        self,
        capability: str,
        symbol: Optional[str] = None,
        market: Optional[str] = None,
    ) -> Any:
        """Dynamic query matching requested capability and symbol/market."""
        pass


# ============================================================================
# 2. Segregated Domain-Specific Driving Sub-Interfaces
# ============================================================================

class MacroDataDrivingPort(Protocol):
    """Driving interface for macroeconomic, monetary, and systemic risk data."""

    def get_reference_rates(self) -> ReferenceRateSnapshot:
        """Fetch NY Fed overnight reference rates and rate spreads."""
        ...

    def get_treasury_yield_curve(
        self, month_yyyymm: Optional[str] = None
    ) -> TreasuryYieldCurveSnapshot:
        """Fetch US Treasury par yield curve and key spreads (10Y-2Y, 10Y-3M)."""
        ...

    def get_treasury_debt(self, limit: int = 5) -> Tuple[UsNationalDebtSnapshot, ...]:
        """Fetch daily close US Debt to the Penny snapshots."""
        ...

    def get_financial_stress(self) -> FinancialStressSnapshot:
        """Fetch US OFR Financial Stress Index with decomposed categories and T-2 lag."""
        ...

    def get_global_policy_rates(self) -> GlobalPolicyRateSnapshot:
        """Fetch BIS Central Bank Policy Rates across 12 countries with rate spreads."""
        ...


class FixedIncomeDrivingPort(Protocol):
    """Driving interface for sovereign and corporate debt instruments."""

    def get_treasury_auctions(self, limit: int = 10) -> Tuple[TreasuryAuctionResult, ...]:
        """Fetch recent completed US Treasury auction results."""
        ...

    def get_auction_demand_summary(
        self,
        security_type: str,
        security_term: str,
    ) -> AuctionDemandSnapshot:
        """Compute auction demand summary comparing latest auction against prior moving average."""
        ...

    def get_thai_bond_market_stats(self) -> ThaiBondMarketStats:
        """Fetch Thai domestic bond market overview statistics."""
        ...

    def get_thai_corporate_bond_issuance(self) -> ThaiCorporateBondIssuance:
        """Fetch Thai corporate bond offering statistics."""
        ...

    def get_thai_public_debt(self) -> ThaiPublicDebtSnapshot:
        """Fetch monthly Thai public debt to GDP report."""
        ...


class EquityDataDrivingPort(Protocol):
    """Driving interface for corporate equity analytics and intelligence."""

    def get_short_volume(self, symbols: Sequence[str]) -> Dict[str, FinraShortVolumeSnapshot]:
        """Fetch FINRA consolidated daily short-sale volume for symbols."""
        ...

    def get_nasdaq_consensus(self, symbol: str) -> NasdaqEarningsConsensusSnapshot:
        """Fetch Nasdaq earnings surprise history, upcoming date status, and analyst ratings."""
        ...

    def get_sec_financials(self, symbol: str) -> SecCompanyFactsSnapshot:
        """Fetch company-filed XBRL facts and derived financial ratios from SEC EDGAR."""
        ...

    def get_sec_insider_trades(self, symbol: str, limit: int = 20) -> SecInsiderTradeSnapshot:
        """Fetch parsed Form 4 insider transactions and 90-day net buying ratio from SEC EDGAR."""
        ...

    def get_ticker_news_discovery(self, symbol: str, limit: int = 15) -> NewsDiscoverySnapshot:
        """Discover news candidates for a given symbol with source attribution and freshness."""
        ...


class DerivativesDrivingPort(Protocol):
    """Driving interface for equity options, commodity volatility, and prediction markets."""

    def get_options_chain(self, symbol: str) -> OptionsChainSnapshot:
        """Fetch delayed listed equity options chain from Cboe."""
        ...

    def get_options_max_pain(
        self, symbol: str, expiry: Optional[str] = None
    ) -> OptionsMaxPainResult:
        """Calculate analytical Max Pain strike for a specific expiry."""
        ...

    def get_options_put_call_ratios(
        self, symbol: str, expiry: Optional[str] = None
    ) -> OptionsPutCallRatios:
        """Calculate Put/Call volume and OI ratios for a specific expiry."""
        ...

    def get_commodity_volatility(self, symbol: str) -> CommodityVolSnapshot:
        """Fetch commodity volatility snapshot for a specific index (GVZ, VXSLV, OVX)."""
        ...

    def get_all_commodity_volatilities(self) -> Tuple[CommodityVolSnapshot, ...]:
        """Fetch commodity volatility snapshots for all supported indices."""
        ...

    def get_metals_cot(self, commodity: str = "gold") -> MetalsCotPositioningSnapshot:
        """Fetch CFTC Disaggregated Commitments of Traders positioning for metals."""
        ...

    def get_prediction_markets(self, limit: int = 12) -> Tuple[PredictionMarketItem, ...]:
        """Fetch active prediction markets and implied odds from Polymarket."""
        ...


class FundFlowDrivingPort(Protocol):
    """Driving interface for fund flows and institutional holdings."""

    def get_thai_fund_asset_allocation(self) -> ThaiFundAssetAllocationSnapshot:
        """Fetch Thai mutual fund industry asset class distribution."""
        ...

    def get_spot_etf_flows(self, asset: str) -> SpotEtfFlowSnapshot:
        """Fetch US spot ETF daily flows and issuer metrics for BTC or ETH."""
        ...


# ============================================================================
# 3. Composite Driving Port (Single Canonical Contract for Terminal Data)
# ============================================================================

class TerminalDataServicePort(
    MacroDataDrivingPort,
    FixedIncomeDrivingPort,
    EquityDataDrivingPort,
    DerivativesDrivingPort,
    FundFlowDrivingPort,
    Protocol,
):
    """Unified composite driving port combining all institutional and market data capabilities."""
    ...
