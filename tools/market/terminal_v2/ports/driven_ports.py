"""Outbound / Driven Ports for Terminal V2 (Unified SPI Contracts).

These abstract interfaces define what data providers (Adapters) must implement.
All port signatures use only Domain entities or standard Python types.
Zero external library or framework dependencies.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Protocol, Sequence, Tuple

from tools.market.terminal_v2.domain.models import (
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
# 1. Thai Market & Real-time Venue Ports
# ============================================================================

class ThaiMarketPort(ABC):
    """Port for Thai equity market venue-level data."""

    @abstractmethod
    def get_investor_type_flow(self, market: str = "SET") -> ThaiFundFlowSnapshot:
        """Fetch 4-investor-type daily net trading flow."""
        pass

    @abstractmethod
    def get_market_statistics(self, market: str = "SET") -> MarketValuation:
        """Fetch venue aggregate valuation metrics (P/E, P/BV, yield)."""
        pass

    @abstractmethod
    def get_market_breadth(self, market: str = "SET") -> MarketBreadth:
        """Fetch venue advance/decline/unchanged count."""
        pass


class GoldPricePort(ABC):
    """Port for official Thai retail physical gold prices."""

    @abstractmethod
    def get_retail_gold_quote(self) -> ThaiRetailGoldQuote:
        """Fetch official retail gold price from Gold Traders Association."""
        pass


class MacroSeriesPort(ABC):
    """Port for macroeconomic time-series data."""

    @abstractmethod
    def get_macro_series(self, series_id: str) -> MacroSeries:
        """Fetch macroeconomic series points by series ID."""
        pass


class PerpsQuotePort(ABC):
    """Port for crypto and builder-DEX perpetual futures."""

    @abstractmethod
    def get_perps_quote(self, symbol: str) -> LivePerpsQuote:
        """Fetch live perpetual futures quote from DEX orderbook/indexer."""
        pass


# ============================================================================
# 2. Institutional Equity, Short Volume & Options Ports
# ============================================================================

class ShortVolumePort(ABC):
    """Port for daily equity short-sale volume reports."""

    @abstractmethod
    def get_short_volume(self, symbols: Sequence[str]) -> Dict[str, FinraShortVolumeSnapshot]:
        """Fetch consolidated short sale volume for requested symbols."""
        pass


class OptionsChainPort(ABC):
    """Port for US listed equity options chains."""

    @abstractmethod
    def get_options_chain(self, symbol: str) -> OptionsChainSnapshot:
        """Fetch delayed listed options chain for an underlying."""
        pass


class NasdaqEquityIntelligencePort(ABC):
    """Port for fetching Nasdaq Earnings Calendar, Surprise & Consensus."""

    @abstractmethod
    def fetch_earnings_consensus(self, symbol: str) -> NasdaqEarningsConsensusSnapshot:
        """Fetch earnings surprise history, upcoming earnings date, and sell-side consensus."""
        pass


class SecFinancialsPort(Protocol):
    """Port for fetching SEC EDGAR company-filed XBRL facts."""

    def get_company_facts(self, symbol: str) -> SecCompanyFactsSnapshot:
        """Fetch company-filed facts for a US equity symbol."""
        ...


class SecInsiderTradesPort(Protocol):
    """Port for fetching parsed SEC EDGAR Form 4 insider transactions."""

    def get_insider_trades(self, symbol: str, limit: int = 20) -> SecInsiderTradeSnapshot:
        """Fetch Form 4 insider transactions for a US equity symbol."""
        ...


class TickerNewsPort(Protocol):
    """Port for discovering news candidates via RSS feeds."""

    def get_news_candidates(self, symbol: str, limit: int = 15) -> NewsDiscoverySnapshot:
        """Fetch news candidates for a symbol with status and attribution."""
        ...


# ============================================================================
# 3. Sovereign Debt & Benchmark Rates Ports
# ============================================================================

class ReferenceRatesPort(ABC):
    """Port for sovereign benchmark reference rates (SOFR, EFFR, TGCR, BGCR, OBFR)."""

    @abstractmethod
    def get_reference_rates(self) -> ReferenceRateSnapshot:
        """Fetch latest benchmark reference rates and pair spreads."""
        pass


class TreasuryDataPort(ABC):
    """Port for US Treasury yield curves, completed auctions, and national debt."""

    @abstractmethod
    def get_yield_curve(self, month_yyyymm: Optional[str] = None) -> TreasuryYieldCurveSnapshot:
        """Fetch US Treasury par yield curve for a given or current month."""
        pass

    @abstractmethod
    def get_auctions(self, limit: int = 10) -> Tuple[TreasuryAuctionResult, ...]:
        """Fetch recent completed Treasury auction results."""
        pass

    @abstractmethod
    def get_national_debt(self, limit: int = 5) -> Tuple[UsNationalDebtSnapshot, ...]:
        """Fetch recent daily close Debt to the Penny snapshots."""
        pass


class AuctionHistoryPort(Protocol):
    """Port for fetching completed Treasury auction history by security type & term."""

    def fetch_completed_auction_history(
        self,
        security_type: str,
        security_term: str,
        limit: int = 15,
    ) -> Tuple[TreasuryAuctionResult, ...]:
        """Fetch completed auctions of identical type and term."""
        ...


class GlobalPolicyRatesPort(ABC):
    """Port for fetching BIS Central Bank Policy Rates."""

    @abstractmethod
    def fetch_global_policy_rates(self) -> GlobalPolicyRateSnapshot:
        """Fetch policy interest rates across 12 major central banks from the BIS."""
        pass


class OfrFinancialStressPort(ABC):
    """Port for fetching US OFR Financial Stress Index."""

    @abstractmethod
    def fetch_financial_stress(self) -> FinancialStressSnapshot:
        """Fetch the latest OFR Financial Stress Index with T-2 lag and 90-day trend."""
        pass


# ============================================================================
# 4. Commodities, Predictions & Fund Flow Ports
# ============================================================================

class CommodityVolPort(Protocol):
    """Port for fetching Cboe commodity volatility index data (GVZ, VXSLV, OVX)."""

    def get_commodity_vol(self, symbol: str) -> CommodityVolSnapshot:
        """Fetch volatility snapshot for GVZ, VXSLV, or OVX."""
        ...


class MetalsPositioningPort(ABC):
    """Port for fetching CFTC Commitments of Traders (COT) for metals."""

    @abstractmethod
    def fetch_metals_cot(self, commodity: str = "gold") -> MetalsCotPositioningSnapshot:
        """Fetch weekly Disaggregated COT positioning for a metal."""
        pass


class PredictionMarketPort(ABC):
    """Port for prediction market contracts and market-implied outcome odds."""

    @abstractmethod
    def get_prediction_markets(self, limit: int = 12) -> Tuple[PredictionMarketItem, ...]:
        """Fetch active prediction markets ranked by 24h volume."""
        pass


class SpotEtfFlowsPort(ABC):
    """Port for US Spot Bitcoin and Ethereum ETF daily net flows."""

    @abstractmethod
    def get_spot_etf_flows(self, asset: str) -> SpotEtfFlowSnapshot:
        """Fetch ETF flow history and issuer breakdown for BTC or ETH."""
        pass


class ThaiFundAllocationPort(ABC):
    """Port for Thai mutual fund industry asset class distribution."""

    @abstractmethod
    def get_fund_asset_allocation(self) -> ThaiFundAssetAllocationSnapshot:
        """Fetch industry-wide asset class distribution from SEC Thailand."""
        pass


class ThaiBondMarketPort(ABC):
    """Port for Thai domestic bond market statistics and corporate bond offerings."""

    @abstractmethod
    def get_bond_market_stats(self) -> ThaiBondMarketStats:
        """Fetch overall debt market statistics from SEC Thailand."""
        pass

    @abstractmethod
    def get_corporate_bond_issuance(self) -> ThaiCorporateBondIssuance:
        """Fetch corporate bond issuance statistics from SEC Thailand."""
        pass


class ThaiPublicDebtPort(ABC):
    """Port for Thailand Ministry of Finance Public Debt to GDP figures."""

    @abstractmethod
    def get_public_debt(self) -> ThaiPublicDebtSnapshot:
        """Fetch monthly Thai public debt report from MOF Thailand."""
        pass
