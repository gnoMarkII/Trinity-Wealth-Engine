"""Unified Application Service for Terminal V2 Data Engine.

Implements TerminalDataServicePort (and domain sub-interfaces):
- MacroDataDrivingPort
- FixedIncomeDrivingPort
- EquityDataDrivingPort
- DerivativesDrivingPort
- FundFlowDrivingPort

Modular Architecture:
- Coordinates driven ports and pure domain calculations.
- Segregated domain methods to maintain Single Responsibility.
- Zero framework or HTTP dependencies.
"""
from datetime import date
import logging
from typing import Dict, List, Optional, Sequence, Tuple

from tools.market.terminal_v2.domain.calculations import (
    calculate_auction_demand_summary,
    calculate_max_pain,
    calculate_put_call_ratios,
)
from tools.market.terminal_v2.domain.errors import (
    DataUnavailableError,
    InvalidCapabilityError,
    ProviderError,
)
from tools.market.terminal_v2.domain.models import (
    AuctionDemandSnapshot,
    CommodityVolSnapshot,
    FinancialStressSnapshot,
    FinraShortVolumeSnapshot,
    GlobalPolicyRateSnapshot,
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
    ThaiPublicDebtSnapshot,
    TreasuryAuctionResult,
    TreasuryYieldCurveSnapshot,
    UsNationalDebtSnapshot,
)
from tools.market.terminal_v2.ports.driven_ports import (
    AuctionHistoryPort,
    CommodityVolPort,
    GlobalPolicyRatesPort,
    MetalsPositioningPort,
    NasdaqEquityIntelligencePort,
    OfrFinancialStressPort,
    OptionsChainPort,
    PredictionMarketPort,
    ReferenceRatesPort,
    SecFinancialsPort,
    SecInsiderTradesPort,
    ShortVolumePort,
    SpotEtfFlowsPort,
    ThaiBondMarketPort,
    ThaiFundAllocationPort,
    ThaiPublicDebtPort,
    TickerNewsPort,
    TreasuryDataPort,
)
from tools.market.terminal_v2.ports.driving_ports import TerminalDataServicePort

logger = logging.getLogger(__name__)


def _select_nearest_expiry(chain: OptionsChainSnapshot) -> str:
    """Select the nearest listed expiry with non-zero open interest on or after today."""
    today_iso = date.today().isoformat()
    available_expiries = sorted({c.expiry for c in chain.contracts if c.expiry >= today_iso and c.is_standard})
    if not available_expiries:
        # Fallback to any listed standard expiry
        available_expiries = sorted({c.expiry for c in chain.contracts if c.is_standard})

    if not available_expiries:
        raise DataUnavailableError(
            f"No listed option expirations found for {chain.underlying}",
            capability="options-max-pain",
            source="Cboe",
        )

    # Find first expiry with total open interest > 0
    for exp in available_expiries:
        tot_oi = sum(c.open_interest for c in chain.contracts if c.expiry == exp and c.is_standard)
        if tot_oi > 0:
            return exp

    return available_expiries[0]


class TerminalDataService(TerminalDataServicePort):
    """Unified Hexagonal Application Service orchestrating market and institutional data."""

    def __init__(
        self,
        # Equity & Disclosures
        short_volume: ShortVolumePort,
        nasdaq_intelligence: NasdaqEquityIntelligencePort,
        sec_financials: SecFinancialsPort,
        sec_insider_trades: SecInsiderTradesPort,
        ticker_news: TickerNewsPort,
        # Macro & Rates
        reference_rates: ReferenceRatesPort,
        ofr_stress: OfrFinancialStressPort,
        global_policy_rates: GlobalPolicyRatesPort,
        # Fixed Income & Sovereign Debt
        treasury_data: TreasuryDataPort,
        auction_history: AuctionHistoryPort,
        thai_bond_market: ThaiBondMarketPort,
        thai_public_debt: ThaiPublicDebtPort,
        # Derivatives & Commodities
        options_chain: OptionsChainPort,
        commodity_vol: CommodityVolPort,
        metals_cot: MetalsPositioningPort,
        prediction_market: PredictionMarketPort,
        # Fund Flows
        thai_fund_allocation: ThaiFundAllocationPort,
        spot_etf_flows: SpotEtfFlowsPort,
    ):
        self._short_volume = short_volume
        self._nasdaq_intelligence = nasdaq_intelligence
        self._sec_financials = sec_financials
        self._sec_insider_trades = sec_insider_trades
        self._ticker_news = ticker_news

        self._reference_rates = reference_rates
        self._ofr_stress = ofr_stress
        self._global_policy_rates = global_policy_rates

        self._treasury_data = treasury_data
        self._auction_history = auction_history
        self._thai_bond_market = thai_bond_market
        self._thai_public_debt = thai_public_debt

        self._options_chain = options_chain
        self._commodity_vol = commodity_vol
        self._metals_cot = metals_cot
        self._prediction_market = prediction_market

        self._thai_fund_allocation = thai_fund_allocation
        self._spot_etf_flows = spot_etf_flows

    # ========================================================================
    # Macro Domain Capabilities
    # ========================================================================

    def get_reference_rates(self) -> ReferenceRateSnapshot:
        return self._reference_rates.get_reference_rates()

    def get_treasury_yield_curve(
        self, month_yyyymm: Optional[str] = None
    ) -> TreasuryYieldCurveSnapshot:
        return self._treasury_data.get_yield_curve(month_yyyymm=month_yyyymm)

    def get_treasury_debt(self, limit: int = 5) -> Tuple[UsNationalDebtSnapshot, ...]:
        return self._treasury_data.get_national_debt(limit=limit)

    def get_financial_stress(self) -> FinancialStressSnapshot:
        return self._ofr_stress.fetch_financial_stress()

    def get_global_policy_rates(self) -> GlobalPolicyRateSnapshot:
        return self._global_policy_rates.fetch_global_policy_rates()

    # ========================================================================
    # Fixed Income Domain Capabilities
    # ========================================================================

    def get_treasury_auctions(self, limit: int = 10) -> Tuple[TreasuryAuctionResult, ...]:
        return self._treasury_data.get_auctions(limit=limit)

    def get_auction_demand_summary(
        self,
        security_type: str,
        security_term: str,
    ) -> AuctionDemandSnapshot:
        auctions = self._auction_history.fetch_completed_auction_history(
            security_type=security_type,
            security_term=security_term,
            limit=15,
        )

        if not auctions:
            return AuctionDemandSnapshot(
                security_type=security_type,
                security_term=security_term,
                latest_auction_date="",
                latest_bid_to_cover_ratio=None,
                latest_high_yield=None,
                latest_high_investment_rate=None,
                latest_high_discount_rate=None,
                latest_offering_amount_usd=None,
                latest_total_accepted_usd=None,
                prior_mean_bid_to_cover=None,
                demand_delta=None,
                sample_count=0,
                source="US Treasury Fiscal Data",
                as_of_date="",
                fetched_at=0.0,
            )

        latest = auctions[0]
        prior_auctions = auctions[1:9]
        prior_btcs = [a.bid_to_cover_ratio for a in prior_auctions if a.bid_to_cover_ratio is not None]

        prior_mean, demand_delta, sample_count = calculate_auction_demand_summary(
            latest_btc=latest.bid_to_cover_ratio,
            prior_btcs=prior_btcs,
            min_samples=3,
            max_prior=8,
        )

        return AuctionDemandSnapshot(
            security_type=security_type,
            security_term=security_term,
            latest_auction_date=latest.auction_date,
            latest_bid_to_cover_ratio=latest.bid_to_cover_ratio,
            latest_high_yield=latest.high_yield,
            latest_high_investment_rate=latest.high_investment_rate,
            latest_high_discount_rate=latest.high_discount_rate,
            latest_offering_amount_usd=latest.offering_amount_usd,
            latest_total_accepted_usd=latest.total_accepted_usd,
            prior_mean_bid_to_cover=prior_mean,
            demand_delta=demand_delta,
            sample_count=sample_count,
            source="US Treasury Fiscal Data",
            as_of_date=latest.auction_date,
            fetched_at=latest.fetched_at,
            is_stale=latest.is_stale,
            stale_reason=latest.stale_reason,
        )

    def get_thai_bond_market_stats(self) -> ThaiBondMarketStats:
        return self._thai_bond_market.get_bond_market_stats()

    def get_thai_corporate_bond_issuance(self) -> ThaiCorporateBondIssuance:
        return self._thai_bond_market.get_corporate_bond_issuance()

    def get_thai_public_debt(self) -> ThaiPublicDebtSnapshot:
        return self._thai_public_debt.get_public_debt()

    # ========================================================================
    # Equity Domain Capabilities
    # ========================================================================

    def get_short_volume(self, symbols: Sequence[str]) -> Dict[str, FinraShortVolumeSnapshot]:
        return self._short_volume.get_short_volume(symbols)

    def get_nasdaq_consensus(self, symbol: str) -> NasdaqEarningsConsensusSnapshot:
        clean = symbol.strip().upper()
        return self._nasdaq_intelligence.fetch_earnings_consensus(clean)

    def get_sec_financials(self, symbol: str) -> SecCompanyFactsSnapshot:
        clean = symbol.strip().upper()
        return self._sec_financials.get_company_facts(clean)

    def get_sec_insider_trades(self, symbol: str, limit: int = 20) -> SecInsiderTradeSnapshot:
        clean = symbol.strip().upper()
        return self._sec_insider_trades.get_insider_trades(clean, limit=limit)

    def get_ticker_news_discovery(self, symbol: str, limit: int = 15) -> NewsDiscoverySnapshot:
        clean = symbol.strip().upper()
        return self._ticker_news.get_news_candidates(clean, limit=limit)

    # ========================================================================
    # Derivatives & Commodities Domain Capabilities
    # ========================================================================

    def get_options_chain(self, symbol: str) -> OptionsChainSnapshot:
        clean = symbol.strip().upper()
        return self._options_chain.get_options_chain(clean)

    def get_options_max_pain(
        self, symbol: str, expiry: Optional[str] = None
    ) -> OptionsMaxPainResult:
        clean = symbol.strip().upper()
        chain = self._options_chain.get_options_chain(clean)
        target_expiry = expiry or _select_nearest_expiry(chain)
        return calculate_max_pain(
            contracts=chain.contracts,
            underlying=clean,
            expiry=target_expiry,
            spot_price=chain.underlying_price,
        )

    def get_options_put_call_ratios(
        self, symbol: str, expiry: Optional[str] = None
    ) -> OptionsPutCallRatios:
        clean = symbol.strip().upper()
        chain = self._options_chain.get_options_chain(clean)
        target_expiry = expiry or _select_nearest_expiry(chain)
        return calculate_put_call_ratios(
            contracts=chain.contracts,
            underlying=clean,
            expiry=target_expiry,
        )

    def get_commodity_volatility(self, symbol: str) -> CommodityVolSnapshot:
        clean = symbol.strip().upper()
        return self._commodity_vol.get_commodity_vol(clean)

    def get_all_commodity_volatilities(self) -> Tuple[CommodityVolSnapshot, ...]:
        symbols = ("GVZ", "VXSLV", "OVX")
        results = []
        for s in symbols:
            try:
                results.append(self._commodity_vol.get_commodity_vol(s))
            except Exception:
                continue
        return tuple(results)

    def get_metals_cot(self, commodity: str = "gold") -> MetalsCotPositioningSnapshot:
        return self._metals_cot.fetch_metals_cot(commodity)

    def get_prediction_markets(self, limit: int = 12) -> Tuple[PredictionMarketItem, ...]:
        return self._prediction_market.get_prediction_markets(limit=limit)

    # ========================================================================
    # Fund Flow Domain Capabilities
    # ========================================================================

    def get_thai_fund_asset_allocation(self) -> ThaiFundAssetAllocationSnapshot:
        return self._thai_fund_allocation.get_fund_asset_allocation()

    def get_spot_etf_flows(self, asset: str) -> SpotEtfFlowSnapshot:
        clean = asset.strip().upper()
        if clean not in ("BTC", "ETH"):
            raise InvalidCapabilityError(
                f"Unsupported ETF asset '{asset}'. Must be 'BTC' or 'ETH'.",
                capability="spot-etf-flows",
            )
        return self._spot_etf_flows.get_spot_etf_flows(clean)
