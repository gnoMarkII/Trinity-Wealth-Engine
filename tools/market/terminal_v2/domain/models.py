"""Pure Python Domain Models for Terminal V2 (Unified Canonical Specification).

Strict Hexagonal Architecture Rules:
1. Zero external dependencies: Only Python standard library (dataclasses, typing, enum).
2. Pure, immutable domain objects using frozen dataclasses.
3. Explicit provenance and limitations attached to every snapshot model.
4. Strict semantic contracts:
   - SEC facts are company-filed facts (10-K & 10-Q), not audited facts.
   - SEC insider trades are parsed from Form 4 XML, strictly separated from Form 13F.
   - Treasury auction demand snapshot contains NO auction tail.
   - News candidates are discovery items with publisher attribution.
   - COT positioning is based on CFTC Disaggregated reports.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple


# ============================================================================
# Enums
# ============================================================================

class DataStatus(str, Enum):
    """Status indicator for returned market data."""
    LIVE = "live"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class OptionSide(str, Enum):
    CALL = "call"
    PUT = "put"


class EtfAssetType(str, Enum):
    BTC = "BTC"
    ETH = "ETH"


# ============================================================================
# 1. Thai Market & Equity Flow Models
# ============================================================================

@dataclass(frozen=True)
class InvestorTypeRow:
    """Breakdown row for one investor category in SET flow."""
    investor_type: str
    name_en: str
    buy_value: float
    sell_value: float
    net_value: float


@dataclass(frozen=True)
class ThaiFundFlowSnapshot:
    """Daily net trading flow across 4 investor types on the Stock Exchange of Thailand.

    Foreign, Institution, Proprietary (Broker), and Retail individuals.
    Values are denominated in THB.
    """
    market: str
    as_of: str
    total_value: float
    investors: Tuple[InvestorTypeRow, ...]
    source: str = "Settrade"
    is_stale: bool = False
    stale_reason: Optional[str] = None


@dataclass(frozen=True)
class MarketBreadth:
    """Venue-level advance/decline/unchanged count."""
    market: str
    as_of: str
    gainers: int
    losers: int
    unchanged: int
    source: str = "Settrade"
    is_stale: bool = False
    stale_reason: Optional[str] = None


@dataclass(frozen=True)
class MarketValuation:
    """Venue-level aggregate valuation multiples from SET."""
    market: str
    as_of: str
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    pbv_ratio: Optional[float]
    dividend_yield: Optional[float]
    turnover_ratio: Optional[float]
    source: str = "Settrade"
    is_stale: bool = False
    stale_reason: Optional[str] = None


@dataclass(frozen=True)
class GoldPriceDetail:
    """Buy/Sell pair for a specific form of retail gold."""
    buy: float
    sell: float


@dataclass(frozen=True)
class ThaiRetailGoldQuote:
    """Official retail gold prices announced by the Gold Traders Association of Thailand.

    Quotes 96.5% purity gold in standard Thai baht-weight (15.244g).
    Distinct from global paper futures (GC=F) or London bullion spot fix.
    """
    source: str
    unit: str
    bar: GoldPriceDetail
    ornament: GoldPriceDetail
    announced_at: str
    revision: Optional[int] = None
    is_stale: bool = False
    stale_reason: Optional[str] = None


@dataclass(frozen=True)
class MacroSeriesPoint:
    """Single time-series point (date and numerical value)."""
    date: str
    value: float


@dataclass(frozen=True)
class MacroSeries:
    """Official macroeconomic time series from FRED."""
    series_id: str
    label: str
    source: str
    frequency: str
    unit: str
    points: Tuple[MacroSeriesPoint, ...]
    is_stale: bool = False
    stale_reason: Optional[str] = None


@dataclass(frozen=True)
class LivePerpsQuote:
    """Live perpetual futures quote from Hyperliquid."""
    symbol: str
    mark_price: float
    dex_namespace: str
    asset_class: str = "synthetic_crypto_perp"
    contract_type: str = "perpetual_future"
    source: str = "Hyperliquid"
    open_interest: Optional[float] = None
    funding_rate: Optional[float] = None
    day_ntl_vlm: Optional[float] = None
    is_stale: bool = False
    stale_reason: Optional[str] = None


# ============================================================================
# 2. Equity Derivatives, Short Flow & Options Models
# ============================================================================

@dataclass(frozen=True)
class FinraShortVolumeSnapshot:
    """Reported daily consolidated short-sale volume from FINRA TRF/ADF."""
    symbol: str
    report_date: str                   # ISO YYYY-MM-DD
    short_volume: int
    short_exempt_volume: int
    finra_reported_total_volume: int
    short_pct: Optional[float]         # 100 * short_volume / total_volume
    fetched_at: float                  # Wall-clock epoch seconds
    coverage: str = "FINRA consolidated TRF/ADF"
    unit: str = "shares"
    is_stale: bool = False
    stale_reason: str = ""
    source: str = "FINRA"
    limitations: str = "Reported daily short sale volume across TRF/ADF; NOT short interest or institutional accumulation."


@dataclass(frozen=True)
class OptionContract:
    """Individual listed US equity option contract from Cboe delayed quotes."""
    occ_symbol: str                    # e.g. "AAPL260805C00110000"
    underlying: str                    # e.g. "AAPL"
    expiry: str                        # ISO YYYY-MM-DD
    strike: float                      # Dollar strike price, e.g. 110.0
    side: str                          # "call" or "put"
    open_interest: int
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    implied_volatility: Optional[float] = None  # Decimal scale (0.25 = 25%)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    multiplier: int = 100
    is_standard: bool = True           # False for adjusted/non-standard series


@dataclass(frozen=True)
class OptionsChainSnapshot:
    """Complete delayed listed options chain for a US equity underlying."""
    underlying: str
    underlying_price: Optional[float]
    iv30_decimal: Optional[float]      # 30-day IV as decimal
    delay_minutes: int
    contracts: Tuple[OptionContract, ...]
    fetched_at: float
    source: str = "Cboe"
    is_stale: bool = False
    stale_reason: str = ""
    limitations: str = "Delayed 15 minutes by exchange rule. Standard 100-multiplier contracts."


@dataclass(frozen=True)
class OptionsMaxPainResult:
    """Deterministic Max Pain calculation for a specific expiration."""
    underlying: str
    expiry: str                        # ISO YYYY-MM-DD
    strike: float                      # Strike with minimum total cash payout
    minimum_theoretical_payout: float
    candidate_count: int
    excluded_contract_count: int
    spot_price: Optional[float] = None
    distance_from_spot: Optional[float] = None
    distance_pct: Optional[float] = None
    assumptions: str = "Static open interest; standard 100 multiplier; ignores dynamic hedging and trading."
    limitations: str = "Folk/composite analytical indicator; NOT a price target, prediction, or evidence of market maker profits."


@dataclass(frozen=True)
class OptionsPutCallRatios:
    """Put/Call volume and open interest ratios for a specific expiration."""
    underlying: str
    expiry: str
    put_volume: int
    call_volume: int
    volume_ratio: Optional[float]
    put_open_interest: int
    call_open_interest: int
    oi_ratio: Optional[float]


# ============================================================================
# 3. Reference Rates & Sovereign Debt Models
# ============================================================================

@dataclass(frozen=True)
class ReferenceRatePoint:
    """Single reference rate observation from the New York Fed."""
    code: str                          # e.g. "SOFR", "EFFR", "TGCR", "BGCR", "OBFR"
    label: str
    effective_date: str                # ISO YYYY-MM-DD
    rate_percent: Optional[float]      # e.g. 4.90 for 4.90%
    volume_in_billions: Optional[float] = None
    target_rate_from: Optional[float] = None
    target_rate_to: Optional[float] = None


@dataclass(frozen=True)
class ReferenceRateSnapshot:
    """Overnight reference rates and deterministic pair spreads."""
    as_of: str
    rates: Tuple[ReferenceRatePoint, ...]
    spreads_bps: Dict[str, float]      # e.g. {"SOFR-EFFR": 5.0} in basis points
    fetched_at: float
    source: str = "New York Fed"
    unit: str = "percent / basis points"
    is_stale: bool = False
    stale_reason: str = ""
    limitations: str = "Overnight benchmark rates. Spreads explicitly named per rate pair; does not include ON RRP operational balances."


@dataclass(frozen=True)
class TreasuryYieldPoint:
    """Single tenor yield on the US Treasury yield curve."""
    maturity: str                      # e.g. "1 Mo", "3 Mo", "2 Yr", "10 Yr", "30 Yr"
    yield_percent: Optional[float]


@dataclass(frozen=True)
class TreasuryYieldCurveSnapshot:
    """US Treasury Par Yield Curve from Treasury.gov XML feed."""
    observation_date: str              # ISO YYYY-MM-DD
    yields: Tuple[TreasuryYieldPoint, ...]
    spread_10y_2y_bps: Optional[float] = None
    spread_10y_3m_bps: Optional[float] = None
    fetched_at: float = 0.0
    source: str = "US Treasury"
    unit: str = "percent / basis points"
    is_stale: bool = False
    stale_reason: str = ""


@dataclass(frozen=True)
class TreasuryAuctionResult:
    """Completed US Treasury auction result."""
    auction_date: str
    issue_date: str
    security_type: str                 # "Bill", "Note", "Bond", "TIPS"
    security_term: str                 # e.g. "4-Week", "10-Year"
    high_yield: Optional[float]        # Notes/Bonds
    high_investment_rate: Optional[float] = None  # Bills
    high_discount_rate: Optional[float] = None
    bid_to_cover_ratio: Optional[float] = None
    offering_amount_usd: Optional[float] = None
    total_accepted_usd: Optional[float] = None
    fetched_at: float = 0.0
    source: str = "US Treasury Fiscal Data"
    unit: str = "USD / percent"
    is_stale: bool = False
    stale_reason: str = ""


@dataclass(frozen=True)
class UsNationalDebtSnapshot:
    """Daily Debt to the Penny record from US Treasury."""
    record_date: str                   # ISO YYYY-MM-DD
    total_public_debt_usd: float
    debt_held_by_public_usd: Optional[float] = None
    intragovernmental_holdings_usd: Optional[float] = None
    is_daily_close: bool = True
    fetched_at: float = 0.0
    source: str = "US Treasury Fiscal Data"
    unit: str = "USD"
    is_stale: bool = False
    stale_reason: str = ""
    limitations: str = "Daily close accounting snapshot; NOT real-time national debt."


# ============================================================================
# 4. Thai Regulatory & Institutional Macro Models
# ============================================================================

@dataclass(frozen=True)
class ThaiFundAssetAllocationRow:
    """Asset class category breakdown from SEC Thailand MF_PORT_TH.csv."""
    asset_class: str
    domestic_or_foreign: str
    value_thb: float
    share_of_nav_pct: Optional[float] = None


@dataclass(frozen=True)
class ThaiFundAssetAllocationSnapshot:
    """Asset allocation breakdown for the Thai Mutual Fund Industry."""
    reporting_period: str
    total_nav_thb: Optional[float]
    allocations: Tuple[ThaiFundAssetAllocationRow, ...]
    fetched_at: float
    source: str = "SEC Thailand"
    unit: str = "THB / percent"
    is_stale: bool = False
    stale_reason: str = ""
    limitations: str = "Mutual fund industry asset class distribution; NOT individual equity sector allocations."


@dataclass(frozen=True)
class ThaiBondMarketStats:
    """Overview statistics of the Thai domestic bond market from STAT_DEPT_TH.csv."""
    reporting_period: str
    outstanding_thb: float
    trading_value_thb: float
    foreign_holding_thb: float
    foreign_holding_pct: Optional[float]
    fetched_at: float
    source: str = "SEC Thailand"
    unit: str = "THB / percent"
    is_stale: bool = False
    stale_reason: str = ""


@dataclass(frozen=True)
class ThaiCorporateBondIssuance:
    """Corporate bond new issuance statistics from OFFER_DEBT_COR_TH.csv."""
    reporting_period: str
    total_offering_thb: float
    long_term_thb: float
    short_term_thb: float
    top_sectors: Tuple[Tuple[str, float], ...]
    fetched_at: float
    source: str = "SEC Thailand"
    unit: str = "THB"
    is_stale: bool = False
    stale_reason: str = ""


@dataclass(frozen=True)
class ThaiPublicDebtComponent:
    """Individual line component of Thai Public Debt published by MOF."""
    component_number: int
    label_en: str
    label_th: str
    amount_thb: float


@dataclass(frozen=True)
class ThaiPublicDebtSnapshot:
    """Monthly Thai Public Debt to GDP ratio and components from MOF Thailand."""
    reporting_month: str
    total_debt_thb: float
    debt_to_gdp_pct: Optional[float]
    fx_rate_usd_thb: Optional[float]
    components: Tuple[ThaiPublicDebtComponent, ...]
    fetched_at: float
    source: str = "MOF Thailand"
    unit: str = "THB / percent"
    is_stale: bool = False
    stale_reason: str = ""
    limitations: str = "Published monthly by Ministry of Finance Thailand under official statutory debt ceiling definitions."


# ============================================================================
# 5. Prediction Markets & Spot ETF Flows
# ============================================================================

@dataclass(frozen=True)
class PredictionOutcome:
    """Outcome and market-implied outcome price (0.0 to 1.0 probability)."""
    label: str
    price: float


@dataclass(frozen=True)
class PredictionMarketItem:
    """Curated prediction market contract from Polymarket Gamma API."""
    market_id: str
    question: str
    outcomes: Tuple[PredictionOutcome, ...]
    volume_24h_usd: Optional[float]
    end_date: Optional[str]
    source_url: str
    fetched_at: float
    source: str = "Polymarket"
    unit: str = "implied_probability (0-1)"
    is_stale: bool = False
    stale_reason: str = ""
    limitations: str = "Market-implied outcome prices of the specific contracts listed; NOT official economic forecasts."


@dataclass(frozen=True)
class SpotEtfIssuerFlow:
    """Daily and cumulative net flows for a specific ETF issuer (e.g. IBIT, FBTC)."""
    ticker: str
    institute: str
    daily_net_inflow_usd: Optional[float]
    cumulative_net_inflow_usd: Optional[float]
    total_net_assets_usd: Optional[float]


@dataclass(frozen=True)
class SpotEtfFlowSnapshot:
    """US Spot Bitcoin or Ethereum ETF daily flow snapshot from SoSoValue."""
    asset: str
    report_date: str
    daily_total_usd: Optional[float]
    cumulative_total_usd: Optional[float]
    issuers: Tuple[SpotEtfIssuerFlow, ...]
    is_partial: bool = False
    completeness_notes: str = ""
    fetched_at: float = 0.0
    source: str = "SoSoValue"
    unit: str = "USD"
    is_stale: bool = False
    stale_reason: str = ""
    limitations: str = "Aggregated ETF net inflows via best-effort gateway; subject to upstream reporting delays."


# ============================================================================
# 6. Global Intelligence Models (BIS, OFR, CFTC, Nasdaq)
# ============================================================================

@dataclass(frozen=True)
class FinancialStressPoint:
    """Historical observation point for the OFR Financial Stress Index."""
    time_ms: int
    date: str
    value: float
    credit: Optional[float] = None
    equity_valuation: Optional[float] = None
    safe_assets: Optional[float] = None
    funding: Optional[float] = None
    volatility: Optional[float] = None


@dataclass(frozen=True)
class FinancialStressCategory:
    """One decomposed category of the OFR FSI."""
    label: str
    value: float


@dataclass(frozen=True)
class FinancialStressSnapshot:
    """Systemic financial stress reading from US Office of Financial Research (OFR)."""
    as_of_date: str
    published_at: str
    fsi_value: float
    categories: Tuple[FinancialStressCategory, ...]
    trend_90d: Tuple[FinancialStressPoint, ...]
    source: str = "OFR"
    data_lag_days: int = 2
    is_stale: bool = False


@dataclass(frozen=True)
class TraderClassPosition:
    """Positioning breakdown for one trader classification in CFTC COT."""
    class_name: str
    long_contracts: int
    short_contracts: int
    net_contracts: int
    spread_contracts: int = 0
    change_long: int = 0
    change_short: int = 0


@dataclass(frozen=True)
class MetalsCotPositioningSnapshot:
    """CFTC Commitments of Traders (COT) positioning snapshot for metals (Disaggregated)."""
    commodity: str
    commodity_code: str
    as_of_date: str
    published_at: str
    report_type: Literal["disaggregated", "legacy"]
    open_interest: int
    managed_money: TraderClassPosition
    swap_dealers: TraderClassPosition
    producer_merchant: TraderClassPosition
    other_reportables: TraderClassPosition
    non_reportables: TraderClassPosition
    net_managed_money: int
    percentile_52w: float
    source: str = "CFTC"
    is_stale: bool = False


@dataclass(frozen=True)
class PolicyRateItem:
    """Official central bank policy interest rate for one jurisdiction."""
    country: str
    rate_value: float
    rate_type: str
    effective_date: str
    currency: str
    central_bank: str
    previous_rate: Optional[float] = None
    last_change_date: Optional[str] = None
    is_stale: bool = False


@dataclass(frozen=True)
class GlobalPolicyRateSnapshot:
    """Cross-country central bank policy rate board from the BIS."""
    as_of_date: str
    rates: Tuple[PolicyRateItem, ...]
    spreads_vs_bot_repo: Dict[str, float] = field(default_factory=dict)
    source: str = "BIS"
    is_stale: bool = False


@dataclass(frozen=True)
class EarningsDateItem:
    """Scheduled or upcoming earnings release date."""
    earnings_date: str
    date_status: Literal["confirmed", "estimated", "unspecified"]
    report_time: Literal["pre-market", "after-hours", "unknown"]
    consensus_eps: Optional[float] = None
    estimate_count: Optional[int] = None


@dataclass(frozen=True)
class EarningsSurpriseItem:
    """Historical quarter earnings surprise result."""
    fiscal_quarter_end: str
    date_reported: str
    eps: float
    consensus_eps: float
    surprise_pct: float


@dataclass(frozen=True)
class AnalystRatingConsensus:
    """Sell-side analyst ratings consensus from Nasdaq."""
    symbol: str
    consensus: str
    analyst_count: int
    broker_names: Tuple[str, ...]


@dataclass(frozen=True)
class NasdaqEarningsConsensusSnapshot:
    """Equity earnings calendar, surprise history, and sell-side consensus from Nasdaq."""
    symbol: str
    coverage_status: Literal["full", "partial", "no_coverage"]
    has_earnings_surprise: bool
    has_analyst_ratings: bool
    upcoming_earnings: Optional[EarningsDateItem] = None
    surprise_history: Tuple[EarningsSurpriseItem, ...] = field(default_factory=tuple)
    ratings: Optional[AnalystRatingConsensus] = None
    source: str = "Nasdaq"
    is_stale: bool = False


# ============================================================================
# 7. Commodity Volatility, Treasury Demand & SEC Form 4 / XBRL Models
# ============================================================================

@dataclass(frozen=True)
class CommodityVolPoint:
    """Historical observation of a commodity volatility index."""
    date: str
    close: float


@dataclass(frozen=True)
class CommodityVolSnapshot:
    """Commodity Implied Volatility Index Snapshot (GVZ, VXSLV, OVX)."""
    index_symbol: str
    underlying_instrument: str
    close_date: str
    implied_volatility: float
    change_1d_points: Optional[float]
    percentile_52w: Optional[float]
    sample_count: int
    regime_label: Optional[str]
    source: str
    as_of_date: str
    fetched_at: float
    is_stale: bool = False
    stale_reason: Optional[str] = None
    limitations: Tuple[str, ...] = (
        "Measures 30-day annualized implied volatility of ETF options (GLD/SLV/USO), not physical futures directly.",
        "Regime label is a statistical heuristic derived from 52-week close percentile.",
    )


@dataclass(frozen=True)
class AuctionDemandSnapshot:
    """US Treasury Completed Auction Demand Snapshot.

    Reuses existing completed auction fields and computes prior 8-auction moving average.
    Strict Invariant: Does NOT calculate auction tail (which requires When-Issued market yield).
    """
    security_type: str
    security_term: str
    latest_auction_date: str
    latest_bid_to_cover_ratio: Optional[float]
    latest_high_yield: Optional[float]
    latest_high_investment_rate: Optional[float]
    latest_high_discount_rate: Optional[float]
    latest_offering_amount_usd: Optional[float]
    latest_total_accepted_usd: Optional[float]
    prior_mean_bid_to_cover: Optional[float]
    demand_delta: Optional[float]
    sample_count: int
    source: str
    as_of_date: str
    fetched_at: float
    is_stale: bool = False
    stale_reason: Optional[str] = None
    limitations: Tuple[str, ...] = (
        "Calculated from completed auction results published by US Treasury Fiscal Data.",
        "Auction tail is excluded as public API does not provide pre-auction When-Issued market yields.",
        "Moving average requires at least 3 completed auctions of identical type and term.",
    )


@dataclass(frozen=True)
class SecFact:
    """A single company-filed XBRL fact from SEC EDGAR."""
    concept_tag: str
    label: str
    val: Optional[float]
    unit: str
    form: str
    fy: Optional[int]
    fp: Optional[str]
    start: Optional[str]
    end: Optional[str]
    filed: Optional[str]
    accn: Optional[str]


@dataclass(frozen=True)
class SecCompanyFactsSnapshot:
    """Company-filed XBRL financial facts from SEC EDGAR.

    Strict Invariant: Represents company-reported facts across annual Form 10-K and
    quarterly Form 10-Q filings. Must NOT be characterized as universally audited.
    """
    symbol: str
    cik: str
    entity_name: str
    facts: Tuple[SecFact, ...]
    revenue_usd: Optional[float]
    operating_cash_flow_usd: Optional[float]
    capex_usd: Optional[float]
    free_cash_flow_usd: Optional[float]
    free_cash_flow_margin: Optional[float]
    long_term_debt_usd: Optional[float]
    debt_to_ocf_ratio: Optional[float]
    shares_outstanding: Optional[float]
    source: str
    as_of_date: str
    fetched_at: float
    is_stale: bool = False
    stale_reason: Optional[str] = None
    limitations: Tuple[str, ...] = (
        "Derived from company-filed XBRL reports (Form 10-K and 10-Q) via SEC EDGAR.",
        "Quarterly figures (10-Q) are un-audited management disclosures.",
        "Debt is a balance sheet stock as of period end; cash flow is a flow metric over the period.",
    )


@dataclass(frozen=True)
class InsiderTransaction:
    """A parsed Form 4 insider transaction from SEC EDGAR XML."""
    transaction_date: str
    reporting_owner: str
    officer_title: Optional[str]
    is_officer: bool
    is_director: bool
    is_ten_percent_owner: bool
    transaction_code: str
    shares: Optional[float]
    price_per_share: Optional[float]
    notional_usd: Optional[float]
    direct_or_indirect: str
    accession_number: str
    is_amendment: bool


@dataclass(frozen=True)
class SecInsiderTradeSnapshot:
    """SEC EDGAR Form 4 Insider Trades Snapshot.

    Strict Invariant: Parsed directly from Form 4 XML ownership documents.
    Form 13F institutional holdings are strictly segregated and excluded from this snapshot.
    """
    symbol: str
    cik: str
    transactions: Tuple[InsiderTransaction, ...]
    net_buy_ratio_90d: Optional[float]
    p_notional_sum_90d: float
    s_notional_sum_90d: float
    eligible_transaction_count: int
    source: str
    as_of_date: str
    fetched_at: float
    is_stale: bool = False
    stale_reason: Optional[str] = None
    limitations: Tuple[str, ...] = (
        "Parsed from official SEC Form 4 and Form 4/A XML ownership documents.",
        "Net buy ratio reflects non-derivative open market purchases (P) vs sales (S) within 90 days.",
        "Equity awards (A), option exercises (M), and gifts are excluded from open market calculations.",
    )


@dataclass(frozen=True)
class NewsCandidate:
    """A news item candidate discovered via RSS feeds."""
    headline: str
    publisher: str
    source_type: str
    article_url: str
    published_at: str
    discovered_at: float
    symbol: Optional[str] = None
    is_stale: bool = False


@dataclass(frozen=True)
class NewsDiscoverySnapshot:
    """News Discovery and Aggregation Snapshot.

    Strict Invariant: RSS is a discovery/aggregator tool, NOT a primary source or real-time feed.
    """
    query_symbol: str
    items: Tuple[NewsCandidate, ...]
    status: str
    source: str
    as_of_date: str
    fetched_at: float
    limitations: Tuple[str, ...] = (
        "Aggregated via RSS discovery endpoints; not a sub-second real-time news wire.",
        "Articles reflect publisher reported publication times.",
        "Keyless feeds are subject to upstream rate limiting and provider caching.",
    )
