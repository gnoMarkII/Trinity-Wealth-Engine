"""Pydantic Boundary Schemas for Terminal V2 (Hexagonal Architecture).

Strict Rule: Pydantic BaseModels live ONLY here at the API boundary,
never inside the pure Python domain layer.
Maps domain dataclasses to JSON responses for FastAPI.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from tools.market.terminal_v2.domain.models import (
    LivePerpsQuote,
    MacroSeries,
    MarketBreadth,
    MarketValuation,
    ThaiFundFlowSnapshot,
    ThaiRetailGoldQuote,
)


class InvestorTypeRowSchema(BaseModel):
    investor_type: str = Field(..., description="Investor type code, e.g. Foreign, Institution")
    name_en: str = Field(..., description="English name of investor category")
    buy_value: float = Field(..., description="Buy value in THB")
    sell_value: float = Field(..., description="Sell value in THB")
    net_value: float = Field(..., description="Net trading value (buy - sell) in THB")


class ThaiFundFlowResponse(BaseModel):
    market: str
    as_of: str
    total_value: float
    investors: List[InvestorTypeRowSchema]
    source: str = "Settrade"
    is_stale: bool = False
    stale_reason: Optional[str] = None


class GoldPriceDetailSchema(BaseModel):
    buy: float = Field(..., description="Buyback price in THB per baht-weight")
    sell: float = Field(..., description="Selling price in THB per baht-weight")


class ThaiRetailGoldResponse(BaseModel):
    source: str = "Gold Traders Association"
    unit: str = "baht-weight (15.244 g, 96.5%)"
    bar: GoldPriceDetailSchema
    ornament: GoldPriceDetailSchema
    announced_at: str
    revision: Optional[int] = None
    is_stale: bool = False
    stale_reason: Optional[str] = None


class MacroSeriesPointSchema(BaseModel):
    date: str
    value: float


class MacroSeriesResponse(BaseModel):
    series_id: str
    label: str
    source: str = "FRED"
    frequency: str
    unit: str
    points: List[MacroSeriesPointSchema]
    is_stale: bool = False
    stale_reason: Optional[str] = None


class LivePerpsQuoteResponse(BaseModel):
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


class MarketValuationResponse(BaseModel):
    market: str
    as_of: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pbv_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    turnover_ratio: Optional[float] = None
    source: str = "Settrade"
    is_stale: bool = False
    stale_reason: Optional[str] = None


class MarketBreadthResponse(BaseModel):
    market: str
    as_of: str
    gainers: int
    losers: int
    unchanged: int
    source: str = "Settrade"
    is_stale: bool = False
    stale_reason: Optional[str] = None


def from_domain_flow(domain: ThaiFundFlowSnapshot) -> ThaiFundFlowResponse:
    return ThaiFundFlowResponse(
        market=domain.market,
        as_of=domain.as_of,
        total_value=domain.total_value,
        investors=[
            InvestorTypeRowSchema(
                investor_type=row.investor_type,
                name_en=row.name_en,
                buy_value=row.buy_value,
                sell_value=row.sell_value,
                net_value=row.net_value,
            )
            for row in domain.investors
        ],
        source=domain.source,
        is_stale=domain.is_stale,
        stale_reason=domain.stale_reason,
    )


def from_domain_gold(domain: ThaiRetailGoldQuote) -> ThaiRetailGoldResponse:
    return ThaiRetailGoldResponse(
        source=domain.source,
        unit=domain.unit,
        bar=GoldPriceDetailSchema(buy=domain.bar.buy, sell=domain.bar.sell),
        ornament=GoldPriceDetailSchema(buy=domain.ornament.buy, sell=domain.ornament.sell),
        announced_at=domain.announced_at,
        revision=domain.revision,
        is_stale=domain.is_stale,
        stale_reason=domain.stale_reason,
    )


def from_domain_macro(domain: MacroSeries) -> MacroSeriesResponse:
    return MacroSeriesResponse(
        series_id=domain.series_id,
        label=domain.label,
        source=domain.source,
        frequency=domain.frequency,
        unit=domain.unit,
        points=[
            MacroSeriesPointSchema(date=p.date, value=p.value)
            for p in domain.points
        ],
        is_stale=domain.is_stale,
        stale_reason=domain.stale_reason,
    )


def from_domain_perps(domain: LivePerpsQuote) -> LivePerpsQuoteResponse:
    return LivePerpsQuoteResponse(
        symbol=domain.symbol,
        mark_price=domain.mark_price,
        dex_namespace=domain.dex_namespace,
        asset_class=domain.asset_class,
        contract_type=domain.contract_type,
        source=domain.source,
        open_interest=domain.open_interest,
        funding_rate=domain.funding_rate,
        day_ntl_vlm=domain.day_ntl_vlm,
        is_stale=domain.is_stale,
        stale_reason=domain.stale_reason,
    )


def from_domain_valuation(domain: MarketValuation) -> MarketValuationResponse:
    return MarketValuationResponse(
        market=domain.market,
        as_of=domain.as_of,
        market_cap=domain.market_cap,
        pe_ratio=domain.pe_ratio,
        pbv_ratio=domain.pbv_ratio,
        dividend_yield=domain.dividend_yield,
        turnover_ratio=domain.turnover_ratio,
        source=domain.source,
        is_stale=domain.is_stale,
        stale_reason=domain.stale_reason,
    )


def from_domain_breadth(domain: MarketBreadth) -> MarketBreadthResponse:
    return MarketBreadthResponse(
        market=domain.market,
        as_of=domain.as_of,
        gainers=domain.gainers,
        losers=domain.losers,
        unchanged=domain.unchanged,
        source=domain.source,
        is_stale=domain.is_stale,
        stale_reason=domain.stale_reason,
    )


# ============================================================================
# Phase 2 Boundary Schemas
# ============================================================================

from tools.market.terminal_v2.domain.models import (
    FinraShortVolumeSnapshot,
    OptionContract,
    OptionsChainSnapshot,
    OptionsMaxPainResult,
    OptionsPutCallRatios,
    PredictionMarketItem,
    ReferenceRatePoint,
    ReferenceRateSnapshot,
    SpotEtfFlowSnapshot,
    ThaiBondMarketStats,
    ThaiCorporateBondIssuance,
    ThaiFundAssetAllocationRow,
    ThaiFundAssetAllocationSnapshot,
    ThaiPublicDebtComponent,
    ThaiPublicDebtSnapshot,
    TreasuryAuctionResult,
    TreasuryYieldCurveSnapshot,
    TreasuryYieldPoint,
    UsNationalDebtSnapshot,
)


class FinraShortVolumeResponse(BaseModel):
    symbol: str
    report_date: str
    short_volume: int
    short_exempt_volume: int
    finra_reported_total_volume: int
    short_pct: Optional[float]
    coverage: str
    unit: str
    source: str
    limitations: str
    is_stale: bool
    stale_reason: Optional[str] = None


class OptionContractSchema(BaseModel):
    occ_symbol: str
    underlying: str
    expiry: str
    strike: float
    side: str
    open_interest: int
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    multiplier: int
    is_standard: bool


class OptionsChainResponse(BaseModel):
    underlying: str
    underlying_price: Optional[float]
    iv30_decimal: Optional[float]
    delay_minutes: int
    contracts: List[OptionContractSchema]
    fetched_at: float
    source: str
    limitations: str
    is_stale: bool
    stale_reason: Optional[str] = None


class OptionsMaxPainResponse(BaseModel):
    underlying: str
    expiry: str
    strike: float
    minimum_theoretical_payout: float
    candidate_count: int
    excluded_contract_count: int
    spot_price: Optional[float] = None
    distance_from_spot: Optional[float] = None
    distance_pct: Optional[float] = None
    assumptions: str
    limitations: str


class OptionsPutCallRatiosResponse(BaseModel):
    underlying: str
    expiry: str
    put_volume: int
    call_volume: int
    volume_ratio: Optional[float]
    put_open_interest: int
    call_open_interest: int
    oi_ratio: Optional[float]


class ReferenceRatePointSchema(BaseModel):
    code: str
    label: str
    effective_date: str
    rate_percent: Optional[float]
    volume_in_billions: Optional[float] = None
    target_rate_from: Optional[float] = None
    target_rate_to: Optional[float] = None


class ReferenceRatesResponse(BaseModel):
    as_of: str
    rates: List[ReferenceRatePointSchema]
    spreads_bps: dict[str, float]
    fetched_at: float
    source: str
    unit: str
    limitations: str
    is_stale: bool
    stale_reason: Optional[str] = None


class TreasuryYieldPointSchema(BaseModel):
    maturity: str
    yield_percent: Optional[float]


class TreasuryYieldCurveResponse(BaseModel):
    observation_date: str
    yields: List[TreasuryYieldPointSchema]
    spread_10y_2y_bps: Optional[float] = None
    spread_10y_3m_bps: Optional[float] = None
    fetched_at: float
    source: str
    unit: str
    is_stale: bool
    stale_reason: Optional[str] = None


class TreasuryAuctionResponse(BaseModel):
    auction_date: str
    issue_date: str
    security_type: str
    security_term: str
    high_yield: Optional[float]
    high_investment_rate: Optional[float] = None
    high_discount_rate: Optional[float] = None
    bid_to_cover_ratio: Optional[float] = None
    offering_amount_usd: Optional[float] = None
    total_accepted_usd: Optional[float] = None
    fetched_at: float
    source: str
    unit: str
    is_stale: bool
    stale_reason: Optional[str] = None


class UsNationalDebtResponse(BaseModel):
    record_date: str
    total_public_debt_usd: float
    debt_held_by_public_usd: Optional[float] = None
    intragovernmental_holdings_usd: Optional[float] = None
    is_daily_close: bool
    fetched_at: float
    source: str
    unit: str
    limitations: str
    is_stale: bool
    stale_reason: Optional[str] = None


class ThaiFundAssetAllocationRowSchema(BaseModel):
    asset_class: str
    domestic_or_foreign: str
    value_thb: float
    share_of_nav_pct: Optional[float] = None


class ThaiFundAssetAllocationResponse(BaseModel):
    reporting_period: str
    total_nav_thb: Optional[float]
    allocations: List[ThaiFundAssetAllocationRowSchema]
    fetched_at: float
    source: str
    unit: str
    limitations: str
    is_stale: bool
    stale_reason: Optional[str] = None


class ThaiBondMarketStatsResponse(BaseModel):
    reporting_period: str
    outstanding_thb: float
    trading_value_thb: float
    foreign_holding_thb: float
    foreign_holding_pct: Optional[float]
    fetched_at: float
    source: str
    unit: str
    is_stale: bool
    stale_reason: Optional[str] = None


class ThaiCorporateBondIssuanceResponse(BaseModel):
    reporting_period: str
    total_offering_thb: float
    long_term_thb: float
    short_term_thb: float
    top_sectors: List[List[Any]]
    fetched_at: float
    source: str
    unit: str
    is_stale: bool
    stale_reason: Optional[str] = None


class ThaiPublicDebtComponentSchema(BaseModel):
    component_number: int
    label_en: str
    label_th: str
    amount_thb: float


class ThaiPublicDebtResponse(BaseModel):
    reporting_month: str
    total_debt_thb: float
    debt_to_gdp_pct: Optional[float]
    fx_rate_usd_thb: Optional[float]
    components: List[ThaiPublicDebtComponentSchema]
    fetched_at: float
    source: str
    unit: str
    limitations: str
    is_stale: bool
    stale_reason: Optional[str] = None


class PredictionOutcomeSchema(BaseModel):
    label: str
    price: float


class PredictionMarketResponse(BaseModel):
    market_id: str
    question: str
    outcomes: List[PredictionOutcomeSchema]
    volume_24h_usd: Optional[float] = None
    end_date: Optional[str] = None
    source_url: str
    fetched_at: float
    source: str
    unit: str
    limitations: str
    is_stale: bool
    stale_reason: Optional[str] = None


class SpotEtfIssuerFlowSchema(BaseModel):
    ticker: str
    institute: str
    daily_net_inflow_usd: Optional[float] = None
    cumulative_net_inflow_usd: Optional[float] = None
    total_net_assets_usd: Optional[float] = None


class SpotEtfFlowResponse(BaseModel):
    asset: str
    report_date: str
    daily_total_usd: Optional[float] = None
    cumulative_total_usd: Optional[float] = None
    issuers: List[SpotEtfIssuerFlowSchema]
    is_partial: bool
    completeness_notes: str
    fetched_at: float
    source: str
    unit: str
    limitations: str
    is_stale: bool
    stale_reason: Optional[str] = None


# ============================================================================
# Phase 2 Mappers
# ============================================================================

def from_domain_short_volume(d: FinraShortVolumeSnapshot) -> FinraShortVolumeResponse:
    return FinraShortVolumeResponse(
        symbol=d.symbol,
        report_date=d.report_date,
        short_volume=d.short_volume,
        short_exempt_volume=d.short_exempt_volume,
        finra_reported_total_volume=d.finra_reported_total_volume,
        short_pct=d.short_pct,
        coverage=d.coverage,
        unit=d.unit,
        source=d.source,
        limitations=d.limitations,
        is_stale=d.is_stale,
        stale_reason=d.stale_reason,
    )


def from_domain_options_chain(d: OptionsChainSnapshot) -> OptionsChainResponse:
    return OptionsChainResponse(
        underlying=d.underlying,
        underlying_price=d.underlying_price,
        iv30_decimal=d.iv30_decimal,
        delay_minutes=d.delay_minutes,
        contracts=[
            OptionContractSchema(
                occ_symbol=c.occ_symbol,
                underlying=c.underlying,
                expiry=c.expiry,
                strike=c.strike,
                side=c.side,
                open_interest=c.open_interest,
                volume=c.volume,
                bid=c.bid,
                ask=c.ask,
                last_price=c.last_price,
                implied_volatility=c.implied_volatility,
                delta=c.delta,
                gamma=c.gamma,
                vega=c.vega,
                theta=c.theta,
                rho=c.rho,
                multiplier=c.multiplier,
                is_standard=c.is_standard,
            )
            for c in d.contracts
        ],
        fetched_at=d.fetched_at,
        source=d.source,
        limitations=d.limitations,
        is_stale=d.is_stale,
        stale_reason=d.stale_reason,
    )


def from_domain_max_pain(d: OptionsMaxPainResult) -> OptionsMaxPainResponse:
    return OptionsMaxPainResponse(
        underlying=d.underlying,
        expiry=d.expiry,
        strike=d.strike,
        minimum_theoretical_payout=d.minimum_theoretical_payout,
        candidate_count=d.candidate_count,
        excluded_contract_count=d.excluded_contract_count,
        spot_price=d.spot_price,
        distance_from_spot=d.distance_from_spot,
        distance_pct=d.distance_pct,
        assumptions=d.assumptions,
        limitations=d.limitations,
    )


def from_domain_put_call(d: OptionsPutCallRatios) -> OptionsPutCallRatiosResponse:
    return OptionsPutCallRatiosResponse(
        underlying=d.underlying,
        expiry=d.expiry,
        put_volume=d.put_volume,
        call_volume=d.call_volume,
        volume_ratio=d.volume_ratio,
        put_open_interest=d.put_open_interest,
        call_open_interest=d.call_open_interest,
        oi_ratio=d.oi_ratio,
    )


def from_domain_reference_rates(d: ReferenceRateSnapshot) -> ReferenceRatesResponse:
    return ReferenceRatesResponse(
        as_of=d.as_of,
        rates=[
            ReferenceRatePointSchema(
                code=p.code,
                label=p.label,
                effective_date=p.effective_date,
                rate_percent=p.rate_percent,
                volume_in_billions=p.volume_in_billions,
                target_rate_from=p.target_rate_from,
                target_rate_to=p.target_rate_to,
            )
            for p in d.rates
        ],
        spreads_bps=d.spreads_bps,
        fetched_at=d.fetched_at,
        source=d.source,
        unit=d.unit,
        limitations=d.limitations,
        is_stale=d.is_stale,
        stale_reason=d.stale_reason,
    )


def from_domain_yield_curve(d: TreasuryYieldCurveSnapshot) -> TreasuryYieldCurveResponse:
    return TreasuryYieldCurveResponse(
        observation_date=d.observation_date,
        yields=[
            TreasuryYieldPointSchema(maturity=y.maturity, yield_percent=y.yield_percent)
            for y in d.yields
        ],
        spread_10y_2y_bps=d.spread_10y_2y_bps,
        spread_10y_3m_bps=d.spread_10y_3m_bps,
        fetched_at=d.fetched_at,
        source=d.source,
        unit=d.unit,
        is_stale=d.is_stale,
        stale_reason=d.stale_reason,
    )


def from_domain_auction(d: TreasuryAuctionResult) -> TreasuryAuctionResponse:
    return TreasuryAuctionResponse(
        auction_date=d.auction_date,
        issue_date=d.issue_date,
        security_type=d.security_type,
        security_term=d.security_term,
        high_yield=d.high_yield,
        high_investment_rate=d.high_investment_rate,
        high_discount_rate=d.high_discount_rate,
        bid_to_cover_ratio=d.bid_to_cover_ratio,
        offering_amount_usd=d.offering_amount_usd,
        total_accepted_usd=d.total_accepted_usd,
        fetched_at=d.fetched_at,
        source=d.source,
        unit=d.unit,
        is_stale=d.is_stale,
        stale_reason=d.stale_reason,
    )


def from_domain_debt(d: UsNationalDebtSnapshot) -> UsNationalDebtResponse:
    return UsNationalDebtResponse(
        record_date=d.record_date,
        total_public_debt_usd=d.total_public_debt_usd,
        debt_held_by_public_usd=d.debt_held_by_public_usd,
        intragovernmental_holdings_usd=d.intragovernmental_holdings_usd,
        is_daily_close=d.is_daily_close,
        fetched_at=d.fetched_at,
        source=d.source,
        unit=d.unit,
        limitations=d.limitations,
        is_stale=d.is_stale,
        stale_reason=d.stale_reason,
    )


def from_domain_fund_allocation(d: ThaiFundAssetAllocationSnapshot) -> ThaiFundAssetAllocationResponse:
    return ThaiFundAssetAllocationResponse(
        reporting_period=d.reporting_period,
        total_nav_thb=d.total_nav_thb,
        allocations=[
            ThaiFundAssetAllocationRowSchema(
                asset_class=a.asset_class,
                domestic_or_foreign=a.domestic_or_foreign,
                value_thb=a.value_thb,
                share_of_nav_pct=a.share_of_nav_pct,
            )
            for a in d.allocations
        ],
        fetched_at=d.fetched_at,
        source=d.source,
        unit=d.unit,
        limitations=d.limitations,
        is_stale=d.is_stale,
        stale_reason=d.stale_reason,
    )


def from_domain_bond_stats(d: ThaiBondMarketStats) -> ThaiBondMarketStatsResponse:
    return ThaiBondMarketStatsResponse(
        reporting_period=d.reporting_period,
        outstanding_thb=d.outstanding_thb,
        trading_value_thb=d.trading_value_thb,
        foreign_holding_thb=d.foreign_holding_thb,
        foreign_holding_pct=d.foreign_holding_pct,
        fetched_at=d.fetched_at,
        source=d.source,
        unit=d.unit,
        is_stale=d.is_stale,
        stale_reason=d.stale_reason,
    )


def from_domain_bond_issuance(d: ThaiCorporateBondIssuance) -> ThaiCorporateBondIssuanceResponse:
    return ThaiCorporateBondIssuanceResponse(
        reporting_period=d.reporting_period,
        total_offering_thb=d.total_offering_thb,
        long_term_thb=d.long_term_thb,
        short_term_thb=d.short_term_thb,
        top_sectors=[list(item) for item in d.top_sectors],
        fetched_at=d.fetched_at,
        source=d.source,
        unit=d.unit,
        is_stale=d.is_stale,
        stale_reason=d.stale_reason,
    )


def from_domain_public_debt(d: ThaiPublicDebtSnapshot) -> ThaiPublicDebtResponse:
    return ThaiPublicDebtResponse(
        reporting_month=d.reporting_month,
        total_debt_thb=d.total_debt_thb,
        debt_to_gdp_pct=d.debt_to_gdp_pct,
        fx_rate_usd_thb=d.fx_rate_usd_thb,
        components=[
            ThaiPublicDebtComponentSchema(
                component_number=c.component_number,
                label_en=c.label_en,
                label_th=c.label_th,
                amount_thb=c.amount_thb,
            )
            for c in d.components
        ],
        fetched_at=d.fetched_at,
        source=d.source,
        unit=d.unit,
        limitations=d.limitations,
        is_stale=d.is_stale,
        stale_reason=d.stale_reason,
    )


def from_domain_prediction_market(d: PredictionMarketItem) -> PredictionMarketResponse:
    return PredictionMarketResponse(
        market_id=d.market_id,
        question=d.question,
        outcomes=[PredictionOutcomeSchema(label=o.label, price=o.price) for o in d.outcomes],
        volume_24h_usd=d.volume_24h_usd,
        end_date=d.end_date,
        source_url=d.source_url,
        fetched_at=d.fetched_at,
        source=d.source,
        unit=d.unit,
        limitations=d.limitations,
        is_stale=d.is_stale,
        stale_reason=d.stale_reason,
    )


def from_domain_etf_flows(d: SpotEtfFlowSnapshot) -> SpotEtfFlowResponse:
    return SpotEtfFlowResponse(
        asset=d.asset,
        report_date=d.report_date,
        daily_total_usd=d.daily_total_usd,
        cumulative_total_usd=d.cumulative_total_usd,
        issuers=[
            SpotEtfIssuerFlowSchema(
                ticker=i.ticker,
                institute=i.institute,
                daily_net_inflow_usd=i.daily_net_inflow_usd,
                cumulative_net_inflow_usd=i.cumulative_net_inflow_usd,
                total_net_assets_usd=i.total_net_assets_usd,
            )
            for i in d.issuers
        ],
        is_partial=d.is_partial,
        completeness_notes=d.completeness_notes,
        fetched_at=d.fetched_at,
        source=d.source,
        unit=d.unit,
        limitations=d.limitations,
        is_stale=d.is_stale,
        stale_reason=d.stale_reason,
    )


# ============================================================================
# Phase 3 Boundary Schemas & Mappers
# ============================================================================

from typing import Literal
from tools.market.terminal_v2.domain.calculations import classify_fsi_regime
from tools.market.terminal_v2.domain.models import (
    FinancialStressSnapshot,
    GlobalPolicyRateSnapshot,
    MetalsCotPositioningSnapshot,
    NasdaqEarningsConsensusSnapshot,
)


class FinancialStressPointSchema(BaseModel):
    time_ms: int
    date: str
    value: float
    credit: Optional[float] = None
    equity_valuation: Optional[float] = None
    safe_assets: Optional[float] = None
    funding: Optional[float] = None
    volatility: Optional[float] = None


class FinancialStressCategorySchema(BaseModel):
    label: str
    value: float


class FinancialStressResponse(BaseModel):
    as_of_date: str = Field(description="T-2 business days lag observation date")
    published_at: str
    fsi_value: float
    regime: str = Field(description="Systemic risk regime: calm, normal, elevated, severe")
    categories: List[FinancialStressCategorySchema]
    trend_90d: List[FinancialStressPointSchema]
    source: str = "OFR"
    data_lag_days: int = 2
    is_stale: bool = False


class TraderClassPositionSchema(BaseModel):
    class_name: str
    long_contracts: int
    short_contracts: int
    net_contracts: int
    spread_contracts: int = 0
    change_long: int = 0
    change_short: int = 0


class MetalsCotResponse(BaseModel):
    commodity: str
    commodity_code: str
    as_of_date: str = Field(description="Tuesday market close")
    published_at: str = Field(description="Friday afternoon release")
    report_type: str = "disaggregated"
    open_interest: int
    managed_money: TraderClassPositionSchema
    swap_dealers: TraderClassPositionSchema
    producer_merchant: TraderClassPositionSchema
    other_reportables: TraderClassPositionSchema
    non_reportables: TraderClassPositionSchema
    net_managed_money: int
    percentile_52w: float
    source: str = "CFTC"
    is_stale: bool = False


class PolicyRateItemSchema(BaseModel):
    country: str
    rate_value: float
    rate_type: str = Field(description="Specific central bank rate instrument")
    effective_date: str = Field(description="Date rate took effect")
    currency: str
    central_bank: str
    previous_rate: Optional[float] = None
    last_change_date: Optional[str] = None


class GlobalPolicyRatesResponse(BaseModel):
    as_of_date: str
    rates: List[PolicyRateItemSchema]
    spreads_vs_bot_repo: Dict[str, float] = Field(description="Spreads in basis points vs BOT 1-day repo")
    source: str = "BIS"
    is_stale: bool = False


class EarningsDateItemSchema(BaseModel):
    earnings_date: str
    date_status: Literal["confirmed", "estimated", "unspecified"]
    report_time: Literal["pre-market", "after-hours", "unknown"]
    consensus_eps: Optional[float] = None
    estimate_count: Optional[int] = None


class EarningsSurpriseItemSchema(BaseModel):
    fiscal_quarter_end: str
    date_reported: str
    eps: float
    consensus_eps: float
    surprise_pct: float


class AnalystRatingConsensusSchema(BaseModel):
    symbol: str
    consensus: str
    analyst_count: int
    broker_names: List[str]


class NasdaqConsensusResponse(BaseModel):
    symbol: str
    coverage_status: Literal["full", "partial", "no_coverage"]
    has_earnings_surprise: bool
    has_analyst_ratings: bool
    upcoming_earnings: Optional[EarningsDateItemSchema] = None
    surprise_history: List[EarningsSurpriseItemSchema]
    ratings: Optional[AnalystRatingConsensusSchema] = None
    source: str = "Nasdaq"
    is_stale: bool = False


def map_fsi_to_response(snap: FinancialStressSnapshot) -> FinancialStressResponse:
    regime = classify_fsi_regime(snap.fsi_value)
    cats = [FinancialStressCategorySchema(label=c.label, value=c.value) for c in snap.categories]
    pts = [
        FinancialStressPointSchema(
            time_ms=p.time_ms,
            date=p.date,
            value=p.value,
            credit=p.credit,
            equity_valuation=p.equity_valuation,
            safe_assets=p.safe_assets,
            funding=p.funding,
            volatility=p.volatility,
        )
        for p in snap.trend_90d
    ]
    return FinancialStressResponse(
        as_of_date=snap.as_of_date,
        published_at=snap.published_at,
        fsi_value=snap.fsi_value,
        regime=regime,
        categories=cats,
        trend_90d=pts,
        source=snap.source,
        data_lag_days=snap.data_lag_days,
        is_stale=snap.is_stale,
    )


def map_cot_to_response(snap: MetalsCotPositioningSnapshot) -> MetalsCotResponse:
    def _map_pos(p) -> TraderClassPositionSchema:
        return TraderClassPositionSchema(
            class_name=p.class_name,
            long_contracts=p.long_contracts,
            short_contracts=p.short_contracts,
            net_contracts=p.net_contracts,
            spread_contracts=p.spread_contracts,
            change_long=p.change_long,
            change_short=p.change_short,
        )

    return MetalsCotResponse(
        commodity=snap.commodity,
        commodity_code=snap.commodity_code,
        as_of_date=snap.as_of_date,
        published_at=snap.published_at,
        report_type=snap.report_type,
        open_interest=snap.open_interest,
        managed_money=_map_pos(snap.managed_money),
        swap_dealers=_map_pos(snap.swap_dealers),
        producer_merchant=_map_pos(snap.producer_merchant),
        other_reportables=_map_pos(snap.other_reportables),
        non_reportables=_map_pos(snap.non_reportables),
        net_managed_money=snap.net_managed_money,
        percentile_52w=snap.percentile_52w,
        source=snap.source,
        is_stale=snap.is_stale,
    )


def map_bis_rates_to_response(snap: GlobalPolicyRateSnapshot) -> GlobalPolicyRatesResponse:
    rate_schemas = [
        PolicyRateItemSchema(
            country=r.country,
            rate_value=r.rate_value,
            rate_type=r.rate_type,
            effective_date=r.effective_date,
            currency=r.currency,
            central_bank=r.central_bank,
            previous_rate=r.previous_rate,
            last_change_date=r.last_change_date,
        )
        for r in snap.rates
    ]
    return GlobalPolicyRatesResponse(
        as_of_date=snap.as_of_date,
        rates=rate_schemas,
        spreads_vs_bot_repo=snap.spreads_vs_bot_repo,
        source=snap.source,
        is_stale=snap.is_stale,
    )


def map_nasdaq_to_response(snap: NasdaqEarningsConsensusSnapshot) -> NasdaqConsensusResponse:
    upcoming = None
    if snap.upcoming_earnings:
        upcoming = EarningsDateItemSchema(
            earnings_date=snap.upcoming_earnings.earnings_date,
            date_status=snap.upcoming_earnings.date_status,
            report_time=snap.upcoming_earnings.report_time,
            consensus_eps=snap.upcoming_earnings.consensus_eps,
            estimate_count=snap.upcoming_earnings.estimate_count,
        )

    surprises = [
        EarningsSurpriseItemSchema(
            fiscal_quarter_end=s.fiscal_quarter_end,
            date_reported=s.date_reported,
            eps=s.eps,
            consensus_eps=s.consensus_eps,
            surprise_pct=s.surprise_pct,
        )
        for s in snap.surprise_history
    ]

    ratings = None
    if snap.ratings:
        ratings = AnalystRatingConsensusSchema(
            symbol=snap.ratings.symbol,
            consensus=snap.ratings.consensus,
            analyst_count=snap.ratings.analyst_count,
            broker_names=list(snap.ratings.broker_names),
        )

    return NasdaqConsensusResponse(
        symbol=snap.symbol,
        coverage_status=snap.coverage_status,
        has_earnings_surprise=snap.has_earnings_surprise,
        has_analyst_ratings=snap.has_analyst_ratings,
        upcoming_earnings=upcoming,
        surprise_history=surprises,
        ratings=ratings,
        source=snap.source,
        is_stale=snap.is_stale,
    )


# ============================================================================
# Phase 4 Boundary Schemas & Mappers (Non-Crypto)
# ============================================================================

from tools.market.terminal_v2.domain.models import (
    AuctionDemandSnapshot,
    CommodityVolSnapshot,
    NewsDiscoverySnapshot,
    SecCompanyFactsSnapshot,
    SecInsiderTradeSnapshot,
)


class CommodityVolSnapshotSchema(BaseModel):
    index_symbol: str = Field(..., description="Volatility index symbol (GVZ, VXSLV, OVX)")
    underlying_instrument: str = Field(..., description="Underlying ETF options instrument name")
    close_date: str = Field(..., description="Close date in YYYY-MM-DD")
    implied_volatility: float = Field(..., description="30-day annualized implied volatility percentage")
    change_1d_points: Optional[float] = Field(None, description="1-day change in index points")
    percentile_52w: Optional[float] = Field(None, description="52-week percentile (0-100) or null if < 100 samples")
    sample_count: int = Field(..., description="Sample count used for percentile")
    regime_label: Optional[str] = Field(None, description="Statistical regime heuristic (extreme_panic, elevated, normal, complacent)")
    source: str = Field(default="Cboe")
    as_of_date: str
    fetched_at: float
    is_stale: bool = False
    stale_reason: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)


class AuctionDemandSnapshotSchema(BaseModel):
    security_type: str = Field(..., description="Security type (Note, Bill, Bond)")
    security_term: str = Field(..., description="Security term (10-Year, 13-Week, etc.)")
    latest_auction_date: str
    latest_bid_to_cover_ratio: Optional[float] = Field(None, description="Latest auction bid-to-cover ratio")
    latest_high_yield: Optional[float] = Field(None, description="Latest high yield awarded (Note/Bond)")
    latest_high_investment_rate: Optional[float] = Field(None, description="Latest investment rate (Bill)")
    latest_high_discount_rate: Optional[float] = Field(None, description="Latest discount rate (Bill)")
    latest_offering_amount_usd: Optional[float] = None
    latest_total_accepted_usd: Optional[float] = None
    prior_mean_bid_to_cover: Optional[float] = Field(None, description="Moving average of prior 8 completed auctions of same type/term")
    demand_delta: Optional[float] = Field(None, description="latest_btc - prior_mean")
    sample_count: int = Field(..., description="Sample count of prior completed auctions (min 3)")
    source: str = Field(default="US Treasury Fiscal Data")
    as_of_date: str
    fetched_at: float
    is_stale: bool = False
    stale_reason: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)


class SecFactSchema(BaseModel):
    concept_tag: str
    label: str
    val: Optional[float] = None
    unit: str
    form: str
    fy: Optional[int] = None
    fp: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    filed: Optional[str] = None
    accn: Optional[str] = None


class SecCompanyFactsSnapshotSchema(BaseModel):
    symbol: str
    cik: str
    entity_name: str
    facts: List[SecFactSchema] = Field(default_factory=list)
    revenue_usd: Optional[float] = None
    operating_cash_flow_usd: Optional[float] = None
    capex_usd: Optional[float] = None
    free_cash_flow_usd: Optional[float] = None
    free_cash_flow_margin: Optional[float] = None
    long_term_debt_usd: Optional[float] = None
    debt_to_ocf_ratio: Optional[float] = None
    shares_outstanding: Optional[float] = None
    source: str = Field(default="SEC EDGAR (companyfacts)")
    as_of_date: str
    fetched_at: float
    is_stale: bool = False
    stale_reason: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)


class InsiderTransactionSchema(BaseModel):
    transaction_date: str
    reporting_owner: str
    officer_title: Optional[str] = None
    is_officer: bool
    is_director: bool
    is_ten_percent_owner: bool
    transaction_code: str
    shares: Optional[float] = None
    price_per_share: Optional[float] = None
    notional_usd: Optional[float] = None
    direct_or_indirect: str
    accession_number: str
    is_amendment: bool


class SecInsiderTradeSnapshotSchema(BaseModel):
    symbol: str
    cik: str
    transactions: List[InsiderTransactionSchema] = Field(default_factory=list)
    net_buy_ratio_90d: Optional[float] = None
    p_notional_sum_90d: float = 0.0
    s_notional_sum_90d: float = 0.0
    eligible_transaction_count: int = 0
    source: str = Field(default="SEC EDGAR (Form 4 XML)")
    as_of_date: str
    fetched_at: float
    is_stale: bool = False
    stale_reason: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)


class NewsCandidateSchema(BaseModel):
    headline: str
    publisher: str
    source_type: str
    article_url: str
    published_at: str
    discovered_at: float
    symbol: Optional[str] = None
    is_stale: bool = False


class NewsDiscoverySnapshotSchema(BaseModel):
    query_symbol: str
    items: List[NewsCandidateSchema] = Field(default_factory=list)
    status: str = Field(..., description="ok, rate_limited, or feed_unavailable")
    source: str = Field(default="Google News RSS Discovery")
    as_of_date: str
    fetched_at: float
    limitations: List[str] = Field(default_factory=list)


def map_commodity_vol_to_schema(snap: CommodityVolSnapshot) -> CommodityVolSnapshotSchema:
    return CommodityVolSnapshotSchema(
        index_symbol=snap.index_symbol,
        underlying_instrument=snap.underlying_instrument,
        close_date=snap.close_date,
        implied_volatility=snap.implied_volatility,
        change_1d_points=snap.change_1d_points,
        percentile_52w=snap.percentile_52w,
        sample_count=snap.sample_count,
        regime_label=snap.regime_label,
        source=snap.source,
        as_of_date=snap.as_of_date,
        fetched_at=snap.fetched_at,
        is_stale=snap.is_stale,
        stale_reason=snap.stale_reason,
        limitations=list(snap.limitations),
    )


def map_auction_demand_to_schema(snap: AuctionDemandSnapshot) -> AuctionDemandSnapshotSchema:
    return AuctionDemandSnapshotSchema(
        security_type=snap.security_type,
        security_term=snap.security_term,
        latest_auction_date=snap.latest_auction_date,
        latest_bid_to_cover_ratio=snap.latest_bid_to_cover_ratio,
        latest_high_yield=snap.latest_high_yield,
        latest_high_investment_rate=snap.latest_high_investment_rate,
        latest_high_discount_rate=snap.latest_high_discount_rate,
        latest_offering_amount_usd=snap.latest_offering_amount_usd,
        latest_total_accepted_usd=snap.latest_total_accepted_usd,
        prior_mean_bid_to_cover=snap.prior_mean_bid_to_cover,
        demand_delta=snap.demand_delta,
        sample_count=snap.sample_count,
        source=snap.source,
        as_of_date=snap.as_of_date,
        fetched_at=snap.fetched_at,
        is_stale=snap.is_stale,
        stale_reason=snap.stale_reason,
        limitations=list(snap.limitations),
    )


def map_sec_financials_to_schema(snap: SecCompanyFactsSnapshot) -> SecCompanyFactsSnapshotSchema:
    return SecCompanyFactsSnapshotSchema(
        symbol=snap.symbol,
        cik=snap.cik,
        entity_name=snap.entity_name,
        facts=[
            SecFactSchema(
                concept_tag=f.concept_tag,
                label=f.label,
                val=f.val,
                unit=f.unit,
                form=f.form,
                fy=f.fy,
                fp=f.fp,
                start=f.start,
                end=f.end,
                filed=f.filed,
                accn=f.accn,
            )
            for f in snap.facts
        ],
        revenue_usd=snap.revenue_usd,
        operating_cash_flow_usd=snap.operating_cash_flow_usd,
        capex_usd=snap.capex_usd,
        free_cash_flow_usd=snap.free_cash_flow_usd,
        free_cash_flow_margin=snap.free_cash_flow_margin,
        long_term_debt_usd=snap.long_term_debt_usd,
        debt_to_ocf_ratio=snap.debt_to_ocf_ratio,
        shares_outstanding=snap.shares_outstanding,
        source=snap.source,
        as_of_date=snap.as_of_date,
        fetched_at=snap.fetched_at,
        is_stale=snap.is_stale,
        stale_reason=snap.stale_reason,
        limitations=list(snap.limitations),
    )


def map_sec_insider_trades_to_schema(snap: SecInsiderTradeSnapshot) -> SecInsiderTradeSnapshotSchema:
    return SecInsiderTradeSnapshotSchema(
        symbol=snap.symbol,
        cik=snap.cik,
        transactions=[
            InsiderTransactionSchema(
                transaction_date=t.transaction_date,
                reporting_owner=t.reporting_owner,
                officer_title=t.officer_title,
                is_officer=t.is_officer,
                is_director=t.is_director,
                is_ten_percent_owner=t.is_ten_percent_owner,
                transaction_code=t.transaction_code,
                shares=t.shares,
                price_per_share=t.price_per_share,
                notional_usd=t.notional_usd,
                direct_or_indirect=t.direct_or_indirect,
                accession_number=t.accession_number,
                is_amendment=t.is_amendment,
            )
            for t in snap.transactions
        ],
        net_buy_ratio_90d=snap.net_buy_ratio_90d,
        p_notional_sum_90d=snap.p_notional_sum_90d,
        s_notional_sum_90d=snap.s_notional_sum_90d,
        eligible_transaction_count=snap.eligible_transaction_count,
        source=snap.source,
        as_of_date=snap.as_of_date,
        fetched_at=snap.fetched_at,
        is_stale=snap.is_stale,
        stale_reason=snap.stale_reason,
        limitations=list(snap.limitations),
    )


def map_news_discovery_to_schema(snap: NewsDiscoverySnapshot) -> NewsDiscoverySnapshotSchema:
    return NewsDiscoverySnapshotSchema(
        query_symbol=snap.query_symbol,
        items=[
            NewsCandidateSchema(
                headline=item.headline,
                publisher=item.publisher,
                source_type=item.source_type,
                article_url=item.article_url,
                published_at=item.published_at,
                discovered_at=item.discovered_at,
                symbol=item.symbol,
                is_stale=item.is_stale,
            )
            for item in snap.items
        ],
        status=snap.status,
        source=snap.source,
        as_of_date=snap.as_of_date,
        fetched_at=snap.fetched_at,
        limitations=list(snap.limitations),
    )

