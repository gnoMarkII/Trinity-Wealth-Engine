"""FastAPI Inbound Router for Terminal V2 Market Data (/api/v2/market/...).

Exposes keyless market data and institutional data engine endpoints backed by
Terminal V2 Hexagonal Architecture.
Completely decoupled from legacy V1 routes.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.schemas.terminal_v2_schemas import (
    FinraShortVolumeResponse,
    LivePerpsQuoteResponse,
    MacroSeriesResponse,
    MarketBreadthResponse,
    MarketValuationResponse,
    OptionsChainResponse,
    OptionsMaxPainResponse,
    OptionsPutCallRatiosResponse,
    PredictionMarketResponse,
    CryptoBenchmarkResponse,
    CryptoMacroLiquidityResponse,
    ReferenceRatesResponse,
    SpotEtfFlowResponse,
    StablecoinSupplyResponse,
    ThaiBondMarketStatsResponse,
    ThaiCorporateBondIssuanceResponse,
    ThaiFundAssetAllocationResponse,
    ThaiFundFlowResponse,
    ThaiPublicDebtResponse,
    ThaiRetailGoldResponse,
    ThaiYieldCurveResponse,
    TreasuryAuctionResponse,
    TreasuryYieldCurveResponse,
    UsNationalDebtResponse,
    from_domain_auction,
    from_domain_bond_issuance,
    from_domain_bond_stats,
    from_domain_breadth,
    from_domain_crypto_benchmark,
    from_domain_crypto_liquidity,
    from_domain_debt,
    from_domain_etf_flows,
    from_domain_flow,
    from_domain_fund_allocation,
    from_domain_gold,
    from_domain_macro,
    from_domain_max_pain,
    from_domain_options_chain,
    from_domain_perps,
    from_domain_prediction_market,
    from_domain_public_debt,
    from_domain_put_call,
    from_domain_reference_rates,
    from_domain_short_volume,
    from_domain_stablecoins,
    from_domain_thai_yield_curve,
    from_domain_valuation,
    from_domain_yield_curve,
    FinancialStressResponse,
    GlobalPolicyRatesResponse,
    MetalsCotResponse,
    NasdaqConsensusResponse,
    map_bis_rates_to_response,
    map_cot_to_response,
    map_fsi_to_response,
    map_nasdaq_to_response,
    AuctionDemandSnapshotSchema,
    CommodityVolSnapshotSchema,
    NewsDiscoverySnapshotSchema,
    SecCompanyFactsSnapshotSchema,
    SecInsiderTradeSnapshotSchema,
    map_auction_demand_to_schema,
    map_commodity_vol_to_schema,
    map_news_discovery_to_schema,
    map_sec_financials_to_schema,
    map_sec_insider_trades_to_schema,
)
from tools.market.terminal_v2.bootstrap import (
    get_terminal_data_service,
    get_terminal_service,
)
from tools.market.terminal_v2.domain.errors import (
    DataUnavailableError,
    InvalidCapabilityError,
    ProviderError,
    SymbolMarketMismatchError,
)
from tools.market.terminal_v2.ports.driving_ports import (
    MarketTerminalServicePort,
    TerminalDataServicePort,
)

router = APIRouter(prefix="/api/v2/market", tags=["Terminal V2 Market Data"])


# ============================================================================
# Phase 1 Routes
# ============================================================================

@router.get(
    "/thailand/flow",
    response_model=ThaiFundFlowResponse,
    summary="Fetch 4-investor-type daily net trading flow on SET/mai",
)
def get_thai_investor_flow(
    market: str = Query("SET", description="SET or mai"),
    service: MarketTerminalServicePort = Depends(get_terminal_service),
):
    try:
        domain_flow = service.get_investor_flow(market=market)
        return from_domain_flow(domain_flow)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/thailand/gold",
    response_model=ThaiRetailGoldResponse,
    summary="Fetch official Thai retail gold prices from Gold Traders Association",
)
def get_thai_retail_gold(
    service: MarketTerminalServicePort = Depends(get_terminal_service),
):
    try:
        domain_gold = service.get_retail_gold()
        return from_domain_gold(domain_gold)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/macro/series/{series_id}",
    response_model=MacroSeriesResponse,
    summary="Fetch macroeconomic time series from keyless FRED mirror",
)
def get_macro_series(
    series_id: str,
    limit: int = Query(30, ge=1, le=500),
    service: MarketTerminalServicePort = Depends(get_terminal_service),
):
    try:
        domain_series = service.get_macro_series(series_id=series_id, limit=limit)
        return from_domain_macro(domain_series)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/perps/quote/{symbol:path}",
    response_model=LivePerpsQuoteResponse,
    summary="Fetch synthetic perpetuals quote from Hyperliquid L1 clearinghouse",
)
def get_perps_quote(
    symbol: str,
    service: MarketTerminalServicePort = Depends(get_terminal_service),
):
    try:
        domain_perp = service.query_by_capability(capability="perps_quote", symbol=symbol)
        return from_domain_perps(domain_perp)
    except SymbolMarketMismatchError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/thailand/valuation",
    response_model=MarketValuationResponse,
    summary="Fetch venue aggregate valuation multiples from Settrade",
)
def get_market_valuation(
    market: str = Query("SET", description="SET or mai"),
    service: MarketTerminalServicePort = Depends(get_terminal_service),
):
    try:
        domain_val = service.get_market_valuation(market=market)
        return from_domain_valuation(domain_val)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/thailand/breadth",
    response_model=MarketBreadthResponse,
    summary="Fetch market breadth (gainers/losers/unchanged) from Settrade",
)
def get_market_breadth(
    market: str = Query("SET", description="SET or mai"),
    service: MarketTerminalServicePort = Depends(get_terminal_service),
):
    try:
        domain_breadth = service.get_market_breadth(market=market)
        return from_domain_breadth(domain_breadth)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


# ============================================================================
# Phase 2 Routes (Institutional Data Engine)
# ============================================================================

@router.get(
    "/equity/short-volume/{symbol}",
    response_model=FinraShortVolumeResponse,
    summary="Fetch FINRA consolidated daily short sale volume",
)
def get_equity_short_volume(
    symbol: str,
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        res_map = service.get_short_volume([symbol])
        if symbol not in res_map and symbol.upper() not in res_map:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No FINRA short volume reported for symbol '{symbol}'",
            )
        entry = res_map.get(symbol) or res_map[symbol.upper()]
        return from_domain_short_volume(entry)
    except HTTPException:
        raise
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/equity/options/{symbol}",
    response_model=OptionsChainResponse,
    summary="Fetch delayed listed equity options chain from Cboe",
)
def get_equity_options_chain(
    symbol: str,
    expiry: Optional[str] = Query(None, description="Filter for specific expiry ISO YYYY-MM-DD"),
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        chain = service.get_options_chain(symbol)
        if expiry:
            filtered_contracts = tuple(c for c in chain.contracts if c.expiry == expiry.strip())
            chain = OptionsChainSnapshot(
                underlying=chain.underlying,
                underlying_price=chain.underlying_price,
                iv30_decimal=chain.iv30_decimal,
                delay_minutes=chain.delay_minutes,
                contracts=filtered_contracts,
                fetched_at=chain.fetched_at,
                source=chain.source,
                is_stale=chain.is_stale,
                stale_reason=chain.stale_reason,
            )
        return from_domain_options_chain(chain)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/equity/max-pain/{symbol}",
    response_model=OptionsMaxPainResponse,
    summary="Calculate analytical Max Pain strike for a specific or nearest expiry",
)
def get_equity_max_pain(
    symbol: str,
    expiry: Optional[str] = Query(None, description="Expiry ISO YYYY-MM-DD (defaults to nearest)"),
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        res = service.get_options_max_pain(symbol=symbol, expiry=expiry)
        return from_domain_max_pain(res)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/equity/put-call-ratios/{symbol}",
    response_model=OptionsPutCallRatiosResponse,
    summary="Calculate Put/Call volume and OI ratios for a specific or nearest expiry",
)
def get_equity_put_call_ratios(
    symbol: str,
    expiry: Optional[str] = Query(None, description="Expiry ISO YYYY-MM-DD (defaults to nearest)"),
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        res = service.get_options_put_call_ratios(symbol=symbol, expiry=expiry)
        return from_domain_put_call(res)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/macro/nyfed/rates",
    response_model=ReferenceRatesResponse,
    summary="Fetch NY Fed overnight benchmark reference rates and pair spreads",
)
def get_nyfed_reference_rates(
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        rates = service.get_reference_rates()
        return from_domain_reference_rates(rates)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/macro/treasury/yield-curve",
    response_model=TreasuryYieldCurveResponse,
    summary="Fetch US Treasury par yield curve and spreads from Treasury.gov",
)
def get_treasury_yield_curve(
    month: Optional[str] = Query(None, description="Observation month YYYYMM (defaults to current)"),
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        yc = service.get_treasury_yield_curve(month_yyyymm=month)
        return from_domain_yield_curve(yc)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/macro/treasury/auctions",
    response_model=List[TreasuryAuctionResponse],
    summary="Fetch completed US Treasury auctions from Fiscal Data API",
)
def get_treasury_auctions(
    limit: int = Query(10, ge=1, le=50),
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        auctions = service.get_treasury_auctions(limit=limit)
        return [from_domain_auction(a) for a in auctions]
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/macro/treasury/debt",
    response_model=List[UsNationalDebtResponse],
    summary="Fetch daily close Debt to the Penny snapshots from US Treasury",
)
def get_treasury_debt(
    limit: int = Query(5, ge=1, le=30),
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        debt_snapshots = service.get_treasury_debt(limit=limit)
        return [from_domain_debt(d) for d in debt_snapshots]
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/thailand/fund-asset-allocation",
    response_model=ThaiFundAssetAllocationResponse,
    summary="Fetch Thai mutual fund industry asset class distribution from SEC Thailand",
)
def get_thai_fund_allocation(
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        alloc = service.get_thai_fund_asset_allocation()
        return from_domain_fund_allocation(alloc)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/thailand/bonds/market",
    response_model=ThaiBondMarketStatsResponse,
    summary="Fetch Thai domestic bond market overview statistics from SEC Thailand",
)
def get_thai_bonds_market(
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        stats = service.get_thai_bond_market_stats()
        return from_domain_bond_stats(stats)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/thailand/bonds/issuance",
    response_model=ThaiCorporateBondIssuanceResponse,
    summary="Fetch Thai corporate bond offering statistics from SEC Thailand",
)
def get_thai_bonds_issuance(
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        issuance = service.get_thai_corporate_bond_issuance()
        return from_domain_bond_issuance(issuance)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/thailand/public-debt",
    response_model=ThaiPublicDebtResponse,
    summary="Fetch monthly Thai public debt to GDP ratio and components from MOF Thailand",
)
def get_thai_public_debt(
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        debt = service.get_thai_public_debt()
        return from_domain_public_debt(debt)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/thailand/yield-curve",
    response_model=ThaiYieldCurveResponse,
    summary="Fetch Thai Government Bond Model Yield Curve from ThaiBMA",
)
def get_thai_yield_curve(
    date: Optional[str] = Query(None, description="Observation date ISO YYYY-MM-DD (defaults to latest)"),
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        curve = service.get_thai_yield_curve(as_of_date=date)
        return from_domain_thai_yield_curve(curve)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/signals/prediction-markets",
    response_model=List[PredictionMarketResponse],
    summary="Fetch active prediction markets and implied odds from Polymarket",
)
def get_prediction_markets(
    limit: int = Query(12, ge=1, le=50),
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        markets = service.get_prediction_markets(limit=limit)
        return [from_domain_prediction_market(m) for m in markets]
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/crypto/etf-flows/{asset}",
    response_model=SpotEtfFlowResponse,
    summary="Fetch US Spot ETF net flows and issuer breakdown for BTC or ETH",
)
def get_spot_etf_flows(
    asset: str,
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        flow = service.get_spot_etf_flows(asset=asset)
        return from_domain_etf_flows(flow)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/crypto/stablecoins",
    response_model=StablecoinSupplyResponse,
    summary="Fetch global USD stablecoin circulating supply and 7d/30d growth metrics from DeFiLlama",
)
def get_stablecoin_supply(
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        snap = service.get_stablecoin_supply()
        return from_domain_stablecoins(snap)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/crypto/benchmark",
    response_model=CryptoBenchmarkResponse,
    summary="Fetch Bitcoin spot price benchmark, returns, and BTC/Gold valuation ratio",
)
def get_crypto_benchmark(
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        snap = service.get_crypto_benchmark()
        return from_domain_crypto_benchmark(snap)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/macro/crypto-liquidity",
    response_model=CryptoMacroLiquidityResponse,
    summary="Fetch synthesized Level 1 Crypto Macro Liquidity & Risk Appetite proxy",
)
def get_crypto_macro_liquidity(
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        snap = service.get_crypto_macro_liquidity()
        return from_domain_crypto_liquidity(snap)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))



# ============================================================================
# Phase 3 Endpoints (Macro Risk, Metals COT, Policy Rates, Nasdaq Consensus)
# ============================================================================

@router.get(
    "/macro/financial-stress",
    response_model=FinancialStressResponse,
    summary="Fetch US OFR Financial Stress Index with 5 categories and T-2 lag",
)
def get_financial_stress(
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        snap = service.get_financial_stress()
        return map_fsi_to_response(snap)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/commodities/metals/cot",
    response_model=MetalsCotResponse,
    summary="Fetch CFTC Disaggregated COT positioning for metals (gold, silver, etc.)",
)
def get_metals_cot(
    commodity: str = Query("gold", description="Metal name: gold, silver, copper, platinum"),
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        snap = service.get_metals_cot(commodity=commodity)
        return map_cot_to_response(snap)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/macro/global-policy-rates",
    response_model=GlobalPolicyRatesResponse,
    summary="Fetch BIS central bank policy rates across 12 countries with rate spreads",
)
def get_global_policy_rates(
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        snap = service.get_global_policy_rates()
        return map_bis_rates_to_response(snap)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/equity/consensus/{symbol}",
    response_model=NasdaqConsensusResponse,
    summary="Fetch Nasdaq earnings surprise history, upcoming date status, and analyst ratings",
)
def get_nasdaq_consensus(
    symbol: str,
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        snap = service.get_nasdaq_consensus(symbol=symbol)
        return map_nasdaq_to_response(snap)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


# ============================================================================
# Phase 4 Endpoints (Commodity Vol, Treasury Demand, SEC Facts & Form 4, News)
# ============================================================================

@router.get(
    "/commodities/volatility",
    response_model=List[CommodityVolSnapshotSchema],
    summary="Fetch Cboe 30-day commodity volatility indices (GVZ, VXSLV, OVX) with 52-week percentiles",
)
def get_commodity_volatility(
    symbol: Optional[str] = Query(None, description="Optional symbol filter: GVZ, VXSLV, OVX"),
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        if symbol:
            snap = service.get_commodity_volatility(symbol.strip().upper())
            return [map_commodity_vol_to_schema(snap)]
        snaps = service.get_all_commodity_volatilities()
        return [map_commodity_vol_to_schema(s) for s in snaps]
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/macro/treasury/auction-demand",
    response_model=AuctionDemandSnapshotSchema,
    summary="Fetch latest US Treasury auction demand metrics and moving average of prior 8 completed auctions",
)
def get_treasury_auction_demand(
    security_type: str = Query(..., description="Security type (e.g. Note, Bill, Bond)"),
    security_term: str = Query(..., description="Security term (e.g. 10-Year, 13-Week)"),
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        snap = service.get_auction_demand_summary(
            security_type=security_type.strip(),
            security_term=security_term.strip(),
        )
        return map_auction_demand_to_schema(snap)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/equity/sec/financials/{symbol}",
    response_model=SecCompanyFactsSnapshotSchema,
    summary="Fetch SEC company-filed XBRL facts and calculated pure financial ratios",
)
def get_sec_financials(
    symbol: str,
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        snap = service.get_sec_financials(symbol=symbol.strip().upper())
        return map_sec_financials_to_schema(snap)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/equity/sec/insider-trades/{symbol}",
    response_model=SecInsiderTradeSnapshotSchema,
    summary="Fetch SEC Form 4 parsed insider transactions and 90-day net buying ratio",
)
def get_sec_insider_trades(
    symbol: str,
    limit: int = Query(20, ge=1, le=50, description="Max transactions to return"),
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        snap = service.get_sec_insider_trades(symbol=symbol.strip().upper(), limit=limit)
        return map_sec_insider_trades_to_schema(snap)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


@router.get(
    "/equity/news/{symbol}",
    response_model=NewsDiscoverySnapshotSchema,
    summary="Fetch normalized RSS news discovery candidates for equity symbol",
)
def get_equity_news_discovery(
    symbol: str,
    limit: int = Query(15, ge=1, le=30, description="Max news items to return"),
    service: TerminalDataServicePort = Depends(get_terminal_data_service),
):
    try:
        snap = service.get_ticker_news_discovery(symbol=symbol.strip().upper(), limit=limit)
        return map_news_discovery_to_schema(snap)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DataUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ProviderError as exc:
        code = exc.status_code or status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(exc))


