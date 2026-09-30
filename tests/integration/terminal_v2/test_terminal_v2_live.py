"""Live Opt-In Smoke Integration Tests for Terminal V2 Keyless Providers.

Tests live HTTP network connectivity across all Terminal V2 keyless providers:
1. Macro: US Treasury (yield curve, national debt, completed auctions), New York Fed reference rates, OFR Financial Stress Index, BIS central bank policy rates, FRED keyless CSV
2. Fixed Income: MOF Thailand public debt
3. Equity: FINRA short volume, Cboe options chains & Max Pain, Nasdaq consensus & surprise, SEC EDGAR company facts & Form 4 XML, RSS news discovery
4. Derivatives: Cboe commodity volatility (GVZ), CFTC commitments of traders (gold), Hyperliquid synthetic crypto perps
5. Fund Flows: Settrade SET/mai investor flow, Gold Traders Association Thai retail physical gold, SEC Thailand mutual fund asset allocation, Polymarket prediction markets, SoSoValue spot ETF flows

Run explicitly with:
    pytest -m integration tests/integration/terminal_v2/test_terminal_v2_live.py
When offline or firewalled, all tests gracefully skip or handle ProviderError without crash.
"""
import pytest

from tools.market.terminal_v2.adapters.bis_adapter import BisPolicyRatesHttpAdapter
from tools.market.terminal_v2.adapters.cboe_commodity_vol_adapter import CboeCommodityVolAdapter
from tools.market.terminal_v2.adapters.cftc_cot_adapter import CftcCotHttpAdapter
from tools.market.terminal_v2.adapters.nasdaq_adapter import NasdaqHttpAdapter
from tools.market.terminal_v2.adapters.ofr_adapter import OfrHttpAdapter
from tools.market.terminal_v2.adapters.rss_news_adapter import RssNewsDiscoveryAdapter
from tools.market.terminal_v2.adapters.sec_edgar_adapter import SecEdgarAdapter
from tools.market.terminal_v2.adapters.treasury_adapter import TreasuryAdapter
from tools.market.terminal_v2.bootstrap import get_terminal_data_service, get_terminal_service
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import (
    FinraShortVolumeSnapshot,
    LivePerpsQuote,
    MacroSeries,
    OptionsChainSnapshot,
    OptionsMaxPainResult,
    PredictionMarketItem,
    ReferenceRateSnapshot,
    SpotEtfFlowSnapshot,
    ThaiFundAssetAllocationSnapshot,
    ThaiFundFlowSnapshot,
    ThaiPublicDebtSnapshot,
    ThaiRetailGoldQuote,
    TreasuryYieldCurveSnapshot,
    UsNationalDebtSnapshot,
)


# ============================================================================
# Phase 1 Live Smoke Tests
# ============================================================================

@pytest.mark.integration
def test_live_settrade_investor_flow():
    service = get_terminal_service()
    try:
        flow = service.get_investor_flow("SET")
        assert isinstance(flow, ThaiFundFlowSnapshot)
        assert flow.market == "SET"
        assert len(flow.investors) > 0
        assert flow.total_value >= 0
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"Settrade live endpoint unreachable: {exc}")


@pytest.mark.integration
def test_live_goldtraders_retail_gold():
    service = get_terminal_service()
    try:
        gold = service.get_retail_gold()
        assert isinstance(gold, ThaiRetailGoldQuote)
        assert gold.bar.buy > 10000.0
        assert gold.bar.sell >= gold.bar.buy
        assert gold.ornament.sell >= gold.ornament.buy
        assert "Gold Traders Association" in gold.source
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"Gold Traders Association live page unreachable: {exc}")


@pytest.mark.integration
def test_live_fred_keyless_csv():
    service = get_terminal_service()
    try:
        macro = service.get_macro_series("FEDFUNDS")
        assert isinstance(macro, MacroSeries)
        assert macro.series_id == "FEDFUNDS"
        assert len(macro.points) > 10
        assert macro.points[-1].value >= 0
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"FRED live CSV endpoint unreachable: {exc}")


@pytest.mark.integration
def test_live_hyperliquid_perps():
    service = get_terminal_service()
    try:
        perp = service.get_perp_quote("BTC")
        assert isinstance(perp, LivePerpsQuote)
        assert perp.symbol == "BTC"
        assert perp.mark_price > 1000.0
        assert perp.asset_class == "synthetic_crypto_perp"
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"Hyperliquid live API unreachable: {exc}")


# ============================================================================
# Phase 2 Live Smoke Tests
# ============================================================================

@pytest.mark.integration
def test_live_finra_short_volume():
    service = get_terminal_data_service()
    try:
        res = service.get_short_volume(["AAPL", "SPY"])
        assert len(res) > 0
        snap = next(iter(res.values()))
        assert isinstance(snap, FinraShortVolumeSnapshot)
        assert snap.short_volume >= 0
        assert snap.finra_reported_total_volume >= snap.short_volume
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"FINRA live daily file unreachable: {exc}")


@pytest.mark.integration
def test_live_cboe_options():
    service = get_terminal_data_service()
    try:
        chain = service.get_options_chain("AAPL")
        assert isinstance(chain, OptionsChainSnapshot)
        assert chain.underlying == "AAPL"
        assert len(chain.contracts) > 0

        pain = service.get_options_max_pain("AAPL")
        assert isinstance(pain, OptionsMaxPainResult)
        assert pain.strike > 0
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"Cboe live options quote unreachable: {exc}")


@pytest.mark.integration
def test_live_nyfed_reference_rates():
    service = get_terminal_data_service()
    try:
        rates = service.get_reference_rates()
        assert isinstance(rates, ReferenceRateSnapshot)
        assert len(rates.rates) > 0
        codes = {r.code for r in rates.rates}
        assert "SOFR" in codes or "EFFR" in codes
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"New York Fed live API unreachable: {exc}")


@pytest.mark.integration
def test_live_treasury_yield_curve():
    service = get_terminal_data_service()
    try:
        yc = service.get_treasury_yield_curve()
        assert isinstance(yc, TreasuryYieldCurveSnapshot)
        assert len(yc.yields) > 0
        assert yc.observation_date != ""
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"US Treasury yield curve XML unreachable: {exc}")


@pytest.mark.integration
def test_live_treasury_debt():
    service = get_terminal_data_service()
    try:
        debts = service.get_treasury_debt(limit=2)
        assert len(debts) > 0
        assert isinstance(debts[0], UsNationalDebtSnapshot)
        assert debts[0].total_public_debt_usd > 1e12
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"US Treasury Debt to the Penny API unreachable: {exc}")


@pytest.mark.integration
def test_live_sec_th_fund_allocation():
    service = get_terminal_data_service()
    try:
        alloc = service.get_thai_fund_asset_allocation()
        assert isinstance(alloc, ThaiFundAssetAllocationSnapshot)
        assert len(alloc.allocations) > 0
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"SEC Thailand live CSV unreachable: {exc}")


@pytest.mark.integration
def test_live_mof_th_public_debt():
    service = get_terminal_data_service()
    try:
        debt = service.get_thai_public_debt()
        assert isinstance(debt, ThaiPublicDebtSnapshot)
        assert len(debt.components) > 0
        assert debt.total_debt_thb > 1e12
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"MOF Thailand public debt CSV unreachable: {exc}")


@pytest.mark.integration
def test_live_polymarket_prediction_markets():
    service = get_terminal_data_service()
    try:
        markets = service.get_prediction_markets(limit=5)
        assert len(markets) > 0
        assert isinstance(markets[0], PredictionMarketItem)
        assert markets[0].question != ""
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"Polymarket Gamma API unreachable: {exc}")


@pytest.mark.integration
def test_live_sosovalue_etf_flows():
    service = get_terminal_data_service()
    try:
        flow = service.get_spot_etf_flows("BTC")
        assert isinstance(flow, SpotEtfFlowSnapshot)
        assert flow.asset == "BTC"
        assert len(flow.issuers) > 0
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"SoSoValue ETF flows gateway unreachable: {exc}")


# ============================================================================
# Phase 3 Live Smoke Tests
# ============================================================================

@pytest.mark.integration
def test_live_ofr_financial_stress():
    adapter = OfrHttpAdapter(timeout=20)
    try:
        snap = adapter.fetch_financial_stress()
        assert snap.source == "OFR"
        assert snap.data_lag_days == 2
        assert -5.0 < snap.fsi_value < 10.0
        assert len(snap.categories) == 5
        assert len(snap.trend_90d) > 0
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"OFR live endpoint unreachable: {exc}")


@pytest.mark.integration
def test_live_cftc_cot_gold():
    adapter = CftcCotHttpAdapter(timeout=20)
    try:
        snap = adapter.fetch_metals_cot(commodity="gold")
        assert snap.commodity == "GOLD"
        assert snap.commodity_code == "088691"
        assert snap.report_type == "disaggregated"
        assert snap.open_interest > 10000
        assert snap.managed_money.class_name == "Managed Money"
        assert 0.0 <= snap.percentile_52w <= 100.0
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"CFTC live endpoint unreachable: {exc}")


@pytest.mark.integration
def test_live_bis_policy_rates():
    adapter = BisPolicyRatesHttpAdapter(timeout=25)
    try:
        snap = adapter.fetch_global_policy_rates()
        assert snap.source == "BIS"
        assert len(snap.rates) >= 10
        countries = {r.country for r in snap.rates}
        assert "TH" in countries
        assert "US" in countries
        th_rate = next(r for r in snap.rates if r.country == "TH")
        assert 0.0 < th_rate.rate_value < 20.0
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"BIS live endpoint unreachable: {exc}")


@pytest.mark.integration
def test_live_nasdaq_consensus():
    adapter = NasdaqHttpAdapter(timeout=20)
    try:
        snap = adapter.fetch_earnings_consensus("NVDA")
        assert snap.symbol == "NVDA"
        assert snap.source == "Nasdaq"
        assert snap.coverage_status in ("full", "partial", "no_coverage")
        if snap.has_earnings_surprise:
            assert len(snap.surprise_history) > 0
        if snap.has_analyst_ratings:
            assert snap.ratings is not None
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"Nasdaq live endpoint unreachable: {exc}")


# ============================================================================
# Phase 4 Live Smoke Tests
# ============================================================================

@pytest.mark.integration
def test_live_cboe_commodity_volatility():
    adapter = CboeCommodityVolAdapter()
    try:
        snap = adapter.get_commodity_vol("GVZ")
        assert snap.index_symbol == "GVZ"
        assert "GLD" in snap.underlying_instrument
        assert snap.source == "Cboe"
        assert snap.implied_volatility > 0.0
        assert snap.close_date != ""
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"Cboe commodity vol live endpoint unreachable: {exc}")


@pytest.mark.integration
def test_live_treasury_auction_history_demand():
    adapter = TreasuryAdapter()
    try:
        history = adapter.fetch_completed_auction_history("Note", "10-Year", limit=9)
        assert len(history) >= 3
        for a in history:
            assert a.security_type == "Note"
            assert a.security_term == "10-Year"
            assert a.bid_to_cover_ratio is not None and a.bid_to_cover_ratio > 0.0
            assert a.source == "US Treasury Fiscal Data"
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"US Treasury auction history unreachable: {exc}")


@pytest.mark.integration
def test_live_sec_edgar_company_facts_and_form4():
    adapter = SecEdgarAdapter(
        user_agent="InvestAgentsResearch/1.0 (contact: research@investagents.internal)",
    )
    try:
        facts = adapter.get_company_facts("AAPL")
        assert facts.symbol == "AAPL"
        assert facts.cik == "0000320193"
        assert "SEC EDGAR" in facts.source
        assert facts.revenue_usd is not None and facts.revenue_usd > 0

        insider = adapter.get_insider_trades("AAPL", limit=10)
        assert insider.symbol == "AAPL"
        assert "SEC EDGAR" in insider.source
        assert insider.eligible_transaction_count >= 0
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"SEC EDGAR live endpoint unreachable: {exc}")


@pytest.mark.integration
def test_live_rss_news_discovery():
    adapter = RssNewsDiscoveryAdapter()
    try:
        snap = adapter.get_news_candidates("AAPL", limit=10)
        assert snap.query_symbol == "AAPL"
        assert "RSS" in snap.source
        assert snap.status in ("ok", "rate_limited", "feed_unavailable")
        assert len(snap.limitations) > 0
    except (DataUnavailableError, ProviderError) as exc:
        pytest.skip(f"RSS news discovery unreachable: {exc}")
