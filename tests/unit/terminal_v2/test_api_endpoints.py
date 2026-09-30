"""Unit tests for all Terminal V2 FastAPI Endpoints (/api/v2/market/...).

Consolidates all API tests across:
1. Macro: Yield curve, National Debt, NY Fed rates, BIS policy rates, OFR financial stress, Treasury auction demand
2. Fixed Income: Thai corporate bond stats & issuance, Thai public debt
3. Equity: FINRA short volume, Cboe Max Pain & Put/Call, Nasdaq earnings consensus & surprise, SEC XBRL financials, SEC Form 4 insider trades, RSS news discovery
4. Derivatives: Cboe options chains, Cboe commodity volatility (GVZ, VXSLV, OVX), Metals COT positioning (CFTC)
5. Fund Flows: SET/mai 4-investor-type flow, Thai retail gold, Spot ETF flows (BTC/ETH), Polymarket prediction markets, Thai mutual fund asset allocation
6. Robustness: Upstream timeouts (503 Service Unavailable), Bad requests (400), Invalid tickers
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from api.main import app
from tools.market.terminal_v2.adapters.bis_adapter import BisPolicyRatesHttpAdapter
from tools.market.terminal_v2.adapters.cboe_commodity_vol_adapter import CboeCommodityVolAdapter
from tools.market.terminal_v2.adapters.cftc_cot_adapter import CftcCotHttpAdapter
from tools.market.terminal_v2.adapters.nasdaq_adapter import NasdaqHttpAdapter
from tools.market.terminal_v2.adapters.ofr_adapter import OfrHttpAdapter
from tools.market.terminal_v2.adapters.rss_news_adapter import RssNewsDiscoveryAdapter
from tools.market.terminal_v2.adapters.sec_edgar_adapter import SecEdgarAdapter
from tools.market.terminal_v2.adapters.treasury_adapter import TreasuryAdapter
from tools.market.terminal_v2.application.terminal_data_service import TerminalDataService
from tools.market.terminal_v2.bootstrap import (
    create_terminal_data_service,
    get_terminal_data_service,
)
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import (
    FinraShortVolumeSnapshot,
    GoldPriceDetail,
    InvestorTypeRow,
    LivePerpsQuote,
    OptionsMaxPainResult,
    PredictionMarketItem,
    PredictionOutcome,
    ReferenceRatePoint,
    ReferenceRateSnapshot,
    SpotEtfFlowSnapshot,
    SpotEtfIssuerFlow,
    ThaiFundAssetAllocationRow,
    ThaiFundAssetAllocationSnapshot,
    ThaiFundFlowSnapshot,
    ThaiPublicDebtComponent,
    ThaiPublicDebtSnapshot,
    ThaiRetailGoldQuote,
    TreasuryYieldCurveSnapshot,
    TreasuryYieldPoint,
    UsNationalDebtSnapshot,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "terminal_v2"


@pytest.fixture
def mock_service_offline():
    """Create a fully-wired TerminalDataService with offline sanitized fixtures for Global & Terminal V2 intelligence."""
    base = create_terminal_data_service()
    base._commodity_vol = CboeCommodityVolAdapter(fixture_dir=FIXTURES_DIR)
    base._auction_history = TreasuryAdapter(fixture_dir=FIXTURES_DIR)
    sec = SecEdgarAdapter(
        facts_fixture_path=FIXTURES_DIR / "sec_companyfacts_nvda_fixture.json",
        submissions_fixture_path=FIXTURES_DIR / "sec_submissions_nvda_fixture.json",
        form4_fixture_path=FIXTURES_DIR / "sec_form4_nvda_fixture.xml",
    )
    base._sec_financials = sec
    base._sec_insider_trades = sec
    base._ticker_news = RssNewsDiscoveryAdapter(
        fixture_path=FIXTURES_DIR / "google_news_nvda_fixture.xml",
    )
    base._ofr_stress = OfrHttpAdapter(fixture_path=FIXTURES_DIR / "ofr_fsi_fixture.csv")
    base._metals_cot = CftcCotHttpAdapter(fixture_path=FIXTURES_DIR / "cftc_cot_gold_fixture.json")
    base._global_policy_rates = BisPolicyRatesHttpAdapter(fixture_path=FIXTURES_DIR / "bis_cbpol_fixture.csv")
    base._nasdaq_intelligence = NasdaqHttpAdapter(
        surprise_fixture_path=FIXTURES_DIR / "nasdaq_earnings_surprise_nvda.json",
        ratings_fixture_path=FIXTURES_DIR / "nasdaq_ratings_nvda.json",
        calendar_fixture_path=FIXTURES_DIR / "nasdaq_calendar_fixture.json",
    )
    return base


@pytest.fixture
def client(mock_service_offline, monkeypatch):
    monkeypatch.setenv("ENABLE_BACKGROUND_WORKERS", "false")
    monkeypatch.setenv("ENABLE_JOB_WORKERS", "false")
    monkeypatch.setenv("SCHEDULERS_ENABLED", "false")
    app.dependency_overrides[get_terminal_data_service] = lambda: mock_service_offline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_terminal_data_service, None)


# ============================================================================
# Phase 1 Endpoints (Thai Market, Gold, Perps)
# ============================================================================

def test_endpoint_thai_flow(client):
    mock_flow = ThaiFundFlowSnapshot(
        market="SET",
        as_of="26/09/2026",
        total_value=50000000000.0,
        investors=(
            InvestorTypeRow(
                investor_type="Foreign",
                name_en="Foreign Investors",
                buy_value=25000000000.0,
                sell_value=24000000000.0,
                net_value=1000000000.0,
            ),
        ),
        source="Settrade",
    )
    with patch("tools.market.terminal_v2.adapters.settrade_adapter.SettradeAdapter.get_investor_type_flow", return_value=mock_flow):
        resp = client.get("/api/v2/market/thailand/flow?market=SET")
        assert resp.status_code == 200
        data = resp.json()
        assert data["market"] == "SET"
        assert data["total_value"] == 50000000000.0
        assert len(data["investors"]) == 1
        assert data["investors"][0]["net_value"] == 1000000000.0


def test_endpoint_thai_gold(client):
    mock_gold = ThaiRetailGoldQuote(
        source="Gold Traders Association",
        unit="baht-weight (15.244 g, 96.5%)",
        bar=GoldPriceDetail(buy=43000.0, sell=43100.0),
        ornament=GoldPriceDetail(buy=42200.0, sell=43600.0),
        announced_at="26/09/2569 เวลา 09:30 น.",
        revision=1,
    )
    with patch("tools.market.terminal_v2.adapters.goldtraders_adapter.GoldTradersAdapter.get_retail_gold_quote", return_value=mock_gold):
        resp = client.get("/api/v2/market/thailand/gold")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bar"]["buy"] == 43000.0
        assert data["bar"]["sell"] == 43100.0
        assert data["revision"] == 1


def test_endpoint_perps_quote_valid_and_mismatch(client):
    mock_perp = LivePerpsQuote(
        symbol="xyz:TSLA",
        mark_price=255.5,
        dex_namespace="xyz",
        asset_class="synthetic_crypto_perp",
        contract_type="perpetual_future",
        source="Hyperliquid",
    )
    with patch("tools.market.terminal_v2.adapters.hyperliquid_adapter.HyperliquidAdapter.get_perps_quote", return_value=mock_perp):
        resp = client.get("/api/v2/market/perps/quote/xyz:TSLA")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "xyz:TSLA"
        assert data["mark_price"] == 255.5
        assert data["asset_class"] == "synthetic_crypto_perp"

        resp_bad = client.get("/api/v2/market/perps/quote/TSLA")
        assert resp_bad.status_code == 400
        assert "bare ticker" in resp_bad.json()["detail"].lower()


# ============================================================================
# Phase 2 Endpoints (Short Volume, Max Pain, Reference Rates, Treasury, etc.)
# ============================================================================

def test_endpoint_finra_short_volume(client):
    mock_snap = FinraShortVolumeSnapshot(
        symbol="AAPL",
        report_date="2026-09-25",
        short_volume=2000000,
        short_exempt_volume=50000,
        finra_reported_total_volume=5000000,
        short_pct=40.0,
        fetched_at=1758880000.0,
        coverage="FINRA consolidated TRF/ADF",
        unit="shares",
        source="FINRA",
    )
    with patch("tools.market.terminal_v2.application.terminal_data_service.TerminalDataService.get_short_volume", return_value={"AAPL": mock_snap}):
        resp = client.get("/api/v2/market/equity/short-volume/AAPL")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "AAPL"
        assert data["short_volume"] == 2000000
        assert data["short_pct"] == 40.0
        assert "NOT short interest" in data["limitations"]


def test_endpoint_cboe_max_pain(client):
    mock_pain = OptionsMaxPainResult(
        underlying="AAPL",
        expiry="2026-10-16",
        strike=250.0,
        minimum_theoretical_payout=15000000.0,
        candidate_count=20,
        excluded_contract_count=2,
        spot_price=248.5,
        distance_from_spot=-1.5,
        distance_pct=-0.6,
        assumptions="Standard series only.",
        limitations="Analytical composite indicator; NOT a price prediction.",
    )
    with patch("tools.market.terminal_v2.application.terminal_data_service.TerminalDataService.get_options_max_pain", return_value=mock_pain):
        resp = client.get("/api/v2/market/equity/max-pain/AAPL?expiry=2026-10-16")
        assert resp.status_code == 200
        data = resp.json()
        assert data["underlying"] == "AAPL"
        assert data["strike"] == 250.0
        assert data["candidate_count"] == 20
        assert "NOT a price prediction" in data["limitations"]


def test_endpoint_nyfed_rates(client):
    mock_rates = ReferenceRateSnapshot(
        as_of="2026-09-25",
        rates=(
            ReferenceRatePoint(
                code="SOFR",
                label="Secured Overnight Financing Rate",
                effective_date="2026-09-24",
                rate_percent=3.88,
                volume_in_billions=2990.0,
            ),
            ReferenceRatePoint(
                code="EFFR",
                label="Effective Federal Funds Rate",
                effective_date="2026-09-24",
                rate_percent=3.88,
                volume_in_billions=105.0,
            ),
        ),
        spreads_bps={"SOFR-EFFR": 0.0},
        fetched_at=1758880000.0,
        source="New York Fed",
    )
    with patch("tools.market.terminal_v2.application.terminal_data_service.TerminalDataService.get_reference_rates", return_value=mock_rates):
        resp = client.get("/api/v2/market/macro/nyfed/rates")
        assert resp.status_code == 200
        data = resp.json()
        assert data["as_of"] == "2026-09-25"
        assert len(data["rates"]) == 2
        assert data["spreads_bps"]["SOFR-EFFR"] == 0.0


def test_endpoint_treasury_yield_curve(client):
    mock_yc = TreasuryYieldCurveSnapshot(
        observation_date="2026-09-25",
        yields=(
            TreasuryYieldPoint(maturity="2 Yr", yield_percent=4.81),
            TreasuryYieldPoint(maturity="10 Yr", yield_percent=5.17),
        ),
        spread_10y_2y_bps=36.0,
        spread_10y_3m_bps=None,
        fetched_at=1758880000.0,
        source="US Treasury",
    )
    with patch("tools.market.terminal_v2.application.terminal_data_service.TerminalDataService.get_treasury_yield_curve", return_value=mock_yc):
        resp = client.get("/api/v2/market/macro/treasury/yield-curve")
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation_date"] == "2026-09-25"
        assert data["spread_10y_2y_bps"] == 36.0
        assert len(data["yields"]) == 2


def test_endpoint_treasury_debt(client):
    mock_debt = (
        UsNationalDebtSnapshot(
            record_date="2026-09-24",
            total_public_debt_usd=40000000000000.0,
            debt_held_by_public_usd=32000000000000.0,
            intragovernmental_holdings_usd=8000000000000.0,
            fetched_at=1758880000.0,
            source="US Treasury Fiscal Data",
        ),
    )
    with patch("tools.market.terminal_v2.application.terminal_data_service.TerminalDataService.get_treasury_debt", return_value=mock_debt):
        resp = client.get("/api/v2/market/macro/treasury/debt?limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["record_date"] == "2026-09-24"
        assert data[0]["total_public_debt_usd"] == 40000000000000.0


def test_endpoint_thai_fund_allocation(client):
    mock_alloc = ThaiFundAssetAllocationSnapshot(
        reporting_period="2026 Q1",
        total_nav_thb=5000000000000.0,
        allocations=(
            ThaiFundAssetAllocationRow(
                asset_class="Common stock",
                domestic_or_foreign="Domestic",
                value_thb=1500000000000.0,
                share_of_nav_pct=30.0,
            ),
        ),
        fetched_at=1758880000.0,
        source="SEC Thailand",
    )
    with patch("tools.market.terminal_v2.application.terminal_data_service.TerminalDataService.get_thai_fund_asset_allocation", return_value=mock_alloc):
        resp = client.get("/api/v2/market/thailand/fund-asset-allocation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["reporting_period"] == "2026 Q1"
        assert len(data["allocations"]) == 1
        assert data["allocations"][0]["share_of_nav_pct"] == 30.0


def test_endpoint_thai_public_debt(client):
    mock_debt = ThaiPublicDebtSnapshot(
        reporting_month="2026-08",
        total_debt_thb=11500000000000.0,
        debt_to_gdp_pct=64.2,
        fx_rate_usd_thb=35.5,
        components=(
            ThaiPublicDebtComponent(
                component_number=1,
                label_en="Government debt",
                label_th="หนี้รัฐบาล",
                amount_thb=9000000000000.0,
            ),
        ),
        fetched_at=1758880000.0,
        source="MOF Thailand",
    )
    with patch("tools.market.terminal_v2.application.terminal_data_service.TerminalDataService.get_thai_public_debt", return_value=mock_debt):
        resp = client.get("/api/v2/market/thailand/public-debt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["reporting_month"] == "2026-08"
        assert data["debt_to_gdp_pct"] == 64.2
        assert len(data["components"]) == 1


def test_endpoint_polymarket(client):
    mock_items = (
        PredictionMarketItem(
            market_id="12345",
            question="Will Fed cut rates in November?",
            outcomes=(
                PredictionOutcome(label="Yes", price=0.85),
                PredictionOutcome(label="No", price=0.15),
            ),
            volume_24h_usd=500000.0,
            end_date="2026-11-05",
            source_url="https://polymarket.com/market/12345",
            fetched_at=1758880000.0,
            source="Polymarket",
        ),
    )
    with patch("tools.market.terminal_v2.application.terminal_data_service.TerminalDataService.get_prediction_markets", return_value=mock_items):
        resp = client.get("/api/v2/market/signals/prediction-markets?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["question"] == "Will Fed cut rates in November?"
        assert data[0]["outcomes"][0]["price"] == 0.85


def test_endpoint_crypto_etf_flows(client):
    mock_flow = SpotEtfFlowSnapshot(
        asset="BTC",
        report_date="2026-09-25",
        daily_total_usd=120000000.0,
        cumulative_total_usd=25000000000.0,
        issuers=(
            SpotEtfIssuerFlow(
                ticker="IBIT",
                institute="BlackRock",
                daily_net_inflow_usd=90000000.0,
                cumulative_net_inflow_usd=15000000000.0,
                total_net_assets_usd=22000000000.0,
            ),
        ),
        fetched_at=1758880000.0,
        source="SoSoValue",
    )
    with patch("tools.market.terminal_v2.application.terminal_data_service.TerminalDataService.get_spot_etf_flows", return_value=mock_flow):
        resp = client.get("/api/v2/market/crypto/etf-flows/BTC")
        assert resp.status_code == 200
        data = resp.json()
        assert data["asset"] == "BTC"
        assert data["daily_total_usd"] == 120000000.0
        assert len(data["issuers"]) == 1
        assert data["issuers"][0]["ticker"] == "IBIT"


def test_endpoint_upstream_unavailable_returns_503(client):
    with patch(
        "tools.market.terminal_v2.application.terminal_data_service.TerminalDataService.get_short_volume",
        side_effect=DataUnavailableError("Upstream timeout", capability="short-volume", source="FINRA"),
    ):
        resp = client.get("/api/v2/market/equity/short-volume/XYZ")
        assert resp.status_code == 503
        data = resp.json()
        assert "upstream timeout" in data["detail"].lower()


# ============================================================================
# Phase 3 Endpoints (Financial Stress, Metals COT, BIS, Nasdaq Consensus)
# ============================================================================

def test_api_financial_stress(client):
    resp = client.get("/api/v2/market/macro/financial-stress")
    assert resp.status_code == 200
    data = resp.json()
    assert data["as_of_date"] == "2026-09-24"
    assert data["fsi_value"] == -0.25
    assert data["regime"] == "normal"
    assert data["source"] == "OFR"
    assert data["data_lag_days"] == 2
    assert len(data["categories"]) == 5
    assert len(data["trend_90d"]) == 4


def test_api_metals_cot(client):
    resp = client.get("/api/v2/market/commodities/metals/cot?commodity=gold")
    assert resp.status_code == 200
    data = resp.json()
    assert data["commodity"] == "GOLD"
    assert data["commodity_code"] == "088691"
    assert data["as_of_date"] == "2026-09-22"
    assert data["report_type"] == "disaggregated"
    assert data["open_interest"] == 480000
    assert data["managed_money"]["net_contracts"] == 202800
    assert data["managed_money"]["change_long"] == 5200
    assert data["swap_dealers"]["net_contracts"] == -25000


def test_api_global_policy_rates(client):
    resp = client.get("/api/v2/market/macro/global-policy-rates")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "BIS"
    assert len(data["rates"]) == 12
    rates_dict = {r["country"]: r for r in data["rates"]}
    assert rates_dict["TH"]["rate_value"] == 2.50
    assert rates_dict["US"]["rate_value"] == 5.00
    assert data["spreads_vs_bot_repo"]["US"] == 250.0
    assert data["spreads_vs_bot_repo"]["TH"] == 0.0


def test_api_nasdaq_consensus(client):
    resp = client.get("/api/v2/market/equity/consensus/NVDA")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "NVDA"
    assert data["coverage_status"] == "full"
    assert data["has_earnings_surprise"] is True
    assert data["has_analyst_ratings"] is True
    assert data["ratings"]["consensus"] == "Strong Buy"
    assert len(data["surprise_history"]) == 4


# ============================================================================
# Phase 4 Endpoints (Commodity Vol, Treasury Auction Demand, SEC XBRL, News)
# ============================================================================

def test_api_commodity_volatility_all(client):
    resp = client.get("/api/v2/market/commodities/volatility")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 3

    symbols = [item["index_symbol"] for item in data]
    assert "GVZ" in symbols
    assert "VXSLV" in symbols
    assert "OVX" in symbols

    gvz = next(item for item in data if item["index_symbol"] == "GVZ")
    assert "SPDR Gold Shares (GLD)" in gvz["underlying_instrument"]
    assert gvz["close_date"] == "2026-09-25"
    assert gvz["implied_volatility"] == pytest.approx(22.44)
    assert gvz["source"] == "Cboe"
    assert len(gvz["limitations"]) > 0


def test_api_commodity_volatility_single(client):
    resp = client.get("/api/v2/market/commodities/volatility?symbol=GVZ")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["index_symbol"] == "GVZ"


def test_api_commodity_volatility_invalid_symbol(client):
    resp = client.get("/api/v2/market/commodities/volatility?symbol=INVALID")
    assert resp.status_code in (400, 502)


def test_api_treasury_auction_demand(client):
    resp = client.get("/api/v2/market/macro/treasury/auction-demand?security_type=Note&security_term=10-Year")
    assert resp.status_code == 200
    data = resp.json()
    assert data["security_type"] == "Note"
    assert data["security_term"] == "10-Year"
    assert data["latest_auction_date"] == "2026-08-12"
    assert data["latest_bid_to_cover_ratio"] == pytest.approx(2.53)
    assert data["latest_high_yield"] == pytest.approx(4.683)
    assert data["sample_count"] >= 3
    assert data["source"] == "US Treasury Fiscal Data"
    assert "tail_spread" not in data


def test_api_sec_financials(client):
    resp = client.get("/api/v2/market/equity/sec/financials/NVDA")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "NVDA"
    assert data["cik"] == "0001045810"
    assert data["entity_name"] == "NVIDIA CORP"
    assert data["revenue_usd"] == 177837000000.0
    assert data["operating_cash_flow_usd"] is not None
    assert data["capex_usd"] is not None
    assert data["free_cash_flow_usd"] is not None
    assert data["free_cash_flow_margin"] is not None
    assert len(data["facts"]) > 0
    assert "companyfacts" in data["source"]
    assert any("un-audited" in lim.lower() for lim in data["limitations"])


def test_api_sec_insider_trades(client):
    resp = client.get("/api/v2/market/equity/sec/insider-trades/NVDA?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "NVDA"
    assert data["cik"] == "0001045810"
    assert len(data["transactions"]) > 0
    tx = data["transactions"][0]
    assert tx["reporting_owner"] == "Teter Timothy S."
    assert tx["transaction_code"] == "S"
    assert tx["shares"] == 12483.0
    assert tx["price_per_share"] == pytest.approx(222.1932)
    assert data["source"] == "SEC EDGAR (Form 4 XML)"


def test_api_equity_news_discovery(client):
    resp = client.get("/api/v2/market/equity/news/NVDA?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["query_symbol"] == "NVDA"
    assert data["status"] == "ok"
    assert len(data["items"]) == 5
    assert len(data["items"][0]["headline"]) > 0
    assert len(data["items"][0]["publisher"]) > 0
    assert "http" in data["items"][0]["article_url"]
