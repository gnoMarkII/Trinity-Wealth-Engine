"""Unit tests for all Terminal V2 HTTP Adapters against offline sanitized fixtures.

Covers:
1. Short volume (FINRA)
2. Options chains & OCC parsing (Cboe)
3. Reference rates (NY Fed)
4. US Treasury (Yield curve, completed auctions, national debt, auction history)
5. Thai institutional fund allocation (SEC Thailand)
6. Thai public debt (MOF Thailand)
7. Prediction markets (Polymarket)
8. Spot ETF flows (SoSoValue)
9. Financial stress index (OFR)
10. Metals positioning & COT (CFTC)
11. Central bank policy rates (BIS)
12. Equity consensus & earnings surprise (Nasdaq)
13. Commodity volatility indices (Cboe GVZ/VXSLV/OVX)
14. Company-filed XBRL facts & Form 4 ownership (SEC EDGAR)
15. News search & discovery (RSS Google News)
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from tools.market.terminal_v2.adapters.bis_adapter import BisPolicyRatesHttpAdapter
from tools.market.terminal_v2.adapters.cboe_commodity_vol_adapter import CboeCommodityVolAdapter
from tools.market.terminal_v2.adapters.cboe_options_adapter import (
    CboeOptionsAdapter,
    parse_occ_symbol,
)
from tools.market.terminal_v2.adapters.cftc_cot_adapter import CftcCotHttpAdapter
from tools.market.terminal_v2.adapters.etf_flows_adapter import SoSoValueEtfFlowsAdapter
from tools.market.terminal_v2.adapters.finra_adapter import FinraAdapter
from tools.market.terminal_v2.adapters.mof_th_adapter import MofThailandAdapter
from tools.market.terminal_v2.adapters.nasdaq_adapter import NasdaqHttpAdapter
from tools.market.terminal_v2.adapters.nyfed_adapter import NyFedAdapter
from tools.market.terminal_v2.adapters.ofr_adapter import OfrHttpAdapter
from tools.market.terminal_v2.adapters.polymarket_adapter import PolymarketAdapter
from tools.market.terminal_v2.adapters.rss_news_adapter import RssNewsDiscoveryAdapter
from tools.market.terminal_v2.adapters.sec_edgar_adapter import SecEdgarAdapter
from tools.market.terminal_v2.adapters.sec_th_adapter import SecThailandAdapter
from tools.market.terminal_v2.adapters.treasury_adapter import TreasuryAdapter
from tools.market.terminal_v2.application.cache import ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "terminal_v2"


def _read_fixture(filename: str, subfolder: Path = FIXTURES_DIR) -> str:
    path = subfolder / filename
    assert path.is_file(), f"Fixture missing: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================================
# 1. FINRA Short Volume Tests
# ============================================================================

def test_finra_adapter_parses_fixture():
    fixture_text = _read_fixture("finra_short_volume.txt")
    cache = ThreadSafeTTLCache(default_ttl_seconds=3600.0)
    adapter = FinraAdapter(cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = fixture_text

    with patch("requests.get", return_value=mock_resp):
        res = adapter.get_short_volume(["A", "AA", "AAL"])

    assert "A" in res
    a = res["A"]
    assert a.symbol == "A"
    assert a.report_date == "2026-09-25"
    assert a.short_volume == 420830
    assert a.short_exempt_volume == 0
    assert a.finra_reported_total_volume == 1173148
    expected_pct = (420830 / 1173148) * 100.0
    assert round(a.short_pct, 2) == round(expected_pct, 2)
    assert a.coverage == "FINRA consolidated TRF/ADF"
    assert "NOT short interest" in a.limitations
    assert a.is_stale is False

    assert "AA" in res
    aa = res["AA"]
    assert aa.short_volume == 632170
    assert aa.short_exempt_volume == 2210


def test_finra_serves_stale_on_upstream_failure():
    fixture_text = _read_fixture("finra_short_volume.txt")
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0)
    adapter = FinraAdapter(cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = fixture_text

    with patch("requests.get", return_value=mock_resp):
        first = adapter.get_short_volume(["A"])
        assert first["A"].is_stale is False

    cache._entries["finra:shortvol:latest"].expires_at = 0.0

    with patch("requests.get", side_effect=Exception("CDN 503 Service Unavailable")):
        second = adapter.get_short_volume(["A"])
        assert second["A"].is_stale is True
        assert second["A"].symbol == "A"
        assert "503" in second["A"].stale_reason or "Upstream error" in second["A"].stale_reason


# ============================================================================
# 2. Cboe Options Adapter Tests
# ============================================================================

def test_occ_symbol_parser():
    parsed = parse_occ_symbol("AAPL261016C00250000", expected_root="AAPL")
    assert parsed is not None
    root, expiry, side, strike, is_standard = parsed
    assert root == "AAPL"
    assert expiry == "2026-10-16"
    assert side == "call"
    assert strike == 250.0
    assert is_standard is True

    parsed_put = parse_occ_symbol("AAPL261016P00240000", expected_root="AAPL")
    assert parsed_put is not None
    assert parsed_put[2] == "put"
    assert parsed_put[3] == 240.0

    parsed_adj = parse_occ_symbol("AAPL1261016C00250000", expected_root="AAPL")
    assert parsed_adj is not None
    assert parsed_adj[4] is False

    assert parse_occ_symbol("INVALID", expected_root="AAPL") is None


def test_cboe_options_adapter_parses_fixture():
    fixture_json = json.loads(_read_fixture("cboe_options_aapl.json"))
    cache = ThreadSafeTTLCache(default_ttl_seconds=300.0)
    adapter = CboeOptionsAdapter(cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fixture_json
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        snap = adapter.get_options_chain("AAPL")

    assert snap.underlying == "AAPL"
    assert snap.underlying_price == 341.4603
    assert snap.iv30_decimal == 0.22215
    assert snap.delay_minutes == 15
    assert len(snap.contracts) > 0

    first_contract = snap.contracts[0]
    assert first_contract.underlying == "AAPL"
    assert first_contract.strike == 110.0
    assert first_contract.side == "call"
    assert first_contract.implied_volatility is None


# ============================================================================
# 3. New York Fed Reference Rates Tests
# ============================================================================

def test_nyfed_adapter_parses_fixture():
    fixture_json = json.loads(_read_fixture("nyfed_rates.json"))
    cache = ThreadSafeTTLCache(default_ttl_seconds=720.0)
    adapter = NyFedAdapter(cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fixture_json
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        snap = adapter.get_reference_rates()

    assert snap.as_of == "2026-09-25"
    codes = {r.code for r in snap.rates}
    assert "SOFR" in codes
    assert "EFFR" in codes
    assert "TGCR" in codes

    sofr = next(r for r in snap.rates if r.code == "SOFR")
    effr = next(r for r in snap.rates if r.code == "EFFR")
    assert sofr.rate_percent == 3.88
    assert sofr.volume_in_billions == 2990.0
    assert effr.rate_percent == 3.88

    assert "SOFR-EFFR" in snap.spreads_bps
    assert snap.spreads_bps["SOFR-EFFR"] == 0.0
    assert "does not include ON RRP" in snap.limitations


# ============================================================================
# 4. US Treasury Adapter Tests
# ============================================================================

def test_treasury_yield_curve_parses_xml():
    fixture_xml = _read_fixture("treasury_yield_curve.xml")
    cache = ThreadSafeTTLCache(default_ttl_seconds=10800.0)
    adapter = TreasuryAdapter(cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = fixture_xml

    with patch("requests.get", return_value=mock_resp):
        yc = adapter.get_yield_curve("202609")

    assert yc.observation_date == "2026-09-25"
    tenors = {p.maturity: p.yield_percent for p in yc.yields}
    assert tenors["1 Mo"] == 4.04
    assert tenors["2 Yr"] == 4.81
    assert tenors["10 Yr"] == 5.17
    assert tenors["30 Yr"] == 5.49

    assert yc.spread_10y_2y_bps == 36.0
    assert yc.spread_10y_3m_bps == 93.0


def test_treasury_auctions_parses_json():
    fixture_json = json.loads(_read_fixture("treasury_auctions.json"))
    cache = ThreadSafeTTLCache(default_ttl_seconds=10800.0)
    adapter = TreasuryAdapter(cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fixture_json
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        auctions = adapter.get_auctions(limit=5)

    assert len(auctions) > 0
    first = auctions[0]
    assert first.security_type == "Note"
    assert first.bid_to_cover_ratio == 2.42
    assert first.high_yield == 5.085


def test_treasury_debt_to_penny_parses_json():
    fixture_json = json.loads(_read_fixture("treasury_debt_to_penny.json"))
    cache = ThreadSafeTTLCache(default_ttl_seconds=10800.0)
    adapter = TreasuryAdapter(cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fixture_json
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        debts = adapter.get_national_debt(limit=5)

    assert len(debts) > 0
    debt = debts[0]
    assert debt.record_date == "2026-09-24"
    assert debt.total_public_debt_usd == 40068807991924.84
    assert debt.debt_held_by_public_usd == 32362728656693.39
    assert debt.intragovernmental_holdings_usd == 7706079335231.45
    assert "Daily close accounting snapshot" in debt.limitations


def test_treasury_auction_history_note_10y():
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0)
    adapter = TreasuryAdapter(cache=cache, fixture_dir=FIXTURES_DIR)

    results = adapter.fetch_completed_auction_history(security_type="Note", security_term="10-Year", limit=15)
    assert len(results) == 15
    latest = results[0]
    assert latest.security_type == "Note"
    assert latest.security_term == "10-Year"
    assert latest.high_yield == 4.683
    assert latest.bid_to_cover_ratio == 2.53
    assert latest.offering_amount_usd is not None

    # Strict Invariant: Does NOT have any auction tail field in TreasuryAuctionResult
    assert not hasattr(latest, "tail_spread")
    assert not hasattr(latest, "auction_tail_bps")


def test_treasury_auction_history_bill_13w():
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0)
    adapter = TreasuryAdapter(cache=cache, fixture_dir=FIXTURES_DIR)

    results = adapter.fetch_completed_auction_history(security_type="Bill", security_term="13-Week", limit=15)
    assert len(results) == 15
    latest = results[0]
    assert latest.security_type == "Bill"
    assert latest.security_term == "13-Week"
    assert latest.high_discount_rate is not None
    assert latest.high_investment_rate is not None


# ============================================================================
# 5. SEC Thailand Adapter Tests
# ============================================================================

def test_sec_th_fund_allocation_parses_fixture():
    fixture_csv = _read_fixture("sec_th_mf_port.csv")
    cache = ThreadSafeTTLCache(default_ttl_seconds=43200.0)
    adapter = SecThailandAdapter(cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = fixture_csv
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        snap = adapter.get_fund_asset_allocation()

    assert "2026" in snap.reporting_period or "2569" in snap.reporting_period
    assert snap.total_nav_thb is not None
    assert snap.total_nav_thb > 0
    assert len(snap.allocations) > 0

    classes = {a.asset_class for a in snap.allocations}
    assert "Common stock" in classes
    assert "Government bonds & treasury bills" in classes
    assert "NOT individual equity sector allocations" in snap.limitations


def test_sec_th_detects_waf_rejection():
    waf_html = "<html><head><title>Request Rejected</title></head><body><p>Your support ID is: 12345</p></body></html>"
    cache = ThreadSafeTTLCache(default_ttl_seconds=43200.0)
    adapter = SecThailandAdapter(cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = waf_html

    with patch("requests.get", return_value=mock_resp):
        with pytest.raises(ProviderError) as exc_info:
            adapter.get_fund_asset_allocation()

        assert "WAF rejected" in str(exc_info.value)
        assert exc_info.value.status_code == 403


# ============================================================================
# 6. MOF Thailand Public Debt Adapter Tests
# ============================================================================

def test_mof_th_public_debt_parses_fixture():
    fixture_csv = _read_fixture("mof_th_public_debt.csv")
    cache = ThreadSafeTTLCache(default_ttl_seconds=43200.0)
    adapter = MofThailandAdapter(cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = fixture_csv
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        debt = adapter.get_public_debt()

    assert debt.reporting_month != ""
    assert debt.total_debt_thb > 0
    assert debt.debt_to_gdp_pct is not None
    assert len(debt.components) > 0
    labels = {c.label_en for c in debt.components}
    assert "Government debt" in labels


# ============================================================================
# 7. Polymarket Adapter Tests
# ============================================================================

def test_polymarket_adapter_parses_fixture():
    fixture_json = json.loads(_read_fixture("polymarket_markets.json"))
    cache = ThreadSafeTTLCache(default_ttl_seconds=240.0)
    adapter = PolymarketAdapter(cache=cache)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fixture_json
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        items = adapter.get_prediction_markets(limit=5)

    assert len(items) <= 5
    for item in items:
        assert item.market_id != ""
        assert item.question != ""
        for outcome in item.outcomes:
            assert 0.0 <= outcome.price <= 1.0


# ============================================================================
# 8. SoSoValue ETF Flows Adapter Tests
# ============================================================================

def test_sosovalue_etf_flows_parses_fixtures():
    hist_json = json.loads(_read_fixture("sosovalue_history_btc.json"))
    metrics_json = json.loads(_read_fixture("sosovalue_metrics_btc.json"))
    cache = ThreadSafeTTLCache(default_ttl_seconds=21600.0)
    adapter = SoSoValueEtfFlowsAdapter(cache=cache, enabled=True)

    def mock_post(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if "historicalInflowChart" in url:
            resp.json.return_value = hist_json
        else:
            resp.json.return_value = metrics_json
        return resp

    with patch("requests.post", side_effect=mock_post):
        flow = adapter.get_spot_etf_flows("BTC")

    assert flow.asset == "BTC"
    assert flow.report_date != ""
    assert len(flow.issuers) > 0
    tickers = {i.ticker for i in flow.issuers}
    assert "IBIT" in tickers or "FBTC" in tickers


def test_sosovalue_feature_flag_disabled():
    cache = ThreadSafeTTLCache(default_ttl_seconds=21600.0)
    adapter = SoSoValueEtfFlowsAdapter(cache=cache, enabled=False)

    with pytest.raises(DataUnavailableError) as exc_info:
        adapter.get_spot_etf_flows("BTC")

    assert "disabled by feature flag" in str(exc_info.value)


# ============================================================================
# 9. OFR Financial Stress Adapter Tests
# ============================================================================

def test_ofr_adapter_with_fixture():
    fixture_path = FIXTURES_DIR / "ofr_fsi_fixture.csv"
    assert fixture_path.exists(), f"Missing OFR fixture: {fixture_path}"

    adapter = OfrHttpAdapter(fixture_path=fixture_path)
    snapshot = adapter.fetch_financial_stress()

    assert snapshot.as_of_date == "2026-09-24"
    assert snapshot.fsi_value == -0.25
    assert snapshot.data_lag_days == 2
    assert snapshot.source == "OFR"
    assert len(snapshot.categories) == 5

    cat_map = {c.label: c.value for c in snapshot.categories}
    assert "Credit" in cat_map
    assert "Equity valuation" in cat_map
    assert "Safe assets" in cat_map
    assert "Funding" in cat_map
    assert "Volatility" in cat_map
    assert cat_map["Credit"] == -0.09
    assert cat_map["Equity valuation"] == -0.05

    assert len(snapshot.trend_90d) == 4
    assert snapshot.trend_90d[-1].value == -0.25


# ============================================================================
# 10. CFTC Metals COT Adapter Tests
# ============================================================================

def test_cftc_cot_adapter_with_fixture():
    fixture_path = FIXTURES_DIR / "cftc_cot_gold_fixture.json"
    assert fixture_path.exists(), f"Missing CFTC COT fixture: {fixture_path}"

    adapter = CftcCotHttpAdapter(fixture_path=fixture_path)
    snapshot = adapter.fetch_metals_cot(commodity="gold")

    assert snapshot.commodity == "GOLD"
    assert snapshot.commodity_code == "088691"
    assert snapshot.as_of_date == "2026-09-22"
    assert snapshot.report_type == "disaggregated"
    assert snapshot.open_interest == 480000

    mm = snapshot.managed_money
    assert mm.class_name == "Managed Money"
    assert mm.long_contracts == 245100
    assert mm.short_contracts == 42300
    assert mm.net_contracts == 202800
    assert mm.change_long == 5200
    assert mm.change_short == -1800
    assert snapshot.net_managed_money == 202800

    swap = snapshot.swap_dealers
    assert swap.class_name == "Swap Dealers"
    assert swap.long_contracts == 120000
    assert swap.short_contracts == 145000
    assert swap.net_contracts == -25000

    prod = snapshot.producer_merchant
    assert prod.class_name == "Producer/Merchant/Processor/User"
    assert prod.long_contracts == 78500
    assert prod.short_contracts == 298000

    assert 0.0 <= snapshot.percentile_52w <= 100.0


# ============================================================================
# 11. BIS Central Bank Policy Rates Tests
# ============================================================================

def test_bis_policy_rates_adapter_with_fixture():
    fixture_path = FIXTURES_DIR / "bis_cbpol_fixture.csv"
    assert fixture_path.exists(), f"Missing BIS fixture: {fixture_path}"

    adapter = BisPolicyRatesHttpAdapter(fixture_path=fixture_path)
    snapshot = adapter.fetch_global_policy_rates()

    assert snapshot.source == "BIS"
    assert len(snapshot.rates) == 12

    rates_map = {r.country: r for r in snapshot.rates}
    assert "TH" in rates_map
    assert "US" in rates_map
    assert "XM" in rates_map
    assert "JP" in rates_map

    th = rates_map["TH"]
    assert th.rate_value == 2.50
    assert "bilateral repurchase" in th.rate_type.lower()
    assert th.effective_date == "2026-09"
    assert th.currency == "THB"

    us = rates_map["US"]
    assert us.rate_value == 5.00
    assert "federal funds" in us.rate_type.lower()
    assert us.previous_rate == 5.50
    assert us.last_change_date == "2026-08"

    assert "US" in snapshot.spreads_vs_bot_repo
    assert snapshot.spreads_vs_bot_repo["US"] == 250.0
    assert snapshot.spreads_vs_bot_repo["TH"] == 0.0


# ============================================================================
# 12. Nasdaq Equity Intelligence Tests
# ============================================================================

def test_nasdaq_adapter_with_fixtures():
    surprise_fixture = FIXTURES_DIR / "nasdaq_earnings_surprise_nvda.json"
    ratings_fixture = FIXTURES_DIR / "nasdaq_ratings_nvda.json"
    assert surprise_fixture.exists()
    assert ratings_fixture.exists()

    adapter = NasdaqHttpAdapter(
        surprise_fixture_path=surprise_fixture,
        ratings_fixture_path=ratings_fixture,
    )
    snapshot = adapter.fetch_earnings_consensus("NVDA")

    assert snapshot.symbol == "NVDA"
    assert snapshot.coverage_status == "full"
    assert snapshot.has_earnings_surprise is True
    assert snapshot.has_analyst_ratings is True

    assert len(snapshot.surprise_history) == 4
    latest_s = snapshot.surprise_history[0]
    assert latest_s.fiscal_quarter_end == "Jul 2026"
    assert latest_s.date_reported == "2026-08-26"
    assert latest_s.eps == 0.68
    assert latest_s.consensus_eps == 0.64
    assert latest_s.surprise_pct == 6.25

    assert snapshot.ratings is not None
    r = snapshot.ratings
    assert r.consensus == "Strong Buy"
    assert r.analyst_count == 42
    assert "GOLDMAN SACHS" in r.broker_names


def test_nasdaq_adapter_no_coverage_fallback():
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    with patch("requests.get", return_value=mock_resp):
        adapter = NasdaqHttpAdapter(
            surprise_fixture_path=Path("non_existent_surprise.json"),
            ratings_fixture_path=Path("non_existent_ratings.json"),
        )
        snapshot = adapter.fetch_earnings_consensus("SMALLCAP")
        assert snapshot.symbol == "SMALLCAP"
        assert snapshot.coverage_status == "no_coverage"
        assert snapshot.has_earnings_surprise is False
        assert snapshot.has_analyst_ratings is False
        assert snapshot.ratings is None
        assert snapshot.surprise_history == ()


# ============================================================================
# 13. Cboe Commodity Volatility Tests
# ============================================================================

def test_cboe_gvz_adapter_offline():
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0)
    adapter = CboeCommodityVolAdapter(cache=cache, fixture_dir=FIXTURES_DIR)

    snap = adapter.get_commodity_vol("GVZ")
    assert snap.index_symbol == "GVZ"
    assert "SPDR Gold Shares (GLD)" in snap.underlying_instrument
    assert snap.implied_volatility == 22.44
    assert snap.close_date == "2026-09-25"
    assert snap.change_1d_points is not None
    assert snap.sample_count >= 100
    assert snap.percentile_52w is not None
    assert snap.regime_label in ("extreme_panic", "elevated", "normal", "complacent")
    assert snap.source == "Cboe"


def test_cboe_vxslv_adapter_offline():
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0)
    adapter = CboeCommodityVolAdapter(cache=cache, fixture_dir=FIXTURES_DIR)

    snap = adapter.get_commodity_vol("VXSLV")
    assert snap.index_symbol == "VXSLV"
    assert "iShares Silver Trust (SLV)" in snap.underlying_instrument
    assert snap.implied_volatility == 35.99
    assert snap.close_date == "2026-09-25"


def test_cboe_ovx_adapter_offline():
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0)
    adapter = CboeCommodityVolAdapter(cache=cache, fixture_dir=FIXTURES_DIR)

    snap = adapter.get_commodity_vol("OVX")
    assert snap.index_symbol == "OVX"
    assert "United States Oil Fund (USO)" in snap.underlying_instrument
    assert snap.implied_volatility == 55.09


def test_cboe_unsupported_symbol():
    adapter = CboeCommodityVolAdapter()
    with pytest.raises(ProviderError) as exc_info:
        adapter.get_commodity_vol("VIX")
    assert "Unsupported" in str(exc_info.value)


# ============================================================================
# 14. SEC EDGAR Company Facts & Form 4 XML Tests
# ============================================================================

def test_sec_edgar_company_facts_offline():
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0)
    adapter = SecEdgarAdapter(
        cache=cache,
        facts_fixture_path=FIXTURES_DIR / "sec_companyfacts_nvda_fixture.json",
    )

    snap = adapter.get_company_facts("NVDA")
    assert snap.symbol == "NVDA"
    assert snap.cik == "0001045810"
    assert "NVIDIA" in snap.entity_name
    assert snap.revenue_usd is not None
    assert snap.operating_cash_flow_usd is not None
    assert snap.free_cash_flow_usd is not None
    assert snap.free_cash_flow_margin is not None
    assert len(snap.facts) >= 5

    assert any("un-audited" in lim for lim in snap.limitations)


def test_sec_edgar_form4_xml_offline():
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0)
    adapter = SecEdgarAdapter(
        cache=cache,
        submissions_fixture_path=FIXTURES_DIR / "sec_submissions_nvda_fixture.json",
        form4_fixture_path=FIXTURES_DIR / "sec_form4_nvda_fixture.xml",
    )

    snap = adapter.get_insider_trades("NVDA", limit=5)
    assert snap.symbol == "NVDA"
    assert len(snap.transactions) >= 1

    first_tx = snap.transactions[0]
    assert first_tx.reporting_owner == "Teter Timothy S."
    assert first_tx.officer_title == "EVP, General Counsel and Sec"
    assert first_tx.is_officer is True
    assert first_tx.transaction_code == "S"
    assert first_tx.shares == 12483.0
    assert first_tx.price_per_share == 222.1932
    assert first_tx.notional_usd is not None


# ============================================================================
# 15. RSS News Discovery Tests
# ============================================================================

def test_rss_news_discovery_offline():
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0)
    adapter = RssNewsDiscoveryAdapter(
        cache=cache,
        fixture_path=FIXTURES_DIR / "google_news_nvda_fixture.xml",
    )

    snap = adapter.get_news_candidates("NVDA", limit=10)
    assert snap.query_symbol == "NVDA"
    assert snap.status == "ok"
    assert len(snap.items) == 10

    first = snap.items[0]
    assert first.headline
    assert first.publisher
    assert first.article_url.startswith("http")
    assert first.published_at


def test_rss_news_discovery_empty_or_degraded():
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0)
    adapter = RssNewsDiscoveryAdapter(
        cache=cache,
        fixture_path=FIXTURES_DIR / "google_news_empty_fixture.xml",
    )

    snap = adapter.get_news_candidates("UNKNOWN", limit=10)
    assert snap.query_symbol == "UNKNOWN"
    assert len(snap.items) == 0
