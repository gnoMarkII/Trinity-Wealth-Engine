"""Test market_tools: ticker normalization, currency-aware formatting, TH/US market routing"""
import pytest


import tools.market.core as core
import tools.market.fundamentals as fundamentals
import tools.market.financials as financials
import tools.market.news as news
import tools.market.technical as technical
import tools.market.consensus as consensus
from types import SimpleNamespace
mt = SimpleNamespace(
    ingest_stock_fundamentals=fundamentals.ingest_stock_fundamentals,
    ingest_financial_health=fundamentals.ingest_financial_health,
    ingest_financial_trends=financials.ingest_financial_trends,
    ingest_stock_news=news.ingest_stock_news,
    ingest_stock_momentum=technical.ingest_stock_momentum,
    ingest_stock_consensus=consensus.ingest_stock_consensus,
)



# --- Pure helpers ---




class TestNormalizeYfTicker:
    def test_us_unchanged(self):
        assert core._normalize_yf_ticker("AAPL", "US") == "AAPL"

    def test_th_adds_bk_suffix(self):
        assert core._normalize_yf_ticker("PTT", "TH") == "PTT.BK"

    def test_th_no_double_suffix(self):
        assert core._normalize_yf_ticker("PTT.BK", "TH") == "PTT.BK"

    def test_strip_and_upper(self):
        assert core._normalize_yf_ticker(" ptt ", "TH") == "PTT.BK"
        assert core._normalize_yf_ticker(" aapl ", "US") == "AAPL"


class TestCurrencyFor:
    def test_th_returns_thb(self):
        assert core._currency_for("TH") == "THB"

    def test_us_returns_usd(self):
        assert core._currency_for("US") == "USD"


class TestFmtLargeWithCurrency:
    def test_default_currency_usd(self):
        assert core._fmt_large(1_500_000_000) == "1.50B USD"

    def test_custom_currency_thb(self):
        assert core._fmt_large(1_500_000_000, "THB") == "1.50B THB"

    def test_trillion(self):
        assert core._fmt_large(2.5e12, "USD") == "2.50T USD"

    def test_million(self):
        assert core._fmt_large(5_000_000, "THB") == "5.00M THB"

    def test_none_returns_na(self):
        assert core._fmt_large(None, "THB") == "N/A"


class TestFmtFinWithCurrency:
    def test_positive_billions_thb(self):
        assert core._fmt_fin(2_500_000_000, "THB") == "2.50B THB"

    def test_negative_billions_usd(self):
        assert core._fmt_fin(-3_000_000_000, "USD") == "-3.00B USD"

    def test_default_back_compat(self):
        assert "USD" in core._fmt_fin(1_000_000_000)

