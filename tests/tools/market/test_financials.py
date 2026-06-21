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




class TestIngestFinancialHealthThaiStock:
    def test_th_market_currency_in_output(self, mock_yf_ticker):
        result = fundamentals.ingest_financial_health.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        # Cash flow ต้องแสดงเป็น THB
        assert "100.00B THB" in result  # operatingCashflow
        assert "80.00B THB" in result   # freeCashflow
        assert "USD" not in result.replace("USDT", "")  # ไม่มี USD label (ยกเว้น token USDT ถ้ามี)


class TestIngestFinancialTrendsThaiStock:
    def test_th_market_currency(self, mock_yf_ticker):
        result = financials.ingest_financial_trends.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        assert "B THB" in result  # revenue/income formatted in THB
        assert "market: TH" in result

