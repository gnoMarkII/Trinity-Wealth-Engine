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




class TestIngestFundamentalsThaiStock:
    def test_th_market_normalizes_ticker(self, mock_yf_ticker):
        result = fundamentals.ingest_stock_fundamentals.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        # Display ใช้ bare ticker
        assert "PTT Fundamentals" in result
        assert "(PTT, TH)" in result
        # Currency suffix ต้องเป็น THB
        assert "THB" in result
        assert "1.50T THB" in result  # market cap formatted

    def test_us_market_default(self, mock_yf_ticker):
        result = fundamentals.ingest_stock_fundamentals.invoke({"ticker": "AAPL"})
        assert mock_yf_ticker["ticker"] == "AAPL"
        assert "USD" in result
        assert "(AAPL, US)" in result



import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from tools.market.fundamentals import ingest_stock_fundamentals, ingest_financial_health

class TestFundamentalsExceptions:
    @patch("tools.market.fundamentals._with_retry")
    def test_fundamentals_exception(self, mock_retry):
        mock_retry.side_effect = Exception("yf error")
        
        res = ingest_stock_fundamentals.func("AAPL", "US")
        assert "ERROR: ไม่สามารถดึงข้อมูล AAPL (US) ได้:" in res

    @patch("tools.market.fundamentals.yf.Ticker")
    def test_fundamentals_no_info(self, mock_ticker):
        # Line 35
        mock_instance = MagicMock()
        mock_instance.info = {}
        mock_ticker.return_value = mock_instance
        
        res = ingest_stock_fundamentals.func("AAPL", "US")
        assert "ERROR: ไม่พบข้อมูลสำหรับ ticker" in res

    @patch("tools.market.fundamentals.yf.Ticker")
    def test_fundamentals_esg(self, mock_ticker):
        # Lines 65-97
        mock_instance = MagicMock()
        mock_instance.info = {"quoteType": "EQUITY", "shortName": "Apple"}
        
        # Create a mock dataframe for sustainability
        df = pd.DataFrame(
            {"Value": [15.0, 5.0, 5.0, 5.0]},
            index=["totalEsg", "environmentScore", "socialScore", "governanceScore"]
        )
        df["Value"] = df["Value"].astype(object)
        # Mock dataframe with weird type to trigger typeerror branch
        df.loc["socialScore", "Value"] = "invalid_string"
        
        mock_instance.sustainability = df
        mock_ticker.return_value = mock_instance
        
        res = ingest_stock_fundamentals.func("AAPL", "US")
        assert "## คะแนนความยั่งยืน ESG" in res
        assert "15.0" in res

    @patch("tools.market.fundamentals._yf_info")
    def test_financial_health_exception(self, mock_info):
        # Lines 218-220
        mock_info.side_effect = Exception("yf info error")
        res = ingest_financial_health.func("AAPL", "US")
        assert "ERROR: ไม่สามารถดึงข้อมูล AAPL (US) ได้:" in res

    @patch("tools.market.fundamentals._yf_info")
    def test_financial_health_no_info(self, mock_info):
        # Line 223
        mock_info.return_value = {}
        res = ingest_financial_health.func("AAPL", "US")
        assert "ERROR: ไม่พบข้อมูลสำหรับ ticker" in res
