import pytest
from unittest.mock import patch, MagicMock
from tools.market.financials import ingest_financial_trends
from tools.market.consensus import ingest_stock_consensus
from tools.market.news import ingest_stock_news
from tools.market.core import _normalize_yf_ticker, _currency_for, _yf_info, _yf_financials, _yf_news, _fmt_number, _fmt_large, _fmt_fin

class TestMarketCoreAddons:
    def test_normalize_yf_ticker(self):
        assert _normalize_yf_ticker("AAPL", "US") == "AAPL"
        assert _normalize_yf_ticker("PTT", "TH") == "PTT.BK"
        assert _normalize_yf_ticker("PTT.BK", "TH") == "PTT.BK"

    def test_currency_for(self):
        assert _currency_for("TH") == "THB"
        assert _currency_for("US") == "USD"
        assert _currency_for("OTHER") == "USD"

    @patch("tools.market.core.yf.Ticker")
    def test_yf_info(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.info = {"test": 1}
        mock_ticker.return_value = mock_instance
        assert _yf_info("AAPL") == {"test": 1}
        
    @patch("tools.market.core.yf.Ticker")
    def test_yf_financials(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.financials = {"test": 1}
        mock_ticker.return_value = mock_instance
        assert _yf_financials("AAPL") == {"test": 1}

    @patch("tools.market.core.yf.Ticker")
    def test_yf_news(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.news = [{"test": 1}]
        mock_ticker.return_value = mock_instance
        assert _yf_news("AAPL") == [{"test": 1}]

    def test_fmt_fin(self):
        assert _fmt_fin(1_500_000_000, "USD") == "1.50B USD"
        assert _fmt_fin(1_500_000, "USD") == "1.50M USD"
        assert _fmt_fin(1_500, "USD") == "1500 USD"
        assert _fmt_fin(None, "USD") == "N/A"

class TestFinancialsAddons:
    @patch("tools.market.financials._with_retry")
    def test_financial_trends_exception(self, mock_retry):
        mock_retry.side_effect = Exception("yf error")
        
        res = ingest_financial_trends.func("AAPL", "US")
        assert "ERROR: ไม่สามารถดึงข้อมูล AAPL (US) ได้:" in res

    @patch("tools.market.financials.yf.Ticker")
    def test_financial_trends_no_info(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.info = {}
        mock_instance.financials = None
        mock_ticker.return_value = mock_instance
        
        res = ingest_financial_trends.func("AAPL", "US")
        assert "ERROR: ไม่พบข้อมูลงบการเงินสำหรับ AAPL (US)" in res

    @patch("tools.market.financials.yf.Ticker")
    def test_financial_trends_missing_rows(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.info = {"shortName": "Apple"}
        # Create a financials dataframe missing Revenue and Net Income
        import pandas as pd
        mock_instance.financials = pd.DataFrame(
            [[100, 200]], index=["Operating Expense"], columns=["2023", "2022"]
        )
        mock_ticker.return_value = mock_instance
        res = ingest_financial_trends.func("AAPL", "US")
        assert "ไม่พบแถว 'Total Revenue'" in res
        assert "ไม่พบแถว 'Net Income'" in res

class TestConsensusAddons:
    @patch("tools.market.consensus._yf_info")
    def test_consensus_exception(self, mock_info):
        mock_info.side_effect = Exception("yf error")
        
        res = ingest_stock_consensus.func("AAPL", "US")
        assert "ERROR: ไม่สามารถดึงข้อมูล AAPL (US) ได้:" in res

    @patch("tools.market.consensus.yf.Ticker")
    def test_consensus_no_info(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.info = {}
        mock_ticker.return_value = mock_instance
        
        res = ingest_stock_consensus.func("AAPL", "US")
        assert "ERROR: ไม่พบข้อมูลสำหรับ ticker" in res
        
    @patch("tools.market.consensus.yf.Ticker")
    def test_consensus_no_recommendations(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.info = {"quoteType": "EQUITY", "shortName": "Apple"}
        mock_instance.recommendations = None
        mock_ticker.return_value = mock_instance
        
        res = ingest_stock_consensus.func("AAPL", "US")
        assert "## 2. คำแนะนำนักวิเคราะห์ (Recommendations)" not in res

class TestNewsAddons:
    @patch("tools.market.news._yf_news")
    def test_news_exception(self, mock_news):
        mock_news.side_effect = Exception("news error")
        
        res = ingest_stock_news.func("AAPL", "US")
        assert "ERROR: ไม่สามารถดึงข่าวของ AAPL (US) ได้:" in res

    @patch("tools.market.news._yf_news")
    def test_news_empty(self, mock_news):
        mock_news.return_value = []
        res = ingest_stock_news.func("AAPL", "US")
        assert "ไม่พบข่าวสำหรับ AAPL (US)" in res
