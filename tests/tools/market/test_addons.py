import pytest
from unittest.mock import patch, MagicMock
from tools.market.consensus import ingest_stock_consensus
from tools.market.news import ingest_stock_news, save_company_news_to_vault
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
    @patch("tools.market.news._rss_news_candidates")
    @patch("tools.market.news._yf_news")
    def test_news_exception(self, mock_news, mock_rss):
        mock_news.side_effect = Exception("news error")
        mock_rss.return_value = MagicMock(items=[])

        res = ingest_stock_news.func("AAPL", "US")
        assert "ERROR: ไม่สามารถดึงข่าวของ AAPL (US) ได้:" in res
        assert "news error" in res

    @patch("tools.market.news._rss_news_candidates")
    @patch("tools.market.news._yf_news")
    def test_news_empty(self, mock_news, mock_rss):
        mock_news.return_value = []
        mock_rss.return_value = MagicMock(items=[])
        res = ingest_stock_news.func("AAPL", "US")
        assert "ไม่พบข่าวสำหรับ AAPL (US)" in res

    @patch("tools.market.news._rss_news_candidates")
    @patch("tools.market.news._yf_news")
    def test_news_rss_supplement_when_yf_empty(self, mock_news, mock_rss):
        mock_news.return_value = []
        fake_item = MagicMock(
            headline="Apple introduces groundbreaking AI capabilities",
            publisher="TechCrunch",
            article_url="https://techcrunch.com/apple-ai",
            published_at="Sat, 17 Oct 2026 12:00:00 +0000",
        )
        mock_rss.return_value = MagicMock(items=[fake_item])

        res = ingest_stock_news.func("AAPL", "US")
        assert "schema_version: 2" in res
        assert "AAPL Latest News" in res
        assert "Apple introduces groundbreaking AI capabilities" in res
        assert "TechCrunch" in res

    @patch("tools.archivist.writer.write_raw_markdown")
    def test_save_company_news_to_vault_frontmatter_fallback(self, mock_tool):
        mock_tool.invoke.return_value = "saved"
        sample_md = """---
title: NVDA Latest News 2026-10-18
ticker: NVDA
date: 2026-10-18
---
# Content
"""
        res = save_company_news_to_vault(sample_md)
        assert res == "saved"
        mock_tool.invoke.assert_called_once_with({
            "content": sample_md,
            "folder_path": "30_Knowledge_Base/News",
            "filename": "2026-10-18 NVDA Latest News 2026-10-18",
        })
