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




class TestIngestStockMomentumThaiStock:
    def test_th_market_currency(self, mock_yf_ticker):
        result = technical.ingest_stock_momentum.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        assert "40.00 THB" in result  # current price
        assert "39.00 THB" in result  # MA50



import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from tools.market.technical import ingest_stock_momentum, _summarize_insider_transactions

class TestTechnicalExceptions:
    @patch("tools.market.technical._with_retry")
    def test_momentum_exception(self, mock_with_retry):
        # Lines 64-66
        mock_with_retry.side_effect = Exception("yf error")
        
        res = ingest_stock_momentum.func("AAPL", "US")
        assert "ERROR: ไม่สามารถดึงข้อมูล AAPL (US) ได้:" in res

    @patch("tools.market.technical.yf.Ticker")
    def test_momentum_no_info(self, mock_ticker):
        # Line 69
        mock_instance = MagicMock()
        mock_instance.info = {}
        mock_ticker.return_value = mock_instance
        
        res = ingest_stock_momentum.func("AAPL", "US")
        assert "ERROR: ไม่พบข้อมูลสำหรับ ticker" in res

class TestSummarizeInsiderTransactions:
    def test_summarize_insider_transactions_empty(self):
        mock_tk = MagicMock()
        mock_tk.insider_transactions = pd.DataFrame()
        assert _summarize_insider_transactions(mock_tk) == "ไม่พบข้อมูล"
        
    def test_summarize_insider_transactions_buys_sells(self):
        # Create a mock dataframe for insider transactions
        recent_date = datetime.now() - timedelta(days=10)
        df = pd.DataFrame({
            "Start Date": [recent_date, recent_date, recent_date],
            "Transaction": ["Buy", "Sell", "Purchase"]
        })
        mock_tk = MagicMock()
        mock_tk.insider_transactions = df
        
        res = _summarize_insider_transactions(mock_tk)
        assert "ซื้อมากกว่าขาย" in res
        
    def test_summarize_insider_transactions_more_sells(self):
        recent_date = datetime.now() - timedelta(days=10)
        df = pd.DataFrame({
            "Start Date": [recent_date, recent_date, recent_date],
            "Transaction": ["Sell", "Sell", "Buy"]
        })
        mock_tk = MagicMock()
        mock_tk.insider_transactions = df
        
        res = _summarize_insider_transactions(mock_tk)
        assert "ขายมากกว่าซื้อ" in res
        
    def test_summarize_insider_transactions_equal(self):
        recent_date = datetime.now() - timedelta(days=10)
        df = pd.DataFrame({
            "Start Date": [recent_date, recent_date],
            "Transaction": ["Buy", "Sell"]
        })
        mock_tk = MagicMock()
        mock_tk.insider_transactions = df
        
        res = _summarize_insider_transactions(mock_tk)
        assert "ซื้อและขายเท่ากัน" in res
        
    def test_summarize_insider_transactions_no_tx_col(self):
        recent_date = datetime.now() - timedelta(days=10)
        df = pd.DataFrame({
            "Start Date": [recent_date, recent_date]
        })
        mock_tk = MagicMock()
        mock_tk.insider_transactions = df
        
        res = _summarize_insider_transactions(mock_tk)
        assert "ไม่สามารถแยกประเภทได้" in res
        
    def test_summarize_insider_transactions_old_dates(self):
        old_date = datetime.now() - timedelta(days=200)
        df = pd.DataFrame({
            "Start Date": [old_date],
            "Transaction": ["Buy"]
        })
        mock_tk = MagicMock()
        mock_tk.insider_transactions = df
        
        res = _summarize_insider_transactions(mock_tk)
        assert "ไม่มีรายการใน 6 เดือนล่าสุด" in res
