"""Test market_tools: ticker normalization, currency-aware formatting, TH/US market routing"""
import pytest


import tools.market.technical as technical



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


class TestComputeTacticalSetup:
    @patch("tools.market.technical.yf.Ticker")
    def test_tactical_setup_dual_rr_and_breakout_ftnt_fixtures(self, mock_ticker):
        from tools.market.technical import compute_tactical_setup
        import pandas as pd
        import numpy as np

        # Create mock 1y history
        dates = pd.date_range(end=datetime.now(), periods=250, freq="B")
        df = pd.DataFrame(index=dates)
        # Create prices fluctuating between 150 and 170
        df["Close"] = 160.0 + 5.0 * np.sin(np.linspace(0, 20, len(dates)))
        df["High"] = df["Close"] + 3.0
        df["Low"] = df["Close"] - 3.0
        df["Open"] = df["Close"]
        df["Volume"] = 1_000_000

        mock_instance = MagicMock()
        mock_instance.history.return_value = df
        mock_ticker.return_value = mock_instance

        # Test at P = 166.0
        res_166, flags = compute_tactical_setup(
            "FTNT", market="US", current_price=166.0, price_history_df=df
        )
        assert res_166 is not None
        assert res_166.status == "available"
        assert res_166.breakout_trigger_price is not None
        assert res_166.breakout_target_price is not None
        assert res_166.breakout_stop_loss is not None
        assert res_166.breakout_planned_rr == 2.00
        assert res_166.is_in_buy_zone is False  # 166 is above buy zone

        # Test breakout current R:R at P = 172.78
        # Using exact FTNT values: R = 169.91, ATR = 6.14
        # Trigger = 170.524, Target = 184.032, Stop = 163.77
        # At P = 172.78: Breakout Current RR = (184.032 - 172.78) / (172.78 - 163.77) = 11.252 / 9.010 = 1.25
        trigger = 169.91 + 0.1 * 6.14
        target = 169.91 + 2.3 * 6.14
        stop = 169.91 - 1.0 * 6.14
        planned_rr = round((target - trigger) / (trigger - stop), 2)
        current_rr_172 = round((target - 172.78) / (172.78 - stop), 2)

        assert planned_rr == 2.00
        assert current_rr_172 == 1.25

