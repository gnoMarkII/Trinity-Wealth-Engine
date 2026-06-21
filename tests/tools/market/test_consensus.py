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




class TestIngestStockConsensusThaiStock:
    def test_th_market_currency(self, mock_yf_ticker):
        result = consensus.ingest_stock_consensus.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        assert "45.00 THB" in result  # target mean
        assert "market: TH" in result

