"""Unit tests สำหรับ agents/equity_narrative_agent.py (RunnableLambda, ไม่ใช้ ReAct)"""
from unittest.mock import MagicMock, patch

from agents.equity_narrative_agent import create_equity_narrative
from schemas.micro_quant_schemas import EquitySentimentContext


class TestCreateEquityNarrative:
    @patch("tools.market.news.ingest_stock_news")
    @patch("tools.archivist.search.search_memories_with_evidence")
    def test_gathers_context_from_tools_directly_then_forces_schema(self, mock_search, mock_news):
        mock_search.invoke.return_value = "Vault snippet about AAPL"
        mock_news.invoke.return_value = "AAPL beats earnings"

        expected = EquitySentimentContext(
            evaluated_at="2026-07-20T00:00:00+00:00",
            market_sentiment="bullish",
            sources_summary="based on vault + news",
        )
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = expected
        mock_model.with_structured_output.return_value = mock_structured

        runnable = create_equity_narrative(mock_model)
        result = runnable.invoke({"ticker": "AAPL", "market": "US"})

        # tool ถูกเรียกตรงๆ ในโค้ด ไม่ผ่าน LLM tool-calling loop
        mock_search.invoke.assert_called_once_with(
            {
                "keyword": "AAPL",
                "retrieval_namespace": "primary",
                "production_mode": False,
            }
        )
        mock_news.invoke.assert_called_once_with({"ticker": "AAPL", "market": "US"})
        mock_model.with_structured_output.assert_called_once_with(EquitySentimentContext)

        content = result["messages"][0].content
        assert "bullish" in content
        assert result["messages"][0].name == "equity_narrative"
        assert result["equity_news_raw"] == "AAPL beats earnings"

    @patch("tools.market.news.ingest_stock_news")
    @patch("tools.archivist.search.search_memories_with_evidence")
    def test_uses_company_name_in_vault_search_when_available(self, mock_search, mock_news):
        mock_search.invoke.return_value = "Vault snippet"
        mock_news.invoke.return_value = "News"

        expected = EquitySentimentContext(
            evaluated_at="2026-07-20T00:00:00+00:00",
            market_sentiment="neutral",
            sources_summary="test",
        )
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = expected
        mock_model.with_structured_output.return_value = mock_structured

        runnable = create_equity_narrative(mock_model)
        runnable.invoke({"ticker": "AAPL", "market": "US", "company_name": "Apple Inc."})

        mock_search.invoke.assert_called_once_with(
            {
                "keyword": "AAPL Apple Inc.",
                "retrieval_namespace": "primary",
                "production_mode": False,
            }
        )

    @patch("tools.market.news.ingest_stock_news")
    @patch("tools.archivist.search.search_memories_with_evidence")
    def test_tool_failure_does_not_crash_still_calls_llm(self, mock_search, mock_news):
        mock_search.invoke.side_effect = Exception("vault unavailable")
        mock_news.invoke.side_effect = Exception("network error")

        expected = EquitySentimentContext(
            evaluated_at="2026-07-20T00:00:00+00:00",
            market_sentiment="neutral",
            sources_summary="no data found",
        )
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = expected
        mock_model.with_structured_output.return_value = mock_structured

        runnable = create_equity_narrative(mock_model)
        result = runnable.invoke({"ticker": "BROKEN", "market": "US"})

        mock_structured.invoke.assert_called_once()
        assert "neutral" in result["messages"][0].content
