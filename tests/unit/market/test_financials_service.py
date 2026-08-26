"""Unit tests for FinancialsService dependency injection and caching."""
import time
import pytest
from unittest.mock import MagicMock
from tools.market.financials.ports.cache_port import FinancialCachePort, CacheEntry
from tools.market.financials.ports.provider_port import FinancialStatementProviderPort
from tools.market.financials.domain.models import FinancialStatementsDTO
from tools.market.financials.service import FinancialsService


def test_financials_service_cache_hit():
    mock_cache = MagicMock(spec=FinancialCachePort)
    mock_us_provider = MagicMock(spec=FinancialStatementProviderPort)
    mock_th_provider = MagicMock(spec=FinancialStatementProviderPort)

    dummy_dto = FinancialStatementsDTO(
        ticker="AAPL",
        provider_symbol="AAPL",
        company_name="Apple Inc.",
        currency="USD",
        market="US",
        data_status="ok",
    )
    mock_cache.get.return_value = CacheEntry(
        statements=dummy_dto,
        provider="edgartools",
        synced_at=time.time() - 60,
    )

    service = FinancialsService(
        cache_port=mock_cache,
        us_provider=mock_us_provider,
        th_provider=mock_th_provider,
    )

    result = service.get_financial_statements(ticker="AAPL", market="US", force_refresh=False)
    assert result.ticker == "AAPL"
    mock_cache.get.assert_called_once_with("US", "AAPL")
    mock_us_provider.fetch_statements.assert_not_called()
