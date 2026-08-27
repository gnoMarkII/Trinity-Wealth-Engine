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


def test_financials_service_uses_injected_us_fallback_provider():
    """The application service must use a port, never construct a concrete fallback."""
    mock_cache = MagicMock(spec=FinancialCachePort)
    mock_cache.get.return_value = None
    primary = MagicMock(spec=FinancialStatementProviderPort)
    fallback = MagicMock(spec=FinancialStatementProviderPort)
    primary.fetch_statements.return_value = None
    fallback.fetch_statements.return_value = None

    service = FinancialsService(
        cache_port=mock_cache,
        us_provider=primary,
        us_fallback_provider=fallback,
    )

    result = service.get_financial_statements(ticker="USFALLBACK", market="US")

    primary.fetch_statements.assert_called_once_with("USFALLBACK", "USFALLBACK")
    fallback.fetch_statements.assert_called_once_with("USFALLBACK", "USFALLBACK")
    assert result.data_status == "empty"


def test_legacy_financials_entrypoint_uses_composition_root(monkeypatch):
    """Compatibility API must not instantiate an under-wired FinancialsService."""
    import tools.market.financials as financials

    expected = MagicMock()
    expected.get_financial_statements.return_value = "result"
    monkeypatch.setattr(
        "tools.market.financials.bootstrap.build_default_financials_service",
        lambda: expected,
    )

    result = financials.get_financial_statements("AAPL", provider_symbol="AAPL-US")

    assert result == "result"
    expected.get_financial_statements.assert_called_once_with(
        ticker="AAPL",
        market="US",
        provider_symbol="AAPL-US",
        force_refresh=False,
    )
