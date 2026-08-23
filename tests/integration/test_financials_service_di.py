"""Integration tests for FinancialsService using pure Dependency Injection (No Monkeypatching)."""
import time
from typing import Optional
from tools.market.financials.adapters.in_memory_cache_adapter import InMemoryCacheAdapter
from tools.market.financials.domain.models import (
    FinancialCellDTO,
    FinancialPeriodDTO,
    FinancialStatementCategoryDTO,
    FinancialStatementsDTO,
)
from tools.market.financials.ports.provider_port import FinancialStatementProviderPort
from tools.market.financials.service import FinancialsService


class FakeUsFinancialProvider(FinancialStatementProviderPort):
    def __init__(self, fetch_count: int = 0):
        self.call_count = 0

    def fetch_statements(self, ticker: str, provider_symbol: str) -> Optional[FinancialStatementsDTO]:
        self.call_count += 1
        p_inc = FinancialPeriodDTO(
            period_key="2024-FY",
            fiscal_year=2024,
            period_end_date="2024-12-31",
            period_kind="duration",
            form_type="10-K",
            items={"revenue": FinancialCellDTO(value=1000.0)},
        )
        cat_inc = FinancialStatementCategoryDTO(statement_type="income", period_kind="duration", periods=[p_inc], line_items=[])
        return FinancialStatementsDTO(
            schema_version=6,
            ticker=ticker,
            market="US",
            currency="USD",
            provider="edgartools",
            provider_symbol=provider_symbol,
            data_status="ok",
            coverage_status="complete",
            core_coverage_status="complete",
            expanded_coverage_status="complete",
            expanded_data_status="complete",
            annual=[cat_inc],
            quarterly=[cat_inc],
        )


def test_financials_service_caching_and_di():
    cache = InMemoryCacheAdapter()
    fake_provider = FakeUsFinancialProvider()

    service = FinancialsService(cache_port=cache, us_provider=fake_provider)

    # 1. First call -> Live fetch from provider
    res1 = service.get_financial_statements(ticker="NVDA", market="US")
    assert res1.ticker == "NVDA"
    assert fake_provider.call_count == 1

    # 2. Second call -> Fresh cached data (Provider should NOT be called again)
    res2 = service.get_financial_statements(ticker="NVDA", market="US")
    assert res2.ticker == "NVDA"
    assert fake_provider.call_count == 1

    # 3. Force refresh -> Provider is called again
    res3 = service.get_financial_statements(ticker="NVDA", market="US", force_refresh=True)
    assert res3.ticker == "NVDA"
    assert fake_provider.call_count == 2
