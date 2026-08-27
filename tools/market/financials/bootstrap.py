"""Financials Service Composition Root."""
from typing import Optional

from tools.market.financials.ports.cache_port import FinancialCachePort
from tools.market.financials.ports.provider_port import FinancialStatementProviderPort
from tools.market.financials.service import FinancialsService


def build_financials_service(
    cache_port: Optional[FinancialCachePort] = None,
    us_provider: Optional[FinancialStatementProviderPort] = None,
    th_provider: Optional[FinancialStatementProviderPort] = None,
    us_fallback_provider: Optional[FinancialStatementProviderPort] = None,
) -> FinancialsService:
    """Build FinancialsService with injected or default concrete adapters."""
    from tools.market.financials.adapters.sqlite_cache_adapter import SQLiteCacheAdapter
    from tools.market.financials.adapters.composite_us_provider import CompositeUsFinancialProvider
    from tools.market.financials.adapters.thai_set_provider import ThaiSetFinancialProvider

    return FinancialsService(
        cache_port=cache_port or SQLiteCacheAdapter(),
        us_provider=us_provider or CompositeUsFinancialProvider(),
        us_fallback_provider=us_fallback_provider or ThaiSetFinancialProvider(market="US"),
        th_provider=th_provider or ThaiSetFinancialProvider(),
    )


def build_default_financials_service() -> FinancialsService:
    """Build default FinancialsService wiring default concrete adapters."""
    return build_financials_service()
