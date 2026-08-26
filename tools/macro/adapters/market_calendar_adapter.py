"""Driven adapters for market symbol resolution and calendar fetching."""
from typing import Any, Dict

class MarketAssetResolverAdapter:
    def resolve(self, symbol: str) -> Any:
        from tools.market import asset_resolver

        return asset_resolver.resolve_asset(symbol)


class MarketCalendarAdapter:
    def fetch(self, provider_symbol: str) -> Dict[str, Any]:
        from tools.market import calendar

        result = calendar.get_asset_calendar(provider_symbol)
        return result if isinstance(result, dict) else {}
