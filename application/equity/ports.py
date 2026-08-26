"""Outbound ports for Equity application use cases.

Ports intentionally expose plain mappings rather than FastAPI/Pydantic
objects.  HTTP response models belong to the inbound adapter.
"""
from typing import Any, Dict, List, Optional, Protocol


class AssetResolverPort(Protocol):
    def resolve(self, ticker: str) -> Any:
        ...


class AnalystCachePort(Protocol):
    def get(self, ticker: str) -> Optional[Dict[str, Any]]:
        ...

    def upsert(self, ticker: str, data: Dict[str, Any]) -> None:
        ...


class AnalystProviderPort(Protocol):
    def price_targets(self, provider_symbol: str) -> Dict[str, Any]:
        ...

    def calendar(self, provider_symbol: str) -> Dict[str, Any]:
        ...

    def earnings_history(self, provider_symbol: str, exchange_tz: str) -> Any:
        ...


class ValuationLedgerPort(Protocol):
    def latest(self, ticker: str) -> Optional[Dict[str, Any]]:
        ...

    def record(self, **kwargs: Any) -> None:
        ...


class ValuationSidecarPort(Protocol):
    def latest(self, ticker: str) -> Any:
        ...


class InsiderLedgerPort(Protocol):
    def list_records(self, ticker: str, since_date: str) -> List[Dict[str, Any]]:
        ...


class InsiderSyncPort(Protocol):
    def sync(self, ticker: str) -> None:
        ...


class InsiderHistoryProviderPort(Protocol):
    """External insider-history provider returning normalized plain mappings."""

    def fetch(self, ticker: str) -> List[Dict[str, Any]]:
        ...


class EquityResearchQueryPort(Protocol):
    """Read-only equity research query boundary.

    The port deliberately returns plain mappings.  Filesystem paths, Pydantic
    response models, and Markdown/JSON parsing belong to the driven adapter or
    the inbound HTTP adapter respectively.
    """

    def list_latest(self) -> List[Dict[str, Any]]:
        ...

    def get_detail(self, ticker: str) -> Optional[Dict[str, Any]]:
        ...

    def get_news(self, ticker: str) -> Optional[Dict[str, Any]]:
        ...

    def list_notes(self, ticker: str, days: int = 3) -> Dict[str, Any]:
        ...

    def read_note(self, relative_path: str) -> Dict[str, Any]:
        ...


class FinancialsQueryPort(Protocol):
    """Application-facing boundary for financial statement retrieval.

    The concrete financials service owns cache/provider policy.  Equity
    application code only needs this narrow query operation and must not know
    which provider or cache implementation is used.
    """

    def get_financial_statements(
        self,
        ticker: str,
        market: str = "US",
        provider_symbol: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Any:
        ...
