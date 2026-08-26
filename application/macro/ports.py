"""Outbound ports for Macro and portfolio calendar use cases."""
from typing import Any, Dict, List, Optional, Protocol


class StrategySnapshotPort(Protocol):
    def latest(self) -> Dict[str, Any]:
        ...


class IndicatorSeriesPort(Protocol):
    def load(self, series_key: str, range_name: str) -> List[Dict[str, Any]]:
        ...


class NewsFunnelPort(Protocol):
    def pending(self) -> List[Dict[str, Any]]:
        ...

    def filtered(self) -> List[Dict[str, Any]]:
        ...

    def reject(self, event_id: str) -> int:
        """Reject an event and return the remaining pending count."""
        ...


class NewsFunnelCardPort(Protocol):
    """Persistence port for the approval card emitted by the funnel."""

    def upsert_open_card(self, card: Dict[str, Any]) -> Dict[str, Any]:
        """Atomically create or update the open card for a flow."""
        ...

    def find_open_card(self, flow: str) -> Optional[Dict[str, Any]]:
        ...

    def create_card(self, card: Dict[str, Any]) -> None:
        ...

    def update_card(self, card_id: str, card: Dict[str, Any]) -> None:
        ...


class NewsFunnelPromptPort(Protocol):
    def format_prompt(self, period: str, events: List[Dict[str, Any]]) -> str:
        ...


class PortfolioReadPort(Protocol):
    def get_state(self, portfolio_id: str) -> Any:
        ...

    def get_watchlist(self, portfolio_id: str) -> Any:
        ...


class AssetResolverPort(Protocol):
    def resolve(self, symbol: str) -> Any:
        ...


class MarketCalendarPort(Protocol):
    def fetch(self, provider_symbol: str) -> Dict[str, Any]:
        ...
