"""Application use cases for read-only equity research views."""
from typing import Any, Dict, List, Optional

from application.equity.ports import EquityResearchQueryPort


class EquityDataCorruptError(RuntimeError):
    """The newest authoritative sidecar exists but cannot be read safely."""


class EquityNoteAccessError(ValueError):
    """A note path violates the Vault boundary or file type policy."""


class EquityResearchQueryService:
    """Coordinates equity read models without knowing their storage format."""

    def __init__(self, query_port: EquityResearchQueryPort) -> None:
        self._query_port = query_port

    def list_latest(self) -> List[Dict[str, Any]]:
        return self._query_port.list_latest()

    def get_detail(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self._query_port.get_detail(ticker.upper())

    def get_news(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self._query_port.get_news(ticker.upper())

    def list_notes(self, ticker: str, days: int = 3) -> Dict[str, Any]:
        return self._query_port.list_notes(ticker.upper(), days=days)

    def read_note(self, relative_path: str) -> Dict[str, Any]:
        return self._query_port.read_note(relative_path)
