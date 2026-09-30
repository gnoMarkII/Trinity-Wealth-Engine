"""Domain Exceptions for Terminal V2.

Pure Python exceptions representing domain invariants and failure states.
No external dependencies.
"""
from typing import Optional


class DomainError(Exception):
    """Base class for all domain errors."""
    pass


class DataUnavailableError(DomainError):
    """Raised when the requested data is completely unavailable and no cache exists."""

    def __init__(self, message: str, capability: Optional[str] = None, source: Optional[str] = None):
        super().__init__(message)
        self.capability = capability
        self.source = source


class ProviderError(DomainError):
    """Raised when a data provider fails to respond or returns malformed data."""

    def __init__(self, message: str, source: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.source = source
        self.status_code = status_code


class InvalidCapabilityError(DomainError):
    """Raised when an unknown or unsupported capability is requested."""
    pass


class SymbolMarketMismatchError(DomainError):
    """Raised when a symbol is requested under an incompatible capability or market.

    For example, requesting a cash equity quote for a DEX perp symbol (xyz:TSLA)
    or trying to route a cash equity quote to a perpetuals provider.
    """
    pass
