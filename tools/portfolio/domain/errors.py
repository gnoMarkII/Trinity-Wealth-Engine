class PortfolioDomainError(ValueError):
    """Base exception for all portfolio domain errors."""
    pass


class InsufficientCashError(PortfolioDomainError):
    """Raised when available cash is insufficient for a buy trade or withdrawal."""
    pass


class InvalidTradeError(PortfolioDomainError):
    """Raised when trade parameters are logically invalid."""
    pass


class HoldingNotFoundError(PortfolioDomainError):
    """Raised when a specified holding does not exist in the portfolio."""
    pass


class PortfolioNotFoundError(PortfolioDomainError):
    """Raised when the specified portfolio ID does not exist."""
    pass


class RecoveryConflictError(PortfolioDomainError):
    """Raised when multi-file commit recovery encounters unresolvable hash divergence on disk."""
    pass


class StagedScanExpiredError(PortfolioDomainError):
    """Raised when access to a staged scan batch has expired (TTL exceeded)."""
    pass


class StagedScanForbiddenError(PortfolioDomainError):
    """Raised when a session attempts to access staged data belonging to another session."""
    pass


class StagedScanNotFoundError(PortfolioDomainError):
    """Raised when a requested staged scan batch ID is not found."""
    pass


class TradeDuplicateError(PortfolioDomainError):
    """Raised when a trade matches an existing ledger row or duplicate within batch."""
    pass


class TradeReconciliationError(PortfolioDomainError):
    """Raised when a trade confirmation violates reconciliation invariants."""
    pass
