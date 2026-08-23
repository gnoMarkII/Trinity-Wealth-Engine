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
