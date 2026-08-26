"""Domain and Application Exceptions for Earnings Call Context."""


class EarningsCallError(Exception):
    """Base exception for all earnings call domain errors."""


class EarningsCallValidationError(EarningsCallError):
    """Raised when input validation for earnings call summarization fails."""


class EarningsCallProviderUnavailableError(EarningsCallError):
    """Raised when external LLM or upstream provider is unreachable, timed out, or rate-limited."""


class EarningsCallProcessingError(EarningsCallError):
    """Raised when processing a step in the earnings call pipeline fails."""


class EarningsCallRunNotFoundError(EarningsCallError):
    """Raised when a requested run_id is not found in the workflow repository."""


class EarningsCallTickerMismatchError(EarningsCallError):
    """Raised when a run does not belong to the specified ticker."""


class EarningsCallLeaseExpiredError(EarningsCallError):
    """Raised when an execution or outbox lease has expired or was acquired by another caller."""
