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


class EarningsCallRunNotReadyError(EarningsCallError):
    """Raised when a run cannot be retried because its Saga state is not deliverable."""


class EarningsCallLeaseExpiredError(EarningsCallError):
    """Raised when an execution or outbox lease has expired or was acquired by another caller."""


class EarningsCallRunInProgressError(EarningsCallError):
    """Control-flow signal that another owner currently holds the run lease.

    Lease contention is an expected Saga state, not an internal server fault.
    The inbound adapter uses ``run_id`` to read the latest state and return
    ``202 Accepted`` without exposing lease tokens or infrastructure details.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"Earnings call run '{run_id}' is currently in progress")
