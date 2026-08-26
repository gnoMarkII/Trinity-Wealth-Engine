"""Centralized Domain Exception to HTTP Exception Mapping."""
from contextlib import contextmanager
from typing import Iterator
from fastapi import HTTPException
from filelock import Timeout
from pydantic import ValidationError

from tools.portfolio.domain.errors import (
    PortfolioNotFoundError,
    HoldingNotFoundError,
    InsufficientCashError,
    InvalidTradeError,
    RecoveryConflictError,
)


@contextmanager
def handle_domain_exceptions(timeout_detail: str = "Portfolio lock timeout") -> Iterator[None]:
    """Translate domain and infrastructure exceptions into structured HTTPExceptions."""
    try:
        yield
    except PortfolioNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HoldingNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InsufficientCashError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InvalidTradeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RecoveryConflictError as exc:
        raise HTTPException(status_code=500, detail=f"Database Recovery Conflict: {exc}") from exc
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail=f"Internal DTO validation error: {exc}") from exc
    except Timeout as exc:
        raise HTTPException(status_code=503, detail=timeout_detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
