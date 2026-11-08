"""Hexagonal Ports for Structured Thesis & As-Of Research State.

Architecture Rule (Layer 2 - Application):
Depends only on Core Domain and standard library Protocols.
Zero imports from infrastructure, SQLite, or API layers.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, Sequence

from core.thesis.models import (
    ThesisClaimRecord,
    ThesisExceptionRow,
)


class ThesisStorePort(Protocol):
    """Port for persisting and loading append-only thesis claim records."""

    def save_claim(self, claim: ThesisClaimRecord) -> None:
        """Saves a verified thesis claim record."""
        ...

    def get_active_claim(self, ticker: str) -> Optional[ThesisClaimRecord]:
        """Loads latest active claim for the ticker."""
        ...

    def load_claims(self, ticker: str) -> List[ThesisClaimRecord]:
        """Loads all historical claims for the ticker."""
        ...

    def list_all_active_claims(self) -> List[ThesisClaimRecord]:
        """Lists active claims across all tracked tickers."""
        ...


class ThesisExceptionStorePort(Protocol):
    """Port for persisting and managing watchdog exceptions."""

    def insert_exception(self, exc: ThesisExceptionRow) -> None:
        ...

    def list_pending_exceptions(self, ticker: Optional[str] = None, limit: int = 50) -> List[ThesisExceptionRow]:
        ...

    def acknowledge_exception(self, exception_id: str) -> None:
        ...


class MarketObservationPort(Protocol):
    """Port for fetching latest point-in-time metrics for watchdog verification."""

    def get_financial_metric(self, ticker: str, metric_name: str) -> Optional[Decimal]:
        ...

    def get_upcoming_catalyst_date(self, ticker: str, catalyst_name: str) -> Optional[str]:
        ...

