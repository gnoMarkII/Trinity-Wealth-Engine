"""Application use case for dispatching one prerequisite equity research request."""
from __future__ import annotations

from typing import Any, Dict

from application.roundtable.ports import RoundtableResearchDispatchPort


class RoundtableResearchDispatchService:
    """Builds stable research identity and delegates the atomic persistence boundary."""

    def __init__(self, dispatcher: RoundtableResearchDispatchPort) -> None:
        self._dispatcher = dispatcher

    def dispatch_prerequisite_research(
        self,
        ticker: str,
        *,
        portfolio_id: str = "global",
        scope: str = "both",
    ) -> str:
        normalized_ticker = (ticker or "").strip().upper()
        normalized_portfolio = (portfolio_id or "global").strip().casefold() or "global"
        normalized_scope = (scope or "both").strip().casefold() or "both"
        if not normalized_ticker:
            raise ValueError("ticker is required for prerequisite research")

        research_type = "equity_intel"
        instruction = f"\u0e27\u0e34\u0e40\u0e04\u0e23\u0e32\u0e30\u0e2b\u0e4c\u0e2b\u0e38\u0e49\u0e19 {normalized_ticker} \u0e41\u0e1a\u0e1a\u0e25\u0e30\u0e40\u0e2d\u0e35\u0e22\u0e14"
        source_key = (
            f"roundtable:{research_type}:{normalized_ticker}:"
            f"portfolio:{normalized_portfolio}:scope:{normalized_scope}"
        )
        approval_metadata: Dict[str, Any] = {
            "ticker": normalized_ticker,
            "research_type": research_type,
            "portfolio_id": normalized_portfolio,
            "source_key": source_key,
        }
        return self._dispatcher.dispatch_approval_with_card(
            source_key=source_key,
            title=instruction,
            instruction=instruction,
            flow="manager",
            scope=normalized_scope,
            approval_metadata=approval_metadata,
        )
